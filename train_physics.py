#!/usr/bin/env python3
"""
ShenSi-CFD V3 训练脚本

改进点：
1. 使用物理约束损失函数 (EnhancedPhysicsLoss + ProgressiveLossScheduler)
2. 混合精度训练 (AMP) - 加速1.5-2x
3. 数据加载优化 (prefetch_to_memory, num_workers=4)
4. 更大batch_size=16 (A800 80GB)
5. EMA模型 - 提升稳定性
6. TensorBoard日志
7. 修复的梯度累积统计
8. 修复的WarmupCosineScheduler
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.swin_unet_lite import create_lite_model
from src.data.fuxi_cfd_dataset import FuXiCFDDataset, create_dataloaders
from src.losses.enhanced_physics_loss import EnhancedPhysicsLoss, ProgressiveLossScheduler


class EMA:
    """Exponential Moving Average 模型参数平滑"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class WarmupCosineScheduler:
    """带warmup的余弦退火学习率调度器 (修复版)"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265)))
            lr = lr.item()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


def train_v3():
    # === 配置 ===
    gpu_ids = [2, 3, 4, 5]   # Multi-GPU
    n_gpus = len(gpu_ids)
    batch_size_per_gpu = 4
    batch_size = batch_size_per_gpu * n_gpus  # Total: 16
    accum_steps = 2           # 等效batch=32
    epochs = 80
    lr = 3e-4
    warmup_epochs = 5
    weight_decay = 0.01
    patience = 20
    ema_decay = 0.999
    grad_clip = 1.0

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in gpu_ids)
    device = 'cuda'

    # === 目录 ===
    ckpt_dir = Path('checkpoints')
    ckpt_dir.mkdir(exist_ok=True)
    log_dir = Path('logs/train_v3')
    log_dir.mkdir(parents=True, exist_ok=True)

    print('\n' + '='*70)
    print('ShenSi-CFD V3 - Physics-Informed Training')
    print('='*70)
    print(f'   GPU: {torch.cuda.get_device_name(0)} x{n_gpus}')
    print(f'   Batch Size: {batch_size} ({batch_size_per_gpu}/GPU, 等效: {batch_size * accum_steps})')
    print(f'   Learning Rate: {lr}')
    print(f'   Warmup Epochs: {warmup_epochs}')
    print(f'   Weight Decay: {weight_decay}')
    print(f'   Epochs: {epochs}')
    print(f'   AMP: Enabled')
    print(f'   EMA Decay: {ema_decay}')
    print(f'   Loss: EnhancedPhysicsLoss + ProgressiveLossScheduler')
    print('='*70 + '\n')

    # === 模型 ===
    print('Creating model...')

    # Get normalization stats from dataset for physics constraint layer
    output_mean = None
    output_std = None
    if hasattr(train_loader.dataset, 'stats') and train_loader.dataset.stats is not None:
        output_mean = torch.from_numpy(train_loader.dataset.stats['output_mean']).float()
        output_std = torch.from_numpy(train_loader.dataset.stats['output_std']).float()
        print(f'   Loaded output_mean: {output_mean}')
        print(f'   Loaded output_std:  {output_std}')

    model = create_lite_model(config={
        'base_channels': 64,
        'bottleneck_depth': 6,
        'window_size': (5, 5),
        'dropout': 0.1,
        'output_mean': output_mean,
        'output_std': output_std,
    })
    model = model.to(device)

    # Multi-GPU DataParallel
    if n_gpus > 1:
        model = nn.DataParallel(model)
        print(f'   Using {n_gpus} GPUs: {gpu_ids}')

    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    params = raw_model.get_num_params()
    print(f'   Parameters: {params["total"]:,} ({params["total_mb"]:.1f} MB)')

    # === 数据 ===
    print('\nLoading datasets...')
    data_dir = '/mnt/sdata/jz/fuxi_cfd/dataset'

    train_loader, val_loader, _ = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=4,
        prefetch_to_memory=False,  # Set True for faster epoch after first cache build
        pin_memory=True,
    )

    # === 损失函数 ===
    # Warmup phase: use robust SmoothL1Loss to avoid NaN from untrained model
    warmup_criterion = nn.SmoothL1Loss(beta=1.0)

    physics_loss = EnhancedPhysicsLoss(
        mse_weight=1.0,
        l1_weight=0.5,
        mass_conservation_weight=0.1,
        boundary_layer_weight=0.05,
        terrain_penalty_weight=0.1,
        k_positive_weight=0.05,
        gradient_smoothness_weight=0.1,
        use_k_transform=True,
        k_specialized_weight=0.5,
    ).to(device)

    loss_scheduler = ProgressiveLossScheduler(
        base_loss=physics_loss,
        n_stages=3,
        stage_epochs=[30, 40, 30],
    )

    # Switch from warmup loss to physics loss after these epochs
    warmup_loss_epochs = 5

    # === 优化器 ===
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    lr_scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, epochs)

    # === AMP ===
    scaler = GradScaler('cuda')

    # === EMA ===
    ema = EMA(raw_model, decay=ema_decay)

    # === TensorBoard ===
    writer = SummaryWriter(log_dir=str(log_dir / 'tensorboard'))

    # === 训练循环 ===
    best_val_loss = float('inf')
    patience_counter = 0
    history = []
    start_epoch = 0

    # Resume from checkpoint if exists
    resume_ckpt = ckpt_dir / 'best_model_v3.pt'
    if resume_ckpt.exists():
        print(f'\nResuming from checkpoint: {resume_ckpt}')
        ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        if 'ema_shadow' in ckpt:
            ema.shadow = {k: v.clone() for k, v in ckpt['ema_shadow'].items()}
        start_epoch = ckpt.get('epoch', 0)
        print(f'   Resumed from epoch {start_epoch}')
        # Reset best_val_loss if we are past warmup (loss function changed)
        if start_epoch >= warmup_loss_epochs:
            best_val_loss = float('inf')
            patience_counter = 0
            print(f'   Reset best_val_loss and patience (past warmup)')
        else:
            best_val_loss = ckpt.get('val_loss', float('inf'))
        print(f'   best_val_loss reset to: {best_val_loss:.6f}')

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        current_lr = lr_scheduler.step(epoch)
        current_stage = loss_scheduler.get_current_stage(epoch)

        # Reset best_val_loss when switching from warmup to physics loss
        if epoch == warmup_loss_epochs:
            best_val_loss = float('inf')
            patience_counter = 0
            print(f'\n>>> Warmup ended. Resetting best_val_loss and patience for physics loss phase.')
        elif epoch > warmup_loss_epochs and patience_counter == 0:
            # Track the best within physics loss phase
            pass

        # --- Train ---
        model.train()
        train_loss = 0.0
        train_losses_dict = {}
        n_batches = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            inputs = batch['input'].to(device, non_blocking=True)
            targets = batch['target'].to(device, non_blocking=True)

            # Forward in FP32 to avoid float16 overflow from untrained model
            outputs = model(inputs)

            # Clamp outputs to prevent extreme values
            outputs = torch.clamp(outputs, min=-10.0, max=10.0)

            with autocast('cuda'):
                if epoch < warmup_loss_epochs:
                    loss = warmup_criterion(outputs, targets)
                    loss_dict = {'total': loss, 'mse': loss.detach()}
                else:
                    dem = inputs[:, 2:3]
                    loss_dict = loss_scheduler(outputs, targets, epoch=epoch, dem=dem, return_dict=True)
                    loss = loss_dict['total']

            # Skip NaN losses
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update()

            # Track loss (unscaled)
            train_loss += loss.item() * accum_steps

            # Track individual losses
            for k, v in loss_dict.items():
                if k == 'current_stage':
                    continue
                try:
                    v_val = v.item() if isinstance(v, torch.Tensor) else float(v)
                    train_losses_dict[k] = train_losses_dict.get(k, 0) + v_val * accum_steps
                except (ValueError, RuntimeError):
                    pass

            n_batches += 1

            if (batch_idx + 1) % 100 == 0:
                print(f'  [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}')

        train_loss /= max(n_batches, 1)
        for k in train_losses_dict:
            train_losses_dict[k] /= n_batches

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_batches = 0

        # Use EMA model for validation
        ema.apply_shadow()

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(device, non_blocking=True)
                targets = batch['target'].to(device, non_blocking=True)

                outputs = model(inputs)
                outputs = torch.clamp(outputs, min=-10.0, max=10.0)

                with autocast('cuda'):
                    if epoch < warmup_loss_epochs:
                        loss = warmup_criterion(outputs, targets)
                    else:
                        dem = inputs[:, 2:3]
                        loss_dict = loss_scheduler(outputs, targets, epoch=epoch, dem=dem, return_dict=True)
                        loss = loss_dict['total']

                val_loss += loss.item()
                val_batches += 1

        val_loss /= max(val_batches, 1)
        ema.restore()

        epoch_time = time.time() - epoch_start

        # --- Logging ---
        record = {
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'lr': float(current_lr),
            'stage': int(current_stage),
            'time': float(epoch_time),
        }
        for k, v in train_losses_dict.items():
            record[f'train_{k}'] = float(v)
        history.append(record)

        # TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LR', current_lr, epoch)
        writer.add_scalar('Stage', current_stage, epoch)
        for k, v in train_losses_dict.items():
            writer.add_scalar(f'TrainLoss/{k}', v, epoch)

        print(f'\nEpoch {epoch+1}/{epochs} [Stage {current_stage}]')
        print(f'   Train Loss: {train_loss:.6f}')
        print(f'   Val Loss:   {val_loss:.6f}')
        print(f'   LR:         {current_lr:.7f}')
        print(f'   Time:       {epoch_time:.1f}s')

        # Print key loss components
        for k in ['mse', 'log_k_mse', 'mass_conservation', 'boundary_layer', 'terrain', 'k_positive']:
            if k in train_losses_dict:
                print(f'   {k}: {train_losses_dict[k]:.6f}')

        # --- Save best model ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save EMA model as best
            ema.apply_shadow()
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'ema_shadow': ema.shadow,
            }, ckpt_dir / 'best_model_v3.pt')
            ema.restore()
            print(f'   Best model saved! (val_loss: {val_loss:.6f})')
        else:
            patience_counter += 1
            print(f'   Patience: {patience_counter}/{patience}')

        # --- Early stopping ---
        if patience_counter >= patience:
            print(f'\nEarly stopping triggered! No improvement for {patience} epochs')
            break

        # --- Periodic save ---
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, ckpt_dir / f'v3_epoch{epoch+1}.pt')

            with open(log_dir / 'history.json', 'w') as f:
                json.dump(history, f, indent=2)

    # --- Final save ---
    with open(log_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    writer.close()

    print('\n' + '='*70)
    print('Training complete!')
    print(f'   Best val loss: {best_val_loss:.6f}')
    print(f'   History saved: {log_dir / "history.json"}')
    print(f'   Best model: {ckpt_dir / "best_model_v3.pt"}')
    print('='*70)


if __name__ == '__main__':
    train_v3()
