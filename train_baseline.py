#!/usr/bin/env python3
"""
ShenSi-CFD Baseline 训练脚本（无物理约束）

特点：
1. 纯数据驱动 - 只用 MSE Loss
2. 无物理约束层 (PhysicsConstraintLayer 关闭)
3. 无物理损失项 (mass_conservation, terrain, boundary_layer 等)
4. 保留 AMP + EMA + Multi-GPU 基础设施以便公平对比
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
    """带warmup的余弦退火学习率调度器"""
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


def train_baseline():
    # === 配置 ===
    gpu_ids = [2, 3, 4, 5]
    n_gpus = len(gpu_ids)
    batch_size_per_gpu = 4
    batch_size = batch_size_per_gpu * n_gpus
    accum_steps = 2
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
    log_dir = Path('logs/train_baseline')
    log_dir.mkdir(parents=True, exist_ok=True)

    print('\n' + '='*70)
    print('ShenSi-CFD Baseline - Data-Driven Training (No Physics)')
    print('='*70)
    print(f'   GPU: {torch.cuda.get_device_name(0)} x{n_gpus}')
    print(f'   Batch Size: {batch_size} ({batch_size_per_gpu}/GPU, 等效: {batch_size * accum_steps})')
    print(f'   Learning Rate: {lr}')
    print(f'   Epochs: {epochs}')
    print(f'   AMP: Enabled')
    print(f'   EMA Decay: {ema_decay}')
    print(f'   Loss: MSELoss only')
    print(f'   PhysicsConstraint: DISABLED')
    print('='*70 + '\n')

    # === 数据 ===
    print('Loading datasets...')
    data_dir = '/mnt/sdata/jz/fuxi_cfd/dataset'

    train_loader, val_loader, _ = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=4,
        prefetch_to_memory=False,
        pin_memory=True,
    )

    # === 模型（关闭物理约束）===
    print('Creating model...')
    model = create_lite_model(config={
        'base_channels': 64,
        'bottleneck_depth': 6,
        'window_size': (5, 5),
        'dropout': 0.1,
        'use_physics_constraint': False,  # 关键：关闭物理约束层
    })
    model = model.to(device)

    if n_gpus > 1:
        model = nn.DataParallel(model)
        print(f'   Using {n_gpus} GPUs: {gpu_ids}')

    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    params = raw_model.get_num_params()
    print(f'   Parameters: {params["total"]:,} ({params["total_mb"]:.1f} MB)')

    # === 损失函数（纯MSE）===
    criterion = nn.MSELoss()

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

    resume_ckpt = ckpt_dir / 'best_model_baseline.pt'
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
        best_val_loss = ckpt.get('val_loss', float('inf'))
        print(f'   Resumed from epoch {start_epoch}')

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        current_lr = lr_scheduler.step(epoch)

        # --- Train ---
        model.train()
        train_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            inputs = batch['input'].to(device, non_blocking=True)
            targets = batch['target'].to(device, non_blocking=True)

            outputs = model(inputs)
            outputs = torch.clamp(outputs, min=-10.0, max=10.0)

            with autocast('cuda'):
                loss = criterion(outputs, targets)

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

            train_loss += loss.item() * accum_steps
            n_batches += 1

            if (batch_idx + 1) % 100 == 0:
                print(f'  [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}')

        train_loss /= max(n_batches, 1)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_batches = 0

        ema.apply_shadow()

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(device, non_blocking=True)
                targets = batch['target'].to(device, non_blocking=True)

                outputs = model(inputs)
                outputs = torch.clamp(outputs, min=-10.0, max=10.0)

                with autocast('cuda'):
                    loss = criterion(outputs, targets)

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
            'time': float(epoch_time),
        }
        history.append(record)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LR', current_lr, epoch)

        print(f'\nEpoch {epoch+1}/{epochs}')
        print(f'   Train Loss: {train_loss:.6f}')
        print(f'   Val Loss:   {val_loss:.6f}')
        print(f'   LR:         {current_lr:.7f}')
        print(f'   Time:       {epoch_time:.1f}s')

        # --- Save best model ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            ema.apply_shadow()
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'ema_shadow': ema.shadow,
            }, ckpt_dir / 'best_model_baseline.pt')
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
            }, ckpt_dir / f'baseline_epoch{epoch+1}.pt')

            with open(log_dir / 'history.json', 'w') as f:
                json.dump(history, f, indent=2)

    # --- Final save ---
    with open(log_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    writer.close()

    print('\n' + '='*70)
    print('Baseline Training complete!')
    print(f'   Best val loss: {best_val_loss:.6f}')
    print(f'   History saved: {log_dir / "history.json"}')
    print(f'   Best model: {ckpt_dir / "best_model_baseline.pt"}')
    print('='*70)


if __name__ == '__main__':
    train_baseline()
