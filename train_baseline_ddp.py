#!/usr/bin/env python3
"""
ShenSi-CFD Baseline Training (DDP Version)

纯数据驱动对照实验：
- 无物理约束层 (use_physics_constraint=False)
- 无物理损失项 (MSE only)
- 其他超参与 physics 版本完全一致，确保公平对比

Usage:
  torchrun --nproc_per_node=4 --master_port=29501 train_baseline_ddp.py
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.swin_unet_lite import create_lite_model
from src.data.fuxi_cfd_dataset import FuXiCFDDataset, RandomFlipTransform


class EMA:
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
                self.shadow[name] = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]

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


def setup_ddp():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def train():
    local_rank = setup_ddp()
    is_main = local_rank == 0

    batch_size = 4
    accum_steps = 2
    epochs = 80
    lr = 3e-4
    warmup_epochs = 5
    weight_decay = 0.05
    patience = 15
    ema_decay = 0.999
    grad_clip = 1.0
    device = torch.device(f'cuda:{local_rank}')

    ckpt_dir = Path('checkpoints')
    log_dir = Path('logs/train_baseline')
    if is_main:
        ckpt_dir.mkdir(exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

    if is_main:
        print('\n' + '='*70)
        print('ShenSi-CFD Baseline Training (DDP) - Data-Driven Only')
        print('='*70)
        print(f'   Local Rank: {local_rank}')
        print(f'   Batch Size: {batch_size}/GPU, Effective: {batch_size * accum_steps * dist.get_world_size()}')
        print(f'   Epochs: {epochs}')
        print('='*70 + '\n')

    data_dir = '/mnt/sdata/jz/fuxi_cfd/dataset'
    train_dataset = FuXiCFDDataset(data_dir, split='train', normalize=True, prefetch_to_memory=False,
                                   transform=RandomFlipTransform(p_horizontal=0.5, p_vertical=0.5))
    val_dataset = FuXiCFDDataset(data_dir, split='val', normalize=True, prefetch_to_memory=False)

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler,
                            num_workers=0, pin_memory=True, drop_last=False)

    output_mean = None
    output_std = None
    if hasattr(train_dataset, 'stats') and train_dataset.stats is not None:
        output_mean = torch.from_numpy(train_dataset.stats['output_mean']).float()
        output_std = torch.from_numpy(train_dataset.stats['output_std']).float()

    # C. 增大模型: base_channels 32 -> 48
    model = create_lite_model(config={
        'base_channels': 48,
        'bottleneck_depth': 4,
        'window_size': (5, 5),
        'dropout': 0.2,
        'drop_path_rate': 0.1,
        'use_physics_constraint': False,
        'output_mean': output_mean,
        'output_std': output_std,
    }).to(device)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    raw_model = model.module

    params = raw_model.get_num_params()
    if is_main:
        print(f'   Parameters: {params["total"]:,} ({params["total_mb"]:.1f} MB)\n')

    # A+B: 加权 MSE Loss（k 峰值加权 + w 地形边缘加权）
    class WeightedMSELoss(nn.Module):
        def __init__(self, k_threshold=2.0, k_weight=10.0, w_edge_weight=5.0, edge_slope_threshold=5.0):
            super().__init__()
            self.k_threshold = k_threshold
            self.k_weight = k_weight
            self.w_edge_weight = w_edge_weight
            self.edge_slope_threshold = edge_slope_threshold

            self.sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3)
            self.sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3)

        def forward(self, pred, target, dem=None):
            mse = (pred - target) ** 2
            weights = torch.ones_like(mse)

            # A. k 峰值加权: 对 k > threshold 的区域加权
            k_target = target[:, :, 3:4]  # (B, L, 1, H, W)
            k_mask = k_target > self.k_threshold
            k_weights = torch.where(k_mask, self.k_weight, 1.0)
            weights[:, :, 3:4] = k_weights

            # B. w 地形边缘加权
            if dem is not None:
                device = dem.device
                dtype = dem.dtype
                sobel_x = self.sobel_x.to(device=device, dtype=dtype)
                sobel_y = self.sobel_y.to(device=device, dtype=dtype)
                dz_dx = F.conv2d(dem, sobel_x, padding=1)
                dz_dy = F.conv2d(dem, sobel_y, padding=1)
                slope = torch.sqrt(dz_dx ** 2 + dz_dy ** 2 + 1e-6)  # (B, 1, H, W)
                edge_mask = slope > self.edge_slope_threshold  # (B, 1, H, W)
                edge_mask = edge_mask.unsqueeze(2)  # (B, 1, 1, H, W)
                w_weights = torch.where(edge_mask, self.w_edge_weight, 1.0)
                weights[:, :, 2:3] = w_weights

            return (weights * mse).mean()

    criterion = WeightedMSELoss(k_threshold=2.0, k_weight=10.0, w_edge_weight=5.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8)
    lr_scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, epochs)
    scaler = GradScaler('cuda')
    ema = EMA(raw_model, decay=ema_decay) if is_main else None

    writer = SummaryWriter(log_dir=str(log_dir / 'tensorboard')) if is_main else None

    best_val_loss = float('inf')
    patience_counter = 0
    history = []
    start_epoch = 0

    # NOTE: best_model_baseline.pt is 32-channel old model, not compatible with 48-channel
    # Do NOT resume from old checkpoint
    resume_ckpt = ckpt_dir / 'best_model_baseline_v2.pt'

    if resume_ckpt.exists():
        if is_main:
            print(f'Resuming from checkpoint: {resume_ckpt}')
        map_location = {'cuda:0': f'cuda:{local_rank}', 'cuda:1': f'cuda:{local_rank}',
                        'cuda:2': f'cuda:{local_rank}', 'cuda:3': f'cuda:{local_rank}'}
        checkpoint = torch.load(resume_ckpt, map_location=map_location)
        raw_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        if 'ema_shadow' in checkpoint and ema and checkpoint['ema_shadow']:
            ema.shadow = checkpoint['ema_shadow']
        if is_main:
            print(f'   Resumed from epoch {start_epoch}, val_loss: {best_val_loss:.6f}')

    dist.barrier()

    var_names = ['u', 'v', 'w', 'k']

    for epoch in range(start_epoch, epochs):
        train_sampler.set_epoch(epoch)
        epoch_start = time.time()
        current_lr = lr_scheduler.step(epoch)

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
                dem = inputs[:, 2:3]
                loss = criterion(outputs, targets, dem=dem)

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue

            scaler.scale(loss / accum_steps).backward()

            if (batch_idx + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update()

            train_loss += loss.item()
            n_batches += 1

            if is_main and (batch_idx + 1) % 100 == 0:
                print(f'  [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}')

        train_loss /= max(n_batches, 1)

        train_tensor = torch.tensor([train_loss], device=device)
        dist.all_reduce(train_tensor, op=dist.ReduceOp.AVG)
        train_loss = train_tensor.item()

        # Validate
        model.eval()
        val_loss = 0.0
        val_batches = 0
        val_sum_pred = torch.zeros(4, device=device)
        val_sum_target = torch.zeros(4, device=device)
        val_sum_pred_sq = torch.zeros(4, device=device)
        val_sum_target_sq = torch.zeros(4, device=device)
        val_sum_pred_target = torch.zeros(4, device=device)
        val_count = torch.zeros(1, device=device)
        if ema:
            ema.apply_shadow()

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(device, non_blocking=True)
                targets = batch['target'].to(device, non_blocking=True)
                outputs = model(inputs)
                outputs = torch.clamp(outputs, min=-10.0, max=10.0)

                with autocast('cuda'):
                    dem = inputs[:, 2:3]
                    loss = criterion(outputs, targets, dem=dem)

                val_loss += loss.item()
                val_batches += 1

                pred = outputs
                tgt = targets
                if hasattr(train_dataset, 'denormalize_output') and train_dataset.stats is not None:
                    pred = train_dataset.denormalize_output(outputs)
                    tgt = train_dataset.denormalize_output(targets)

                B, L, C, H, W = pred.shape
                n = B * L * H * W
                for c in range(C):
                    pc = pred[:, :, c]
                    tc = tgt[:, :, c]
                    val_sum_pred[c] += pc.sum()
                    val_sum_target[c] += tc.sum()
                    val_sum_pred_sq[c] += (pc ** 2).sum()
                    val_sum_target_sq[c] += (tc ** 2).sum()
                    val_sum_pred_target[c] += (pc * tc).sum()
                val_count += n

        val_loss /= max(val_batches, 1)
        if ema:
            ema.restore()

        val_tensor = torch.tensor([val_loss], device=device)
        dist.all_reduce(val_tensor, op=dist.ReduceOp.AVG)
        val_loss = val_tensor.item()

        for t in [val_sum_pred, val_sum_target, val_sum_pred_sq, val_sum_target_sq, val_sum_pred_target, val_count]:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

        val_r2 = {}
        val_rmse = {}
        for c in range(4):
            n = val_count.item()
            if n > 0:
                sp = val_sum_pred[c].item()
                st = val_sum_target[c].item()
                spsq = val_sum_pred_sq[c].item()
                stsq = val_sum_target_sq[c].item()
                spt = val_sum_pred_target[c].item()
                cov = n * spt - sp * st
                var_p = n * spsq - sp * sp
                var_t = n * stsq - st * st
                if var_p > 1e-12 and var_t > 1e-12:
                    r2 = (cov ** 2) / (var_p * var_t)
                    r2 = max(0.0, min(1.0, r2))
                else:
                    r2 = 0.0
                mse = (spsq + stsq - 2 * spt) / n
                rmse = mse ** 0.5
                val_r2[var_names[c]] = r2
                val_rmse[var_names[c]] = rmse

        epoch_time = time.time() - epoch_start

        if is_main:
            record = {
                'epoch': epoch + 1, 'train_loss': float(train_loss),
                'val_loss': float(val_loss), 'lr': float(current_lr),
                'time': float(epoch_time),
            }
            for name in var_names:
                record[f'val_r2_{name}'] = val_r2.get(name, 0.0)
                record[f'val_rmse_{name}'] = val_rmse.get(name, 0.0)
            history.append(record)

            if writer:
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('LR', current_lr, epoch)
                for name in var_names:
                    writer.add_scalar(f'ValR2/{name}', val_r2.get(name, 0.0), epoch)
                    writer.add_scalar(f'ValRMSE/{name}', val_rmse.get(name, 0.0), epoch)

            print(f'\nEpoch {epoch+1}/{epochs}')
            print(f'   Train Loss: {train_loss:.6f}')
            print(f'   Val Loss:   {val_loss:.6f}')
            print(f'   LR:         {current_lr:.7f}')
            print(f'   Time:       {epoch_time:.1f}s')
            r2_str = ', '.join(f'{n}={val_r2.get(n, 0.0):.3f}' for n in var_names)
            rmse_str = ', '.join(f'{n}={val_rmse.get(n, 0.0):.3f}' for n in var_names)
            print(f'   Val R2:     {r2_str}')
            print(f'   Val RMSE:   {rmse_str}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if ema:
                    ema.apply_shadow()
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': raw_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                    'ema_shadow': ema.shadow if ema else {},
                }, ckpt_dir / 'best_model_baseline_v2.pt')
                if ema:
                    ema.restore()
                print(f'   Best model saved! (val_loss: {val_loss:.6f})')
            else:
                patience_counter += 1
                print(f'   Patience: {patience_counter}/{patience}')

            if patience_counter >= patience:
                print(f'\nEarly stopping triggered!')
                break

            if (epoch + 1) % 20 == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': raw_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, ckpt_dir / f'baseline_epoch{epoch+1}.pt')

    if is_main:
        with open(log_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
        if writer:
            writer.close()
        print('\nTraining complete!')
        print(f'   Best val loss: {best_val_loss:.6f}')

    dist.barrier()
    cleanup_ddp()


if __name__ == '__main__':
    train()
