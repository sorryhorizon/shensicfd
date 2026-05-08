#!/usr/bin/env python3
"""
ShenSi-CFD V4 Multi-GPU Training (DDP + AMP + Gradient Accumulation)

Usage:
  CUDA_VISIBLE_DEVICES=5 torchrun --nproc_per_node=1 --master_port=29500 train_v4.py
  CUDA_VISIBLE_DEVICES=3,4,5 torchrun --nproc_per_node=3 --master_port=29500 train_v4.py --batch-size 16

Resume:
  CUDA_VISIBLE_DEVICES=5 torchrun --nproc_per_node=1 --master_port=29500 train_v4.py --resume checkpoints/latest_v4.pt

Changes from previous version:
1. AMP mixed precision training (2x faster, 50% less memory)
2. Gradient accumulation (effective larger batch without OOM)
3. Fixed signal handler deadlock (removed dist.barrier())
4. num_workers=4 for faster data loading
5. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
6. RMSGroupNorm memory optimization
7. Vertical decoder chunk_size=3 (was 9)
8. Terrain penalty uses DEM data
9. Disabled k_height_profile (questionable physics assumption)
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import sys
import time
import json
import signal
import argparse
import datetime
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.swin_unet_lite import create_lite_model
from src.data.fuxi_cfd_dataset import FuXiCFDDataset
from src.losses.enhanced_physics_loss import EnhancedPhysicsLoss, ProgressiveLossScheduler


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

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict):
        self.shadow = {k: v.clone() for k, v in state_dict.items()}


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
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=7200))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def train_v4():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint path for resuming')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=25, help='Early stopping patience')
    parser.add_argument('--grad-accum-steps', type=int, default=2, help='Gradient accumulation steps')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader num_workers')
    parser.add_argument('--no-amp', action='store_true', help='Disable AMP (force FP32)')
    args = parser.parse_args()

    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')
    is_main = (rank == 0)

    batch_size_per_gpu = args.batch_size
    epochs = args.epochs
    lr = args.lr
    warmup_epochs = 5
    weight_decay = 0.01
    patience = args.patience
    ema_decay = 0.999
    grad_clip = 1.0
    warmup_loss_epochs = 5
    grad_accum_steps = args.grad_accum_steps
    num_workers = args.num_workers
    use_amp = not args.no_amp

    ckpt_dir = Path('checkpoints')
    log_dir = Path('logs/train_v4')
    if is_main:
        ckpt_dir.mkdir(exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

    if is_main:
        print('\n' + '='*70)
        print('ShenSi-CFD V4 - DDP Multi-GPU Training')
        print('='*70)
        print(f'   World Size:       {world_size}')
        print(f'   Batch/GPU:        {batch_size_per_gpu}')
        print(f'   Grad Accum:       {grad_accum_steps}')
        print(f'   Effective Batch:  {batch_size_per_gpu * world_size * grad_accum_steps}')
        print(f'   Learning Rate:    {lr}')
        print(f'   Warmup Epochs:    {warmup_epochs}')
        print(f'   Epochs:           {epochs}')
        print(f'   AMP:              {use_amp}')
        print(f'   EMA Decay:        {ema_decay}')
        print(f'   num_workers:      {num_workers}')
        print(f'   Physics Loss:     After epoch {warmup_loss_epochs}')
        if args.resume:
            print(f'   Resuming from:    {args.resume}')
        print('='*70 + '\n')

    if is_main:
        print('Creating model...')
    model = create_lite_model(config={
        'base_channels': 64,
        'bottleneck_depth': 6,
        'window_size': (5, 5),
        'dropout': 0.1,
        'use_physics_constraint': False,
    })
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    raw_model = model.module

    if is_main:
        params = raw_model.get_num_params()
        print(f'   Parameters: {params["total"]:,} ({params["total_mb"]:.1f} MB)')

    if is_main:
        print('\nLoading datasets...')
    data_dir = '/mnt/sdata/jz/fuxi_cfd/dataset'

    train_dataset = FuXiCFDDataset(
        data_dir=data_dir, split='train', normalize=True, prefetch_to_memory=False,
    )
    val_dataset = FuXiCFDDataset(
        data_dir=data_dir, split='val', normalize=True, prefetch_to_memory=False,
    )

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size_per_gpu, sampler=train_sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size_per_gpu, sampler=val_sampler,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    output_mean = torch.from_numpy(train_dataset.stats['output_mean']).to(device)
    output_std = torch.from_numpy(train_dataset.stats['output_std']).to(device)
    input_mean = torch.from_numpy(train_dataset.stats['input_mean']).to(device)
    input_std = torch.from_numpy(train_dataset.stats['input_std']).to(device)

    if is_main:
        print(f'   output_mean: {train_dataset.stats["output_mean"]}')
        print(f'   output_std:  {train_dataset.stats["output_std"]}')
        print(f'   input_mean:  {train_dataset.stats["input_mean"]}')
        print(f'   input_std:   {train_dataset.stats["input_std"]}')

    warmup_criterion = nn.SmoothL1Loss(beta=1.0)

    physics_loss = EnhancedPhysicsLoss(
        mse_weight=1.0, l1_weight=0.5,
        mass_conservation_weight=0.05, boundary_layer_weight=0.02,
        terrain_penalty_weight=0.05, k_positive_weight=0.1,
        gradient_smoothness_weight=0.05,
        use_k_transform=True, k_specialized_weight=0.5,
        use_k_height_profile=False,
    ).to(device)

    loss_scheduler = ProgressiveLossScheduler(
        base_loss=physics_loss, n_stages=3, stage_epochs=[25, 35, 20],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
        betas=(0.9, 0.999), eps=1e-8,
    )
    lr_scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, epochs)
    ema = EMA(raw_model, decay=ema_decay)
    scaler = GradScaler(init_scale=2**14, enabled=use_amp)

    best_val_loss = float('inf')
    patience_counter = 0
    history = []
    nan_count = 0
    start_epoch = 0
    epoch_times = []

    if args.resume and os.path.exists(args.resume):
        if is_main:
            print(f'\nResuming from checkpoint: {args.resume}')
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        if 'ema_shadow' in ckpt:
            ema.load_state_dict(ckpt['ema_shadow'])
        if 'history' in ckpt:
            history = ckpt['history']
        if 'scaler_state' in ckpt and use_amp:
            scaler.load_state_dict(ckpt['scaler_state'])
        if is_main:
            print(f'   Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.6f}')

    if is_main:
        writer = SummaryWriter(log_dir=str(log_dir / 'tensorboard'))
    else:
        writer = None

    training_state = {'epoch': start_epoch - 1}

    def save_handler(signum, frame):
        if is_main:
            print('\nSaving checkpoint before exit...')
            torch.save({
                'epoch': training_state['epoch'] + 1,
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'ema_shadow': ema.state_dict(),
                'history': history,
                'scaler_state': scaler.state_dict(),
            }, ckpt_dir / 'latest_v4.pt')
            with open(log_dir / 'history.json', 'w') as f:
                json.dump(history, f, indent=2)
            print('Checkpoint saved. Exiting.')
        sys.exit(0)

    signal.signal(signal.SIGTERM, save_handler)
    signal.signal(signal.SIGINT, save_handler)

    if is_main:
        print(f'\nStarting training from epoch {start_epoch+1}...\n')

    for epoch in range(start_epoch, epochs):
        train_sampler.set_epoch(epoch)
        training_state['epoch'] = epoch

        epoch_start = time.time()
        current_lr = lr_scheduler.step(epoch)
        current_stage = loss_scheduler.get_current_stage(epoch) if epoch >= warmup_loss_epochs else -1

        model.train()
        train_loss = 0.0
        train_losses_dict = {}
        n_batches = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            inputs = batch['input'].to(device, non_blocking=True)
            targets = batch['target'].to(device, non_blocking=True)

            with autocast('cuda', enabled=use_amp):
                outputs = model(inputs)

                if epoch < warmup_loss_epochs:
                    loss = warmup_criterion(outputs, targets)
                    loss_dict = {'total': loss, 'mse': loss.detach()}
                else:
                    pred_phys = outputs * output_std.view(1, 1, 4, 1, 1) + output_mean.view(1, 1, 4, 1, 1)
                    target_phys = targets * output_std.view(1, 1, 4, 1, 1) + output_mean.view(1, 1, 4, 1, 1)

                    roughness_phys = inputs[:, 3:4] * input_std[3].view(1, 1, 1) + input_mean[3].view(1, 1, 1)
                    dem_phys = inputs[:, 2:3] * input_std[2].view(1, 1, 1) + input_mean[2].view(1, 1, 1)

                    with torch.amp.autocast('cuda', enabled=False):
                        pred_fp32 = pred_phys.float()
                        target_fp32 = target_phys.float()
                        roughness_fp32 = roughness_phys.float().unsqueeze(1)
                        dem_fp32 = dem_phys.float().unsqueeze(1)
                        loss_dict = loss_scheduler(
                            pred_fp32, target_fp32, epoch=epoch,
                            dem=dem_fp32,
                            roughness=roughness_fp32,
                            return_dict=True,
                        )
                    loss = loss_dict['total']

            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                if is_main and nan_count <= 20:
                    print(f'  NaN/Inf at batch {batch_idx}, skipping (total: {nan_count})')
                optimizer.zero_grad()
                continue

            loss_scaled = loss / grad_accum_steps
            scaler.scale(loss_scaled).backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update()

            train_loss += loss.item()
            for k, v in loss_dict.items():
                if k == 'current_stage':
                    continue
                try:
                    v_val = v.item() if isinstance(v, torch.Tensor) else float(v)
                    train_losses_dict[k] = train_losses_dict.get(k, 0) + v_val
                except (ValueError, RuntimeError):
                    pass

            n_batches += 1

            if is_main and (batch_idx + 1) % 50 == 0:
                mem_used = torch.cuda.max_memory_allocated(device) / 1024**3
                print(f'  [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f} GPU: {mem_used:.1f}GB')

        if n_batches % grad_accum_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            ema.update()

        train_loss /= max(n_batches, 1)
        for k in train_losses_dict:
            train_losses_dict[k] /= max(n_batches, 1)

        train_loss_tensor = torch.tensor([train_loss], device=device)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        train_loss = train_loss_tensor.item() / world_size

        torch.cuda.empty_cache()

        model.eval()
        val_loss = 0.0
        val_batches = 0

        ema.apply_shadow()

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(device, non_blocking=True)
                targets = batch['target'].to(device, non_blocking=True)

                with autocast('cuda', enabled=use_amp):
                    outputs = model(inputs)

                    if epoch < warmup_loss_epochs:
                        loss = warmup_criterion(outputs, targets)
                    else:
                        pred_phys = outputs * output_std.view(1, 1, 4, 1, 1) + output_mean.view(1, 1, 4, 1, 1)
                        target_phys = targets * output_std.view(1, 1, 4, 1, 1) + output_mean.view(1, 1, 4, 1, 1)
                        roughness_phys = inputs[:, 3:4] * input_std[3].view(1, 1, 1) + input_mean[3].view(1, 1, 1)
                        dem_phys = inputs[:, 2:3] * input_std[2].view(1, 1, 1) + input_mean[2].view(1, 1, 1)

                        with torch.amp.autocast('cuda', enabled=False):
                            pred_fp32 = pred_phys.float()
                            target_fp32 = target_phys.float()
                            roughness_fp32 = roughness_phys.float().unsqueeze(1)
                            dem_fp32 = dem_phys.float().unsqueeze(1)
                            loss_dict_val = loss_scheduler(
                                pred_fp32, target_fp32, epoch=epoch,
                                dem=dem_fp32,
                                roughness=roughness_fp32,
                                return_dict=True,
                            )
                        loss = loss_dict_val['total']

                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_loss += loss.item()
                    val_batches += 1

        val_loss /= max(val_batches, 1)

        val_loss_tensor = torch.tensor([val_loss], device=device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        val_loss = val_loss_tensor.item() / world_size

        ema.restore()

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        if is_main:
            record = {
                'epoch': epoch + 1,
                'train_loss': float(train_loss),
                'val_loss': float(val_loss),
                'lr': float(current_lr),
                'stage': int(current_stage) if current_stage >= 0 else -1,
                'time': float(epoch_time),
                'nan_count': nan_count,
                'gpu_mem_gb': torch.cuda.max_memory_allocated(device) / 1024**3,
            }
            for k, v in train_losses_dict.items():
                record[f'train_{k}'] = float(v)
            history.append(record)

            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('LR', current_lr, epoch)

            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            remaining_epochs = epochs - (epoch + 1)
            eta_hours = avg_epoch_time * remaining_epochs / 3600

            stage_str = f' [Stage {current_stage}]' if current_stage >= 0 else ' [Warmup]'
            print(f'\nEpoch {epoch+1}/{epochs}{stage_str}')
            print(f'   Train Loss: {train_loss:.6f}')
            print(f'   Val Loss:   {val_loss:.6f}')
            print(f'   LR:         {current_lr:.7f}')
            print(f'   Time:       {epoch_time:.1f}s')
            print(f'   ETA:        {eta_hours:.1f}h ({remaining_epochs} epochs left)')
            print(f'   NaN count:  {nan_count}')
            print(f'   GPU Peak:   {record["gpu_mem_gb"]:.1f}GB')

            for k in ['mse', 'log_k_mse', 'mass_conservation', 'boundary_layer', 'terrain', 'k_positive']:
                if k in train_losses_dict:
                    print(f'   {k}: {train_losses_dict[k]:.6f}')

        if is_main:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                ema.apply_shadow()
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': raw_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'train_loss': train_loss,
                    'ema_shadow': ema.state_dict(),
                    'output_mean': train_dataset.stats['output_mean'],
                    'output_std': train_dataset.stats['output_std'],
                    'input_mean': train_dataset.stats['input_mean'],
                    'input_std': train_dataset.stats['input_std'],
                    'history': history,
                    'scaler_state': scaler.state_dict(),
                }, ckpt_dir / 'best_model_v4.pt')
                ema.restore()
                print(f'   Best model saved! (val_loss: {val_loss:.6f})')
            else:
                patience_counter += 1
                print(f'   Patience: {patience_counter}/{patience}')

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'ema_shadow': ema.state_dict(),
                'history': history,
                'scaler_state': scaler.state_dict(),
            }, ckpt_dir / 'latest_v4.pt')

            with open(log_dir / 'history.json', 'w') as f:
                json.dump(history, f, indent=2)

        dist.barrier()

        patience_tensor = torch.tensor([patience_counter], device=device)
        dist.broadcast(patience_tensor, src=0)
        patience_counter = patience_tensor.item()

        if patience_counter >= patience:
            if is_main:
                print(f'\nEarly stopping triggered!')
            break

    if is_main:
        writer.close()
        print('\n' + '='*70)
        print('Training complete!')
        print(f'   Best val loss: {best_val_loss:.6f}')
        print(f'   History: {log_dir / "history.json"}')
        print(f'   Best model: {ckpt_dir / "best_model_v4.pt"}')
        print('='*70)

    cleanup_ddp()


if __name__ == '__main__':
    train_v4()
