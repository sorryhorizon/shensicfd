#!/usr/bin/env python3
"""
ShenSi-CFD Physics-Informed Training (DDP Version)

Usage:
  torchrun --nproc_per_node=4 --master_port=29500 train_physics_ddp.py
  torchrun --nproc_per_node=4 --master_port=29500 train_physics_ddp.py --resume --batch-size 8 --epochs 80
"""

import os
import sys
import time
import json
import signal
import argparse
import torch
import torch.nn as nn
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
from src.losses.enhanced_physics_loss import EnhancedPhysicsLoss, PhysicsLossWarmupScheduler


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


def parse_args():
    parser = argparse.ArgumentParser(description='ShenSi-CFD DDP Training')
    parser.add_argument('--resume', action='store_true', help='Resume from latest_ddp.pt')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=80, help='Total training epochs')
    parser.add_argument('--num-workers', type=int, default=0, help='DataLoader num_workers')
    return parser.parse_args()


def train():
    args = parse_args()
    local_rank = setup_ddp()
    is_main = local_rank == 0

    batch_size = args.batch_size
    accum_steps = 2
    epochs = args.epochs
    lr = 3e-4
    warmup_epochs = 5
    weight_decay = 0.05
    patience = 15
    ema_decay = 0.999
    grad_clip = 1.0
    device = torch.device(f'cuda:{local_rank}')

    ckpt_dir = Path('checkpoints')
    log_dir = Path('logs/train_physics')
    if is_main:
        ckpt_dir.mkdir(exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

    if is_main:
        print('\n' + '='*70)
        print('ShenSi-CFD Physics-Informed Training (DDP)')
        print('='*70)
        print(f'   Local Rank: {local_rank}')
        print(f'   Batch Size: {batch_size}/GPU, Effective: {batch_size * accum_steps * dist.get_world_size()}')
        print(f'   Epochs: {epochs}')
        print(f'   Resume: {args.resume}')
        print('='*70 + '\n')

    data_dir = '/mnt/sdata/jz/fuxi_cfd/dataset'
    train_dataset = FuXiCFDDataset(data_dir, split='train', normalize=True, prefetch_to_memory=False,
                                   transform=RandomFlipTransform(p_horizontal=0.5, p_vertical=0.5))
    val_dataset = FuXiCFDDataset(data_dir, split='val', normalize=True, prefetch_to_memory=False)

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler,
                            num_workers=args.num_workers, pin_memory=True)

    output_mean = None
    output_std = None
    input_mean = None
    input_std = None
    if hasattr(train_dataset, 'stats') and train_dataset.stats is not None:
        output_mean = torch.from_numpy(train_dataset.stats['output_mean']).float()
        output_std = torch.from_numpy(train_dataset.stats['output_std']).float()
        input_mean = torch.from_numpy(train_dataset.stats['input_mean']).float()
        input_std = torch.from_numpy(train_dataset.stats['input_std']).float()

    model = create_lite_model(config={
        'base_channels': 48,
        'bottleneck_depth': 4,
        'num_heads': 4,
        'window_size': (5, 5),
        'dropout': 0.2,
        'drop_path_rate': 0.1,
        'output_mean': output_mean,
        'output_std': output_std,
    }).to(device)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    raw_model = model.module

    params = raw_model.get_num_params()
    if is_main:
        print(f'   Parameters: {params["total"]:,} ({params["total_mb"]:.1f} MB)\n')

    warmup_criterion = nn.SmoothL1Loss(beta=1.0)
    physics_loss = EnhancedPhysicsLoss(
        mse_weight=2.0, l1_weight=0.3,
        mass_conservation_weight=0.001,
        boundary_layer_weight=0.0,
        terrain_penalty_weight=0.0,
        k_positive_weight=0.1,
        gradient_smoothness_weight=0.02,
        use_k_transform=True,
        k_specialized_weight=0.5,
        use_k_height_profile=False,
    ).to(device)
    loss_scheduler = PhysicsLossWarmupScheduler(physics_loss, warmup_epochs=10, start_epoch=5)
    warmup_loss_epochs = 5

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8)
    lr_scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, epochs)
    scaler = GradScaler('cuda')
    ema = EMA(raw_model, decay=ema_decay) if is_main else None

    writer = SummaryWriter(log_dir=str(log_dir / 'tensorboard')) if is_main else None

    best_val_loss = float('inf')
    patience_counter = 0
    history = []
    start_epoch = 0
    epoch_times = []
    nan_count = 0

    training_state = {'epoch': start_epoch - 1}

    def save_handler(signum, frame):
        if is_main:
            print('\nSaving checkpoint before exit...')
            torch.save({
                'epoch': training_state['epoch'] + 1,
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_val_loss': best_val_loss,
                'train_loss': 0.0,
                'ema_shadow': ema.shadow if ema else {},
                'output_mean': train_dataset.stats['output_mean'],
                'output_std': train_dataset.stats['output_std'],
                'input_mean': train_dataset.stats['input_mean'],
                'input_std': train_dataset.stats['input_std'],
            }, ckpt_dir / 'latest_ddp.pt')
            sys.exit(0)

    signal.signal(signal.SIGTERM, save_handler)
    signal.signal(signal.SIGINT, save_handler)

    resume_path = None
    if args.resume:
        resume_path = ckpt_dir / 'latest_ddp.pt'

    if resume_path is not None and resume_path.exists():
        if is_main:
            try:
                ckpt = torch.load(resume_path, map_location=device, weights_only=False)
                raw_model.load_state_dict(ckpt['model_state_dict'], strict=False)
                print(f'   Resumed from {resume_path} (epoch {ckpt.get("epoch", 0)})')
            except Exception as e:
                print(f'   Resume failed (incompatible checkpoint): {e}, starting fresh')
        if dist.is_initialized():
            dist.barrier()
        if not is_main:
            try:
                ckpt = torch.load(resume_path, map_location=device, weights_only=False)
                raw_model.load_state_dict(ckpt['model_state_dict'], strict=False)
            except Exception:
                pass

    if is_main:
        print('Starting training loop...')

    for epoch in range(start_epoch, epochs):
        train_sampler.set_epoch(epoch)
        epoch_start = time.time()
        current_lr = lr_scheduler.step(epoch)
        if epoch == warmup_loss_epochs and is_main:
            best_val_loss = float('inf')
            patience_counter = 0

        model.train()
        train_loss = 0.0
        train_losses_dict = {}
        n_batches = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            inputs = batch['input'].to(device, non_blocking=True)
            targets = batch['target'].to(device, non_blocking=True)

            outputs = model(inputs)

            if epoch < warmup_loss_epochs:
                loss = warmup_criterion(outputs, targets)
                loss_dict = {'total': loss, 'mse': loss.detach()}
            else:
                pred_phys = outputs * output_std.view(1, 1, 4, 1, 1) + output_mean.view(1, 1, 4, 1, 1)
                target_phys = targets * output_std.view(1, 1, 4, 1, 1) + output_mean.view(1, 1, 4, 1, 1)
                dem_phys = inputs[:, 2:3] * input_std[2].view(1, 1, 1) + input_mean[2].view(1, 1, 1)
                roughness_phys = inputs[:, 3:4] * input_std[3].view(1, 1, 1) + input_mean[3].view(1, 1, 1)

                with torch.amp.autocast('cuda', enabled=False):
                    loss_dict = loss_scheduler(
                        pred_phys.float(), target_phys.float(), epoch=epoch,
                        dem=dem_phys.float().unsqueeze(1),
                        roughness=roughness_phys.float().unsqueeze(1),
                        return_dict=True,
                    )
                loss = loss_dict['total']

            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                if is_main and nan_count <= 20:
                    print(f'  NaN/Inf at batch {batch_idx}, skipping (total: {nan_count})')
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
            for k, v in loss_dict.items():
                v_val = v.item() if isinstance(v, torch.Tensor) else float(v)
                train_losses_dict[k] = train_losses_dict.get(k, 0) + v_val
            n_batches += 1

            if is_main and (batch_idx + 1) % 100 == 0:
                print(f'  [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}')

        train_loss /= max(n_batches, 1)
        for k in train_losses_dict:
            train_losses_dict[k] /= n_batches

        train_tensor = torch.tensor([train_loss], device=device)
        dist.all_reduce(train_tensor, op=dist.ReduceOp.AVG)
        train_loss = train_tensor.item()

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

                if epoch < warmup_loss_epochs:
                    loss = warmup_criterion(outputs, targets)
                else:
                    pred_phys = outputs * output_std.view(1, 1, 4, 1, 1) + output_mean.view(1, 1, 4, 1, 1)
                    target_phys = targets * output_std.view(1, 1, 4, 1, 1) + output_mean.view(1, 1, 4, 1, 1)
                    dem_phys = inputs[:, 2:3] * input_std[2].view(1, 1, 1) + input_mean[2].view(1, 1, 1)
                    roughness_phys = inputs[:, 3:4] * input_std[3].view(1, 1, 1) + input_mean[3].view(1, 1, 1)

                    with torch.amp.autocast('cuda', enabled=False):
                        loss_dict = loss_scheduler(
                            pred_phys.float(), target_phys.float(), epoch=epoch,
                            dem=dem_phys.float().unsqueeze(1),
                            roughness=roughness_phys.float().unsqueeze(1),
                            return_dict=True,
                        )
                    loss = loss_dict['total']

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

        var_names = ['u', 'v', 'w', 'k']
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
        epoch_times.append(epoch_time)
        training_state['epoch'] = epoch

        if is_main:
            record = {
                'epoch': epoch + 1, 'train_loss': float(train_loss),
                'val_loss': float(val_loss), 'lr': float(current_lr),
                'time': float(epoch_time),
            }
            for k, v in train_losses_dict.items():
                record[f'train_{k}'] = float(v)
            history.append(record)

            if writer:
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('LR', current_lr, epoch)
                for k, v in train_losses_dict.items():
                    writer.add_scalar(f'TrainLoss/{k}', v, epoch)
                for name in var_names:
                    writer.add_scalar(f'ValR2/{name}', val_r2.get(name, 0.0), epoch)
                    writer.add_scalar(f'ValRMSE/{name}', val_rmse.get(name, 0.0), epoch)

            mem_used = torch.cuda.max_memory_allocated(device) / 1024**3
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            remaining = epochs - (epoch + 1)
            eta_hours = avg_epoch_time * remaining / 3600

            print(f'\nEpoch {epoch+1}/{epochs}')
            print(f'   Train Loss: {train_loss:.6f}')
            print(f'   Val Loss:   {val_loss:.6f}')
            print(f'   LR:         {current_lr:.7f}')
            print(f'   Time:       {epoch_time:.1f}s')
            print(f'   GPU Peak: {mem_used:.1f}GB, ETA: {eta_hours:.1f}h')
            r2_str = ', '.join(f'{n}={val_r2.get(n, 0.0):.3f}' for n in var_names)
            rmse_str = ', '.join(f'{n}={val_rmse.get(n, 0.0):.3f}' for n in var_names)
            print(f'   Val R2:     {r2_str}')
            print(f'   Val RMSE:   {rmse_str}')
            for name in var_names:
                record[f'val_r2_{name}'] = val_r2.get(name, 0.0)
                record[f'val_rmse_{name}'] = val_rmse.get(name, 0.0)

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
                    'best_val_loss': best_val_loss,
                    'train_loss': train_loss,
                    'ema_shadow': ema.shadow if ema else {},
                    'output_mean': train_dataset.stats['output_mean'],
                    'output_std': train_dataset.stats['output_std'],
                    'input_mean': train_dataset.stats['input_mean'],
                    'input_std': train_dataset.stats['input_std'],
                }, ckpt_dir / 'best_model_ddp.pt')
                if ema:
                    ema.restore()
                print(f'   Best model saved! (val_loss: {val_loss:.6f})')
            else:
                patience_counter += 1
                print(f'   Patience: {patience_counter}/{patience}')

            if patience_counter >= patience:
                print(f'\nEarly stopping triggered!')
                break

            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': raw_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss,
                    'train_loss': train_loss,
                    'ema_shadow': ema.shadow if ema else {},
                    'output_mean': train_dataset.stats['output_mean'],
                    'output_std': train_dataset.stats['output_std'],
                    'input_mean': train_dataset.stats['input_mean'],
                    'input_std': train_dataset.stats['input_std'],
                }, ckpt_dir / f'ddp_epoch{epoch+1}.pt')

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'train_loss': train_loss,
                'ema_shadow': ema.shadow if ema else {},
                'output_mean': train_dataset.stats['output_mean'],
                'output_std': train_dataset.stats['output_std'],
                'input_mean': train_dataset.stats['input_mean'],
                'input_std': train_dataset.stats['input_std'],
            }, ckpt_dir / 'latest_ddp.pt')

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
