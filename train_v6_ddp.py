#!/usr/bin/env python3
"""
ShenSi-CFD V6 Training (DDP Version)

Architecture: Enhanced decoder + vertical smoother + per-level normalization
- Same Swin-UNet encoder as V5
- EnhancedVerticalDecoder: hidden_dim=192, 4-layer u/v heads, 5-layer w/k heads
- VerticalSmoother: 1D depthwise conv along height dimension
- Per-level normalization (27,4) mean/std

Loss: Charbonnier + Spectral + Spatially-weighted k + L1

Usage:
  NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=4 --master_port=29502 train_v6_ddp.py
  NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=4 --master_port=29502 train_v6_ddp.py --resume
"""

import os
import sys
import time
import json
import signal
import argparse

os.environ['PYTHONUNBUFFERED'] = '1'
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

from src.models.swin_unet_v6 import SwinUNetV6
from src.data.fuxi_cfd_dataset import FuXiCFDDataset, RandomFlipTransform


# ─── Loss Functions ───────────────────────────────────────────────────────────

class CharbonnierLoss(nn.Module):
    """Smooth L1 variant: sqrt((pred-tgt)^2 + eps^2) - eps, more robust to outliers than MSE."""
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2) - self.eps)


class SpectralLoss(nn.Module):
    """Frequency-domain loss: penalizes errors in spatial frequency structure."""
    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)
        return self.weight * torch.mean(torch.abs(pred_fft - target_fft))


class SpatiallyWeightedKLoss(nn.Module):
    """Weighted k loss: higher weight in regions with large terrain gradient."""
    def __init__(self, alpha=2.0, eps=1e-3):
        super().__init__()
        self.alpha = alpha
        self.charb = CharbonnierLoss(eps=eps)
        # Sobel filters for terrain gradient
        self.register_buffer('sobel_x', torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0)
        self.register_buffer('sobel_y', torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0)

    def forward(self, pred_k, target_k, dem):
        """pred_k: (B, 27, H, W), target_k: (B, 27, H, W), dem: (B, 1, H, W)"""
        dz_dx = F.conv2d(dem, self.sobel_x, padding=1)
        dz_dy = F.conv2d(dem, self.sobel_y, padding=1)
        grad_mag = torch.sqrt(dz_dx ** 2 + dz_dy ** 2 + 1e-8)
        # Normalize to [0, 1] range per sample
        grad_max = grad_mag.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        grad_norm = grad_mag / grad_max
        weight = 1.0 + self.alpha * grad_norm  # (B, 1, H, W)
        # Expand to match pred_k shape (B, 27, H, W)
        weight = weight.expand(-1, pred_k.shape[1], -1, -1)
        return torch.mean(weight * (torch.sqrt((pred_k - target_k) ** 2 + 1e-3 ** 2) - 1e-3))


# ─── Utilities ────────────────────────────────────────────────────────────────

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
    parser = argparse.ArgumentParser(description='ShenSi-CFD V6 DDP Training')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=40, help='Total training epochs')
    parser.add_argument('--num-workers', type=int, default=0, help='DataLoader num_workers')
    parser.add_argument('--charb-weight', type=float, default=1.0, help='Charbonnier loss weight')
    parser.add_argument('--l1-weight', type=float, default=0.3, help='L1 loss weight')
    parser.add_argument('--spectral-weight', type=float, default=0.01, help='Spectral loss weight')
    parser.add_argument('--k-weight', type=float, default=2.0, help='Additional k loss weight')
    parser.add_argument('--w-weight', type=float, default=1.5, help='Additional w loss weight')
    parser.add_argument('--k-spatial-alpha', type=float, default=2.0, help='Spatial weighting alpha for k')
    parser.add_argument('--accum-steps', type=int, default=2, help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--ckpt-dir', type=str, default=None, help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default=None, help='Tensorboard log directory')
    return parser.parse_args()


def train():
    args = parse_args()
    local_rank = setup_ddp()
    is_main = local_rank == 0

    batch_size = args.batch_size
    accum_steps = args.accum_steps
    epochs = args.epochs
    lr = args.lr
    warmup_epochs = 5
    weight_decay = 0.05
    patience = 20
    ema_decay = 0.999
    grad_clip = 1.0
    device = torch.device(f'cuda:{local_rank}')

    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else Path('checkpoints/shensiv6_main')
    log_dir = Path(args.log_dir) if args.log_dir else Path('logs/shensiv6_main')
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

    if is_main:
        print('\n' + '='*70)
        print('ShenSi-CFD V6 Training (DDP)')
        print('  Enhanced decoder (hidden_dim=192) + VerticalSmoother')
        print('  Per-level normalization (27,4)')
        print('  Loss: Charbonnier + Spectral + Spatially-weighted k + L1')
        print('='*70)
        print(f'   Local Rank: {local_rank}')
        print(f'   Batch Size: {batch_size}/GPU, Effective: {batch_size * accum_steps * dist.get_world_size()}')
        print(f'   Epochs: {epochs}')
        print(f'   Loss: charb={args.charb_weight}, l1={args.l1_weight}, '
              f'spectral={args.spectral_weight}, k_extra={args.k_weight}, w_extra={args.w_weight}')
        print(f'   k spatial alpha: {args.k_spatial_alpha}')
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

    # Build model with per-level normalization stats
    output_mean = None
    output_std = None
    if hasattr(train_dataset, 'stats') and train_dataset.stats is not None:
        output_mean = torch.from_numpy(train_dataset.stats['output_mean']).float()
        output_std = torch.from_numpy(train_dataset.stats['output_std']).float()
        if is_main:
            print(f'   Output norm shape: mean={output_mean.shape}, std={output_std.shape}')

    model = SwinUNetV6(
        in_channels=6,
        n_levels=27,
        base_channels=48,
        channel_multipliers=[1, 2, 4, 8],
        bottleneck_depth=4,
        num_heads=4,
        window_size=(5, 5),
        dropout=0.2,
        drop_path_rate=0.1,
        use_cross_attention=True,
        output_mean=output_mean,
        output_std=output_std,
    ).to(device)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    raw_model = model.module

    params = raw_model.get_num_params()
    if is_main:
        print(f'   Parameters: total={params["total"]:,}, trainable={params["trainable"]:,}')
        print(f'   Decoder: {params["decoder"]:,} ({params["decoder"]/params["total"]*100:.1f}%)')
        print(f'   C_μ initial: {raw_model.vertical_decoder.C_mu.item():.4f}')
        print(f'   VerticalSmoother alpha: {raw_model.vertical_decoder.vertical_smoother.alpha.item():.4f}\n')

    # Loss functions
    charb_loss = CharbonnierLoss(eps=1e-3).to(device)
    spectral_loss = SpectralLoss(weight=args.spectral_weight).to(device)
    weighted_k_loss = SpatiallyWeightedKLoss(alpha=args.k_spatial_alpha, eps=1e-3).to(device)
    l1_loss = nn.L1Loss()

    charb_w = args.charb_weight
    l1_w = args.l1_weight
    w_extra_w = args.w_weight
    k_extra_w = args.k_weight

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8)
    lr_scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, epochs)
    scaler = GradScaler('cuda')
    ema = EMA(raw_model, decay=ema_decay) if is_main else None

    writer = SummaryWriter(log_dir=str(log_dir / 'tensorboard')) if is_main else None

    log_file = None
    if is_main:
        log_file = open(log_dir / 'train.log', 'a')

    def log_print(msg):
        print(msg, flush=True)
        if log_file:
            log_file.write(msg + '\n')
            log_file.flush()

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
            }, ckpt_dir / 'latest_v6.pt')
            sys.exit(0)

    signal.signal(signal.SIGTERM, save_handler)
    signal.signal(signal.SIGINT, save_handler)

    resume_path = None
    if args.resume:
        resume_path = ckpt_dir / 'latest_v6.pt'

    if resume_path is not None and resume_path.exists():
        try:
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            raw_model.load_state_dict(ckpt['model_state_dict'], strict=False)
            if 'optimizer_state_dict' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scaler_state_dict' in ckpt:
                scaler.load_state_dict(ckpt['scaler_state_dict'])
            if 'best_val_loss' in ckpt:
                best_val_loss = ckpt['best_val_loss']
            if 'ema_shadow' in ckpt and ema is not None:
                ema.shadow = ckpt['ema_shadow']
            if 'epoch' in ckpt:
                start_epoch = ckpt['epoch']
            if is_main:
                print(f'   Resumed from {resume_path} (epoch {start_epoch})', flush=True)
        except Exception as e:
            if is_main:
                print(f'   Resume failed: {e}, starting fresh', flush=True)

    if is_main:
        print('Starting training loop...', flush=True)

    var_names = ['u', 'v', 'w', 'k']

    for epoch in range(start_epoch, epochs):
        train_sampler.set_epoch(epoch)
        epoch_start = time.time()
        current_lr = lr_scheduler.step(epoch)

        model.train()
        train_loss = 0.0
        train_losses_dict = {}
        n_batches = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            inputs = batch['input'].to(device, non_blocking=True)
            targets = batch['target'].to(device, non_blocking=True)

            outputs = model(inputs)  # (B, 27, 4, H, W)

            # ─── Charbonnier loss (per variable) ─────────────────────────
            u_charb = charb_loss(outputs[:, :, 0], targets[:, :, 0])
            v_charb = charb_loss(outputs[:, :, 1], targets[:, :, 1])
            w_charb = charb_loss(outputs[:, :, 2], targets[:, :, 2])
            k_charb = charb_loss(outputs[:, :, 3], targets[:, :, 3])

            charb_total = (u_charb + v_charb + w_charb * w_extra_w + k_charb * k_extra_w) * charb_w

            # ─── L1 loss (per variable) ──────────────────────────────────
            u_l1 = l1_loss(outputs[:, :, 0], targets[:, :, 0])
            v_l1 = l1_loss(outputs[:, :, 1], targets[:, :, 1])
            w_l1 = l1_loss(outputs[:, :, 2], targets[:, :, 2])
            k_l1 = l1_loss(outputs[:, :, 3], targets[:, :, 3])

            l1_total = (u_l1 + v_l1 + w_l1 * w_extra_w + k_l1 * k_extra_w) * l1_w

            # ─── Spectral loss (all variables) ───────────────────────────
            spec_u = spectral_loss(outputs[:, :, 0], targets[:, :, 0])
            spec_v = spectral_loss(outputs[:, :, 1], targets[:, :, 1])
            spec_w = spectral_loss(outputs[:, :, 2], targets[:, :, 2])
            spec_k = spectral_loss(outputs[:, :, 3], targets[:, :, 3])
            spec_total = spec_u + spec_v + spec_w + spec_k

            # ─── Spatially-weighted k loss ───────────────────────────────
            dem = inputs[:, 2:3]  # DEM channel
            k_weighted = weighted_k_loss(outputs[:, :, 3], targets[:, :, 3], dem) * k_extra_w

            # ─── Total loss ──────────────────────────────────────────────
            loss = charb_total + l1_total + spec_total + k_weighted

            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                if is_main and nan_count <= 20:
                    print(f'  NaN/Inf at batch {batch_idx}, skipping (total: {nan_count})', flush=True)
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
            train_losses_dict['charb'] = train_losses_dict.get('charb', 0) + charb_total.item()
            train_losses_dict['l1'] = train_losses_dict.get('l1', 0) + l1_total.item()
            train_losses_dict['spectral'] = train_losses_dict.get('spectral', 0) + spec_total.item()
            train_losses_dict['k_weighted'] = train_losses_dict.get('k_weighted', 0) + k_weighted.item()
            train_losses_dict['u_charb'] = train_losses_dict.get('u_charb', 0) + u_charb.item()
            train_losses_dict['v_charb'] = train_losses_dict.get('v_charb', 0) + v_charb.item()
            train_losses_dict['w_charb'] = train_losses_dict.get('w_charb', 0) + w_charb.item()
            train_losses_dict['k_charb'] = train_losses_dict.get('k_charb', 0) + k_charb.item()
            n_batches += 1

            if is_main and (batch_idx + 1) % 100 == 0:
                print(f'  [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f} '
                      f'(charb={charb_total.item():.4f}, spec={spec_total.item():.4f}, '
                      f'k_w={k_weighted.item():.4f})')

        train_loss /= max(n_batches, 1)
        for k in train_losses_dict:
            train_losses_dict[k] /= n_batches

        train_tensor = torch.tensor([train_loss], device=device)
        dist.all_reduce(train_tensor, op=dist.ReduceOp.AVG)
        train_loss = train_tensor.item()

        # ─── Validation ──────────────────────────────────────────────────
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

                outputs = raw_model(inputs)  # (B, 27, 4, H, W)

                # Validation loss (same components as training)
                u_charb = charb_loss(outputs[:, :, 0], targets[:, :, 0])
                v_charb = charb_loss(outputs[:, :, 1], targets[:, :, 1])
                w_charb = charb_loss(outputs[:, :, 2], targets[:, :, 2])
                k_charb = charb_loss(outputs[:, :, 3], targets[:, :, 3])
                charb_total = (u_charb + v_charb + w_charb * w_extra_w + k_charb * k_extra_w) * charb_w

                u_l1 = l1_loss(outputs[:, :, 0], targets[:, :, 0])
                v_l1 = l1_loss(outputs[:, :, 1], targets[:, :, 1])
                w_l1 = l1_loss(outputs[:, :, 2], targets[:, :, 2])
                k_l1 = l1_loss(outputs[:, :, 3], targets[:, :, 3])
                l1_total = (u_l1 + v_l1 + w_l1 * w_extra_w + k_l1 * k_extra_w) * l1_w

                spec_u = spectral_loss(outputs[:, :, 0], targets[:, :, 0])
                spec_v = spectral_loss(outputs[:, :, 1], targets[:, :, 1])
                spec_w = spectral_loss(outputs[:, :, 2], targets[:, :, 2])
                spec_k = spectral_loss(outputs[:, :, 3], targets[:, :, 3])
                spec_total = spec_u + spec_v + spec_w + spec_k

                dem = inputs[:, 2:3]
                k_weighted = weighted_k_loss(outputs[:, :, 3], targets[:, :, 3], dem) * k_extra_w

                val_loss_batch = charb_total + l1_total + spec_total + k_weighted

                val_loss += val_loss_batch.item()
                val_batches += 1

                # R² computation in normalized space (equivalent to denormalized)
                # Linear transform doesn't change R² ratio, avoids slow denormalize per batch
                pred = outputs
                tgt = targets

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
        epoch_times.append(epoch_time)
        training_state['epoch'] = epoch

        c_mu_val = raw_model.vertical_decoder.C_mu.item()
        vs_alpha = raw_model.vertical_decoder.vertical_smoother.alpha.item()

        if is_main:
            record = {
                'epoch': epoch + 1, 'train_loss': float(train_loss),
                'val_loss': float(val_loss), 'lr': float(current_lr),
                'time': float(epoch_time),
                'c_mu': float(c_mu_val),
                'vs_alpha': float(vs_alpha),
            }
            for k, v in train_losses_dict.items():
                record[f'train_{k}'] = float(v)
            for name in var_names:
                record[f'val_r2_{name}'] = val_r2.get(name, 0.0)
                record[f'val_rmse_{name}'] = val_rmse.get(name, 0.0)
            history.append(record)

            if writer:
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('LR', current_lr, epoch)
                writer.add_scalar('Physics/C_mu', c_mu_val, epoch)
                writer.add_scalar('Physics/VS_alpha', vs_alpha, epoch)
                for k, v in train_losses_dict.items():
                    writer.add_scalar(f'TrainLoss/{k}', v, epoch)
                for name in var_names:
                    writer.add_scalar(f'ValR2/{name}', val_r2.get(name, 0.0), epoch)
                    writer.add_scalar(f'ValRMSE/{name}', val_rmse.get(name, 0.0), epoch)

            mem_used = torch.cuda.max_memory_allocated(device) / 1024**3
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            remaining = epochs - (epoch + 1)
            eta_hours = avg_epoch_time * remaining / 3600

            log_print(f'\nEpoch {epoch+1}/{epochs}')
            log_print(f'   Train Loss: {train_loss:.6f}')
            log_print(f'   Val Loss:   {val_loss:.6f}')
            log_print(f'   LR:         {current_lr:.7f}')
            log_print(f'   Time:       {epoch_time:.1f}s')
            log_print(f'   GPU Peak: {mem_used:.1f}GB, ETA: {eta_hours:.1f}h')
            r2_str = ', '.join(f'{n}={val_r2.get(n, 0.0):.3f}' for n in var_names)
            rmse_str = ', '.join(f'{n}={val_rmse.get(n, 0.0):.3f}' for n in var_names)
            log_print(f'   Val R2:     {r2_str}')
            log_print(f'   Val RMSE:   {rmse_str}')
            loss_str = ', '.join(f'{k}={v:.4f}' for k, v in train_losses_dict.items())
            log_print(f'   Train Losses: {loss_str}')
            log_print(f'   C_μ = {c_mu_val:.4f}, VS_α = {vs_alpha:.4f}')

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
                }, ckpt_dir / 'best_model_v6.pt')
                if ema:
                    ema.restore()
                log_print(f'   Best model saved! (val_loss: {val_loss:.6f})')
            else:
                patience_counter += 1
                log_print(f'   Patience: {patience_counter}/{patience}')

            if patience_counter >= patience:
                log_print(f'\nEarly stopping triggered!')
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
                }, ckpt_dir / f'shensiv6_main_epoch{epoch+1}.pt')

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
            }, ckpt_dir / 'latest_v6.pt')

    if is_main:
        with open(log_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
        if writer:
            writer.close()
        print('\nTraining complete!')
        log_print(f'   Best val loss: {best_val_loss:.6f}')

    if log_file:
        log_file.close()

    dist.barrier()
    cleanup_ddp()


if __name__ == '__main__':
    train()
