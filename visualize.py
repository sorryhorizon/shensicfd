#!/usr/bin/env python3
"""Visualize model predictions vs ground truth on test set.

Generates comparison figures for each version:
  - Spatial distribution (pred vs truth at selected levels)
  - Vertical profile (R² and RMSE per level)
  - Residual distribution (histogram)
  - Parity plot (pred vs truth scatter)
  - Training curves (from TensorBoard)
  - FuXi-CFD comparison (side-by-side with baseline)

Usage:
    python visualize.py --version v4
    python visualize.py --version v5 --ckpt checkpoints/shensiv5_main/best_model_v5.pt
    python visualize.py --version v4 --no-diffusion
    python visualize.py --version v5 --with-fuxi   # include FuXi-CFD comparison
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import zoom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.fuxi_cfd_dataset import FuXiCFDDataset


def r2(pred, target):
    p, t = pred.flatten().astype(np.float64), target.flatten().astype(np.float64)
    ss_res = np.sum((p - t) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return max(0.0, min(1.0, 1 - ss_res / ss_tot))


def rmse(pred, target):
    return np.sqrt(np.mean((pred - target) ** 2))


def load_model(version, ckpt_path, device, output_mean, output_std):
    """Load model based on version string."""
    if version == 'v4':
        from src.models.hybrid_swin_unet_diffusion import create_hybrid_model
        model = create_hybrid_model(config={
            'base_channels': 48, 'bottleneck_depth': 4, 'window_size': (5, 5),
            'dropout': 0.2, 'drop_path_rate': 0.1,
            'k_diffusion_steps': 1000, 'k_ddim_steps': 20,
            'output_mean': output_mean, 'output_std': output_std,
        }).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        return model, 'forward_inference'

    elif version == 'v5':
        from src.models.swin_unet_v5 import SwinUNetV5
        model = SwinUNetV5(
            in_channels=6, n_levels=27, base_channels=48,
            bottleneck_depth=4, num_heads=4, window_size=(5, 5),
            dropout=0.2, drop_path_rate=0.1,
            output_mean=output_mean, output_std=output_std,
        ).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        return model, 'direct'

    else:
        raise ValueError(f"Unknown version: {version}. Supported: v4, v5")


def run_inference(model, test_loader, device, call_type, use_diffusion=False):
    """Run inference and collect predictions."""
    all_pred, all_target = [], []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)

            if call_type == 'forward_inference':
                outputs = model.forward_inference(inputs, use_diffusion=use_diffusion)
            else:
                outputs = model(inputs)

            if hasattr(test_loader.dataset, 'denormalize_output'):
                outputs = test_loader.dataset.denormalize_output(outputs)
                targets = test_loader.dataset.denormalize_output(targets)

            all_pred.append(outputs.cpu().numpy())
            all_target.append(targets.cpu().numpy())
            if (i + 1) % 50 == 0:
                print(f'  {i+1}/{len(test_loader)} batches done')

    return np.concatenate(all_pred, axis=0), np.concatenate(all_target, axis=0)


# ---- Plotting functions ----

LEVEL_HEIGHTS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80,
                 85, 90, 95, 100, 106.5, 114.95, 125.94, 140.22, 158.78, 182.91, 214.29]
VAR_NAMES = ['u', 'v', 'w', 'k']
VAR_UNITS = ['m/s', 'm/s', 'm/s', 'm²/s²']


def plot_spatial_comparison(pred, target, version, output_dir, sample_idx=0):
    """Plot pred vs truth spatial distribution at selected levels."""
    N, L, C, H, W = pred.shape
    selected_levels = [0, 10, 20, 26]
    level_labels = [f'L{i} ({LEVEL_HEIGHTS[i]}m)' for i in selected_levels]

    for c, (var, unit) in enumerate(zip(VAR_NAMES, VAR_UNITS)):
        fig, axes = plt.subplots(len(selected_levels), 3, figsize=(15, 5 * len(selected_levels)))

        for row, lvl in enumerate(selected_levels):
            truth = target[sample_idx, lvl, c]
            pred_slice = pred[sample_idx, lvl, c]
            residual = pred_slice - truth

            vmin = min(truth.min(), pred_slice.min())
            vmax = max(truth.max(), pred_slice.max())

            # Truth
            im0 = axes[row, 0].imshow(truth, cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')
            axes[row, 0].set_title(f'Truth {var} {level_labels[row]}')
            axes[row, 0].set_aspect('equal')
            plt.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04, label=unit)

            # Prediction
            im1 = axes[row, 1].imshow(pred_slice, cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')
            axes[row, 1].set_title(f'Pred {var} {level_labels[row]}')
            axes[row, 1].set_aspect('equal')
            plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04, label=unit)

            # Residual
            abs_max = max(abs(residual.min()), abs(residual.max()))
            if abs_max < 1e-8:
                abs_max = 1.0
            im2 = axes[row, 2].imshow(residual, cmap='RdBu_r', vmin=-abs_max, vmax=abs_max, origin='lower')
            lvl_r2 = r2(pred[:, lvl, c], target[:, lvl, c])
            axes[row, 2].set_title(f'Residual {var} {level_labels[row]} (R²={lvl_r2:.3f})')
            axes[row, 2].set_aspect('equal')
            plt.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04, label=unit)

        plt.suptitle(f'shensiv{version} — {var} Spatial Comparison (Sample #{sample_idx})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        path = os.path.join(output_dir, f'spatial_comparison_{var}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {path}')


def plot_vertical_profile(pred, target, version, output_dir):
    """Plot R² and RMSE as function of height level."""
    N, L, C, H, W = pred.shape

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for c, (var, unit) in enumerate(zip(VAR_NAMES, VAR_UNITS)):
        r2_per_level = []
        rmse_per_level = []
        for lvl in range(L):
            r2_per_level.append(r2(pred[:, lvl, c], target[:, lvl, c]))
            rmse_per_level.append(rmse(pred[:, lvl, c], target[:, lvl, c]))

        axes[0].plot(LEVEL_HEIGHTS, r2_per_level, label=f'{var}', linewidth=1.5)
        axes[1].plot(LEVEL_HEIGHTS, rmse_per_level, label=f'{var} ({unit})', linewidth=1.5)

    axes[0].set_xlabel('Height (m)')
    axes[0].set_ylabel('R²')
    axes[0].set_title('R² vs Height')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(bottom=0)

    axes[1].set_xlabel('Height (m)')
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('RMSE vs Height')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f'shensiv{version} — Vertical Profile', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'vertical_profile.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


def plot_residual_distribution(pred, target, version, output_dir):
    """Plot residual histogram for each variable."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for c, (var, unit) in enumerate(zip(VAR_NAMES, VAR_UNITS)):
        residual = (pred[:, :, c] - target[:, :, c]).flatten()
        axes[c].hist(residual, bins=100, density=True, alpha=0.7, color=f'C{c}')
        axes[c].set_title(f'{var} residual')
        axes[c].set_xlabel(f'Error ({unit})')
        axes[c].set_ylabel('Density')
        mean_err = np.mean(residual)
        std_err = np.std(residual)
        axes[c].axvline(mean_err, color='k', linestyle='--', label=f'mean={mean_err:.3f}')
        axes[c].axvline(mean_err + std_err, color='gray', linestyle=':', label=f'std={std_err:.3f}')
        axes[c].axvline(mean_err - std_err, color='gray', linestyle=':')
        axes[c].legend(fontsize=8)

    plt.suptitle(f'shensiv{version} — Residual Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'residual_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


def plot_parity(pred, target, version, output_dir):
    """Parity plot: pred vs truth scatter at selected levels."""
    selected_levels = [0, 10, 20, 26]
    level_labels = [f'L{i} ({LEVEL_HEIGHTS[i]}m)' for i in selected_levels]

    fig, axes = plt.subplots(4, len(selected_levels), figsize=(4 * len(selected_levels), 16))

    for c, (var, unit) in enumerate(zip(VAR_NAMES, VAR_UNITS)):
        for col, lvl in enumerate(selected_levels):
            p = pred[:, lvl, c].flatten()
            t = target[:, lvl, c].flatten()
            # Subsample for speed
            if len(p) > 50000:
                idx = np.random.choice(len(p), 50000, replace=False)
                p, t = p[idx], t[idx]

            axes[c, col].scatter(t, p, alpha=0.1, s=1, color=f'C{c}')
            vmin = min(t.min(), p.min())
            vmax = max(t.max(), p.max())
            axes[c, col].plot([vmin, vmax], [vmin, vmax], 'k--', linewidth=1)
            lvl_r2 = r2(pred[:, lvl, c], target[:, lvl, c])
            axes[c, col].set_title(f'{var} {level_labels[col]} (R²={lvl_r2:.3f})')
            axes[c, col].set_xlabel(f'Truth ({unit})')
            axes[c, col].set_ylabel(f'Pred ({unit})')
            axes[c, col].set_aspect('equal')

    plt.suptitle(f'shensiv{version} — Parity Plot', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'parity_plot.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


def plot_training_curves(tb_dir, version, output_dir):
    """Plot training curves from TensorBoard logs."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    # Find all event files
    all_data = {}
    for root, dirs, files in os.walk(tb_dir):
        for f in files:
            if f.startswith('events'):
                ea = EventAccumulator(os.path.join(root, f))
                ea.Reload()
                tags = ea.Tags()['scalars']
                for tag in tags:
                    if tag not in all_data:
                        all_data[tag] = {}
                    for e in ea.Scalars(tag):
                        if e.step not in all_data[tag]:
                            all_data[tag][e.step] = e.value

    if not all_data:
        print(f'  No TensorBoard data found in {tb_dir}')
        return

    def get_series(tag):
        d = all_data.get(tag, {})
        steps = sorted(d.keys())
        return np.array(steps), np.array([d[s] for s in steps])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Train/Val Loss
    ax = axes[0, 0]
    if 'Loss/train' in all_data:
        ep, v = get_series('Loss/train')
        ax.plot(ep, v, label='Train Loss', linewidth=1.5)
    if 'Loss/val' in all_data:
        ep, v = get_series('Loss/val')
        ax.plot(ep, v, label='Val Loss', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Train & Val Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Train Loss Components
    ax = axes[0, 1]
    loss_tags = [t for t in all_data if 'TrainLoss' in t or 'train_loss' in t.lower()]
    for tag in sorted(loss_tags):
        ep, v = get_series(tag)
        ax.plot(ep, v, label=tag.split('/')[-1], linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Component')
    ax.set_title('Train Loss Components')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Val R²
    ax = axes[1, 0]
    for var in ['u', 'v', 'w', 'k']:
        tag = f'ValR2/{var}'
        if tag in all_data:
            ep, v = get_series(tag)
            ax.plot(ep, v, label=f'{var} R²', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('R²')
    ax.set_title('Validation R²')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # 4. Val RMSE
    ax = axes[1, 1]
    for var in ['u', 'v', 'w', 'k']:
        tag = f'ValRMSE/{var}'
        if tag in all_data:
            ep, v = get_series(tag)
            ax.plot(ep, v, label=f'{var} RMSE', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.set_title('Validation RMSE')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'shensiv{version} — Training Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


# ---- FuXi-CFD inference ----

FUXI_MODEL_PATH = '/mnt/sdata/jz/fuxi_cfd/model/fuxicfd_model.onnx'
FUXI_NORM_IN_PATH = '/mnt/sdata/jz/fuxi_cfd/inference_example/normalization/scaler_input.npy'
FUXI_NORM_OUT_PATH = '/mnt/sdata/jz/fuxi_cfd/inference_example/normalization/scaler_output.npy'
FUXI_DATA_DIR = '/mnt/sdata/jz/fuxi_cfd/dataset'


def run_fuxi_inference(test_indices):
    """Run FuXi-CFD ONNX model on test set cases (batched, GPU-accelerated)."""
    import onnxruntime as ort

    in_stats = np.load(FUXI_NORM_IN_PATH, allow_pickle=True).item()
    out_stats = np.load(FUXI_NORM_OUT_PATH, allow_pickle=True).item()

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess = ort.InferenceSession(FUXI_MODEL_PATH, sess_options=sess_options, providers=providers)
    input_name = sess.get_inputs()[0].name

    high_mean = in_stats['high_mean'][:, None, None]
    high_std = in_stats['high_std'][:, None, None]
    low_mean = in_stats['low_mean'][:, None, None]
    low_std = in_stats['low_std'][:, None, None]

    out_mean = out_stats['mean'][:, :, None, None]  # (27, 4, 1, 1)
    out_std = out_stats['std'][:, :, None, None]

    # Pre-load all inputs and targets
    print(f'  Loading {len(test_indices)} cases...')
    all_inputs = []
    all_targets = []
    for idx in test_indices:
        case_dir = os.path.join(FUXI_DATA_DIR, f'case_{idx+1:06d}')
        inputs = np.load(os.path.join(case_dir, 'inputs.npz'))
        outputs = np.load(os.path.join(case_dir, 'outputs.npz'))

        dem_rough = np.stack([inputs['dem'], inputs['roughness']], axis=0)
        uv_100m = np.stack([inputs['u_100m'], inputs['v_100m']], axis=0)

        dem_rough = (dem_rough - high_mean) / high_std
        uv_100m = (uv_100m - low_mean) / low_std
        uv_100m = zoom(uv_100m, (1, 300 / uv_100m.shape[1], 300 / uv_100m.shape[2]), order=1)

        x = np.concatenate([uv_100m, dem_rough], axis=0).astype(np.float32)
        all_inputs.append(x)

        target = np.stack([outputs['u'], outputs['v'], outputs['w'], outputs['k']], axis=1)
        all_targets.append(target)

    # Batch inference (batch_size=1 to avoid OOM)
    batch_size = 1
    all_pred = []
    for i in range(0, len(all_inputs), batch_size):
        batch = np.stack(all_inputs[i:i+batch_size], axis=0)  # (B, 4, 300, 300)
        pred = sess.run(None, {input_name: batch})[0]  # (B, 27, 4, 300, 300)
        pred = pred * out_std + out_mean
        all_pred.append(pred)
        if (i + batch_size) % 200 == 0 or i + batch_size >= len(all_inputs):
            print(f'  FuXi inference: {min(i+batch_size, len(all_inputs))}/{len(test_indices)} done')

    pred = np.concatenate(all_pred, axis=0)  # (N, 27, 4, 300, 300)
    target = np.stack(all_targets, axis=0)    # (N, 27, 4, 300, 300)

    return pred, target


# ---- FuXi-CFD comparison plots ----

def plot_spatial_comparison_fuxi(pred_shensi, pred_fuxi, target, version, output_dir, sample_idx=0):
    """3-column comparison: Truth | ShenSi | FuXi-CFD at selected levels."""
    selected_levels = [0, 10, 20, 26]
    level_labels = [f'L{i} ({LEVEL_HEIGHTS[i]}m)' for i in selected_levels]

    for c, (var, unit) in enumerate(zip(VAR_NAMES, VAR_UNITS)):
        fig, axes = plt.subplots(len(selected_levels), 3, figsize=(15, 5 * len(selected_levels)))

        for row, lvl in enumerate(selected_levels):
            truth = target[sample_idx, lvl, c]
            shensi = pred_shensi[sample_idx, lvl, c]
            fuxi = pred_fuxi[sample_idx, lvl, c]

            vmin = min(truth.min(), shensi.min(), fuxi.min())
            vmax = max(truth.max(), shensi.max(), fuxi.max())

            # Truth
            im0 = axes[row, 0].imshow(truth, cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')
            axes[row, 0].set_title(f'Truth {var} {level_labels[row]}')
            axes[row, 0].set_aspect('equal')
            plt.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04, label=unit)

            # ShenSi
            im1 = axes[row, 1].imshow(shensi, cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')
            shensi_r2 = r2(pred_shensi[:, lvl, c], target[:, lvl, c])
            axes[row, 1].set_title(f'ShenSi v{version} {var} {level_labels[row]} (R²={shensi_r2:.3f})')
            axes[row, 1].set_aspect('equal')
            plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04, label=unit)

            # FuXi-CFD
            im2 = axes[row, 2].imshow(fuxi, cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')
            fuxi_r2 = r2(pred_fuxi[:, lvl, c], target[:, lvl, c])
            axes[row, 2].set_title(f'FuXi-CFD {var} {level_labels[row]} (R²={fuxi_r2:.3f})')
            axes[row, 2].set_aspect('equal')
            plt.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04, label=unit)

        plt.suptitle(f'{var} Comparison: Truth | ShenSi v{version} | FuXi-CFD', fontsize=14, fontweight='bold')
        plt.tight_layout()
        path = os.path.join(output_dir, f'fuxi_comparison_{var}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {path}')


def plot_vertical_profile_comparison(pred_shensi, pred_fuxi, target, version, output_dir):
    """Vertical profile: ShenSi vs FuXi-CFD R² and RMSE."""
    N, L, C, H, W = pred_shensi.shape

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for c, (var, unit) in enumerate(zip(VAR_NAMES, VAR_UNITS)):
        r2_shensi = [r2(pred_shensi[:, lvl, c], target[:, lvl, c]) for lvl in range(L)]
        r2_fuxi = [r2(pred_fuxi[:, lvl, c], target[:, lvl, c]) for lvl in range(L)]
        rmse_shensi = [rmse(pred_shensi[:, lvl, c], target[:, lvl, c]) for lvl in range(L)]
        rmse_fuxi = [rmse(pred_fuxi[:, lvl, c], target[:, lvl, c]) for lvl in range(L)]

        axes[0].plot(LEVEL_HEIGHTS, r2_shensi, label=f'{var} ShenSi', linewidth=1.5, linestyle='-')
        axes[0].plot(LEVEL_HEIGHTS, r2_fuxi, label=f'{var} FuXi', linewidth=1.5, linestyle='--')
        axes[1].plot(LEVEL_HEIGHTS, rmse_shensi, label=f'{var} ShenSi', linewidth=1.5, linestyle='-')
        axes[1].plot(LEVEL_HEIGHTS, rmse_fuxi, label=f'{var} FuXi', linewidth=1.5, linestyle='--')

    axes[0].set_xlabel('Height (m)')
    axes[0].set_ylabel('R²')
    axes[0].set_title('R² vs Height: ShenSi vs FuXi-CFD')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(bottom=0)

    axes[1].set_xlabel('Height (m)')
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('RMSE vs Height: ShenSi vs FuXi-CFD')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f'ShenSi v{version} vs FuXi-CFD — Vertical Profile', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'fuxi_comparison_vertical_profile.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


def plot_residual_comparison(pred_shensi, pred_fuxi, target, version, output_dir):
    """Residual histogram: ShenSi vs FuXi-CFD overlaid."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for c, (var, unit) in enumerate(zip(VAR_NAMES, VAR_UNITS)):
        res_shensi = (pred_shensi[:, :, c] - target[:, :, c]).flatten()
        res_fuxi = (pred_fuxi[:, :, c] - target[:, :, c]).flatten()

        axes[c].hist(res_shensi, bins=100, density=True, alpha=0.5, color='C0', label='ShenSi')
        axes[c].hist(res_fuxi, bins=100, density=True, alpha=0.5, color='C1', label='FuXi')
        axes[c].set_title(f'{var} residual')
        axes[c].set_xlabel(f'Error ({unit})')
        axes[c].set_ylabel('Density')
        axes[c].legend(fontsize=8)

    plt.suptitle(f'ShenSi v{version} vs FuXi-CFD — Residual Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'fuxi_comparison_residual.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


def plot_metrics_comparison(pred_shensi, pred_fuxi, target, version, output_dir):
    """Bar chart: overall R² and RMSE comparison."""
    shensi_r2 = [r2(pred_shensi[:, :, c], target[:, :, c]) for c in range(4)]
    fuxi_r2 = [r2(pred_fuxi[:, :, c], target[:, :, c]) for c in range(4)]
    shensi_rmse = [rmse(pred_shensi[:, :, c], target[:, :, c]) for c in range(4)]
    fuxi_rmse = [rmse(pred_fuxi[:, :, c], target[:, :, c]) for c in range(4)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    x = np.arange(4)
    width = 0.35

    axes[0].bar(x - width/2, shensi_r2, width, label=f'ShenSi v{version}', color='C0')
    axes[0].bar(x + width/2, fuxi_r2, width, label='FuXi-CFD', color='C1')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(VAR_NAMES)
    axes[0].set_ylabel('R²')
    axes[0].set_title('Overall R² Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].bar(x - width/2, shensi_rmse, width, label=f'ShenSi v{version}', color='C0')
    axes[1].bar(x + width/2, fuxi_rmse, width, label='FuXi-CFD', color='C1')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(VAR_NAMES)
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('Overall RMSE Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'ShenSi v{version} vs FuXi-CFD — Metrics Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'fuxi_comparison_metrics.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


def main():
    parser = argparse.ArgumentParser(description='Visualize model predictions vs ground truth')
    parser.add_argument('--version', type=str, required=True, choices=['v4', 'v5'],
                        help='Model version')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Checkpoint path (auto-detected if not specified)')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--sample-idx', type=int, default=0,
                        help='Sample index for spatial comparison')
    parser.add_argument('--no-diffusion', action='store_true',
                        help='For v4: use regression k instead of DDIM')
    parser.add_argument('--with-fuxi', action='store_true',
                        help='Include FuXi-CFD comparison figures')
    parser.add_argument('--skip-inference', action='store_true',
                        help='Skip inference, only plot training curves')
    args = parser.parse_args()

    # Auto-detect checkpoint
    if args.ckpt is None:
        if args.version == 'v4':
            args.ckpt = 'checkpoints/shensiv4_main/best_model.pt'
        elif args.version == 'v5':
            args.ckpt = 'checkpoints/shensiv5_main/best_model_v5.pt'

    # Output directory (under checkpoints)
    version_num = args.version.lstrip('v')  # 'v4' -> '4'
    ckpt_dir = os.path.dirname(args.ckpt) if args.ckpt else f'checkpoints/shensiv{version_num}_main'
    output_dir = os.path.join(ckpt_dir, 'figures')
    os.makedirs(output_dir, exist_ok=True)

    print(f'=== shensiv{version_num} Visualization ===')
    print(f'Checkpoint: {args.ckpt}')
    print(f'Output: {output_dir}')

    # Plot training curves first (no GPU needed)
    tb_dirs = {
        'v4': 'logs/shensiv4_main/tensorboard/tensorboard',
        'v5': 'logs/shensiv5_main/tensorboard',
    }
    tb_dir = tb_dirs.get(args.version)
    if tb_dir and os.path.exists(tb_dir):
        print('\nPlotting training curves...')
        plot_training_curves(tb_dir, version_num, output_dir)

    if args.skip_inference:
        print('\nSkipping inference (--skip-inference)')
        return

    # Load dataset and model
    device = torch.device(args.device)
    data_dir = '/mnt/sdata/jz/fuxi_cfd/dataset'

    print('\nLoading dataset...')
    test_dataset = FuXiCFDDataset(data_dir, split='test', normalize=True, prefetch_to_memory=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    output_mean = output_std = None
    if hasattr(test_dataset, 'stats') and test_dataset.stats is not None:
        output_mean = torch.from_numpy(test_dataset.stats['output_mean']).float()
        output_std = torch.from_numpy(test_dataset.stats['output_std']).float()

    print('Loading model...')
    model, call_type = load_model(args.version, args.ckpt, device, output_mean, output_std)
    model.eval()

    # Run inference
    print(f'Running inference on {len(test_dataset)} samples...')
    use_diffusion = not args.no_diffusion
    pred, target = run_inference(model, test_loader, device, call_type, use_diffusion=use_diffusion)

    # Generate all plots
    print('\nGenerating visualizations...')

    # 1. Spatial comparison
    plot_spatial_comparison(pred, target, version_num, output_dir, sample_idx=args.sample_idx)

    # 2. Vertical profile
    plot_vertical_profile(pred, target, version_num, output_dir)

    # 3. Residual distribution
    plot_residual_distribution(pred, target, version_num, output_dir)

    # 4. Parity plot
    plot_parity(pred, target, version_num, output_dir)

    # Print summary metrics
    N, L, C, H, W = pred.shape
    print(f'\n=== Test Set Metrics ===')
    for c, var in enumerate(VAR_NAMES):
        overall_r2 = r2(pred[:, :, c], target[:, :, c])
        overall_rmse = rmse(pred[:, :, c], target[:, :, c])
        print(f'  {var}: R²={overall_r2:.4f}, RMSE={overall_rmse:.4f}')

    # FuXi-CFD comparison
    if args.with_fuxi:
        print('\n=== FuXi-CFD Comparison ===')
        print('Running FuXi-CFD inference on test set...')

        # Get test case indices from dataset
        test_indices = test_dataset.indices if hasattr(test_dataset, 'indices') else range(N)

        pred_fuxi, target_fuxi = run_fuxi_inference(test_indices)

        # Verify targets match
        if np.allclose(target, target_fuxi, atol=1e-3):
            print('  Targets match between ShenSi and FuXi datasets')
        else:
            print('  Warning: targets differ slightly, using ShenSi targets')

        print('Generating FuXi-CFD comparison figures...')
        plot_spatial_comparison_fuxi(pred, pred_fuxi, target, version_num, output_dir, sample_idx=args.sample_idx)
        plot_vertical_profile_comparison(pred, pred_fuxi, target, version_num, output_dir)
        plot_residual_comparison(pred, pred_fuxi, target, version_num, output_dir)
        plot_metrics_comparison(pred, pred_fuxi, target, version_num, output_dir)

        # Print FuXi-CFD metrics
        print(f'\n=== FuXi-CFD Test Set Metrics ===')
        for c, var in enumerate(VAR_NAMES):
            fuxi_r2 = r2(pred_fuxi[:, :, c], target[:, :, c])
            fuxi_rmse = rmse(pred_fuxi[:, :, c], target[:, :, c])
            shensi_r2 = r2(pred[:, :, c], target[:, :, c])
            diff = shensi_r2 - fuxi_r2
            print(f'  {var}: ShenSi R²={shensi_r2:.4f}, FuXi R²={fuxi_r2:.4f}, diff={diff:+.4f}')

    print(f'\nAll figures saved to: {output_dir}')


if __name__ == '__main__':
    main()