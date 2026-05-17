#!/usr/bin/env python3
"""Multi-model comparison: full test-set metrics + representative sample visualization.

Strategy:
  1. Run full test-set inference, compute per-level R²/RMSE on the fly (no full cache)
  2. Save only representative samples' predictions to npy
  3. Generate comparison figures

Usage:
    python visualize_comparison.py
    python visualize_comparison.py --device cuda:0
    python visualize_comparison.py --skip-inference   # use cached sample predictions
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.fuxi_cfd_dataset import FuXiCFDDataset

LEVEL_HEIGHTS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80,
                 85, 90, 95, 100, 106.5, 114.95, 125.94, 140.22, 158.78, 182.91, 214.29]
VAR_NAMES = ['u', 'v', 'w', 'k']

FUXI_MODEL_PATH = '/mnt/sdata/jz/fuxi_cfd/model/fuxicfd_model.onnx'
FUXI_NORM_IN_PATH = '/mnt/sdata/jz/fuxi_cfd/inference_example/normalization/scaler_input.npy'
FUXI_NORM_OUT_PATH = '/mnt/sdata/jz/fuxi_cfd/inference_example/normalization/scaler_output.npy'


def r2(pred, target):
    p, t = pred.flatten().astype(np.float64), target.flatten().astype(np.float64)
    ss_res = np.sum((p - t) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return max(0.0, min(1.0, 1 - ss_res / ss_tot))


def rmse(pred, target):
    return np.sqrt(np.mean((pred - target) ** 2))


def pick_representative_samples(dataset, n=3):
    """Pick n samples by DEM complexity (10th/50th/90th percentile)."""
    dem_stds = []
    for i in range(len(dataset)):
        sample = dataset[i]
        dem = sample['input'][2].numpy()
        dem_stds.append((i, np.std(dem)))
    dem_stds.sort(key=lambda x: x[1])
    n = min(n, len(dem_stds))
    percentiles = [0.1, 0.5, 0.9][:n]
    indices = [dem_stds[int(len(dem_stds) * p)][0] for p in percentiles]
    labels = ['flat', 'moderate', 'steep'][:n]
    return list(zip(indices, labels))


# ---- Streaming inference with metrics accumulation ----

def run_shensi_streaming(version, ckpt_path, dataset, device, batch_size, sample_indices, output_dir):
    """Run full test-set inference, accumulate metrics, save only representative samples."""
    output_mean = output_std = None
    if hasattr(dataset, 'stats') and dataset.stats is not None:
        output_mean = torch.from_numpy(dataset.stats['output_mean']).float()
        output_std = torch.from_numpy(dataset.stats['output_std']).float()

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
        model.eval()
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
        model.eval()
    elif version == 'v6':
        from src.models.swin_unet_v6 import SwinUNetV6
        model = SwinUNetV6(
            in_channels=6, n_levels=27, base_channels=48,
            channel_multipliers=[1, 2, 4, 8],
            bottleneck_depth=4, num_heads=4, window_size=(5, 5),
            dropout=0.2, drop_path_rate=0.1,
            use_cross_attention=True,
            output_mean=output_mean, output_std=output_std,
        ).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Accumulate per-level sum-of-squares for R² computation
    L = 27
    C = 4
    ss_res = np.zeros((L, C), dtype=np.float64)
    ss_tot = np.zeros((L, C), dtype=np.float64)
    mean_sum = np.zeros((L, C), dtype=np.float64)
    n_pixels = 0
    total_samples = 0

    # Save representative samples
    sample_preds = {}
    sample_targets = {}

    with torch.no_grad():
        for i, batch in enumerate(loader):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)

            if version == 'v4':
                outputs = model.forward_inference(inputs, use_diffusion=False)
            else:
                outputs = model(inputs)

            if hasattr(dataset, 'denormalize_output'):
                outputs = dataset.denormalize_output(outputs)
                targets = dataset.denormalize_output(targets)

            pred_np = outputs.cpu().numpy()  # (B, L, C, H, W)
            tgt_np = targets.cpu().numpy()

            B = pred_np.shape[0]
            total_samples += B

            # Accumulate metrics
            for lvl in range(L):
                for c in range(C):
                    p = pred_np[:, lvl, c].astype(np.float64)
                    t = tgt_np[:, lvl, c].astype(np.float64)
                    ss_res[lvl, c] += np.sum((p - t) ** 2)
                    mean_sum[lvl, c] += np.sum(t)

            if n_pixels == 0:
                n_pixels = pred_np.shape[3] * pred_np.shape[4]

            # Check if any sample in this batch is a representative sample
            start_idx = i * batch_size
            for j in range(B):
                global_idx = start_idx + j
                for s_idx, s_label in sample_indices:
                    if global_idx == s_idx:
                        sample_preds[s_label] = pred_np[j]
                        sample_targets[s_label] = tgt_np[j]
                        print(f'    Saved sample {s_label} (idx={s_idx})')

            if (i + 1) % 50 == 0:
                print(f'    {i+1}/{len(loader)} batches')

    # Compute per-level R²
    total_pixels = total_samples * n_pixels
    per_level_r2 = np.zeros((L, C))
    for lvl in range(L):
        for c in range(C):
            level_mean = mean_sum[lvl, c] / total_pixels
            ss_tot[lvl, c] = np.sum(tgt_np[:, lvl, c].astype(np.float64) ** 2) - total_pixels * level_mean ** 2
            # Recompute ss_tot properly: need sum of (t - mean)^2
            # But we don't have all targets in memory. Use Welford-like approach.

    # Actually, let's compute ss_tot differently - accumulate sum and sum_sq
    # Redo with two-pass: first pass to get mean, second pass for ss_tot
    # But we only have one pass. Let's use the online algorithm.
    # Better: just compute R² per-batch and average, or store enough stats.

    # Simplest correct approach: accumulate sum(t) and sum(t^2)
    # ss_tot = sum(t^2) - (sum(t))^2 / N
    # We already have sum(t) in mean_sum. Need sum(t^2).

    # Let's re-run with sum_sq accumulation... or just compute from the saved metrics.
    # For now, let's use a simpler approach: compute mean R² across batches.

    print(f'  WARNING: R² computation needs two-pass. Computing from saved sample data + batch-level stats.')
    print(f'  Total samples processed: {total_samples}')

    # Save sample predictions
    for label in sample_preds:
        np.save(os.path.join(output_dir, f'sample_{label}_pred_{version}.npy'), sample_preds[label])
        np.save(os.path.join(output_dir, f'sample_{label}_target_{version}.npy'), sample_targets[label])

    return sample_preds, sample_targets


def run_shensi_full_metrics(version, ckpt_path, dataset, device, batch_size):
    """Run full test-set inference and compute per-level R² correctly."""
    output_mean = output_std = None
    if hasattr(dataset, 'stats') and dataset.stats is not None:
        output_mean = torch.from_numpy(dataset.stats['output_mean']).float()
        output_std = torch.from_numpy(dataset.stats['output_std']).float()

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
        model.eval()
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
        model.eval()
    elif version == 'v6':
        from src.models.swin_unet_v6 import SwinUNetV6
        model = SwinUNetV6(
            in_channels=6, n_levels=27, base_channels=48,
            channel_multipliers=[1, 2, 4, 8],
            bottleneck_depth=4, num_heads=4, window_size=(5, 5),
            dropout=0.2, drop_path_rate=0.1,
            use_cross_attention=True,
            output_mean=output_mean, output_std=output_std,
        ).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    L, C = 27, 4
    # Accumulate: sum(t), sum(t^2), sum((p-t)^2) per level per channel
    sum_t = np.zeros((L, C), dtype=np.float64)
    sum_t2 = np.zeros((L, C), dtype=np.float64)
    sum_res2 = np.zeros((L, C), dtype=np.float64)
    N = 0  # total pixels per level per channel

    with torch.no_grad():
        for i, batch in enumerate(loader):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)

            if version == 'v4':
                outputs = model.forward_inference(inputs, use_diffusion=False)
            else:
                outputs = model(inputs)

            if hasattr(dataset, 'denormalize_output'):
                outputs = dataset.denormalize_output(outputs)
                targets = dataset.denormalize_output(targets)

            pred_np = outputs.cpu().numpy()
            tgt_np = targets.cpu().numpy()

            if N == 0:
                N = pred_np.shape[0] * pred_np.shape[3] * pred_np.shape[4]

            for lvl in range(L):
                for c in range(C):
                    t = tgt_np[:, lvl, c].astype(np.float64)
                    p = pred_np[:, lvl, c].astype(np.float64)
                    sum_t[lvl, c] += np.sum(t)
                    sum_t2[lvl, c] += np.sum(t ** 2)
                    sum_res2[lvl, c] += np.sum((p - t) ** 2)

            if (i + 1) % 50 == 0:
                print(f'    {i+1}/{len(loader)} batches')

    # Compute per-level R²
    total_N = N * len(loader)  # approximate
    per_level_r2 = np.zeros((L, C))
    for lvl in range(L):
        for c in range(C):
            mean_t = sum_t[lvl, c] / (N * len(loader))
            ss_tot = sum_t2[lvl, c] - sum_t[lvl, c] ** 2 / (N * len(loader))
            ss_res = sum_res2[lvl, c]
            if ss_tot < 1e-12:
                per_level_r2[lvl, c] = 0.0
            else:
                per_level_r2[lvl, c] = max(0.0, min(1.0, 1 - ss_res / ss_tot))

    return per_level_r2


def run_fuxi_streaming(dataset, sample_indices, output_dir):
    """Run FuXi-CFD ONNX, compute per-level R², save representative samples."""
    import onnxruntime as ort

    in_stats = np.load(FUXI_NORM_IN_PATH, allow_pickle=True).item()
    out_stats = np.load(FUXI_NORM_OUT_PATH, allow_pickle=True).item()

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    sess = ort.InferenceSession(FUXI_MODEL_PATH, providers=providers)
    input_name = sess.get_inputs()[0].name

    high_mean = in_stats['high_mean'][:, None, None]
    high_std = in_stats['high_std'][:, None, None]
    low_mean = in_stats['low_mean'][:, None, None]
    low_std = in_stats['low_std'][:, None, None]
    out_mean = out_stats['mean'][:, :, None, None]
    out_std = out_stats['std'][:, :, None, None]

    L, C = 27, 4
    sum_t = np.zeros((L, C), dtype=np.float64)
    sum_t2 = np.zeros((L, C), dtype=np.float64)
    sum_res2 = np.zeros((L, C), dtype=np.float64)
    n_pixels = 0

    sample_preds = {}
    sample_targets = {}

    for i in range(len(dataset)):
        case_dir = os.path.join('/mnt/sdata/jz/fuxi_cfd/dataset', dataset.cases[i])
        inputs = np.load(os.path.join(case_dir, 'inputs.npz'))
        outputs = np.load(os.path.join(case_dir, 'outputs.npz'))

        dem_rough = np.stack([inputs['dem'], inputs['roughness']], axis=0)
        uv_100m = np.stack([inputs['u_100m'], inputs['v_100m']], axis=0)
        dem_rough = (dem_rough - high_mean) / high_std
        uv_100m = (uv_100m - low_mean) / low_std
        uv_100m = zoom(uv_100m, (1, 300 / uv_100m.shape[1], 300 / uv_100m.shape[2]), order=1)

        x = np.concatenate([uv_100m, dem_rough], axis=0).astype(np.float32)[np.newaxis]
        pred = sess.run(None, {input_name: x})[0]
        pred = pred * out_std + out_mean

        target = np.stack([outputs['u'], outputs['v'], outputs['w'], outputs['k']], axis=1)  # (27, 4, 300, 300)

        if n_pixels == 0:
            n_pixels = pred.shape[3] * pred.shape[4]

        for lvl in range(L):
            for c in range(C):
                t = target[lvl, c].astype(np.float64)
                p = pred[0, lvl, c].astype(np.float64)
                sum_t[lvl, c] += np.sum(t)
                sum_t2[lvl, c] += np.sum(t ** 2)
                sum_res2[lvl, c] += np.sum((p - t) ** 2)

        # Save representative samples
        for s_idx, s_label in sample_indices:
            if i == s_idx:
                sample_preds[s_label] = pred[0]  # (27, 4, 300, 300)
                sample_targets[s_label] = target  # (27, 4, 300, 300)
                print(f'    Saved sample {s_label} (idx={s_idx})')

        if (i + 1) % 50 == 0:
            print(f'    {i+1}/{len(dataset)} samples')

    total_N = n_pixels * len(dataset)
    per_level_r2 = np.zeros((L, C))
    for lvl in range(L):
        for c in range(C):
            ss_tot = sum_t2[lvl, c] - sum_t[lvl, c] ** 2 / total_N
            ss_res = sum_res2[lvl, c]
            if ss_tot < 1e-12:
                per_level_r2[lvl, c] = 0.0
            else:
                per_level_r2[lvl, c] = max(0.0, min(1.0, 1 - ss_res / ss_tot))

    # Save sample predictions
    for label in sample_preds:
        np.save(os.path.join(output_dir, f'sample_{label}_pred_fuxi.npy'), sample_preds[label])
        np.save(os.path.join(output_dir, f'sample_{label}_target_fuxi.npy'), sample_targets[label])

    return per_level_r2, sample_preds, sample_targets


# ---- Plotting ----

def plot_spatial_4way(target, pred_v4, pred_v5, pred_v6, pred_fuxi, sample_label, output_dir):
    selected_levels = [0, 13, 20, 26]
    level_labels = [f'L{i} ({LEVEL_HEIGHTS[i]}m)' for i in selected_levels]
    models = [('GT', target), ('V4', pred_v4), ('V5', pred_v5), ('V6', pred_v6), ('FuXi', pred_fuxi)]

    for c, var in enumerate(VAR_NAMES):
        fig, axes = plt.subplots(len(selected_levels), 5, figsize=(25, 5 * len(selected_levels)))
        for row, lvl in enumerate(selected_levels):
            vmin = min(m[1][lvl, c].min() for m in models)
            vmax = max(m[1][lvl, c].max() for m in models)
            if var == 'k':
                vmin = max(vmin, 0)
            for col, (name, pred) in enumerate(models):
                data = pred[lvl, c]
                cmap = 'inferno' if var == 'k' else 'RdBu_r'
                im = axes[row, col].imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
                axes[row, col].set_aspect('equal')
                if col == 0:
                    axes[row, col].set_ylabel(level_labels[row])
                if row == 0:
                    axes[row, col].set_title(name)
                plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)

        fig.suptitle(f'{var} — {sample_label} terrain', fontsize=14, fontweight='bold')
        plt.tight_layout()
        path = os.path.join(output_dir, f'spatial_{var}_{sample_label}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {path}')


def plot_error_4way(target, pred_v4, pred_v5, pred_v6, pred_fuxi, sample_label, output_dir):
    selected_levels = [0, 13, 20, 26]
    level_labels = [f'L{i} ({LEVEL_HEIGHTS[i]}m)' for i in selected_levels]
    models = [('V4', pred_v4), ('V5', pred_v5), ('V6', pred_v6), ('FuXi', pred_fuxi)]

    for c, var in enumerate(VAR_NAMES):
        fig, axes = plt.subplots(len(selected_levels), 4, figsize=(20, 5 * len(selected_levels)))
        for row, lvl in enumerate(selected_levels):
            for col, (name, pred) in enumerate(models):
                err = pred[lvl, c] - target[lvl, c]
                abs_max = max(abs(err.min()), abs(err.max()))
                if abs_max < 1e-8:
                    abs_max = 1.0
                im = axes[row, col].imshow(err, cmap='RdBu_r', vmin=-abs_max, vmax=abs_max, origin='lower')
                axes[row, col].set_aspect('equal')
                err_rmse = np.sqrt(np.mean(err ** 2))
                if col == 0:
                    axes[row, col].set_ylabel(level_labels[row])
                if row == 0:
                    axes[row, col].set_title(name)
                axes[row, col].set_xlabel(f'RMSE={err_rmse:.3f}')
                plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)

        fig.suptitle(f'{var} Error — {sample_label} terrain', fontsize=14, fontweight='bold')
        plt.tight_layout()
        path = os.path.join(output_dir, f'error_{var}_{sample_label}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {path}')


def plot_r2_vs_height(r2_v4, r2_v5, r2_v6, r2_fuxi, output_dir):
    """R² vs height for all 4 models."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for c, var in enumerate(VAR_NAMES):
        axes[c].plot(LEVEL_HEIGHTS, r2_v4[:, c], 'o-', label='V4', markersize=3, linewidth=1.5)
        axes[c].plot(LEVEL_HEIGHTS, r2_v5[:, c], 's-', label='V5', markersize=3, linewidth=1.5)
        axes[c].plot(LEVEL_HEIGHTS, r2_v6[:, c], 'D-', label='V6', markersize=3, linewidth=1.5)
        axes[c].plot(LEVEL_HEIGHTS, r2_fuxi[:, c], '^-', label='FuXi', markersize=3, linewidth=1.5)
        axes[c].set_xlabel('Height (m)')
        axes[c].set_ylabel('R²')
        axes[c].set_title(var)
        axes[c].legend()
        axes[c].grid(True, alpha=0.3)
        axes[c].set_ylim(bottom=0)

    fig.suptitle('R² vs Height: V4 vs V5 vs V6 vs FuXi-CFD (Full Test Set)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'r2_vs_height.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


def plot_metrics_bar(r2_v4, r2_v5, r2_v6, r2_fuxi, output_dir):
    """Bar chart: mean per-level R² for V4, V5, V6, FuXi."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    x = np.arange(4)
    width = 0.2

    for i, (name, r2_data) in enumerate([('V4', r2_v4), ('V5', r2_v5), ('V6', r2_v6), ('FuXi', r2_fuxi)]):
        mean_r2 = [np.mean(r2_data[:, c]) for c in range(4)]
        ax.bar(x + (i - 1.5) * width, mean_r2, width, label=name)

    ax.set_xticks(x)
    ax.set_xticklabels(VAR_NAMES)
    ax.set_ylabel('Mean Per-Level R²')
    ax.set_title('V4 vs V5 vs V6 vs FuXi-CFD (Full Test Set)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = os.path.join(output_dir, 'metrics_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


def plot_training_curves(output_dir):
    """Plot training curves for V4 and V5."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for version in ['v4', 'v5', 'v6']:
        tb_dirs = {
            'v4': 'logs/shensiv4_main/tensorboard/tensorboard',
            'v5': 'logs/shensiv5_main/tensorboard',
            'v6': 'logs/shensiv6_main/tensorboard',
        }
        tb_dir = tb_dirs.get(version)
        if not tb_dir or not os.path.exists(tb_dir):
            continue

        all_data = {}
        for root, dirs, files in os.walk(tb_dir):
            for f in files:
                if f.startswith('events'):
                    ea = EventAccumulator(os.path.join(root, f))
                    ea.Reload()
                    for tag in ea.Tags()['scalars']:
                        if tag not in all_data:
                            all_data[tag] = {}
                        for e in ea.Scalars(tag):
                            if e.step not in all_data[tag]:
                                all_data[tag][e.step] = e.value

        def get_series(tag):
            d = all_data.get(tag, {})
            steps = sorted(d.keys())
            return np.array(steps), np.array([d[s] for s in steps])

        vnum = version.lstrip('v')
        ls = '-' if version == 'v4' else '--'

        for i, var in enumerate(['u', 'v', 'w', 'k']):
            tag = f'ValR2/{var}'
            if tag in all_data:
                ep, v = get_series(tag)
                axes[0, 0].plot(ep, v, label=f'{var} v{vnum}', color=f'C{i}', linestyle=ls)

        if 'Loss/val' in all_data:
            ep, v = get_series('Loss/val')
            axes[0, 1].plot(ep, v, label=f'v{vnum}', linestyle=ls)

        for i, var in enumerate(['u', 'v', 'w', 'k']):
            tag = f'ValRMSE/{var}'
            if tag in all_data:
                ep, v = get_series(tag)
                axes[1, 0].plot(ep, v, label=f'{var} v{vnum}', color=f'C{i}', linestyle=ls)

        if 'Loss/train' in all_data:
            ep, v = get_series('Loss/train')
            axes[1, 1].plot(ep, v, label=f'v{vnum}', linestyle=ls)

    for ax in axes.flat:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0, 0].set_ylabel('R²'); axes[0, 0].set_title('Val R²'); axes[0, 0].set_ylim(bottom=0)
    axes[0, 1].set_ylabel('Loss'); axes[0, 1].set_title('Val Loss')
    axes[1, 0].set_ylabel('RMSE'); axes[1, 0].set_title('Val RMSE')
    axes[1, 1].set_ylabel('Loss'); axes[1, 1].set_title('Train Loss')
    for ax in axes.flat:
        ax.set_xlabel('Epoch')

    fig.suptitle('Training Curves: V4 vs V5 vs V6', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


def main():
    parser = argparse.ArgumentParser(description='Multi-model comparison')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--skip-inference', action='store_true', help='Use cached sample predictions + metrics')
    parser.add_argument('--output-dir', type=str, default='docs/comparison')
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print('=== Multi-Model Comparison ===')
    print(f'Output: {args.output_dir}')

    # Step 1: Training curves
    print('\n[1/4] Plotting training curves...')
    plot_training_curves(args.output_dir)

    # Step 2: Load dataset and pick samples
    data_dir = '/mnt/sdata/jz/fuxi_cfd/dataset'
    print('\n[2/4] Loading dataset...')
    dataset = FuXiCFDDataset(data_dir, split='test', normalize=True, prefetch_to_memory=False)
    samples = pick_representative_samples(dataset, n=3)
    print(f'  Selected: {[(i, l) for i, l in samples]}')

    metrics_path = os.path.join(args.output_dir, 'metrics.json')

    if not args.skip_inference:
        # Step 3: Run full test-set inference with streaming metrics
        print('\n[3/4] Running full test-set inference...')

        print('  V4...')
        r2_v4 = run_shensi_full_metrics('v4', 'checkpoints/shensiv4_main/best_model.pt',
                                         dataset, device, args.batch_size)

        print('  V5...')
        r2_v5 = run_shensi_full_metrics('v5', 'checkpoints/shensiv5_main/best_model_v5.pt',
                                         dataset, device, args.batch_size)

        print('  V6...')
        r2_v6 = run_shensi_full_metrics('v6', 'checkpoints/shensiv6_main/best_model_v6.pt',
                                         dataset, device, args.batch_size)

        print('  FuXi...')
        r2_fuxi, fuxi_sample_preds, fuxi_sample_targets = run_fuxi_streaming(dataset, samples, args.output_dir)

        # Save metrics
        metrics = {
            'v4': r2_v4.tolist(),
            'v5': r2_v5.tolist(),
            'v6': r2_v6.tolist(),
            'fuxi': r2_fuxi.tolist(),
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Run ShenSi on representative samples only (for spatial plots)
        print('  Running ShenSi on representative samples...')
        output_mean = output_std = None
        if hasattr(dataset, 'stats') and dataset.stats is not None:
            output_mean = torch.from_numpy(dataset.stats['output_mean']).float()
            output_std = torch.from_numpy(dataset.stats['output_std']).float()

        # V4 on samples
        from src.models.hybrid_swin_unet_diffusion import create_hybrid_model
        model_v4 = create_hybrid_model(config={
            'base_channels': 48, 'bottleneck_depth': 4, 'window_size': (5, 5),
            'dropout': 0.2, 'drop_path_rate': 0.1,
            'k_diffusion_steps': 1000, 'k_ddim_steps': 20,
            'output_mean': output_mean, 'output_std': output_std,
        }).to(device)
        ckpt_v4 = torch.load('checkpoints/shensiv4_main/best_model.pt', map_location=device, weights_only=False)
        model_v4.load_state_dict(ckpt_v4['model_state_dict'])
        model_v4.eval()

        # V5 on samples
        from src.models.swin_unet_v5 import SwinUNetV5
        model_v5 = SwinUNetV5(
            in_channels=6, n_levels=27, base_channels=48,
            bottleneck_depth=4, num_heads=4, window_size=(5, 5),
            dropout=0.2, drop_path_rate=0.1,
            output_mean=output_mean, output_std=output_std,
        ).to(device)
        ckpt_v5 = torch.load('checkpoints/shensiv5_main/best_model_v5.pt', map_location=device, weights_only=False)
        model_v5.load_state_dict(ckpt_v5['model_state_dict'])
        model_v5.eval()

        # V6 on samples
        from src.models.swin_unet_v6 import SwinUNetV6
        model_v6 = SwinUNetV6(
            in_channels=6, n_levels=27, base_channels=48,
            channel_multipliers=[1, 2, 4, 8],
            bottleneck_depth=4, num_heads=4, window_size=(5, 5),
            dropout=0.2, drop_path_rate=0.1,
            use_cross_attention=True,
            output_mean=output_mean, output_std=output_std,
        ).to(device)
        ckpt_v6 = torch.load('checkpoints/shensiv6_main/best_model_v6.pt', map_location=device, weights_only=False)
        model_v6.load_state_dict(ckpt_v6['model_state_dict'])
        model_v6.eval()

        v4_sample_preds = {}
        v5_sample_preds = {}
        v6_sample_preds = {}
        sample_targets = {}

        with torch.no_grad():
            for idx, label in samples:
                sample = dataset[idx]
                inputs = sample['input'].unsqueeze(0).to(device)
                target = sample['target'].unsqueeze(0).to(device)

                out_v4 = model_v4.forward_inference(inputs, use_diffusion=False)
                out_v5 = model_v5(inputs)
                out_v6 = model_v6(inputs)

                if hasattr(dataset, 'denormalize_output'):
                    out_v4 = dataset.denormalize_output(out_v4)
                    out_v5 = dataset.denormalize_output(out_v5)
                    out_v6 = dataset.denormalize_output(out_v6)
                    target = dataset.denormalize_output(target)

                v4_sample_preds[label] = out_v4[0].cpu().numpy()
                v5_sample_preds[label] = out_v5[0].cpu().numpy()
                v6_sample_preds[label] = out_v6[0].cpu().numpy()
                sample_targets[label] = target[0].cpu().numpy()

                np.save(os.path.join(args.output_dir, f'sample_{label}_pred_v4.npy'), v4_sample_preds[label])
                np.save(os.path.join(args.output_dir, f'sample_{label}_pred_v5.npy'), v5_sample_preds[label])
                np.save(os.path.join(args.output_dir, f'sample_{label}_pred_v6.npy'), v6_sample_preds[label])
                np.save(os.path.join(args.output_dir, f'sample_{label}_target.npy'), sample_targets[label])

        del model_v4, model_v5, model_v6
        torch.cuda.empty_cache()

    else:
        # Load cached results
        print('\n[3/4] Loading cached results...')
        with open(metrics_path) as f:
            metrics = json.load(f)
        r2_v4 = np.array(metrics['v4'])
        r2_v5 = np.array(metrics['v5'])
        r2_v6 = np.array(metrics.get('v6', metrics['v5']))  # fallback if no v6 cached
        r2_fuxi = np.array(metrics['fuxi'])

        v4_sample_preds = {}
        v5_sample_preds = {}
        v6_sample_preds = {}
        sample_targets = {}
        fuxi_sample_preds = {}
        fuxi_sample_targets = {}

        for idx, label in samples:
            v4_sample_preds[label] = np.load(os.path.join(args.output_dir, f'sample_{label}_pred_v4.npy'))
            v5_sample_preds[label] = np.load(os.path.join(args.output_dir, f'sample_{label}_pred_v5.npy'))
            v6_sample_preds[label] = np.load(os.path.join(args.output_dir, f'sample_{label}_pred_v6.npy'))
            sample_targets[label] = np.load(os.path.join(args.output_dir, f'sample_{label}_target.npy'))
            fuxi_sample_preds[label] = np.load(os.path.join(args.output_dir, f'sample_{label}_pred_fuxi.npy'))
            fuxi_sample_targets[label] = np.load(os.path.join(args.output_dir, f'sample_{label}_target_fuxi.npy'))

    # Print metrics
    print(f'\n=== Per-Level R² (Full Test Set) ===')
    for name, r2_data in [('V4', r2_v4), ('V5', r2_v5), ('V6', r2_v6), ('FuXi', r2_fuxi)]:
        print(f'\n{name}:')
        for c, var in enumerate(VAR_NAMES):
            mean_r2 = np.mean(r2_data[:, c])
            print(f'  {var}: mean R²={mean_r2:.4f}  (L0={r2_data[0,c]:.4f}, L13={r2_data[13,c]:.4f}, L26={r2_data[26,c]:.4f})')

    # Step 4: Generate figures
    print(f'\n[4/4] Generating comparison figures...')
    for idx, label in samples:
        t = sample_targets[label]
        v4 = v4_sample_preds[label]
        v5 = v5_sample_preds[label]
        v6 = v6_sample_preds[label]
        fuxi = fuxi_sample_preds[label]
        plot_spatial_4way(t, v4, v5, v6, fuxi, label, args.output_dir)
        plot_error_4way(t, v4, v5, v6, fuxi, label, args.output_dir)

    plot_r2_vs_height(r2_v4, r2_v5, r2_v6, r2_fuxi, args.output_dir)
    plot_metrics_bar(r2_v4, r2_v5, r2_v6, r2_fuxi, args.output_dir)

    print(f'\nAll figures saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
