#!/usr/bin/env python3
"""
Baseline 模型在 Test Set 上的评估脚本
对比 FuXi-CFD 真实数据

Usage:
    python eval_baseline.py

输出:
    - results/eval_baseline.json: 定量指标
    - results/eval_baseline/figures/: 可视化图
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.swin_unet_lite import create_lite_model
from src.data.fuxi_cfd_dataset import FuXiCFDDataset


def compute_r2(pred: np.ndarray, target: np.ndarray) -> float:
    """计算 Pearson R^2"""
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    mask = ~np.isnan(pred_flat) & ~np.isnan(target_flat)
    p, t = pred_flat[mask], target_flat[mask]
    if len(p) < 2:
        return 0.0
    cov = np.cov(p, t)[0, 1]
    var_p = np.var(p)
    var_t = np.var(t)
    if var_p < 1e-12 or var_t < 1e-12:
        return 0.0
    r2 = (cov ** 2) / (var_p * var_t)
    return max(0.0, min(1.0, r2))


def compute_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return np.sqrt(np.mean((pred - target) ** 2))


def compute_mae(pred: np.ndarray, target: np.ndarray) -> float:
    return np.mean(np.abs(pred - target))


def eval_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = '/mnt/sdata/jz/fuxi_cfd/dataset'
    ckpt_path = 'checkpoints/best_model_baseline.pt'

    print('\n' + '='*70)
    print('ShenSi-CFD Baseline Evaluation vs FuXi-CFD')
    print('='*70)

    # Load dataset
    test_dataset = FuXiCFDDataset(data_dir, split='test', normalize=True, prefetch_to_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=False)

    # Load model
    output_mean = None
    output_std = None
    if hasattr(test_dataset, 'stats') and test_dataset.stats is not None:
        output_mean = torch.from_numpy(test_dataset.stats['output_mean']).float()
        output_std = torch.from_numpy(test_dataset.stats['output_std']).float()

    model = create_lite_model(config={
        'base_channels': 32,
        'bottleneck_depth': 4,
        'window_size': (5, 5),
        'dropout': 0.2,
        'drop_path_rate': 0.1,
        'use_physics_constraint': False,
        'output_mean': output_mean,
        'output_std': output_std,
    }).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'   Loaded checkpoint: {ckpt_path} (epoch {ckpt.get("epoch", "?")})')

    # Collect predictions
    all_pred = []
    all_target = []
    all_case_ids = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            outputs = model(inputs)

            # Denormalize to physical space
            if hasattr(test_dataset, 'denormalize_output'):
                outputs = test_dataset.denormalize_output(outputs)
                targets = test_dataset.denormalize_output(targets)

            all_pred.append(outputs.cpu().numpy())
            all_target.append(targets.cpu().numpy())
            all_case_ids.extend(batch['case_id'])

    pred = np.concatenate(all_pred, axis=0)   # (N, 27, 4, 300, 300)
    target = np.concatenate(all_target, axis=0)
    N, L, C, H, W = pred.shape

    print(f'   Test samples: {N}')
    print(f'   Shape: {pred.shape}')

    # Compute metrics
    var_names = ['u', 'v', 'w', 'k']
    level_heights = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
                     55, 60, 65, 70, 75, 80, 85, 90, 95, 100,
                     106.5, 114.95, 125.94, 140.22, 158.78,
                     182.91, 214.29]

    results = {
        'checkpoint': ckpt_path,
        'n_samples': N,
        'overall': {},
        'per_level': {},
        'per_case': {},
    }

    # Overall per-variable
    print('\n' + '-'*70)
    print('Overall Metrics (Physical Space)')
    print('-'*70)
    for c, name in enumerate(var_names):
        p = pred[:, :, c]
        t = target[:, :, c]
        r2 = compute_r2(p, t)
        rmse = compute_rmse(p, t)
        mae = compute_mae(p, t)
        bias = np.mean(p - t)
        results['overall'][name] = {
            'r2': float(r2),
            'rmse': float(rmse),
            'mae': float(mae),
            'bias': float(bias),
        }
        print(f'   {name}: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, Bias={bias:+.4f}')

    # Per-level metrics
    print('\n' + '-'*70)
    print('Per-Level R²')
    print('-'*70)
    for l in range(L):
        h = level_heights[l] if l < len(level_heights) else l
        level_r2 = {}
        for c, name in enumerate(var_names):
            p = pred[:, l, c]
            t = target[:, l, c]
            level_r2[name] = float(compute_r2(p, t))
        results['per_level'][f'level_{l}_h{h}m'] = level_r2
        if l % 5 == 0 or l == L - 1:
            r2_str = ', '.join(f'{n}={level_r2[n]:.3f}' for n in var_names)
            print(f'   Level {l:2d} ({h:6.1f}m): {r2_str}')

    # Per-case sample (first 5)
    for i in range(min(5, N)):
        case_r2 = {}
        for c, name in enumerate(var_names):
            case_r2[name] = float(compute_r2(pred[i, :, c], target[i, :, c]))
        results['per_case'][all_case_ids[i]] = case_r2

    # Save results
    out_dir = Path('results/eval_baseline')
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n   Results saved: {out_dir / "metrics.json"}')

    # Visualize: first sample, level 0 and level 13
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        sample_idx = 0
        levels_to_plot = [0, 9, 18, 26]

        for level in levels_to_plot:
            fig, axes = plt.subplots(4, 3, figsize=(15, 20))
            h = level_heights[level] if level < len(level_heights) else level

            for c, name in enumerate(var_names):
                p = pred[sample_idx, level, c]
                t = target[sample_idx, level, c]
                err = p - t

                vmax = max(np.abs(p).max(), np.abs(t).max())
                emax = np.abs(err).max()

                im0 = axes[c, 0].imshow(t, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
                axes[c, 0].set_title(f'{name} - Ground Truth')
                plt.colorbar(im0, ax=axes[c, 0])

                im1 = axes[c, 1].imshow(p, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
                axes[c, 1].set_title(f'{name} - Prediction (R²={compute_r2(p, t):.3f})')
                plt.colorbar(im1, ax=axes[c, 1])

                im2 = axes[c, 2].imshow(err, cmap='RdBu_r', vmin=-emax, vmax=emax)
                axes[c, 2].set_title(f'{name} - Error')
                plt.colorbar(im2, ax=axes[c, 2])

            plt.suptitle(f'Sample {all_case_ids[sample_idx]} - Level {level} ({h}m)', fontsize=14)
            plt.tight_layout()
            plt.savefig(out_dir / f'level_{level}_sample_{sample_idx}.png', dpi=150)
            plt.close()
            print(f'   Figure saved: level_{level}_sample_{sample_idx}.png')

        # R² vs Height plot
        fig, ax = plt.subplots(figsize=(10, 6))
        for c, name in enumerate(var_names):
            r2_levels = [results['per_level'][f'level_{l}_h{level_heights[l]}m'][name] for l in range(L)]
            ax.plot(level_heights[:L], r2_levels, marker='o', label=name, markersize=3)
        ax.set_xlabel('Height (m)')
        ax.set_ylabel('R²')
        ax.set_title('R² vs Height (Baseline vs FuXi-CFD)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / 'r2_vs_height.png', dpi=150)
        plt.close()
        print(f'   Figure saved: r2_vs_height.png')

    except ImportError:
        print('   matplotlib not available, skipping figures')

    print('\n' + '='*70)
    print('Evaluation Complete!')
    print('='*70)


if __name__ == '__main__':
    eval_model()
