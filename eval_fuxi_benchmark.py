#!/usr/bin/env python3
"""
ShenSi-CFD Baseline vs FuXi-CFD Benchmark

全面对标评估：
- 定量指标（R², RMSE, MAE, Bias, Correlation, RelErr）
- 分层分析（按高度、地形类型）
- 空间分布误差（山坡/背风/平地）
- 可视化对比图

Usage:
    python eval_fuxi_benchmark.py

输出:
    results/benchmark/metrics.json
    results/benchmark/figures/
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.swin_unet_lite import create_lite_model
from src.data.fuxi_cfd_dataset import FuXiCFDDataset


def r2(pred, target):
    p, t = pred.flatten(), target.flatten()
    mask = ~np.isnan(p) & ~np.isnan(t)
    p, t = p[mask], t[mask]
    if len(p) < 2:
        return 0.0
    cov = np.cov(p, t)[0, 1]
    vp, vt = np.var(p), np.var(t)
    if vp < 1e-12 or vt < 1e-12:
        return 0.0
    return max(0.0, min(1.0, (cov ** 2) / (vp * vt)))


def rmse(pred, target):
    return np.sqrt(np.mean((pred - target) ** 2))


def mae(pred, target):
    return np.mean(np.abs(pred - target))


def bias(pred, target):
    return np.mean(pred - target)


def rel_err(pred, target, threshold=0.1):
    """Relative error (MAPE variant)"""
    p, t = pred.flatten(), target.flatten()
    mask = np.abs(t) > threshold
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs(p[mask] - t[mask]) / np.abs(t[mask])) * 100


def pearson_r(pred, target):
    p, t = pred.flatten(), target.flatten()
    mask = ~np.isnan(p) & ~np.isnan(t)
    p, t = p[mask], t[mask]
    if len(p) < 2:
        return 0.0
    return np.corrcoef(p, t)[0, 1]


def classify_terrain(dem):
    """简单地形分类"""
    # dem: (H, W)
    grad_y, grad_x = np.gradient(dem)
    slope = np.sqrt(grad_x**2 + grad_y**2)

    flat = slope < 2.0
    moderate = (slope >= 2.0) & (slope < 10.0)
    steep = slope >= 10.0

    return {
        'flat': flat,
        'moderate': moderate,
        'steep': steep,
    }


def main():
    device = torch.device('cuda:0')
    data_dir = '/mnt/sdata/jz/fuxi_cfd/dataset'
    ckpt_path = 'checkpoints/best_model_baseline.pt'
    out_dir = Path('results/benchmark')
    fig_dir = out_dir / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print('\n' + '='*70)
    print('SHENSI-CFD vs FUXI-CFD BENCHMARK')
    print('='*70)

    # Load data
    print('\n[1/4] Loading test dataset...')
    test_dataset = FuXiCFDDataset(data_dir, split='test', normalize=True, prefetch_to_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
    print(f'      Samples: {len(test_dataset)}')

    # Load model
    print('\n[2/4] Loading baseline model...')
    output_mean = output_std = None
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
    print(f'      Checkpoint: {ckpt_path} (epoch {ckpt.get("epoch", "?")})')

    # Inference
    print('\n[3/4] Running inference...')
    all_pred = []
    all_target = []
    all_dem = []
    all_case_ids = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            outputs = model(inputs)

            if hasattr(test_dataset, 'denormalize_output'):
                outputs = test_dataset.denormalize_output(outputs)
                targets = test_dataset.denormalize_output(targets)

            all_pred.append(outputs.cpu().numpy())
            all_target.append(targets.cpu().numpy())
            all_dem.append(inputs[:, 2].cpu().numpy())  # DEM channel
            all_case_ids.extend(batch['case_id'])

            if (i + 1) % 100 == 0:
                print(f'      {i+1}/{len(test_loader)} batches')

    pred = np.concatenate(all_pred, axis=0)       # (N, 27, 4, H, W)
    target = np.concatenate(all_target, axis=0)
    dem_batch = np.concatenate(all_dem, axis=0)   # (N, H, W)
    N, L, C, H, W = pred.shape
    print(f'\n      Done. Shape: {pred.shape}')

    # Level heights
    level_heights = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
                     55, 60, 65, 70, 75, 80, 85, 90, 95, 100,
                     106.5, 114.95, 125.94, 140.22, 158.78,
                     182.91, 214.29]

    var_names = ['u', 'v', 'w', 'k']
    results = {}

    # ============================================================
    # 1. OVERALL METRICS
    # ============================================================
    print('\n' + '='*70)
    print('OVERALL METRICS (Physical Space)')
    print('='*70)

    for c, name in enumerate(var_names):
        p = pred[:, :, c]
        t = target[:, :, c]
        metrics = {
            'r2': float(r2(p, t)),
            'rmse': float(rmse(p, t)),
            'mae': float(mae(p, t)),
            'bias': float(bias(p, t)),
            'rel_err_%': float(rel_err(p, t)),
            'pearson_r': float(pearson_r(p, t)),
        }
        results[f'overall_{name}'] = metrics
        print(f'\n{name}:')
        print(f'  R²          = {metrics["r2"]:.4f}')
        print(f'  Pearson r   = {metrics["pearson_r"]:.4f}')
        print(f'  RMSE        = {metrics["rmse"]:.4f}')
        print(f'  MAE         = {metrics["mae"]:.4f}')
        print(f'  Bias        = {metrics["bias"]:+.4f}')
        print(f'  Rel Err     = {metrics["rel_err_%"]:.2f}%')

    # ============================================================
    # 2. PER-LEVEL METRICS
    # ============================================================
    print('\n' + '='*70)
    print('PER-LEVEL R²')
    print('='*70)

    per_level = {}
    for l in range(L):
        h = level_heights[l] if l < len(level_heights) else l
        lvl_data = {}
        for c, name in enumerate(var_names):
            p = pred[:, l, c]
            t = target[:, l, c]
            lvl_data[name] = {
                'r2': float(r2(p, t)),
                'rmse': float(rmse(p, t)),
            }
        per_level[f'L{l}_h{h}m'] = lvl_data

    print(f'\n{"Level":>8} {"Height":>8} {"u_R2":>8} {"v_R2":>8} {"w_R2":>8} {"k_R2":>8}')
    print('-' * 60)
    for l in range(L):
        h = level_heights[l] if l < len(level_heights) else l
        key = f'L{l}_h{h}m'
        d = per_level[key]
        print(f'{l:>8} {h:>7.1f}m {d["u"]["r2"]:>8.4f} {d["v"]["r2"]:>8.4f} {d["w"]["r2"]:>8.4f} {d["k"]["r2"]:>8.4f}')

    results['per_level'] = per_level

    # ============================================================
    # 3. TERRAIN-TYPE ANALYSIS
    # ============================================================
    print('\n' + '='*70)
    print('TERRAIN-TYPE ANALYSIS')
    print('='*70)

    terrain_metrics = defaultdict(lambda: defaultdict(list))
    n_samples = min(50, N)  # 采样部分样本做地形分类

    for i in range(n_samples):
        dem = dem_batch[i]
        terrain = classify_terrain(dem)

        for ttype, mask in terrain.items():
            mask_3d = np.broadcast_to(mask[np.newaxis, np.newaxis, :, :], (L, C, H, W))
            for c, name in enumerate(var_names):
                p = pred[i, :, c][mask]
                t = target[i, :, c][mask]
                if len(p) > 0:
                    terrain_metrics[ttype][name].append((p, t))

    for ttype in ['flat', 'moderate', 'steep']:
        print(f'\n{ttype.upper()} terrain:')
        for c, name in enumerate(var_names):
            pairs = terrain_metrics[ttype][name]
            if len(pairs) == 0:
                continue
            p_all = np.concatenate([p for p, t in pairs])
            t_all = np.concatenate([t for p, t in pairs])
            r2_val = r2(p_all, t_all)
            rmse_val = rmse(p_all, t_all)
            print(f'  {name}: R²={r2_val:.4f}, RMSE={rmse_val:.4f}')
            results[f'terrain_{ttype}_{name}'] = {'r2': float(r2_val), 'rmse': float(rmse_val)}

    # ============================================================
    # 4. SAVE RESULTS
    # ============================================================
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n{"="*70}')
    print(f'Results saved: {out_dir / "metrics.json"}')
    print(f'{"="*70}\n')

    # ============================================================
    # 5. VISUALIZATION
    # ============================================================
    print('[5/5] Generating figures...')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # (a) R² vs Height
        fig, ax = plt.subplots(figsize=(10, 6))
        for c, name in enumerate(var_names):
            r2_levels = [per_level[f'L{l}_h{level_heights[l]}m'][name]['r2'] for l in range(L)]
            ax.plot(level_heights[:L], r2_levels, marker='o', label=name, markersize=3)
        ax.set_xlabel('Height (m)', fontsize=12)
        ax.set_ylabel('R²', fontsize=12)
        ax.set_title('ShenSi-CFD vs FuXi-CFD: R² vs Height', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        plt.tight_layout()
        plt.savefig(fig_dir / 'r2_vs_height.png', dpi=150)
        plt.close()
        print(f'      Saved: r2_vs_height.png')

        # (b) RMSE vs Height
        fig, ax = plt.subplots(figsize=(10, 6))
        for c, name in enumerate(var_names):
            rmse_levels = [per_level[f'L{l}_h{level_heights[l]}m'][name]['rmse'] for l in range(L)]
            ax.plot(level_heights[:L], rmse_levels, marker='o', label=name, markersize=3)
        ax.set_xlabel('Height (m)', fontsize=12)
        ax.set_ylabel('RMSE', fontsize=12)
        ax.set_title('ShenSi-CFD vs FuXi-CFD: RMSE vs Height', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / 'rmse_vs_height.png', dpi=150)
        plt.close()
        print(f'      Saved: rmse_vs_height.png')

        # (c) Sample comparison: first test case, multiple levels
        sample_idx = 0
        levels_plot = [0, 5, 10, 15, 20, 26]
        for lvl in levels_plot:
            h = level_heights[lvl] if lvl < len(level_heights) else lvl
            fig, axes = plt.subplots(4, 3, figsize=(15, 20))
            fig.suptitle(f'Sample: {all_case_ids[sample_idx]} | Level {lvl} ({h}m)', fontsize=16)

            for c, name in enumerate(var_names):
                p = pred[sample_idx, lvl, c]
                t = target[sample_idx, lvl, c]
                err = p - t

                vmax = max(np.abs(p).max(), np.abs(t).max(), 0.1)
                emax = max(np.abs(err).max(), 0.1)

                im0 = axes[c, 0].imshow(t, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
                axes[c, 0].set_title(f'{name} - FuXi-CFD Ground Truth')
                plt.colorbar(im0, ax=axes[c, 0], fraction=0.046)

                im1 = axes[c, 1].imshow(p, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
                axes[c, 1].set_title(f'{name} - ShenSi-CFD Prediction')
                plt.colorbar(im1, ax=axes[c, 1], fraction=0.046)

                im2 = axes[c, 2].imshow(err, cmap='RdBu_r', vmin=-emax, vmax=emax, aspect='auto')
                r2_val = r2(p, t)
                axes[c, 2].set_title(f'{name} - Error (R²={r2_val:.3f})')
                plt.colorbar(im2, ax=axes[c, 2], fraction=0.046)

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(fig_dir / f'sample_L{lvl}_h{h}m.png', dpi=120)
            plt.close()
            print(f'      Saved: sample_L{lvl}_h{h}m.png')

        # (d) Scatter plots (overall)
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        for idx, (c, name) in enumerate(zip(range(4), var_names)):
            ax = axes[idx // 2, idx % 2]
            p = pred[:, :, c].flatten()
            t = target[:, :, c].flatten()

            # Sample for speed
            n = min(len(p), 50000)
            indices = np.random.choice(len(p), n, replace=False)
            p_s, t_s = p[indices], t[indices]

            ax.scatter(t_s, p_s, alpha=0.1, s=1)
            lim = max(np.abs(t_s).max(), np.abs(p_s).max())
            ax.plot([-lim, lim], [-lim, lim], 'r--', lw=2, label='Perfect')
            ax.set_xlabel('FuXi-CFD (Ground Truth)')
            ax.set_ylabel('ShenSi-CFD (Prediction)')
            r2_val = r2(p, t)
            rmse_val = rmse(p, t)
            ax.set_title(f'{name}: R²={r2_val:.3f}, RMSE={rmse_val:.3f}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle('ShenSi-CFD vs FuXi-CFD: Scatter Plots', fontsize=14)
        plt.tight_layout()
        plt.savefig(fig_dir / 'scatter_overall.png', dpi=150)
        plt.close()
        print(f'      Saved: scatter_overall.png')

    except ImportError:
        print('      matplotlib not available, skipping figures')

    print(f'\n{"="*70}')
    print('BENCHMARK COMPLETE')
    print(f'{"="*70}\n')


if __name__ == '__main__':
    main()
