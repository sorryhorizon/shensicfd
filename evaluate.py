#!/usr/bin/env python3
"""
ShenSi-CFD V2 评估脚本

在测试集上评估模型，计算物理空间指标，与 FuXi-CFD 基准对比
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.swin_unet_lite import create_lite_model
from src.data.fuxi_cfd_dataset import FuXiCFDDataset

VERTICAL_LEVELS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
                   55, 60, 65, 70, 75, 80, 85, 90, 95, 100,
                   106.5, 114.95, 125.94, 140.22, 158.78,
                   182.91, 214.29]

FUXI_BENCHMARK = {
    'u': {'r2': 0.998, 'rmse': 0.16},
    'v': {'r2': 0.998, 'rmse': 0.16},
    'w': {'r2': 0.998, 'rmse': 0.05},
    'k': {'r2': 0.990, 'rmse': 0.12},
}


def compute_rmse(pred, target):
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def compute_mae(pred, target):
    return float(np.mean(np.abs(pred - target)))


def compute_r2(pred, target):
    ss_res = np.sum((pred - target) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-10))


def compute_pearson(pred, target):
    p = pred.flatten()
    t = target.flatten()
    if len(p) < 2:
        return 0.0
    corr, _ = pearsonr(p, t)
    return float(corr)


def compute_bias(pred, target):
    return float(np.mean(pred - target))


def evaluate_model(checkpoint_path, data_dir, gpu_id=3, batch_size=8, max_samples=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = 'cuda'
    
    print('\n' + '='*70)
    print('📊 ShenSi-CFD V2 模型评估')
    print('='*70)
    
    print(f'\n📦 加载模型: {checkpoint_path}')
    model = create_lite_model(config={
        'base_channels': 64,
        'bottleneck_depth': 6,
        'window_size': (5, 5),
    })
    
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).eval()
    
    print(f'   Epoch: {ckpt["epoch"]}, Val Loss: {ckpt["val_loss"]:.6f}')
    
    print(f'\n📂 加载测试集: {data_dir}')
    test_dataset = FuXiCFDDataset(
        data_dir=data_dir,
        split='test',
        normalize=True,
        prefetch_to_memory=False,
    )
    
    stats = test_dataset.stats
    output_mean = stats['output_mean']
    output_std = stats['output_std']
    print(f'   output_mean: {output_mean}')
    print(f'   output_std:  {output_std}')
    
    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    print(f'\n🔬 开始评估 (测试集: {len(test_dataset)} 样本)...')
    
    all_preds = {name: [] for name in ['u', 'v', 'w', 'k']}
    all_targets = {name: [] for name in ['u', 'v', 'w', 'k']}
    all_level_preds = {i: [] for i in range(27)}
    all_level_targets = {i: [] for i in range(27)}
    
    n_eval = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            outputs = model(inputs)
            
            outputs_np = outputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            
            for i, name in enumerate(['u', 'v', 'w', 'k']):
                pred_phys = outputs_np[:, :, i] * output_std[i] + output_mean[i]
                target_phys = targets_np[:, :, i] * output_std[i] + output_mean[i]
                all_preds[name].append(pred_phys)
                all_targets[name].append(target_phys)
            
            for l in range(27):
                pred_l = outputs_np[:, l, :, :, :] * output_std[np.newaxis, :, np.newaxis, np.newaxis] + output_mean[np.newaxis, :, np.newaxis, np.newaxis]
                target_l = targets_np[:, l, :, :, :] * output_std[np.newaxis, :, np.newaxis, np.newaxis] + output_mean[np.newaxis, :, np.newaxis, np.newaxis]
                all_level_preds[l].append(pred_l)
                all_level_targets[l].append(target_l)
            
            n_eval += inputs.shape[0]
            if (batch_idx + 1) % 20 == 0:
                print(f'   [{n_eval}/{len(test_dataset)}] 已评估')
            
            if max_samples and n_eval >= max_samples:
                break
    
    print(f'\n📈 计算指标...')
    
    results = {'per_variable': {}, 'per_level': {}, 'overall': {}}
    
    for name in ['u', 'v', 'w', 'k']:
        pred = np.concatenate(all_preds[name], axis=0)
        target = np.concatenate(all_targets[name], axis=0)
        
        results['per_variable'][name] = {
            'rmse': compute_rmse(pred, target),
            'mae': compute_mae(pred, target),
            'r2': compute_r2(pred, target),
            'pearson': compute_pearson(pred, target),
            'bias': compute_bias(pred, target),
        }
    
    for l in range(27):
        pred_l = np.concatenate(all_level_preds[l], axis=0)
        target_l = np.concatenate(all_level_targets[l], axis=0)
        height = VERTICAL_LEVELS[l]
        results['per_level'][f'{height}m'] = {
            'rmse': compute_rmse(pred_l, target_l),
            'mae': compute_mae(pred_l, target_l),
            'r2': compute_r2(pred_l, target_l),
        }
    
    all_pred = np.concatenate([np.concatenate(all_preds[n], axis=0) for n in ['u', 'v', 'w', 'k']], axis=0)
    all_target = np.concatenate([np.concatenate(all_targets[n], axis=0) for n in ['u', 'v', 'w', 'k']], axis=0)
    results['overall'] = {
        'rmse': compute_rmse(all_pred, all_target),
        'mae': compute_mae(all_pred, all_target),
        'r2': compute_r2(all_pred, all_target),
    }
    
    report = generate_report(results)
    
    report_path = 'logs/evaluation_report_v2.md'
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(report)
    print(f'\n📄 报告已保存: {report_path}')
    
    results_path = 'logs/evaluation_results_v2.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


def generate_report(results):
    lines = []
    lines.append('# ShenSi-CFD V2 评估报告')
    lines.append('')
    
    lines.append('## 整体指标')
    lines.append('')
    lines.append('| 指标 | 值 |')
    lines.append('|------|-----|')
    for k, v in results['overall'].items():
        lines.append(f'| {k} | {v:.6f} |')
    
    lines.append('')
    lines.append('## 各变量指标')
    lines.append('')
    lines.append('| 变量 | RMSE | MAE | R² | Pearson | Bias |')
    lines.append('|------|------|-----|----|---------|------|')
    for name in ['u', 'v', 'w', 'k']:
        m = results['per_variable'][name]
        lines.append(f'| {name} | {m["rmse"]:.4f} | {m["mae"]:.4f} | {m["r2"]:.6f} | {m["pearson"]:.6f} | {m["bias"]:.4f} |')
    
    lines.append('')
    lines.append('## 与 FuXi-CFD 基准对比')
    lines.append('')
    lines.append('| 变量 | Our R² | Target R² | 状态 | Our RMSE | Target RMSE |')
    lines.append('|------|--------|-----------|------|----------|-------------|')
    for name in ['u', 'v', 'w', 'k']:
        our_r2 = results['per_variable'][name]['r2']
        our_rmse = results['per_variable'][name]['rmse']
        target_r2 = FUXI_BENCHMARK[name]['r2']
        target_rmse = FUXI_BENCHMARK[name]['rmse']
        status = '✅ PASS' if our_r2 >= target_r2 * 0.95 else '⚠️ GAP'
        lines.append(f'| {name} | {our_r2:.6f} | {target_r2:.3f} | {status} | {our_rmse:.4f} | {target_rmse:.2f} |')
    
    lines.append('')
    lines.append('## 各高度层 R²')
    lines.append('')
    lines.append('| 高度(m) | RMSE | MAE | R² |')
    lines.append('|---------|------|-----|-----|')
    for h_key in [f'{h}m' for h in [5, 10, 50, 100, 150, 214.29]]:
        if h_key in results['per_level']:
            m = results['per_level'][h_key]
            lines.append(f'| {h_key} | {m["rmse"]:.4f} | {m["mae"]:.4f} | {m["r2"]:.6f} |')
    
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model_v2.pt')
    parser.add_argument('--data_dir', type=str, default='/mnt/sdata/jz/fuxi_cfd/dataset')
    parser.add_argument('--gpu_id', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_samples', type=int, default=None)
    args = parser.parse_args()
    
    evaluate_model(args.checkpoint, args.data_dir, args.gpu_id, args.batch_size, args.max_samples)
