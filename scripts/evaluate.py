#!/usr/bin/env python3
"""
ShenSi-CFD 评估脚本

生成详细的性能报告，对标 FuXi-CFD 的评估体系

用法:
    python scripts/evaluate.py --checkpoint experiments/checkpoints/best.ckpt \
                              --data_dir /path/to/fuxi_data
    python scripts/evaluate.py --pred_dir ./predictions/test \
                              --target_dir /path/to/fuxi_data
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import ShenSiCFDDataset, FuXiNormalizer
from src.utils.metrics import CFDMetrics


def evaluate_model(model: torch.nn.Module, device: torch.device,
                   data_root: str, split: str = 'test',
                   output_report: str = './evaluation_report.md'):
    """
    在数据集上全面评估模型
    
    Args:
        model: 训练好的模型
        device: 计算设备
        data_root: 数据根目录
        split: 数据集划分 ('train', 'val', 'test')
        output_report: 报告输出路径
    """
    dataset = ShenSiCFDDataset(data_root, split=split, augment=False)
    
    metrics = CFDMetrics()
    
    print(f"[Evaluate] Evaluating on {split} set ({len(dataset)} samples)")
    
    for idx in range(len(dataset)):
        x, y = dataset[idx]
        x = x.unsqueeze(0).to(device)
        
        with torch.no_grad():
            y_hat = model(x)
        
        metrics.update(y_hat.squeeze(0), y)
        
        if (idx + 1) % 50 == 0 or idx == len(dataset) - 1:
            print(f"  [{idx+1}/{len(dataset)}] Evaluated")
    
    results = metrics.compute()
    report = metrics.generate_report(results)
    
    Path(output_report).parent.mkdir(parents=True, exist_ok=True)
    with open(output_report, 'w') as f:
        f.write(report)
    
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS ({split} set, n={len(dataset)})")
    print(f"{'='*60}")
    print(report)
    print(f"Report saved to: {output_report}")
    
    _print_fuxi_benchmark_comparison(results)
    
    return results


def evaluate_predictions(pred_dir: str, target_data_root: str,
                         output_report: str = './evaluation_report.md'):
    """
    评估已生成的预测结果（无需重新推理）
    
    Args:
        pred_dir: 预测结果目录 (每个子目录包含 outputs.npz)
        target_data_root: 目标数据根目录
        output_report: 报告输出路径
    """
    pred_path = Path(pred_dir)
    target_path = Path(target_data_root)
    
    metrics = CFDMetrics()
    
    pred_cases = sorted([d for d in pred_path.iterdir() if d.is_dir()])
    target_cases = sorted([d for d in target_path.iterdir() 
                          if d.is_dir() and (d / 'outputs.npz').exists()])
    
    print(f"[Evaluate] Comparing predictions vs ground truth")
    print(f"  Predictions: {len(pred_cases)} cases in {pred_dir}")
    print(f"  Targets:     {len(target_cases)} cases in {target_data_root}")
    
    matched = 0
    for target_case in target_cases:
        pred_case = pred_path / target_case.name
        if not pred_case.exists():
            continue
        
        target_file = target_case / 'outputs.npz'
        pred_file = pred_case / 'outputs.npz'
        
        if not pred_file.exists():
            continue
        
        target_data = np.load(target_file)
        pred_data = np.load(pred_file)
        
        u_pred = pred_data['u']
        v_pred = pred_data['v']
        w_pred = pred_data['w']
        k_pred = pred_data['k']
        
        u_target = target_data['u']
        v_target = target_data['v']
        w_target = target_data['w']
        k_target = target_data['k']
        
        pred_stacked = np.stack([u_pred, v_pred, w_pred, k_pred], axis=1)
        target_stacked = np.stack([u_target, v_target, w_target, k_target], axis=1)
        
        metrics.update(pred_stacked, target_stacked)
        matched += 1
    
    print(f"  Matched cases: {matched}")
    
    if matched > 0:
        results = metrics.compute()
        report = metrics.generate_report(results)
        
        with open(output_report, 'w') as f:
            f.write(report)
        
        print(f"\n{report}")
        _print_fuxi_benchmark_comparison(results)
        print(f"\nReport saved to: {output_report}")
        
        return results
    else:
        print("[Error] No matching cases found!")
        return None


def _print_fuxi_benchmark_comparison(results: dict):
    """打印与 FuXi-CFD 基准的对比"""
    fuxi_targets = {
        'u': {'r2': 0.998, 'rmse': 0.16},
        'v': {'r2': 0.998, 'rmse': 0.16},
        'w': {'r2': 0.998, 'rmse': 0.05},
        'k': {'r2': 0.990, 'rmse': 0.12},
    }
    
    print(f"\n{'='*60}")
    print("Benchmark Comparison (vs FuXi-CFD Targets)")
    print(f"{'='*60}")
    print(f"| Variable | Our R² | Target R² | Status   | Our RMSE | Target RMSE |")
    print(f"|----------|--------|-----------|----------|----------|-------------|")
    
    per_var = results.get('per_variable', {})
    for var_name in ['u', 'v', 'w', 'k']:
        our_r2 = per_var.get(var_name, {}).get('r2', 0)
        our_rmse = per_var.get(var_name, {}).get('rmse', float('inf'))
        target_r2 = fuxi_targets[var_name]['r2']
        target_rmse = fuxi_targets[var_name]['rmse']
        
        status = "✅ PASS" if our_r2 >= target_r2 * 0.95 else "⚠️ GAP"
        
        print(f"| {var_name:^8} | {our_r2:.4f} | {target_r2:.3f}     | {status:^8} | {our_rmse:.4f} | {target_rmse:.2f}       |")
    
    overall_r2 = results.get('overall', {}).get('r2', 0)
    print(f"\nOverall R²: {overall_r2:.4f}")


def main():
    parser = argparse.ArgumentParser(description='ShenSi-CFD Evaluation')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Model checkpoint path (for direct evaluation)')
    parser.add_argument('--config', type=str, default=None,
                        help='Model config file')
    parser.add_argument('--data_dir', type=str,
                        default='/mnt/sdata/jz/fuxi_cfd/dataset',
                        help='Data root directory')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate')
    parser.add_argument('--pred_dir', type=str, default=None,
                        help='Predictions directory (alternative to --checkpoint)')
    parser.add_argument('--output', type=str, default='./evaluation_report.md',
                        help='Output report path')
    args = parser.parse_args()
    
    print("=" * 60)
    print("ShenSi-CFD Evaluation")
    print("=" * 60)
    
    if args.pred_dir:
        evaluate_predictions(args.pred_dir, args.data_dir, args.output)
    elif args.checkpoint:
        from src.models.hybrid_model import HybridShenSiCFD
        
        config = {}
        if args.config:
            import yaml
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        
        model = HybridShenSiCFD(**config.get('model', {}))
        
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        state_dict = {k.replace('model.', ''): v for k, v in ckpt['state_dict'].items()}
        model.load_state_dict(state_dict, strict=False)
        
        device = torch.device('cpu')
        model = model.to(device).eval()
        
        evaluate_model(model, device, args.data_dir, args.split, args.output)
    else:
        print("[Error] Please specify either --checkpoint or --pred_dir")
        sys.exit(1)


if __name__ == '__main__':
    main()
