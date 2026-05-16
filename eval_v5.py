#!/usr/bin/env python3
"""Evaluate shensiv5 (SwinUNetV5, physics-informed decoder) on test set.

Usage:
    python eval_v5.py
    python eval_v5.py --ckpt path/to/ckpt.pt
"""

import os
import sys
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.swin_unet_v5 import SwinUNetV5
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


def mae(pred, target):
    return np.mean(np.abs(pred - target))


def main():
    parser = argparse.ArgumentParser(description='Evaluate shensiv5 on test set')
    parser.add_argument('--ckpt', type=str, default='checkpoints/shensiv5_main/best_model_v5.pt',
                        help='Checkpoint path')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=2)
    args = parser.parse_args()

    device = torch.device(args.device)
    data_dir = '/mnt/sdata/jz/fuxi_cfd/dataset'

    print('Loading dataset...')
    test_dataset = FuXiCFDDataset(data_dir, split='test', normalize=True, prefetch_to_memory=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print('Loading model...')
    output_mean = output_std = None
    if hasattr(test_dataset, 'stats') and test_dataset.stats is not None:
        output_mean = torch.from_numpy(test_dataset.stats['output_mean']).float()
        output_std = torch.from_numpy(test_dataset.stats['output_std']).float()

    model = SwinUNetV5(
        in_channels=6, n_levels=27, base_channels=48,
        bottleneck_depth=4, num_heads=4, window_size=(5, 5),
        dropout=0.2, drop_path_rate=0.1,
        output_mean=output_mean, output_std=output_std,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f'Evaluating on {len(test_dataset)} samples...')

    all_pred, all_target = [], []

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
            if (i + 1) % 50 == 0:
                print(f'  {i+1}/{len(test_loader)} batches done')

    pred = np.concatenate(all_pred, axis=0)
    target = np.concatenate(all_target, axis=0)
    N, L, C, H, W = pred.shape

    print(f'\n{"="*60}')
    print('shensiv5 (Physics-Informed Decoder) vs FuXi-CFD (Test Set)')
    print(f'{"="*60}')
    print(f'Checkpoint: {args.ckpt}')
    print(f'Samples: {N}, Levels: {L}')

    var_names = ['u', 'v', 'w', 'k']
    for c, name in enumerate(var_names):
        p = pred[:, :, c]
        t = target[:, :, c]
        print(f'\n{name}:')
        print(f'  Overall  R²={r2(p, t):.4f}  RMSE={rmse(p, t):.4f}  MAE={mae(p, t):.4f}')

        for lvl in [0, 5, 10, 15, 20, 26]:
            if lvl < L:
                print(f'  L{lvl:2d}    R²={r2(pred[:, lvl, c], target[:, lvl, c]):.4f}  RMSE={rmse(pred[:, lvl, c], target[:, lvl, c]):.4f}')

    print(f'\n{"="*60}')


if __name__ == '__main__':
    main()
