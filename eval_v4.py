#!/usr/bin/env python3
"""Evaluate shensiv4 (HybridSwinUNetDiffusion) on test set.

Usage:
    python eval_v4.py                          # DDIM sampling for k
    python eval_v4.py --no-diffusion           # regression k only
    python eval_v4.py --ckpt path/to/ckpt.pt   # custom checkpoint
"""

import os
import sys
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.hybrid_swin_unet_diffusion import create_hybrid_model
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
    parser = argparse.ArgumentParser(description='Evaluate shensiv4 on test set')
    parser.add_argument('--ckpt', type=str, default='checkpoints/shensiv4_main/best_model.pt',
                        help='Checkpoint path')
    parser.add_argument('--no-diffusion', action='store_true',
                        help='Use regression k instead of DDIM sampling')
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

    model = create_hybrid_model(config={
        'base_channels': 48, 'bottleneck_depth': 4, 'window_size': (5, 5),
        'dropout': 0.2, 'drop_path_rate': 0.1,
        'k_diffusion_steps': 1000, 'k_ddim_steps': 20,
        'output_mean': output_mean, 'output_std': output_std,
    }).to(device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    use_diffusion = not args.no_diffusion
    mode_str = 'DDIM' if use_diffusion else 'regression'
    print(f'Evaluating on {len(test_dataset)} samples (k: {mode_str})...')

    all_pred, all_target = [], []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            outputs = model.forward_inference(inputs, use_diffusion=use_diffusion)
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
    print(f'shensiv4 (Hybrid Diffusion, k: {mode_str}) vs FuXi-CFD (Test Set)')
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
