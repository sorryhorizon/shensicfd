#!/usr/bin/env python3
"""快速评估 baseline 模型在 test set 上的指标"""

import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.swin_unet_lite import create_lite_model
from src.data.fuxi_cfd_dataset import FuXiCFDDataset


def r2(pred, target):
    p, t = pred.flatten(), target.flatten()
    cov = np.cov(p, t)[0, 1]
    vp, vt = np.var(p), np.var(t)
    if vp < 1e-12 or vt < 1e-12:
        return 0.0
    return max(0.0, min(1.0, (cov ** 2) / (vp * vt)))


def rmse(pred, target):
    return np.sqrt(np.mean((pred - target) ** 2))


def mae(pred, target):
    return np.mean(np.abs(pred - target))


device = torch.device('cuda:0')
data_dir = '/mnt/sdata/jz/fuxi_cfd/dataset'
ckpt_path = 'checkpoints/shensiv2_baseline/best_model.pt'

print('Loading dataset...')
test_dataset = FuXiCFDDataset(data_dir, split='test', normalize=True, prefetch_to_memory=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)

print('Loading model...')
output_mean = output_std = None
if hasattr(test_dataset, 'stats') and test_dataset.stats is not None:
    output_mean = torch.from_numpy(test_dataset.stats['output_mean']).float()
    output_std = torch.from_numpy(test_dataset.stats['output_std']).float()

model = create_lite_model(config={
    'in_channels': 4, 'base_channels': 32, 'bottleneck_depth': 4, 'window_size': (5, 5),
    'dropout': 0.2, 'drop_path_rate': 0.1, 'num_heads': 4,
    'use_physics_constraint': False,
    'output_mean': output_mean, 'output_std': output_std,
}).to(device)

ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
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
print('BASELINE vs FuXi-CFD (Test Set)')
print(f'{"="*60}')
print(f'Samples: {N}, Levels: {L}')

var_names = ['u', 'v', 'w', 'k']
for c, name in enumerate(var_names):
    p = pred[:, :, c]
    t = target[:, :, c]
    print(f'\n{name}:')
    print(f'  Overall  R²={r2(p, t):.4f}  RMSE={rmse(p, t):.4f}  MAE={mae(p, t):.4f}')

    # Per-level
    for lvl in [0, 5, 10, 15, 20, 26]:
        if lvl < L:
            print(f'  L{lvl:2d}    R²={r2(pred[:, lvl, c], target[:, lvl, c]):.4f}  RMSE={rmse(pred[:, lvl, c], target[:, lvl, c]):.4f}')

print(f'\n{"="*60}')
