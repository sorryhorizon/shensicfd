#!/usr/bin/env python3
"""
优化版训练脚本 v2 - 核心修复：数据归一化
之前的bug: fuxi_cfd_dataset.py 中归一化是 pass (空操作)
修复: 实现了真正的 (x - mean) / std 归一化
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.swin_unet_lite import create_lite_model
from src.data.fuxi_cfd_dataset import FuXiCFDDataset


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
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
            lr = lr.item()
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


def train():
    gpu_id = 3
    batch_size = 8
    accum_steps = 4
    epochs = 100
    lr = 3e-4
    warmup_epochs = 5
    weight_decay = 0.01
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = 'cuda'
    
    print('\n' + '='*70)
    print('🚀 优化版训练 v2 - 数据归一化修复')
    print('='*70)
    print(f'   GPU: {torch.cuda.get_device_name(0)}')
    print(f'   Batch Size: {batch_size} (等效: {batch_size * accum_steps})')
    print(f'   Learning Rate: {lr}')
    print(f'   Warmup Epochs: {warmup_epochs}')
    print(f'   Weight Decay: {weight_decay}')
    print(f'   Epochs: {epochs}')
    print('='*70 + '\n')
    
    print('📦 创建模型...')
    model = create_lite_model(config={
        'base_channels': 64,
        'bottleneck_depth': 6,
        'window_size': (5, 5),
    })
    model = model.to(device)
    
    params = model.get_num_params()
    print(f'   参数量: {params["total"]:,} ({params["total_mb"]:.1f} MB)')
    
    print('\n📂 加载数据集（含归一化）...')
    train_dataset = FuXiCFDDataset(
        data_dir='/mnt/sdata/jz/fuxi_cfd/dataset',
        split='train',
        normalize=True,
        prefetch_to_memory=False,
    )
    val_dataset = FuXiCFDDataset(
        data_dir='/mnt/sdata/jz/fuxi_cfd/dataset',
        split='val',
        normalize=True,
        prefetch_to_memory=False,
    )
    
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    # 验证归一化是否生效
    print('\n🧪 验证归一化效果:')
    sample_batch = next(iter(train_loader))
    for i, name in enumerate(['u_100m', 'v_100m', 'dem', 'roughness']):
        d = sample_batch['input'][:, i]
        print(f'   {name}: mean={d.mean():.4f}, std={d.std():.4f}, range=[{d.min():.4f}, {d.max():.4f}]')
    print(f'   target: mean={sample_batch["target"].mean():.4f}, std={sample_batch["target"].std():.4f}')
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, epochs)
    criterion = nn.SmoothL1Loss(beta=1.0)
    
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    history = []
    
    log_dir = Path('logs/train_v2_normalized')
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path('checkpoints')
    ckpt_dir.mkdir(exist_ok=True)
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        current_lr = scheduler.step(epoch)
        
        model.train()
        train_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch['input'].to(device, non_blocking=True)
            targets = batch['target'].to(device, non_blocking=True)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss_scaled = loss / accum_steps
            loss_scaled.backward()
            
            if (batch_idx + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss += loss.item()
            n_batches += 1
            
            if (batch_idx + 1) % 100 == 0:
                print(f'  [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}')
        
        train_loss /= n_batches
        
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(device, non_blocking=True)
                targets = batch['target'].to(device, non_blocking=True)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_batches += 1
        
        val_loss /= val_batches
        
        epoch_time = time.time() - epoch_start
        
        record = {
            'epoch': epoch,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'lr': float(current_lr),
            'time': float(epoch_time),
        }
        history.append(record)
        
        print(f'\n📅 Epoch {epoch}/{epochs}')
        print(f'   Train Loss: {train_loss:.6f}')
        print(f'   Val Loss:   {val_loss:.6f}')
        print(f'   LR:         {current_lr:.7f}')
        print(f'   Time:       {epoch_time:.1f}s')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
            }, ckpt_dir / 'best_model_v2.pt')
            print(f'   🏆 最佳模型已保存! (val_loss: {val_loss:.6f})')
        else:
            patience_counter += 1
            print(f'   ⏳ 早停计数: {patience_counter}/{patience}')
        
        if patience_counter >= patience:
            print(f'\n⚠️ 早停触发!')
            break
        
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, ckpt_dir / f'v2_epoch{epoch}.pt')
            
            with open(log_dir / 'history.json', 'w') as f:
                json.dump(history, f, indent=2)
    
    with open(log_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print('\n' + '='*70)
    print('🎉 训练完成！')
    print(f'   最佳验证损失: {best_val_loss:.6f}')
    print(f'   训练历史已保存: {log_dir / "history.json"}')
    print('='*70)


if __name__ == '__main__':
    train()
