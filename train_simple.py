#!/usr/bin/env python3
"""
简化版训练脚本 - 专注于基础MSE损失，避免振荡
"""

import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.swin_unet_lite import create_lite_model
from src.data.fuxi_cfd_dataset import FuXiCFDDataset, create_dataloaders


def train_simple():
    """简化训练流程"""
    
    # 配置
    gpu_id = 2
    batch_size = 8
    epochs = 50
    lr = 1e-4
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = 'cuda'
    
    print('\n' + '='*70)
    print('🚀 简化版训练 - 基础MSE Loss')
    print('='*70)
    print(f'   GPU: {torch.cuda.get_device_name(0)}')
    print(f'   Batch Size: {batch_size}')
    print(f'   Learning Rate: {lr}')
    print(f'   Epochs: {epochs}')
    print('='*70 + '\n')
    
    # 创建模型
    print('📦 创建模型...')
    model = create_lite_model(config={
        'base_channels': 64,
        'bottleneck_depth': 6,
        'window_size': (5, 5),
    })
    model = model.to(device)
    
    params = model.get_num_params()
    print(f'   参数量: {params["total"]:,} ({params["total_mb"]:.1f} MB)')
    
    # 加载数据
    print('\n📂 加载数据集...')
    train_loader, val_loader, _ = create_dataloaders(
        data_dir='/mnt/sdata/jz/fuxi_cfd/dataset',
        batch_size=batch_size,
        num_workers=4,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        prefetch_to_memory=False,
    )
    
    # 优化器和损失
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.MSELoss()
    
    # 训练循环
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # 训练
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # 只使用MSE损失
            loss = criterion(outputs, targets)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            train_loss += loss.item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f'  [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}')
        
        train_loss /= len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # 学习率调度
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        print(f'\n📅 Epoch {epoch}/{epochs}')
        print(f'   Train Loss: {train_loss:.6f}')
        print(f'   Val Loss:   {val_loss:.6f}')
        print(f'   LR:         {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'   Time:       {epoch_time:.1f}s')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'checkpoints/best_model_simple.pt')
            print(f'   🏆 最佳模型已保存!')
        
        # 每10个epoch保存一次
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, f'checkpoints/checkpoint_epoch{epoch}.pt')
    
    print('\n' + '='*70)
    print('🎉 训练完成！')
    print(f'   最佳验证损失: {best_val_loss:.6f}')
    print('='*70)


if __name__ == '__main__':
    train_simple()
