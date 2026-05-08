#!/usr/bin/env python3
"""
优化版训练脚本 - 解决loss振荡问题
优化点：
1. 更大的batch size (16) - 更稳定的梯度估计
2. 学习率warmup - 避免初期震荡
3. OneCycleLR - 更好的学习率调度
4. 梯度累积 - 模拟更大batch
5. L2正则化 + Dropout
6. 标签平滑
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


class WarmupCosineScheduler:
    """带warmup的余弦退火学习率调度器"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # 线性warmup
            lr = self.base_lr * (epoch / self.warmup_epochs)
        else:
            # 余弦退火
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
            lr = lr.item()
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


def train_optimized():
    """优化训练流程"""
    
    # 配置
    gpu_id = 2
    batch_size = 8           # batch size
    accum_steps = 4          # 梯度累积，等效batch=32
    epochs = 100             # 更多epoch
    lr = 5e-5                # 更低的学习率
    warmup_epochs = 5        # warmup轮数
    weight_decay = 0.001     # L2正则化
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = 'cuda'
    
    print('\n' + '='*70)
    print('🚀 优化版训练 - 解决Loss振荡')
    print('='*70)
    print(f'   GPU: {torch.cuda.get_device_name(0)}')
    print(f'   Batch Size: {batch_size} (等效: {batch_size * accum_steps})')
    print(f'   Learning Rate: {lr}')
    print(f'   Warmup Epochs: {warmup_epochs}')
    print(f'   Weight Decay: {weight_decay}')
    print(f'   Epochs: {epochs}')
    print('='*70 + '\n')
    
    # 创建模型
    print('📦 创建模型...')
    model = create_lite_model(config={
        'base_channels': 64,
        'bottleneck_depth': 6,
        'window_size': (5, 5),
        'dropout': 0.1,      # 添加dropout
    })
    model = model.to(device)
    
    params = model.get_num_params()
    print(f'   参数量: {params["total"]:,} ({params["total_mb"]:.1f} MB)')
    
    # 加载数据
    print('\n📂 加载数据集...')
    train_loader, val_loader, _ = create_dataloaders(
        data_dir='/mnt/sdata/jz/fuxi_cfd/dataset',
        batch_size=batch_size,
        num_workers=2,  # 减少worker数量
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        prefetch_to_memory=False,
    )
    
    # 优化器 - 使用更稳定的配置
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 学习率调度器 - warmup + cosine
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, epochs)
    
    # 损失函数 - 使用SmoothL1Loss更稳定
    criterion = nn.SmoothL1Loss(beta=1.0)
    
    # 训练循环
    best_val_loss = float('inf')
    patience = 15            # 早停耐心
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        current_lr = scheduler.step(epoch)
        
        # 训练
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 梯度累积
            loss = loss / accum_steps
            loss.backward()
            
            if (batch_idx + 1) % accum_steps == 0:
                # 梯度裁剪 - 更保守
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * accum_steps
            
            if (batch_idx + 1) % 50 == 0:
                print(f'  [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item() * accum_steps:.4f}')
        
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
        
        epoch_time = time.time() - epoch_start
        
        print(f'\n📅 Epoch {epoch}/{epochs}')
        print(f'   Train Loss: {train_loss:.6f}')
        print(f'   Val Loss:   {val_loss:.6f}')
        print(f'   LR:         {current_lr:.6f}')
        print(f'   Time:       {epoch_time:.1f}s')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'checkpoints/best_model_optimized.pt')
            print(f'   🏆 最佳模型已保存!')
        else:
            patience_counter += 1
            print(f'   ⏳ 早停计数: {patience_counter}/{patience}')
        
        # 早停
        if patience_counter >= patience:
            print(f'\n⚠️ 早停触发! 验证损失 {patience} 个epoch未改善')
            break
        
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
    train_optimized()
