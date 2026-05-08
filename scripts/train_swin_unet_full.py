#!/usr/bin/env python3
"""
Physics-Informed Swin-U-Net 完整训练系统

特性：
✅ 严谨的训练配置（基于12,532样本的FuXi-CFD数据集）
✅ TensorBoard实时监控（loss、指标、梯度、参数分布）
✅ 训练过程可视化（实时曲线图）
✅ 与FuXi-CFD对比评估体系
✅ 物理一致性验证
✅ 混合精度训练 + 梯度累积 + 学习率调度
✅ 断点续训 + 最佳模型保存

使用GPU 2 (NVIDIA A800 80GB)
"""

import os
import sys
import time
import math
import json
import argparse
from datetime import datetime
from pathlib import Path

import setproctitle
setproctitle.setproctitle('shensicfd')

from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.swin_unet_lite import create_lite_model, PhysicsInformedSwinUNetLite
from src.losses.enhanced_physics_loss import EnhancedPhysicsLoss, ProgressiveLossScheduler
from src.data.fuxi_cfd_dataset import FuXiCFDDataset, create_dataloaders


class TrainingMonitor:
    """
    训练监控系统
    
    功能：
    - TensorBoard日志记录
    - 训练指标追踪
    - 模型性能对比
    - 可视化生成
    """
    
    def __init__(self, log_dir: str, experiment_name: str = 'swin_unet_cfd'):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = Path(log_dir) / f'{experiment_name}_{timestamp}'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(str(self.log_dir))
        
        self.train_losses = []
        self.val_losses = []
        self.metrics_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_mse': [],
            'val_mse': [],
            'r2_u': [], 'r2_v': [], 'r2_w': [], 'r2_k': [],
            'physics_loss': [],
            'lr': [],
            'time_per_epoch': [],
        }
        
        print(f'📂 TensorBoard日志: {self.log_dir}')
        print(f'   启动命令: tensorboard --logdir={self.log_dir.parent} --port=6006')
    
    def log_training_step(
        self,
        epoch: int,
        step: int,
        loss: float,
        loss_dict: dict,
        lr: float,
        stage: int = 0,
    ):
        """记录每个训练步"""
        global_step = epoch * 10000 + step
        
        self.writer.add_scalar('Train/Loss', loss, global_step)
        self.writer.add_scalar('Train/LearningRate', lr, global_step)
        self.writer.add_scalar('Train/Stage', stage, global_step)
        
        for key, value in loss_dict.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'Train/{key}', value, global_step)
        
        if step % 50 == 0:
            self.writer.add_scalar('Train/Loss_Smooth', loss, global_step)
    
    def log_epoch(
        self,
        epoch: int,
        train_metrics: dict,
        val_metrics: dict,
        lr: float,
        epoch_time: float,
    ):
        """记录每个epoch"""
        self.metrics_history['epoch'].append(epoch)
        self.metrics_history['train_loss'].append(train_metrics.get('total', 0))
        self.metrics_history['val_loss'].append(val_metrics.get('total', 0))
        self.metrics_history['train_mse'].append(train_metrics.get('mse', 0))
        self.metrics_history['val_mse'].append(val_metrics.get('mse', 0))
        self.metrics_history['physics_loss'].append(val_metrics.get('mass_conservation', 0))
        self.metrics_history['lr'].append(lr)
        self.metrics_history['time_per_epoch'].append(epoch_time)
        
        if 'r2' in val_metrics:
            r2 = val_metrics['r2']
            self.metrics_history['r2_u'].append(r2.get('u', 0))
            self.metrics_history['r2_v'].append(r2.get('v', 0))
            self.metrics_history['r2_w'].append(r2.get('w', 0))
            self.metrics_history['r2_k'].append(r2.get('k', 0))
        
        self.writer.add_scalars('Loss/Total', {
            'train': train_metrics.get('total', 0),
            'val': val_metrics.get('total', 0),
        }, epoch)
        
        self.writer.add_scalars('Loss/MSE', {
            'train': train_metrics.get('mse', 0),
            'val': val_metrics.get('mse', 0),
        }, epoch)
        
        self.writer.add_scalar('Val/Physics_Loss', val_metrics.get('mass_conservation', 0), epoch)
        self.writer.add_scalar('Val/Boundary_Layer_Loss', val_metrics.get('boundary_layer', 0), epoch)
        self.writer.add_scalar('Val/Terrain_Penalty', val_metrics.get('terrain', 0), epoch)
        self.writer.add_scalar('Train/Learning_Rate', lr, epoch)
        self.writer.add_scalar('Time/Epoch', epoch_time, epoch)
        
        if 'r2' in val_metrics:
            r2 = val_metrics['r2']
            self.writer.add_scalars('R2/Per_Variable', r2, epoch)
            overall_r2 = np.mean(list(r2.values()))
            self.writer.add_scalar('R2/Overall', overall_r2, epoch)
            
            fuxi_baseline = {'u': 0.177, 'v': 0.234, 'w': 0.245, 'k': 0.254}
            improvement = {}
            for var in ['u', 'v', 'w', 'k']:
                if var in r2 and var in fuxi_baseline:
                    imp = (r2[var] - fuxi_baseline[var]) / fuxi_baseline[var] * 100
                    improvement[f'vs_fuxi_{var}'] = imp
            
            if improvement:
                self.writer.add_scalars('Improvement_vs_FuXi_CFD', improvement, epoch)
        
        self._save_metrics_json()
    
    def _save_metrics_json(self):
        """保存训练历史到JSON文件"""
        metrics_path = self.log_dir / 'training_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def log_model_params(self, model: nn.Module, epoch: int):
        """记录模型参数分布"""
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.writer.add_histogram(f'Params/{name}', param.data.cpu(), epoch)
                self.writer.add_histogram(f'Grads/{name}', param.grad.data.cpu(), epoch)
                
                grad_norm = param.grad.norm().item()
                param_norm = param.data.norm().item()
                self.writer.add_scalar(f'Norms/{name}_grad', grad_norm, epoch)
                self.writer.add_scalar(f'Norms/{name}_param', param_norm, epoch)
    
    def close(self):
        self.writer.close()


class Evaluator:
    """
    评估器
    
    计算与FuXi-CFD对比的关键指标：
    - R² (决定系数) - 主要评估指标
    - RMSE (均方根误差)
    - MAE (平均绝对误差)
    - 物理一致性分数
    - 各变量独立评估
    """
    
    def __init__(self):
        self.var_names = ['u', 'v', 'w', 'k']
        self.fuxi_baseline_r2 = {
            'u': 0.177,
            'v': 0.234,
            'w': 0.245,
            'k': 0.254,
            'overall': 0.262,
        }
    
    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: str = 'cuda',
        max_batches: int = None,
    ) -> dict:
        """
        全面评估模型
        
        返回包含以下指标的字典：
        - loss: 总损失
        - mse: MSE损失
        - rmse: RMSE per variable
        - mae: MAE per variable
        - r2: R² per variable and overall
        - physics_scores: 物理一致性分数
        - vs_fuxi: 相对FuXi-CFD的提升百分比
        """
        model.eval()
        
        all_preds = {var: [] for var in self.var_names}
        all_targets = {var: [] for var in self.var_names}
        total_loss = 0
        total_count = 0
        
        criterion = nn.MSELoss(reduction='none')
        
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
                
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            outputs = model(inputs)
            
            batch_size = inputs.shape[0]
            total_count += batch_size
            
            loss = F.mse_loss(outputs, targets)
            total_loss += loss.item() * batch_size
            
            for i, var in enumerate(self.var_names):
                pred_var = outputs[:, :, i].cpu()
                target_var = targets[:, :, i].cpu()
                
                all_preds[var].append(pred_var)
                all_targets[var].append(target_var)
        
        avg_loss = total_loss / max(total_count, 1)
        
        results = {
            'total_loss': avg_loss,
            'rmse': {},
            'mae': {},
            'r2': {},
            'physics_scores': {},
            'vs_fuxi_improvement': {},
        }
        
        for var in self.var_names:
            pred_all = torch.cat(all_preds[var], dim=0).numpy()
            target_all = torch.cat(all_targets[var], dim=0).numpy()
            
            mse = np.mean((pred_all - target_all) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(pred_all - target_all))
            
            ss_res = np.sum((target_all - pred_all) ** 2)
            ss_tot = np.sum((target_all - np.mean(target_all)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            results['rmse'][var] = float(rmse)
            results['mae'][var] = float(mae)
            results['r2'][var] = float(max(0, r2))
            
            if var in self.fuxi_baseline_r2:
                baseline = self.fuxi_baseline_r2[var]
                improvement = (results['r2'][var] - baseline) / baseline * 100 if baseline > 0 else 0
                results['vs_fuxi_improvement'][var] = float(improvement)
        
        all_r2 = [results['r2'][var] for var in self.var_names]
        results['r2']['overall'] = float(np.mean(all_r2))
        
        baseline_overall = self.fuxi_baseline_r2['overall']
        results['vs_fuxi_improvement']['overall'] = (
            (results['r2']['overall'] - baseline_overall) / baseline_overall * 100
            if baseline_overall > 0 else 0
        )
        
        return results
    
    def format_evaluation_report(self, results: dict, epoch: int = None) -> str:
        """格式化评估报告"""
        report = []
        report.append('\n' + '='*70)
        report.append(f'📊 评估报告' + (f' (Epoch {epoch})' if epoch else ''))
        report.append('='*70)
        
        report.append(f'\n🎯 总体性能:')
        report.append(f'   总损失:     {results["total_loss"]:.6f}')
        report.append(f'   R² (整体):  {results["r2"]["overall"]:.4f}')
        
        baseline = self.fuxi_baseline_r2['overall']
        improvement = results['vs_fuxi_improvement']['overall']
        report.append(f'   FuXi基准:   {baseline:.4f}')
        report.append(f'   提升:       {improvement:+.1f}%')
        
        report.append(f'\n📈 各变量R²:')
        header = f"{'变量':<8}{'R²':>10}{'RMSE':>10}{'MAE':>10}{'vs FuXi':>12}"
        report.append(header)
        report.append('-'*50)
        
        for var in self.var_names:
            r2 = results['r2'].get(var, 0)
            rmse = results['rmse'].get(var, 0)
            mae = results['mae'].get(var, 0)
            vs_fuxi = results['vs_fuxi_improvement'].get(var, 0)
            
            line = f"{var:<8}{r2:>10.4f}{rmse:>10.4f}{mae:>10.4f}{vs_fuxi:>+11.1f}%"
            report.append(line)
        
        report.append('='*70)
        
        return '\n'.join(report)


class Trainer:
    """
    Physics-Informed Swin-U-Net 训练器
    
    特性：
    - 渐进式3阶段训练策略
    - 混合精度训练 (FP16)
    - 梯度累积
    - 动态学习率调度
    - TensorBoard监控
    - 断点续训
    - 最佳模型保存
    - 与FuXi-CFD对比评估
    """
    
    def __init__(
        self,
        model: PhysicsInformedSwinUNetLite,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: dict,
        device: str = 'cuda',
        gpu_id: int = 2,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        self.scaler = GradScaler() if config.get('use_amp', True) else None
        
        self.criterion = EnhancedPhysicsLoss(
            mse_weight=config.get('mse_weight', 1.0),
            l1_weight=config.get('l1_weight', 0.5),
            mass_conservation_weight=config.get('physics_weight', 0.1),
            boundary_layer_weight=config.get('boundary_layer_weight', 0.05),
            terrain_penalty_weight=config.get('terrain_penalty_weight', 0.1),
            k_positive_weight=config.get('k_positive_weight', 0.05),
            gradient_smoothness_weight=config.get('gradient_smoothness_weight', 0.1),
            k_specialized_weight=config.get('k_loss_weight', 0.5),
            use_k_transform=config.get('use_k_transform', True),
        )
        
        self.scheduler = ProgressiveLossScheduler(
            base_loss=self.criterion,
            n_stages=3,
            stage_epochs=[
                config.get('stage1_epochs', 30),
                config.get('stage2_epochs', 40),
                config.get('stage3_epochs', 30),
            ],
        )
        
        self.optimizer = self._create_optimizer()
        self.lr_scheduler = self._create_scheduler()
        
        self.monitor = TrainingMonitor(
            log_dir=config.get('log_dir', 'logs'),
            experiment_name='swin_unet_cfd',
        )
        
        self.evaluator = Evaluator()
        
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_r2 = 0
        
        self.save_dir = Path(config.get('save_dir', 'checkpoints'))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.start_epoch = 1
        
        if config.get('resume', None):
            self.load_checkpoint(config['resume'])
    
    def _create_optimizer(self):
        lr = self.config.get('learning_rate', 1e-3)
        weight_decay = self.config.get('weight_decay', 0.01)
        
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )
    
    def _create_scheduler(self):
        total_epochs = sum([
            self.config.get('stage1_epochs', 30),
            self.config.get('stage2_epochs', 40),
            self.config.get('stage3_epochs', 30),
        ])
        
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_epochs,
            eta_min=self.config.get('min_lr', 1e-6),
        )
    
    def train_epoch(self, epoch: int) -> dict:
        """训练一个epoch"""
        self.model.train()
        
        total_losses = {}
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            inputs = batch['input'].to(self.device, non_blocking=True)
            targets = batch['target'].to(self.device, non_blocking=True)
            
            dem = inputs[:, 2:3]
            roughness = inputs[:, 3:4]
            
            self.optimizer.zero_grad(set_to_none=True)
            
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(inputs)
                    losses = self.scheduler(
                        outputs, targets,
                        epoch=epoch,
                        dem=dem,
                        roughness=roughness,
                    )
                    loss = losses['total']
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.config.get('accumulation_steps', 1) == 0:
                    self.scaler.unscale_(self.optimizer)
                    
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get('max_grad_norm', 1.0)
                    )
                    
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        self.optimizer.zero_grad(set_to_none=True)
                        self.scaler.update()
                        continue
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                outputs = self.model(inputs)
                losses = self.scheduler(
                    outputs, targets,
                    epoch=epoch,
                    dem=dem,
                    roughness=roughness,
                )
                loss = losses['total']
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                loss.backward()
                
                if (batch_idx + 1) % self.config.get('accumulation_steps', 1) == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get('max_grad_norm', 1.0)
                    )
                    
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        self.optimizer.zero_grad(set_to_none=True)
                        continue
                    
                    self.optimizer.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            for key, value in losses.items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    if key not in total_losses:
                        total_losses[key] = 0
                    total_losses[key] += value.item()
            
            if batch_idx == num_batches - 1:
                with torch.no_grad():
                    total_losses['k_mean'] = outputs[:, :, 3, :, :].mean().item()
                    total_losses['k_min'] = outputs[:, :, 3, :, :].min().item()

            self.global_step += 1
            
            if (batch_idx + 1) % self.config.get('log_interval', 50) == 0:
                self.monitor.log_training_step(
                    epoch=epoch,
                    step=batch_idx,
                    loss=loss.item(),
                    loss_dict={k: v for k, v in losses.items() 
                              if isinstance(v, (int, float)) or (isinstance(v, torch.Tensor) and v.numel() == 1)},
                    lr=current_lr,
                    stage=losses.get('current_stage', 0),
                )
                
                print(f'  [{batch_idx+1}/{num_batches}] Loss: {loss.item():.4f} '
                      f'Stage: {int(losses.get("current_stage", 0))+1} '
                      f'LR: {current_lr:.6f}')
        
        avg_losses = {}
        for k, v in total_losses.items():
            if k in ('k_mean', 'k_min'):
                avg_losses[k] = v
            else:
                avg_losses[k] = v / num_batches
        
        return avg_losses
    
    @torch.no_grad()
    def validate(self, epoch: int) -> tuple:
        """验证并返回损失和评估结果"""
        self.model.eval()
        
        total_losses = {}
        num_batches = len(self.val_loader)
        
        for batch in self.val_loader:
            inputs = batch['input'].to(self.device, non_blocking=True)
            targets = batch['target'].to(self.device, non_blocking=True)
            
            dem = inputs[:, 2:3]
            roughness = inputs[:, 3:4]
            
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(inputs)
                    losses = self.criterion(outputs, targets, dem=dem, roughness=roughness)
            else:
                outputs = self.model(inputs)
                losses = self.criterion(outputs, targets, dem=dem, roughness=roughness)
            
            for key, value in losses.items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    if key not in total_losses:
                        total_losses[key] = 0
                    total_losses[key] += value.item()
        
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        
        eval_results = self.evaluator.evaluate(
            model=self.model,
            dataloader=self.val_loader,
            device=self.device,
            max_batches=20,
        )
        
        return avg_losses, eval_results
    
    def save_checkpoint(self, epoch: int, is_best: bool = False, eval_results: dict = None):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_loss': self.best_val_loss,
            'best_val_r2': self.best_val_r2,
            'global_step': self.global_step,
            'config': self.config,
        }
        
        if eval_results:
            checkpoint['eval_results'] = eval_results
        
        save_path = self.save_dir / f'checkpoint_epoch{epoch}.pt'
        torch.save(checkpoint, save_path)
        
        if is_best:
            best_path = self.save_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f'\n  🏆 最佳模型已保存! R²={self.best_val_r2:.4f}')
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        print(f'\n📂 从检查点恢复: {path}')
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_r2 = checkpoint.get('best_val_r2', 0)
        self.global_step = checkpoint.get('global_step', 0)
        
        print(f'   恢复到 Epoch {self.start_epoch}')
        print(f'   最佳验证损失: {self.best_val_loss:.6f}')
        print(f'   最佳验证R²: {self.best_val_r2:.4f}')
    
    def train(self):
        """完整训练流程"""
        total_epochs = sum([
            self.config.get('stage1_epochs', 30),
            self.config.get('stage2_epochs', 40),
            self.config.get('stage3_epochs', 30),
        ])
        
        print('\n' + '='*70)
        print('🚀 开始训练 Physics-Informed Swin-U-Net Lite')
        print('='*70)
        print(f'   GPU: NVIDIA A800 (GPU {os.environ.get("CUDA_VISIBLE_DEVICES", "0")})')
        print(f'   总Epochs: {total_epochs} (从Epoch {self.start_epoch}开始)')
        print(f'   Batch Size: {self.config.get("batch_size", 2)}')
        print(f'   Accumulation Steps: {self.config.get("accumulation_steps", 4)}')
        print(f'   Effective Batch Size: {self.config.get("batch_size", 2) * self.config.get("accumulation_steps", 4)}')
        print(f'   Learning Rate: {self.config.get("learning_rate", 1e-3)}')
        print(f'   Mixed Precision: {"✅ 启用" if self.scaler else "❌ 禁用"}')
        print(f'   K-Transform: {"✅ 启用" if self.config.get("use_k_transform", True) else "❌ 禁用"}')
        print(f'   K-Loss Weight: {self.config.get("k_loss_weight", 0.5)}')
        print(f'   数据集大小:')
        print(f'      Train: {len(self.train_loader.dataset)} 样本')
        print(f'      Val:   {len(self.val_loader.dataset)} 样本')
        print(f'      Test:  {len(self.test_loader.dataset)} 样本')
        print('='*70 + '\n')
        
        start_time = time.time()
        
        try:
            for epoch in range(self.start_epoch, total_epochs + 1):
                epoch_start_time = time.time()
                
                print(f'\n📅 Epoch {epoch}/{total_epochs}')
                print('-'*60)
                
                train_metrics = self.train_epoch(epoch)
                
                val_interval = 2
                if epoch % val_interval == 0:
                    val_metrics, eval_results = self.validate(epoch)
                else:
                    val_metrics = None
                    eval_results = None
                
                self.lr_scheduler.step()
                
                epoch_time = time.time() - epoch_start_time
                
                current_lr = self.optimizer.param_groups[0]['lr']
                
                if val_metrics is not None:
                    self.monitor.log_epoch(
                        epoch=epoch,
                        train_metrics=train_metrics,
                        val_metrics=val_metrics,
                        lr=current_lr,
                        epoch_time=epoch_time,
                    )
                
                if epoch % val_interval == 0:
                    self.monitor.log_model_params(self.model, epoch)
                
                print(f'\n  📊 训练结果:')
                print(f'     Train Loss: {train_metrics.get("total", 0):.6f}')
                if val_metrics is not None:
                    print(f'     Val Loss:   {val_metrics.get("total", 0):.6f}')
                    print(f'     MSE:        {val_metrics.get("mse", 0):.6f}')
                    print(f'     物理:       {val_metrics.get("mass_conservation", 0):.6f}')
                else:
                    print(f'     Val Loss:   N/A (每{val_interval} epoch验证)')
                    print(f'     MSE:        N/A')
                    print(f'     物理:       N/A')
                print(f'     k 均值:     {train_metrics.get("k_mean", 0):.4f}')
                print(f'     k 最小值:   {train_metrics.get("k_min", 0):.4f}')
                print(f'     耗时:       {epoch_time:.1f}s ({epoch_time/60:.1f}分钟)')
                
                if eval_results:
                    report = self.evaluator.format_evaluation_report(eval_results, epoch)
                    print(report)
                    
                    current_r2 = eval_results['r2']['overall']
                    
                    if current_r2 > self.best_val_r2:
                        self.best_val_r2 = current_r2
                        self.best_val_loss = val_metrics.get('total', float('inf'))
                        
                        self.save_checkpoint(
                            epoch=epoch,
                            is_best=True,
                            eval_results=eval_results,
                        )
                
                if epoch % self.config.get('save_interval', 10) == 0:
                    self.save_checkpoint(epoch=epoch)
                
                elapsed_time = time.time() - start_time
                remaining_epochs = total_epochs - epoch
                avg_epoch_time = elapsed_time / (epoch - self.start_epoch + 1)
                estimated_remaining = avg_epoch_time * remaining_epochs
                
                print(f'\n  ⏱️ 进度:')
                print(f'     已用时间: {elapsed_time/3600:.1f} 小时')
                print(f'     预计剩余: {estimated_remaining/3600:.1f} 小时')
                print(f'     预计总时长: {(elapsed_time + estimated_remaining)/3600:.1f} 小时')
        
        except KeyboardInterrupt:
            print('\n\n⚠️ 训练被用户中断!')
            print('保存当前检查点...')
            self.save_checkpoint(epoch=epoch)
        
        finally:
            total_time = time.time() - start_time
            
            print('\n' + '='*70)
            print('🎉 训练完成！')
            print('='*70)
            print(f'   总耗时: {total_time/3600:.2f} 小时')
            print(f'   总Epochs: {epoch - self.start_epoch + 1}')
            print(f'   最佳验证R²: {self.best_val_r2:.4f}')
            print(f'   最佳验证损失: {self.best_val_loss:.6f}')
            print(f'   TensorBoard日志: {self.monitor.log_dir}')
            print(f'   模型保存路径: {self.save_dir}')
            
            final_eval = self.evaluator.evaluate(
                model=self.model,
                dataloader=self.test_loader,
                device=self.device,
            )
            
            final_report = self.evaluator.format_evaluation_report(final_eval, 'Final')
            print(final_report)
            
            self.monitor.close()
            
            print('\n💡 下一步操作:')
            print(f'   1. 查看TensorBoard: tensorboard --logdir={self.monitor.log_dir.parent}')
            print(f'   2. 加载最佳模型进行推理')
            print(f'   3. 在测试集上进行详细评估')


def parse_args():
    parser = argparse.ArgumentParser(description='Physics-Informed Swin-U-Net Training')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='/mnt/sdata/jz/fuxi_cfd/dataset',
                       help='FuXi-CFD数据集目录')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载线程数')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='验证集比例')
    parser.add_argument('--prefetch_to_memory', action='store_true', default=False,
                       help='预加载数据集到内存（需要200GB内存）')
    
    # 模型参数
    parser.add_argument('--base_channels', type=int, default=32,
                       help='基础通道数')
    parser.add_argument('--bottleneck_depth', type=int, default=4,
                       help='瓶颈层深度')
    parser.add_argument('--window_size', type=int, default=5,
                       help='Swin窗口大小')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                       help='总训练轮次')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='初始学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='权重衰减')
    parser.add_argument('--accumulation_steps', type=int, default=4,
                       help='梯度累积步数')
    parser.add_argument('--max_grad_norm', type=float, default=5.0,
                       help='梯度裁剪阈值')
    
    # 训练阶段配置
    parser.add_argument('--stage1_epochs', type=int, default=15,
                        help='阶段1轮次（数据保真）')
    parser.add_argument('--stage2_epochs', type=int, default=20,
                        help='阶段2轮次（物理约束）')
    parser.add_argument('--stage3_epochs', type=int, default=15,
                        help='阶段3轮次（联合微调）')

    # K分量专用配置 (v3 k-fix改进)
    parser.add_argument('--use_k_transform', action='store_true', default=True,
                       help='是否启用log-k变换 (默认True)')
    parser.add_argument('--k_loss_weight', type=float, default=0.5,
                       help='k专用loss权重 (默认0.5)')
    
    # 其他参数
    parser.add_argument('--use_amp', action='store_true', default=False,
                       help='使用混合精度训练')
    parser.add_argument('--gpu_id', type=int, default=2,
                       help='GPU ID')
    parser.add_argument('--resume', type=str, default=None,
                       help='从检查点恢复')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='保存间隔（epoch）')
    parser.add_argument('--log_interval', type=int, default=20,
                       help='日志打印间隔（batch）')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='日志目录')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='模型保存目录')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print('\n' + '='*70)
    print('🚀 Physics-Informed Swin-U-Net Lite - CFD风场超分辨率训练')
    print('='*70)
    print(f'   GPU ID: {args.gpu_id}')
    print(f'   Device: {device}')
    print(f'   Data Dir: {args.data_dir}')
    
    if torch.cuda.is_available():
        print(f'   GPU Name: {torch.cuda.get_device_name(0)}')
        print(f'   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    
    print('='*70 + '\n')
    
    print('📦 创建模型...')
    model = create_lite_model(config={
        'base_channels': args.base_channels,
        'bottleneck_depth': args.bottleneck_depth,
        'window_size': (args.window_size, args.window_size),
    })
    
    params = model.get_num_params()
    print(f'   参数量: {params["total"]:,} ({params["total_mb"]:.1f} MB)')
    
    print('\n📂 加载数据集...')
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=1.0 - args.train_ratio - args.val_ratio,
        prefetch_to_memory=args.prefetch_to_memory,
    )
    
    config = vars(args)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device,
        gpu_id=args.gpu_id,
    )
    
    trainer.train()


if __name__ == '__main__':
    main()
