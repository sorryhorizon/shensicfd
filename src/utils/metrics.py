import numpy as np
import torch
from typing import Dict, Optional
from scipy.stats import pearsonr


def compute_rmse(pred: np.ndarray, target: np.ndarray, axis: Optional[tuple] = None) -> float:
    """计算均方根误差"""
    return float(np.sqrt(np.mean((pred - target) ** 2, axis=axis)))


def compute_mae(pred: np.ndarray, target: np.ndarray, axis: Optional[tuple] = None) -> float:
    """计算平均绝对误差"""
    return float(np.mean(np.abs(pred - target), axis=axis))


def compute_r_squared(pred: np.ndarray, target: np.ndarray, axis: Optional[tuple] = None) -> float:
    """计算决定系数 R²"""
    ss_res = np.sum((pred - target) ** 2, axis=axis)
    ss_tot = np.sum((target - np.mean(target, axis=axis, keepdims=True)) ** 2, axis=axis)
    r2 = 1 - ss_res / (ss_tot + 1e-10)
    return float(r2)


def compute_pearson(pred: np.ndarray, target: np.ndarray) -> float:
    """计算皮尔逊相关系数"""
    p = pred.flatten()
    t = target.flatten()
    corr, _ = pearsonr(p, t)
    return float(corr)


def compute_bias(pred: np.ndarray, target: np.ndarray) -> float:
    """计算平均偏差 Bias = mean(pred - target)"""
    return float(np.mean(pred - target))


def compute_max_error(pred: np.ndarray, target: np.ndarray) -> float:
    """计算最大绝对误差"""
    return float(np.max(np.abs(pred - target)))


def compute_median_ae(pred: np.ndarray, target: np.ndarray) -> float:
    """计算中位数绝对误差"""
    return float(np.median(np.abs(pred.flatten() - target.flatten())))


def compute_p90_error(pred: np.ndarray, target: np.ndarray) -> float:
    """计算90分位误差"""
    errors = np.abs(pred.flatten() - target.flatten())
    return float(np.percentile(errors, 90))


class CFDMetrics:
    """
    CFD 风场评估指标集合
    
    与 FuXi-CFD 性能报告对标
    """
    
    VAR_NAMES = ['u', 'v', 'w', 'k']
    N_LEVELS = 27
    VERTICAL_LEVELS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
                       55, 60, 65, 70, 75, 80, 85, 90, 95, 100,
                       106.5, 114.95, 125.94, 140.22, 158.78,
                       182.91, 214.29]
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置累积统计"""
        self.all_preds = []
        self.all_targets = []
        self.all_var_preds = {name: [] for name in self.VAR_NAMES}
        self.all_var_targets = {name: [] for name in self.VAR_NAMES}
        self.all_level_preds = {i: [] for i in range(self.N_LEVELS)}
        self.all_level_targets = {i: [] for i in range(self.N_LEVELS)}
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        更新统计数据
        
        Args:
            pred: (B, 27, 4, H, W) 或 (27, 4, H, W) 模型预测
            target: (B, 27, 4, H, W) 或 (27, 4, H, W) 真实值
        """
        if isinstance(pred, torch.Tensor):
            pred_np = pred.cpu().numpy()
        else:
            pred_np = pred
            
        if isinstance(target, torch.Tensor):
            target_np = target.cpu().numpy()
        else:
            target_np = target
        
        self.all_preds.append(pred_np)
        self.all_targets.append(target_np)
        
        for i, name in enumerate(self.VAR_NAMES):
            self.all_var_preds[name].append(pred_np[:, :, i])
            self.all_var_targets[name].append(target_np[:, :, i])
        
        for l in range(self.N_LEVELS):
            self.all_level_preds[l].append(pred_np[:, l])
            self.all_level_targets[l].append(target_np[:, l])
    
    def compute(self) -> Dict:
        """计算所有指标"""
        all_pred = np.concatenate(self.all_preds, axis=0)
        all_target = np.concatenate(self.all_targets, axis=0)
        
        results = {
            'overall': self._compute_overall(all_pred, all_target),
            'per_variable': {},
            'per_level': {},
        }
        
        for name in self.VAR_NAMES:
            var_pred = np.concatenate(self.all_var_preds[name], axis=0)
            var_target = np.concatenate(self.all_var_targets[name], axis=0)
            results['per_variable'][name] = self._compute_single(var_pred, var_target)
        
        for l in range(self.N_LEVELS):
            lvl_pred = np.concatenate(self.all_level_preds[l], axis=0)
            lvl_target = np.concatenate(self.all_level_targets[l], axis=0)
            height = self.VERTICAL_LEVELS[l]
            results['per_level'][f'{height}m'] = self._compute_single(lvl_pred, lvl_target)
        
        return results
    
    def _compute_overall(self, pred: np.ndarray, target: np.ndarray) -> Dict:
        """计算整体指标"""
        return {
            'rmse': compute_rmse(pred, target),
            'mae': compute_mae(pred, target),
            'r2': compute_r_squared(pred, target),
            'pearson': compute_pearson(pred, target),
            'bias': compute_bias(pred, target),
            'max_error': compute_max_error(pred, target),
            'median_ae': compute_median_ae(pred, target),
            'p90_error': compute_p90_error(pred, target),
        }
    
    def _compute_single(self, pred: np.ndarray, target: np.ndarray) -> Dict:
        """计算单个变量/层级的指标"""
        return {
            'rmse': compute_rmse(pred, target),
            'mae': compute_mae(pred, target),
            'r2': compute_r_squared(pred, target),
            'pearson': compute_pearson(pred, target),
            'bias': compute_bias(pred, target),
        }
    
    def generate_report(self, results: Dict) -> str:
        """生成格式化的性能报告（Markdown格式）"""
        lines = []
        lines.append("# ShenSi-CFD Performance Report")
        lines.append("")
        lines.append("## Overall Metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        overall = results['overall']
        for metric_name, value in overall.items():
            lines.append(f"| {metric_name} | {value:.6f} |")
        
        lines.append("")
        lines.append("## Per-Variable Metrics")
        lines.append("")
        lines.append("| Variable | RMSE | MAE | R² | Pearson | Bias |")
        lines.append("|----------|------|-----|----|---------|------|")
        for var_name in self.VAR_NAMES:
            m = results['per_variable'][var_name]
            lines.append(f"| {var_name} | {m['rmse']:.4f} | {m['mae']:.4f} | "
                        f"{m['r2']:.4f} | {m['pearson']:.4f} | {m['bias']:.4f} |")
        
        lines.append("")
        lines.append("## Per-Level R² (Key Levels)")
        lines.append("")
        lines.append("| Height(m) | u-R² | v-R² | w-R² | k-R² |")
        lines.append("|-----------|------|------|------|------|")
        key_heights = ['5', '10', '50', '100', '150', '214.29']
        for h in key_heights:
            if h in results['per_level']:
                m = results['per_level'][h]
                lines.append(f"| {h} | {m['r2']:.4f} | {m['r2']:.4f} | "
                            f"{m['r2']:.4f} | {m['r2']:.4f} |")
        
        return "\n".join(lines)
