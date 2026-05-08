import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class RMSGroupNorm(nn.Module):
    """
    RMS GroupNorm (无偏置的GroupNorm)

    更稳定的归一化方式，对小batch size友好
    """
    def __init__(self, num_channels: int, eps: float = 1e-6, **kwargs):
        super().__init__()
        self.eps = eps
        n_groups = min(32, num_channels)
        while num_channels % n_groups != 0 or num_channels // n_groups < 2:
            n_groups -= 1
            if n_groups <= 0:
                n_groups = 1
                break
        self.num_groups = n_groups
        self.num_channels = num_channels
        self.weight = nn.Parameter(torch.ones(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        B, C, H, W = x_float.shape
        x_group = x_float.view(B, self.num_groups, C // self.num_groups, H, W)
        mean_sq = (x_group * x_group).mean(dim=2, keepdim=True)
        x_norm = x_group * torch.rsqrt(mean_sq + self.eps)
        x_norm = x_norm.view(B, C, H, W)
        weight = self.weight.view(1, -1, 1, 1)
        return (x_norm * weight).type_as(x)


class CrossResolutionFusion(nn.Module):
    """
    跨分辨率注意力融合模块
    
    融合低分辨率风场特征和高分辨率地形特征
    Query来自LR分支（风场趋势），Key/Value来自HR分支（地形细节）
    """
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, lr_feat: torch.Tensor, hr_feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = lr_feat.shape
        
        lr_flat = lr_feat.flatten(2).transpose(1, 2)
        hr_flat = hr_feat.flatten(2).transpose(1, 2)
        
        q = self.norm_q(self.q_proj(lr_flat))
        k = self.norm_k(self.k_proj(hr_flat))
        v = self.v_proj(hr_feat).flatten(2).transpose(1, 2)
        
        q = q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).contiguous().view(B, -1, C)
        out = self.o_proj(out)
        
        out = self.norm_out(out + lr_flat)
        out = out + self.ffn(out)
        
        return out.transpose(1, 2).view(B, C, H, W)


class WindFieldUpsampler(nn.Module):
    """
    低分辨率风场上采样器
    
    将低分辨率风场上采样到目标分辨率
    使用双线性插值 + 学习式卷积细化
    """
    def __init__(self, in_ch: int = 2, hidden: int = 128, target_size: tuple = (300, 300)):
        super().__init__()
        self.target_size = target_size
        
        self.upsampler = nn.Sequential(
            nn.Conv2d(in_ch, hidden // 2, kernel_size=3, padding=1),
            RMSGroupNorm(hidden // 2),
            nn.GELU(),
            nn.Conv2d(hidden // 2, hidden, kernel_size=3, padding=1),
            RMSGroupNorm(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            RMSGroupNorm(hidden),
            nn.GELU(),
        )
    
    def forward(self, wind_lr: torch.Tensor) -> torch.Tensor:
        x_up = F.interpolate(wind_lr, size=self.target_size, mode='bilinear', align_corners=False)
        return self.upsampler(x_up)


class ParallelVerticalDecoder(nn.Module):
    """
    并行垂直层级解码器
    
    为每个垂直层级分配独立的解码头
    支持并行计算所有层级
    """
    def __init__(self, in_dim: int, out_ch: int = 4, n_levels: int = 27, hidden: int = 128):
        super().__init__()
        self.n_levels = n_levels
        
        self.level_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_dim, hidden, kernel_size=3, padding=1),
                RMSGroupNorm(hidden),
                nn.GELU(),
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
                RMSGroupNorm(hidden),
                nn.GELU(),
                nn.Conv2d(hidden, out_ch, kernel_size=1),
            ) for _ in range(n_levels)
        ])
    
    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        outputs = [head(fused_features) for head in self.level_decoders]
        return torch.stack(outputs, dim=1)


class PhysicsConstraintLayer(nn.Module):
    """
    物理约束层

    在网络输出上施加物理约束（在原始物理空间中）：
    - 地面垂直速度为零 (w_ground ≈ 0)
    - TKE 非负 (k >= 0)

    当提供 normalization stats 时，先反归一化到物理空间施加约束，再归一化回来
    """
    def __init__(self, enforce_w_ground: bool = True, enforce_k_positive: bool = True,
                 output_mean: torch.Tensor = None, output_std: torch.Tensor = None):
        super().__init__()
        self.enforce_w_ground = enforce_w_ground
        self.enforce_k_positive = enforce_k_positive

        if output_mean is not None:
            self.register_buffer('output_mean', output_mean)
        else:
            self.output_mean = None
        if output_std is not None:
            self.register_buffer('output_std', output_std)
        else:
            self.output_std = None

    def forward(self, output: torch.Tensor) -> torch.Tensor:
        B, L, C, H, W = output.shape

        has_norm = self.output_mean is not None and self.output_std is not None

        if has_norm:
            mean = self.output_mean.view(1, 1, 4, 1, 1)
            std = self.output_std.view(1, 1, 4, 1, 1)
            output_phys = output * std + mean
        else:
            output_phys = output

        if self.enforce_w_ground:
            decay = torch.linspace(0.1, 1.0, L, device=output.device, dtype=output.dtype).view(1, L, 1, 1)
            w = output_phys[:, :, 2:3, :, :] * decay.unsqueeze(2)
            output_phys = torch.cat([output_phys[:, :, :2, :, :], w, output_phys[:, :, 3:, :, :]], dim=2)

        if self.enforce_k_positive:
            k = F.softplus(output_phys[:, :, 3:4, :, :])
            output_phys = torch.cat([output_phys[:, :, :3, :, :], k], dim=2)

        if has_norm:
            output = (output_phys - mean) / std
        else:
            output = output_phys

        return output
