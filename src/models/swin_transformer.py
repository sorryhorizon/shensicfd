import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from .encoder import RMSGroupNorm


class WindowAttention(nn.Module):
    """
    基于窗口的多头自注意力机制 (Window-based Multi-head Self Attention)
    
    将特征图划分为不重叠的窗口，在每个窗口内独立计算注意力
    复杂度: O(window_size² × num_windows) 而非 O(H²W²)
    
    Args:
        dim: 输入特征维度
        window_size: 窗口大小 (M x M)
        num_heads: 注意力头数
        qkv_bias: 是否使用偏置
        attn_drop: 注意力dropout比例
        proj_drop: 输出投影dropout比例
    """
    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int],
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = coords.view(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
        else:
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block
    
    包含:
    1. Window-based Multi-head Self Attention (W-MSA)
    2. Shifted Window-based Multi-head Self Attention (SW-MSA)
    3. MLP with GELU activation
    
    通过移位窗口实现跨窗口信息交互
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int] = (7, 7),
        shift_size: Tuple[int, int] = (0, 0),
        mlp_ratio: float = 4.0,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads,
            attn_drop=attn_drop, proj_drop=drop
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        shortcut = x
        x_flat = x.flatten(2).transpose(1, 2)
        x_norm = self.norm1(x_flat)
        x_norm = x_norm.transpose(1, 2).reshape(B, H, W, C)
        
        pad_h = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_w = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        
        if pad_h > 0 or pad_w > 0:
            x_norm = F.pad(x_norm, (0, 0, 0, pad_w, 0, pad_h))
        
        Hp, Wp = x_norm.shape[1], x_norm.shape[2]
        
        if min(self.shift_size) > 0:
            shifted_x = torch.roll(x_norm, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        else:
            shifted_x = x_norm
        
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.reshape(-1, self.window_size[0] * self.window_size[1], C)
        
        attn_windows = self.attn(x_windows)
        
        attn_windows = attn_windows.reshape(-1, self.window_size[0], self.window_size[1], C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)
        
        if min(self.shift_size) > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x
        
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :]
        
        x = x.reshape(B, H * W, C)
        
        shortcut_flat = shortcut.reshape(B, C, H * W).permute(0, 2, 1)
        x = shortcut_flat + self.drop_path(x)
        x = x.permute(0, 2, 1).contiguous().reshape(B, H * W, C)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, C, H, W)
        
        return x


class SwinTransformerStage(nn.Module):
    """
    Swin Transformer Stage
    
    每个stage包含多个Swin Transformer Block
    偶数块使用常规窗口，奇数块使用移位窗口
    """
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Tuple[int, int] = (7, 7),
        mlp_ratio: float = 4.0,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0) if (i % 2 == 0) else (window_size[0] // 2, window_size[1] // 2),
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=dpr[i]
            )
            for i in range(depth)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class PatchMerging(nn.Module):
    """
    Patch Merging 层
    
    用于下采样，将相邻的2×2 patch合并为一个
    特征维度翻倍，空间分辨率减半
    """
    def __init__(self, dim: int, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        x0 = x[:, :, 0::2, 0::2]
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]
        
        x = torch.cat([x0, x1, x2, x3], dim=1)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = self.reduction(x)
        
        x = x.transpose(1, 2).contiguous().view(B, 2 * C, H // 2, W // 2)
        
        return x


class PatchExpanding(nn.Module):
    """
    Patch Expanding 层
    
    用于上采样，与Patch Merging相反操作
    特征维度减半，空间分辨率翻倍
    """
    def __init__(self, dim: int, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expansion = nn.Linear(dim, 2 * dim * 4, bias=False)
        self.norm = norm_layer(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = self.expansion(x)
        
        B, L, C_new = x.shape
        C_out = C_new // 4
        x = x.view(B, H, W, 2, 2, C_out)
        x = x.permute(0, 3, 4, 1, 2, 5).contiguous()
        x = x.view(B, C_out, 2 * H, 2 * W)
        
        return x


def window_partition(x: torch.Tensor, window_size: Tuple[int, int]) -> torch.Tensor:
    """将特征图分割为窗口"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: Tuple[int, int], H: int, W: int) -> torch.Tensor:
    """从窗口恢复特征图"""
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class DropPath(nn.Module):
    """随机深度（Stochastic Depth）正则化"""
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep_prob)
        if self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor
