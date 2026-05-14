import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from .encoder import RMSGroupNorm
from .swin_transformer import (
    SwinTransformerStage, WindowAttention,
    window_partition, window_reverse, DropPath
)


class LightweightMultiModalEncoder(nn.Module):
    """
    轻量级多模态输入编码器（4路独立编码，含DEM梯度分支）

    改进：
    - wind: 先在原始9x9分辨率编码，再上采样（避免插值噪声）
    - dem: 单独编码，让模型自己从DEM中学坡度
    - roughness: 单独编码
    - slope: 编码DEM梯度(dz_dx, dz_dy)，显式提供坡度信息
    """
    def __init__(self, in_channels: int = 6, base_dim: int = 32, target_size: Tuple[int, int] = (300, 300)):
        super().__init__()
        self.target_size = target_size

        self.wind_encoder = nn.Sequential(
            nn.Conv2d(2, base_dim, kernel_size=3, padding=1, bias=False),
            RMSGroupNorm(base_dim),
            nn.GELU(),
        )

        self.dem_encoder = nn.Sequential(
            nn.Conv2d(1, base_dim, kernel_size=3, padding=1, bias=False),
            RMSGroupNorm(base_dim),
            nn.GELU(),
        )

        self.roughness_encoder = nn.Sequential(
            nn.Conv2d(1, base_dim, kernel_size=3, padding=1, bias=False),
            RMSGroupNorm(base_dim),
            nn.GELU(),
        )

        self.slope_encoder = nn.Sequential(
            nn.Conv2d(2, base_dim, kernel_size=3, padding=1, bias=False),
            RMSGroupNorm(base_dim),
            nn.GELU(),
        )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(base_dim * 4, base_dim, kernel_size=1, bias=False),
            RMSGroupNorm(base_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wind = x[:, :2]
        dem = x[:, 2:3]
        rough = x[:, 3:4]
        slope = x[:, 4:6]

        wind_feat = self.wind_encoder(wind)
        wind_feat = F.interpolate(wind_feat, size=self.target_size, mode='bilinear', align_corners=False)

        dem_feat = self.dem_encoder(dem)
        rough_feat = self.roughness_encoder(rough)
        slope_feat = self.slope_encoder(slope)

        fused = torch.cat([wind_feat, dem_feat, rough_feat, slope_feat], dim=1)
        output = self.fusion_conv(fused)

        return output


class EfficientUNetDown(nn.Module):
    """高效U-Net下采样块"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            RMSGroupNorm(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            RMSGroupNorm(out_ch),
            nn.GELU(),
        )
        
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        skip = self.conv(x)
        down = self.down(skip)
        return down, skip


class EfficientUNetUp(nn.Module):
    """高效U-Net上采样块"""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, in_ch // 2, kernel_size=1, bias=False),
            RMSGroupNorm(in_ch // 2),
            nn.GELU(),
        )
        
        mid_ch = in_ch // 2
        fusion_ch = mid_ch + skip_ch
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_ch, out_ch, kernel_size=3, padding=1, bias=False),
            RMSGroupNorm(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            RMSGroupNorm(out_ch),
            nn.GELU(),
        )
    
    def forward(self, x, skip):
        x = self.up(x)
        
        if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
            x = F.interpolate(x, size=(skip.size(2), skip.size(3)), mode='bilinear', align_corners=False)
        
        x = torch.cat([x, skip], dim=1)
        x = self.fusion_conv(x)
        
        return x


class PhysicsInformedSwinUNetLite(nn.Module):
    """
    Physics-Informed Swin-U-Net 轻量版 (Lite Version)
    
    优化目标：
    - 参数量: ~50-80M (原版214M)
    - 内存占用: ~200-400MB (原版817MB)
    - GPU显存: <12GB (适合RTX 4090/3090)
    
    架构改进：
    1. 减小base_channels (64→32)
    2. 减少channel_multipliers ([1,2,4,8,16]→[1,2,4,8])
    3. 减少Transformer层数
    4. 使用MaxPool代替Conv下采样（参数更少）
    5. 使用Bilinear上采样代替ConvTranspose（更稳定）
    6. 窗口大小使用5（能整除300）
    """
    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 4,
        n_levels: int = 27,
        base_channels: int = 32,
        channel_multipliers: List[int] = None,
        encoder_depths: List[int] = None,
        decoder_depths: List[int] = None,
        bottleneck_depth: int = 4,
        num_heads: int = 4,
        window_size: Tuple[int, int] = (5, 5),
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
        use_physics_constraint: bool = True,
        use_cross_attention: bool = True,
        output_mean: Optional[torch.Tensor] = None,
        output_std: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_levels = n_levels
        self.base_channels = base_channels

        if channel_multipliers is None:
            channel_multipliers = [1, 2, 4, 8]
        if encoder_depths is None:
            encoder_depths = [1, 1, 1, 1]
        if decoder_depths is None:
            decoder_depths = [1, 1, 1, 1]

        assert 300 % window_size[0] == 0 and 300 % window_size[1] == 0, \
            f"窗口大小{window_size}必须能整除300"

        self.input_encoder = LightweightMultiModalEncoder(
            in_channels=in_channels,
            base_dim=base_channels,
            target_size=(300, 300),
        )
        
        self.enc1 = EfficientUNetDown(base_channels, base_channels * channel_multipliers[0])
        self.enc2 = EfficientUNetDown(base_channels * channel_multipliers[0], base_channels * channel_multipliers[1])
        self.enc3 = EfficientUNetDown(base_channels * channel_multipliers[1], base_channels * channel_multipliers[2])
        self.enc4 = EfficientUNetDown(base_channels * channel_multipliers[2], base_channels * channel_multipliers[3])
        
        bottleneck_dim = base_channels * channel_multipliers[3]
        
        self.bottleneck_swin = SwinTransformerStage(
            dim=bottleneck_dim,
            depth=bottleneck_depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=4.0,
            drop=dropout,
            attn_drop=dropout,
            drop_path_rate=drop_path_rate,
        ) if bottleneck_depth > 0 else None
        
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(bottleneck_dim, bottleneck_dim, kernel_size=3, padding=1, bias=False),
            RMSGroupNorm(bottleneck_dim),
            nn.GELU(),
        )
        
        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.wind_dem_attn = CrossAttentionModule(dim=bottleneck_dim, num_heads=num_heads)
            self.wind_rough_attn = CrossAttentionModule(dim=bottleneck_dim, num_heads=num_heads)
            
            self.dem_proj = nn.Conv2d(base_channels, bottleneck_dim, kernel_size=1, bias=False)
            self.rough_proj = nn.Conv2d(base_channels, bottleneck_dim, kernel_size=1, bias=False)
        
        self.dec4 = EfficientUNetUp(base_channels * channel_multipliers[3], base_channels * channel_multipliers[3], base_channels * channel_multipliers[2])
        self.dec3 = EfficientUNetUp(base_channels * channel_multipliers[2], base_channels * channel_multipliers[2], base_channels * channel_multipliers[1])
        self.dec2 = EfficientUNetUp(base_channels * channel_multipliers[1], base_channels * channel_multipliers[1], base_channels * channel_multipliers[0])
        self.dec1 = EfficientUNetUp(base_channels * channel_multipliers[0], base_channels, base_channels)
        
        self.vertical_decoder = AdaptiveVerticalDecoderLite(
            in_channels=base_channels,
            out_channels=out_channels,
            n_levels=n_levels,
            hidden_dim=base_channels,
        )
        
        if use_physics_constraint:
            from .encoder import PhysicsConstraintLayer
            self.physics_layer = PhysicsConstraintLayer(
                enforce_w_ground=True,
                enforce_k_positive=True,
                output_mean=output_mean,
                output_std=output_std,
            )
        else:
            self.physics_layer = None
        
        self._initialize_weights()

        if hasattr(self, 'vertical_decoder') and hasattr(self.vertical_decoder, 'k_decoder_head'):
            last_layer = self.vertical_decoder.k_decoder_head[-1]
            if hasattr(last_layer, 'bias') and last_layer.bias is not None:
                nn.init.constant_(last_layer.bias, 0.5)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x_raw = x
        
        x_enc = self.input_encoder(x)
        
        x, s1 = self.enc1(x_enc)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)
        x, s4 = self.enc4(x)
        
        if self.bottleneck_swin is not None:
            x = self.bottleneck_swin(x)
        x = self.bottleneck_conv(x)
        
        if self.use_cross_attention:
            dem_input = x_raw[:, 2:3]
            rough_input = x_raw[:, 3:4]

            dem_feat = self.input_encoder.dem_encoder(
                F.interpolate(dem_input, size=x.shape[2:], mode='bilinear', align_corners=False)
            )
            rough_feat = self.input_encoder.roughness_encoder(
                F.interpolate(rough_input, size=x.shape[2:], mode='bilinear', align_corners=False)
            )

            dem_ctx = self.dem_proj(dem_feat)
            rough_ctx = self.rough_proj(rough_feat)

            x_dem = self.wind_dem_attn(x, dem_ctx)
            x_rough = self.wind_rough_attn(x, rough_ctx)

            x = x + (x_dem + x_rough) * 0.5
        
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        
        output = self.vertical_decoder(x)
        
        if self.physics_layer is not None:
            output = self.physics_layer(output)
        
        return output
    
    def get_num_params(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total,
            'trainable': trainable,
            'total_mb': total * 4 / (1024 ** 2),
        }


class CrossAttentionModule(nn.Module):
    """轻量级交叉注意力模块"""
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)
        
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, C, H, W = query.shape
        
        q_flat = query.flatten(2).transpose(1, 2)
        ctx_flat = context.flatten(2).transpose(1, 2)
        
        q = self.q_proj(q_flat).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(ctx_flat).view(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ v).transpose(1, 2).contiguous().view(B, -1, C)
        out = self.out_proj(out)
        
        out = self.norm(q_flat + out)
        
        return out.transpose(1, 2).view(B, C, H, W)


class AdaptiveVerticalDecoderLite(nn.Module):
    """
    轻量级自适应垂直解码器
    
    参数效率优化：
    - 共享基础网络
    - 可学习高度嵌入
    - 轻量级层特定调整
    """
    def __init__(self, in_channels: int = 32, out_channels: int = 4, n_levels: int = 27, hidden_dim: int = 32):
        super().__init__()
        self.n_levels = n_levels
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.shared_encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            RMSGroupNorm(hidden_dim),
            nn.GELU(),
        )
        
        self.height_embeddings = nn.Embedding(n_levels, hidden_dim)
        
        # u,v,w 解码器 (输出3通道)
        self.decoder_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1, bias=False),
            RMSGroupNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 3, kernel_size=1, bias=False),
        )
        
        # k专用解码器 (更深网络，输出1通道)
        self.k_decoder_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            RMSGroupNorm(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1, bias=False),
            RMSGroupNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1, bias=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        shared_feat = self.shared_encoder(x)

        level_indices = torch.arange(self.n_levels, device=x.device)
        height_embs = self.height_embeddings(level_indices)
        height_embs = height_embs.view(self.n_levels, -1, 1, 1)

        chunk_size = 9
        outputs = []
        for start in range(0, self.n_levels, chunk_size):
            end = min(start + chunk_size, self.n_levels)
            chunk_embs = height_embs[start:end].unsqueeze(0)
            chunk_feat = shared_feat.unsqueeze(1) + chunk_embs
            chunk_flat = chunk_feat.view(B * (end - start), -1, H, W)

            uvw_out = self.decoder_head(chunk_flat)
            k_out = self.k_decoder_head(chunk_flat)
            chunk_out = torch.cat([uvw_out, k_out], dim=1)
            outputs.append(chunk_out.view(B, end - start, 4, H, W))

        output = torch.cat(outputs, dim=1)
        return output


def create_lite_model(config: dict = None) -> PhysicsInformedSwinUNetLite:
    """
    工厂函数：创建轻量级模型
    
    默认配置：
    - 参数量: ~55M
    - 内存占用: ~210 MB
    - GPU显存需求: ~6-8 GB (batch_size=1)
    """
    if config is None:
        config = {}
    
    default_config = {
        'in_channels': 6,
        'out_channels': 4,
        'n_levels': 27,
        'base_channels': 64,
        'channel_multipliers': [1, 2, 4, 8],
        'encoder_depths': [1, 1, 1, 1],
        'decoder_depths': [1, 1, 1, 1],
        'bottleneck_depth': 6,
        'num_heads': 8,
        'window_size': (5, 5),
        'dropout': 0.1,
        'drop_path_rate': 0.05,
        'use_physics_constraint': True,
        'use_cross_attention': True,
        'output_mean': None,
        'output_std': None,
    }
    
    default_config.update(config)
    
    return PhysicsInformedSwinUNetLite(**default_config)


if __name__ == "__main__":
    model = create_lite_model()
    params = model.get_num_params()
    
    print(f"✅ 模型创建成功!")
    print(f"   总参数量: {params['total']:,}")
    print(f"   内存占用: {params['total_mb']:.1f} MB")
    
    import torch
    x = torch.randn(1, 6, 300, 300)

    with torch.no_grad():
        y = model(x)

    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {y.shape}")
