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
    def __init__(self, in_channels: int = 6, base_dim: int = 48, target_size: Tuple[int, int] = (300, 300)):
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
        wind_feat = self.wind_encoder(x[:, :2])
        wind_feat = F.interpolate(wind_feat, size=self.target_size, mode='bilinear', align_corners=False)
        dem_feat = self.dem_encoder(x[:, 2:3])
        rough_feat = self.roughness_encoder(x[:, 3:4])
        slope_feat = self.slope_encoder(x[:, 4:6])
        fused = torch.cat([wind_feat, dem_feat, rough_feat, slope_feat], dim=1)
        return self.fusion_conv(fused)


class EfficientUNetDown(nn.Module):
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
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, in_ch // 2, kernel_size=1, bias=False),
            RMSGroupNorm(in_ch // 2),
            nn.GELU(),
        )
        mid_ch = in_ch // 2
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(mid_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
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
        return self.fusion_conv(x)


class CrossAttentionModule(nn.Module):
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


# ============================================================
# Physics-Informed Vertical Decoder
# ============================================================

class PhysicsInformedVerticalDecoder(nn.Module):
    """
    4-head decoder with physics prior injection for w and k.

    - u/v: shallow 2-layer heads (no physics prior)
    - w: 3-layer head with terrain prior (u·dz/dx + v·dz/dy) injected at layer 2
    - k: 3-layer head with turbulence prior (C_μ × |∇u|² + |∇v|²) injected at layer 2

    Physics priors are computed from u/v predictions and injected as extra channels
    into the intermediate layers of w_head and k_head. The model can learn to ignore
    poor priors early in training and leverage them as u/v predictions improve.
    """
    def __init__(self, in_channels: int = 48, n_levels: int = 27, hidden_dim: int = 48):
        super().__init__()
        self.n_levels = n_levels
        self.hidden_dim = hidden_dim

        self.shared_encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            RMSGroupNorm(hidden_dim),
            nn.GELU(),
        )

        # Per-variable height embeddings
        self.u_height_emb = nn.Embedding(n_levels, hidden_dim)
        self.v_height_emb = nn.Embedding(n_levels, hidden_dim)
        self.w_height_emb = nn.Embedding(n_levels, hidden_dim)
        self.k_height_emb = nn.Embedding(n_levels, hidden_dim)

        # u/v: shallow heads (2 conv layers)
        self.u_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1, bias=False),
            RMSGroupNorm(hidden_dim // 2), nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 1, 1),
        )
        self.v_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1, bias=False),
            RMSGroupNorm(hidden_dim // 2), nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 1, 1),
        )

        # w: 3-layer head with terrain prior injection at layer 2
        # Layer 1: hidden_dim → hidden_dim (standard)
        # Layer 2: hidden_dim + 1 (terrain prior) → hidden_dim // 2
        # Layer 3: hidden_dim // 2 → 1
        self.w_head_l1 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            RMSGroupNorm(hidden_dim), nn.GELU(),
        )
        self.w_head_l2 = nn.Sequential(
            nn.Conv2d(hidden_dim + 1, hidden_dim // 2, 3, padding=1, bias=False),
            RMSGroupNorm(hidden_dim // 2), nn.GELU(),
        )
        self.w_head_l3 = nn.Conv2d(hidden_dim // 2, 1, 1)

        # k: 3-layer head with turbulence prior injection at layer 2
        # Layer 1: hidden_dim → hidden_dim (standard)
        # Layer 2: hidden_dim + 1 (turbulence prior) → hidden_dim // 2
        # Layer 3: hidden_dim // 2 → 1
        self.k_head_l1 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            RMSGroupNorm(hidden_dim), nn.GELU(),
        )
        self.k_head_l2 = nn.Sequential(
            nn.Conv2d(hidden_dim + 1, hidden_dim // 2, 3, padding=1, bias=False),
            RMSGroupNorm(hidden_dim // 2), nn.GELU(),
        )
        self.k_head_l3 = nn.Conv2d(hidden_dim // 2, 1, 1, bias=True)

        # Learnable turbulence model coefficient C_μ
        # Initialized to 0.09 (standard k-ε model value)
        self.C_mu = nn.Parameter(torch.tensor(0.09))

        # Sobel kernels for computing velocity gradients (shared across all levels)
        self._sobel_x = nn.Parameter(
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            .view(1, 1, 3, 3) / 4.0,  # normalize
            requires_grad=False
        )
        self._sobel_y = nn.Parameter(
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
            .view(1, 1, 3, 3) / 4.0,
            requires_grad=False
        )

    def _compute_velocity_gradients(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute |∇u|² + |∇v|² using Sobel filters.

        Args:
            u: (B*n, 1, H, W) u velocity at each height level
            v: (B*n, 1, H, W) v velocity at each height level
        Returns:
            (B*n, 1, H, W) velocity gradient magnitude squared
        """
        du_dx = F.conv2d(u, self._sobel_x, padding=1)
        du_dy = F.conv2d(u, self._sobel_y, padding=1)
        dv_dx = F.conv2d(v, self._sobel_x, padding=1)
        dv_dy = F.conv2d(v, self._sobel_y, padding=1)
        return du_dx ** 2 + du_dy ** 2 + dv_dx ** 2 + dv_dy ** 2

    def _compute_terrain_prior(self, u: torch.Tensor, v: torch.Tensor,
                                dz_dx: torch.Tensor, dz_dy: torch.Tensor) -> torch.Tensor:
        """
        Compute w terrain prior: w ≈ u·dz/dx + v·dz/dy.

        Args:
            u: (B*n, 1, H, W) u velocity
            v: (B*n, 1, H, W) v velocity
            dz_dx: (B*n, 1, H, W) DEM horizontal gradient
            dz_dy: (B*n, 1, H, W) DEM vertical gradient
        Returns:
            (B*n, 1, H, W) terrain-based w prior
        """
        return u * dz_dx + v * dz_dy

    def forward(self, x: torch.Tensor, x_raw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: condition features (B, hidden_dim, H, W) from encoder
            x_raw: raw input (B, 6, H, W) — needed for dz_dx, dz_dy (channels 4:5)
        Returns:
            (B, 27, 4, H, W) — [u, v, w, k] at 27 height levels
        """
        B, C, H, W = x.shape
        feat = self.shared_encoder(x)

        dev = x.device
        u_embs = self.u_height_emb(torch.arange(self.n_levels, device=dev)).view(self.n_levels, -1, 1, 1)
        v_embs = self.v_height_emb(torch.arange(self.n_levels, device=dev)).view(self.n_levels, -1, 1, 1)
        w_embs = self.w_height_emb(torch.arange(self.n_levels, device=dev)).view(self.n_levels, -1, 1, 1)
        k_embs = self.k_height_emb(torch.arange(self.n_levels, device=dev)).view(self.n_levels, -1, 1, 1)

        # Extract DEM gradients from raw input
        dz_dx = x_raw[:, 4:5]  # (B, 1, H, W)
        dz_dy = x_raw[:, 5:6]  # (B, 1, H, W)

        chunk_size = 9
        u_out, v_out, w_out, k_out = [], [], [], []

        for start in range(0, self.n_levels, chunk_size):
            end = min(start + chunk_size, self.n_levels)
            n = end - start

            # Compute features with height embeddings
            u_f = (feat.unsqueeze(1) + u_embs[start:end].unsqueeze(0)).view(B * n, -1, H, W)
            v_f = (feat.unsqueeze(1) + v_embs[start:end].unsqueeze(0)).view(B * n, -1, H, W)
            w_f = (feat.unsqueeze(1) + w_embs[start:end].unsqueeze(0)).view(B * n, -1, H, W)
            k_f = (feat.unsqueeze(1) + k_embs[start:end].unsqueeze(0)).view(B * n, -1, H, W)

            # u/v: standard 2-layer heads
            u_pred = self.u_head(u_f).view(B, n, 1, H, W)
            v_pred = self.v_head(v_f).view(B, n, 1, H, W)

            # Extract per-level u/v predictions for physics prior computation
            u_per_level = u_pred.view(B * n, 1, H, W)  # (B*n, 1, H, W)
            v_per_level = v_pred.view(B * n, 1, H, W)  # (B*n, 1, H, W)

            # Expand dz_dx/dz_dy for chunk processing
            dz_dx_exp = dz_dx.unsqueeze(1).expand(B, n, 1, H, W).reshape(B * n, 1, H, W).contiguous()
            dz_dy_exp = dz_dy.unsqueeze(1).expand(B, n, 1, H, W).reshape(B * n, 1, H, W).contiguous()

            # w: 3-layer head with terrain prior injection
            w_l1_out = self.w_head_l1(w_f)  # (B*n, hidden_dim, H, W)
            w_prior = self._compute_terrain_prior(u_per_level, v_per_level, dz_dx_exp, dz_dy_exp)
            w_l2_in = torch.cat([w_l1_out, w_prior], dim=1)  # (B*n, hidden_dim+1, H, W)
            w_l2_out = self.w_head_l2(w_l2_in)  # (B*n, hidden_dim//2, H, W)
            w_pred = self.w_head_l3(w_l2_out).view(B, n, 1, H, W)

            # k: 3-layer head with turbulence prior injection
            k_l1_out = self.k_head_l1(k_f)  # (B*n, hidden_dim, H, W)
            vel_grad_sq = self._compute_velocity_gradients(u_per_level, v_per_level)
            k_prior = self.C_mu * vel_grad_sq  # (B*n, 1, H, W)
            k_l2_in = torch.cat([k_l1_out, k_prior], dim=1)  # (B*n, hidden_dim+1, H, W)
            k_l2_out = self.k_head_l2(k_l2_in)  # (B*n, hidden_dim//2, H, W)
            k_pred = self.k_head_l3(k_l2_out).view(B, n, 1, H, W)

            u_out.append(u_pred)
            v_out.append(v_pred)
            w_out.append(w_pred)
            k_out.append(k_pred)

        output = torch.cat([
            torch.cat(u_out, dim=1),
            torch.cat(v_out, dim=1),
            torch.cat(w_out, dim=1),
            torch.cat(k_out, dim=1),
        ], dim=2)  # (B, 27, 4, H, W)
        return output


# ============================================================
# Main Model: Swin-UNet V5 (全回归 + 物理先验注入)
# ============================================================

class SwinUNetV5(nn.Module):
    """
    V5 model: pure regression with physics-informed decoder.

    - Shared Swin-UNet encoder → condition features
    - PhysicsInformedVerticalDecoder: 4 independent heads with
      w terrain prior and k turbulence prior injected at intermediate layers
    - No Diffusion components — unified regression framework
    """
    def __init__(
        self,
        in_channels: int = 6,
        n_levels: int = 27,
        base_channels: int = 48,
        channel_multipliers: List[int] = None,
        bottleneck_depth: int = 4,
        num_heads: int = 4,
        window_size: Tuple[int, int] = (5, 5),
        dropout: float = 0.2,
        drop_path_rate: float = 0.1,
        use_cross_attention: bool = True,
        output_mean: Optional[torch.Tensor] = None,
        output_std: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.n_levels = n_levels
        self.base_channels = base_channels

        if channel_multipliers is None:
            channel_multipliers = [1, 2, 4, 8]

        assert 300 % window_size[0] == 0 and 300 % window_size[1] == 0

        # Shared encoder
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
            dim=bottleneck_dim, depth=bottleneck_depth, num_heads=num_heads,
            window_size=window_size, mlp_ratio=4.0,
            drop=dropout, attn_drop=dropout, drop_path_rate=drop_path_rate,
        ) if bottleneck_depth > 0 else None

        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(bottleneck_dim, bottleneck_dim, 3, padding=1, bias=False),
            RMSGroupNorm(bottleneck_dim), nn.GELU(),
        )

        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.wind_dem_attn = CrossAttentionModule(dim=bottleneck_dim, num_heads=num_heads)
            self.wind_rough_attn = CrossAttentionModule(dim=bottleneck_dim, num_heads=num_heads)
            self.dem_proj = nn.Conv2d(base_channels, bottleneck_dim, 1, bias=False)
            self.rough_proj = nn.Conv2d(base_channels, bottleneck_dim, 1, bias=False)

        self.dec4 = EfficientUNetUp(base_channels * channel_multipliers[3], base_channels * channel_multipliers[3], base_channels * channel_multipliers[2])
        self.dec3 = EfficientUNetUp(base_channels * channel_multipliers[2], base_channels * channel_multipliers[2], base_channels * channel_multipliers[1])
        self.dec2 = EfficientUNetUp(base_channels * channel_multipliers[1], base_channels * channel_multipliers[1], base_channels * channel_multipliers[0])
        self.dec1 = EfficientUNetUp(base_channels * channel_multipliers[0], base_channels, base_channels)

        # Physics-informed vertical decoder
        self.vertical_decoder = PhysicsInformedVerticalDecoder(
            in_channels=base_channels,
            n_levels=n_levels,
            hidden_dim=base_channels,
        )

        self._initialize_weights()

        # Initialize k_head bias to k mean
        if hasattr(self.vertical_decoder, 'k_head_l3'):
            if hasattr(self.vertical_decoder.k_head_l3, 'bias') and self.vertical_decoder.k_head_l3.bias is not None:
                nn.init.constant_(self.vertical_decoder.k_head_l3.bias, 0.5)

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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Run encoder to get condition features at decoder input resolution."""
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

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input (B, 6, 300, 300)
        Returns:
            (B, 27, 4, 300, 300) — [u, v, w, k] predictions
        """
        cond = self.encode(x)
        output = self.vertical_decoder(cond, x)
        return output

    def get_num_params(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        decoder_params = sum(p.numel() for p in self.vertical_decoder.parameters())
        return {
            'total': total,
            'trainable': trainable,
            'total_mb': total * 4 / (1024 ** 2),
            'decoder_params': decoder_params,
        }


def create_v5_model(config: dict = None) -> SwinUNetV5:
    """Factory function for V5 model."""
    if config is None:
        config = {}

    default_config = {
        'in_channels': 6,
        'n_levels': 27,
        'base_channels': 48,
        'channel_multipliers': [1, 2, 4, 8],
        'bottleneck_depth': 4,
        'num_heads': 4,
        'window_size': (5, 5),
        'dropout': 0.2,
        'drop_path_rate': 0.1,
        'use_cross_attention': True,
        'output_mean': None,
        'output_std': None,
    }
    default_config.update(config)
    return SwinUNetV5(**default_config)


if __name__ == "__main__":
    model = create_v5_model()
    params = model.get_num_params()

    print(f"Model created!")
    print(f"   Total params: {params['total']:,} ({params['total_mb']:.1f} MB)")
    print(f"   Decoder params: {params['decoder_params']:,}")
    print(f"   C_mu initial value: {model.vertical_decoder.C_mu.item():.4f}")

    x = torch.randn(2, 6, 300, 300)

    # Test forward pass
    with torch.no_grad():
        y = model(x)
    print(f"   Input: {x.shape}, Output: {y.shape}")

    # Verify output shape
    assert y.shape == (2, 27, 4, 300, 300), f"Expected (2, 27, 4, 300, 300), got {y.shape}"
    print(f"   Output shape verified!")

    # Check physics prior values
    print(f"   w_prior range: terrain-based, depends on u/v predictions")
    print(f"   k_prior range: C_mu * |grad(u,v)|^2, C_mu={model.vertical_decoder.C_mu.item():.4f}")