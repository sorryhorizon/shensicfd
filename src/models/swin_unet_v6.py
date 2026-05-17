#!/usr/bin/env python3
"""SwinUNetV6: Enhanced decoder + vertical smoother + per-level norm support.

Changes from V5:
  1. Decoder hidden_dim 48→128 (more capacity)
  2. u/v heads: 4-layer conv (128→96→48→24→1)
  3. w/k heads: 5-layer conv + residual + prior projection
  4. VerticalSmoother: 1D depthwise conv along height dimension
  5. Physics priors (w_terrain, k_turbulence) retained
  6. output_mean/std now (27,4) per-level (handled by dataset)
  7. Gradient checkpointing on per-level decode to save VRAM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from .encoder import RMSGroupNorm
from .swin_transformer import SwinTransformerStage, DropPath
from .swin_unet_v5 import (
    LightweightMultiModalEncoder,
    EfficientUNetDown,
    EfficientUNetUp,
    CrossAttentionModule,
)


class VerticalSmoother(nn.Module):
    """1D depthwise conv along height dimension for vertical smoothness."""

    def __init__(self, channels=4, kernel_size=5):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                              padding=kernel_size // 2, groups=channels)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        """x: (B, 27, 4, H, W)"""
        B, L, C, H, W = x.shape
        x_flat = x.permute(0, 3, 4, 2, 1).reshape(B * H * W, C, L)
        x_smooth = self.conv(x_flat)
        x_smooth = x_smooth.reshape(B, H, W, C, L).permute(0, 4, 3, 2, 1)
        return x + self.alpha * (x_smooth - x)


class EnhancedVerticalDecoder(nn.Module):
    """Enhanced vertical decoder: larger capacity + physics priors + vertical smoother."""

    def __init__(self, in_channels=48, n_levels=27, hidden_dim=128):
        super().__init__()
        self.n_levels = n_levels
        self.hidden_dim = hidden_dim

        # Shared encoder: in_channels → hidden_dim
        self.shared_encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1, bias=False),
            RMSGroupNorm(hidden_dim),
            nn.GELU(),
        )

        # Height embeddings
        self.height_emb_uv = nn.Embedding(n_levels, hidden_dim)
        self.height_emb_w = nn.Embedding(n_levels, hidden_dim)
        self.height_emb_k = nn.Embedding(n_levels, hidden_dim)

        # u/v: 4-layer heads
        self.u_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 96, 3, padding=1, bias=False), RMSGroupNorm(96), nn.GELU(),
            nn.Conv2d(96, 48, 3, padding=1, bias=False), RMSGroupNorm(48), nn.GELU(),
            nn.Conv2d(48, 24, 3, padding=1, bias=False), RMSGroupNorm(24), nn.GELU(),
            nn.Conv2d(24, 1, 1),
        )
        self.v_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 96, 3, padding=1, bias=False), RMSGroupNorm(96), nn.GELU(),
            nn.Conv2d(96, 48, 3, padding=1, bias=False), RMSGroupNorm(48), nn.GELU(),
            nn.Conv2d(48, 24, 3, padding=1, bias=False), RMSGroupNorm(24), nn.GELU(),
            nn.Conv2d(24, 1, 1),
        )

        # w: encoder + prior projection + decoder
        self.w_enc = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            RMSGroupNorm(hidden_dim), nn.GELU(),
        )
        self.w_prior_proj = nn.Conv2d(1, hidden_dim, 1)
        self.w_dec = nn.Sequential(
            nn.Conv2d(hidden_dim, 96, 3, padding=1, bias=False), RMSGroupNorm(96), nn.GELU(),
            nn.Conv2d(96, 48, 3, padding=1, bias=False), RMSGroupNorm(48), nn.GELU(),
            nn.Conv2d(48, 24, 3, padding=1, bias=False), RMSGroupNorm(24), nn.GELU(),
            nn.Conv2d(24, 1, 1),
        )

        # k: encoder + prior projection + decoder
        self.k_enc = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            RMSGroupNorm(hidden_dim), nn.GELU(),
        )
        self.k_prior_proj = nn.Conv2d(1, hidden_dim, 1)
        self.k_dec = nn.Sequential(
            nn.Conv2d(hidden_dim, 96, 3, padding=1, bias=False), RMSGroupNorm(96), nn.GELU(),
            nn.Conv2d(96, 48, 3, padding=1, bias=False), RMSGroupNorm(48), nn.GELU(),
            nn.Conv2d(48, 24, 3, padding=1, bias=False), RMSGroupNorm(24), nn.GELU(),
            nn.Conv2d(24, 1, 1),
        )

        # Sobel filters for terrain gradient
        self.register_buffer('sobel_x', torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0)
        self.register_buffer('sobel_y', torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0)

        # Learnable C_mu
        self.C_mu = nn.Parameter(torch.tensor(0.09))

        # Vertical smoother applied after all levels decoded
        self.vertical_smoother = VerticalSmoother(channels=4, kernel_size=5)

    def _compute_terrain_prior(self, x_raw):
        """w_terrain = u_100m * dz/dx + v_100m * dz/dy (from input wind + DEM)."""
        u = x_raw[:, 0:1]
        v = x_raw[:, 1:2]
        dem = x_raw[:, 2:3]
        dz_dx = F.conv2d(dem, self.sobel_x, padding=1)
        dz_dy = F.conv2d(dem, self.sobel_y, padding=1)
        return u * dz_dx + v * dz_dy

    def _compute_turbulence_prior(self, x_raw):
        """k_turb = C_mu * (|grad_u|^2 + |grad_v|^2) from input wind."""
        u = x_raw[:, 0:1]
        v = x_raw[:, 1:2]
        du_dx = F.conv2d(u, self.sobel_x, padding=1)
        du_dy = F.conv2d(u, self.sobel_y, padding=1)
        dv_dx = F.conv2d(v, self.sobel_x, padding=1)
        dv_dy = F.conv2d(v, self.sobel_y, padding=1)
        return self.C_mu * (du_dx ** 2 + du_dy ** 2 + dv_dx ** 2 + dv_dy ** 2)

    def _decode_chunk(self, shared, level_ids_chunk, w_prior, k_prior, bhw_tensor):
        """Decode a chunk of levels. All args must be tensors for checkpointing.
        level_ids_chunk: (n,) tensor of level indices
        bhw_tensor: (3,) tensor storing [B, H, W] as integers
        """
        B = int(bhw_tensor[0].item())
        H = int(bhw_tensor[1].item())
        W = int(bhw_tensor[2].item())
        n = level_ids_chunk.shape[0]
        inject_low = (level_ids_chunk < 14).any()

        # u/v
        h_uv = self.height_emb_uv(level_ids_chunk).view(1, n, self.hidden_dim, 1, 1).expand(B, -1, -1, H, W)
        feat_uv = (shared.unsqueeze(1) + h_uv).reshape(B * n, self.hidden_dim, H, W)
        u_pred = self.u_head(feat_uv).view(B, n, 1, H, W)
        v_pred = self.v_head(feat_uv).view(B, n, 1, H, W)

        # w
        h_w = self.height_emb_w(level_ids_chunk).view(1, n, self.hidden_dim, 1, 1).expand(B, -1, -1, H, W)
        feat_w = self.w_enc((shared.unsqueeze(1) + h_w).reshape(B * n, self.hidden_dim, H, W))
        # Prior mask: 1.0 for levels < 14, 0.0 otherwise
        prior_mask_w = torch.where(level_ids_chunk < 14, 1.0, 0.0).view(1, n, 1, 1).expand(B, -1, -1, H, W)
        feat_w = feat_w + self.w_prior_proj(w_prior.unsqueeze(1).expand(B, n, -1, H, W).reshape(B * n, 1, H, W)) * prior_mask_w.reshape(B * n, 1, H, W)
        w_pred = self.w_dec(feat_w).view(B, n, 1, H, W)

        # k
        h_k = self.height_emb_k(level_ids_chunk).view(1, n, self.hidden_dim, 1, 1).expand(B, -1, -1, H, W)
        feat_k = self.k_enc((shared.unsqueeze(1) + h_k).reshape(B * n, self.hidden_dim, H, W))
        prior_mask_k = torch.where(level_ids_chunk < 14, 1.0, 0.0).view(1, n, 1, 1).expand(B, -1, -1, H, W)
        feat_k = feat_k + self.k_prior_proj(k_prior.unsqueeze(1).expand(B, n, -1, H, W).reshape(B * n, 1, H, W)) * prior_mask_k.reshape(B * n, 1, H, W)
        k_pred = self.k_dec(feat_k).view(B, n, 1, H, W)

        return u_pred, v_pred, w_pred, k_pred

    def forward(self, cond, x_raw):
        """
        cond: (B, in_channels, H, W) from encoder
        x_raw: (B, 6, H, W) raw input for physics priors
        Returns: (B, 27, 4, H, W)
        """
        B, _, H, W = cond.shape
        device = cond.device

        shared = self.shared_encoder(cond)
        w_prior = self._compute_terrain_prior(x_raw)
        k_prior = self._compute_turbulence_prior(x_raw)

        u_all, v_all, w_all, k_all = [], [], [], []

        chunk_size = 3
        bhw_tensor = torch.tensor([B, H, W], device=device)
        for start in range(0, self.n_levels, chunk_size):
            end = min(start + chunk_size, self.n_levels)
            level_ids_chunk = torch.arange(start, end, device=device)

            u_pred, v_pred, w_pred, k_pred = torch.utils.checkpoint.checkpoint(
                self._decode_chunk,
                shared, level_ids_chunk, w_prior, k_prior, bhw_tensor,
                use_reentrant=False,
            )
            u_all.append(u_pred)
            v_all.append(v_pred)
            w_all.append(w_pred)
            k_all.append(k_pred)

        output = torch.cat([
            torch.cat(u_all, dim=1),
            torch.cat(v_all, dim=1),
            torch.cat(w_all, dim=1),
            torch.cat(k_all, dim=1),
        ], dim=2)  # (B, 27, 4, H, W)

        output = self.vertical_smoother(output)
        return output


class SwinUNetV6(nn.Module):
    """V6: Same encoder as V5, enhanced decoder + vertical smoother."""

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

        # Encoder (identical to V5)
        self.input_encoder = LightweightMultiModalEncoder(
            in_channels=in_channels, base_dim=base_channels, target_size=(300, 300),
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

        # Enhanced vertical decoder
        self.vertical_decoder = EnhancedVerticalDecoder(
            in_channels=base_channels, n_levels=n_levels, hidden_dim=128,
        )

        # Per-level denormalization
        self.register_buffer('output_mean', output_mean if output_mean is not None else torch.zeros(1))
        self.register_buffer('output_std', output_std if output_std is not None else torch.ones(1))
        self._has_output_norm = output_mean is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 6, 300, 300) → (B, 27, 4, 300, 300)"""
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

        output = self.vertical_decoder(x, x_raw)

        # Per-level denormalization
        if self._has_output_norm:
            mean = self.output_mean
            std = self.output_std
            if mean.dim() == 2:  # (27, 4) per-level
                mean = mean.view(1, self.n_levels, 4, 1, 1)
                std = std.view(1, self.n_levels, 4, 1, 1)
            elif mean.dim() == 1:  # (4,) global fallback
                mean = mean.view(1, 1, 4, 1, 1)
                std = std.view(1, 1, 4, 1, 1)
            output = output * std + mean

        return output

    def get_num_params(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        dec = sum(p.numel() for p in self.vertical_decoder.parameters())
        return {'total': total, 'trainable': trainable, 'decoder': dec}
