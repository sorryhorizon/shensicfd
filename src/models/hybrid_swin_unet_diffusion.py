import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math

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
# Decoupled Vertical Decoder: 4 independent heads for u/v/w/k
# ============================================================

class DecoupledVerticalDecoder(nn.Module):
    """
    4-head decoupled vertical decoder.

    Each variable (u, v, w, k) has its own decoder head and height embedding,
    ensuring independent gradient pathways. w and k get deeper networks
    since they are harder to learn.
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

        # w: deeper head (3 conv layers) — needs more capacity for terrain-edge features
        self.w_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            RMSGroupNorm(hidden_dim), nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1, bias=False),
            RMSGroupNorm(hidden_dim // 2), nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 1, 1),
        )

        # k: deeper head (3 conv layers) — needs more capacity for sparse peaks
        self.k_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            RMSGroupNorm(hidden_dim), nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1, bias=False),
            RMSGroupNorm(hidden_dim // 2), nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 1, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        feat = self.shared_encoder(x)

        dev = x.device
        u_embs = self.u_height_emb(torch.arange(self.n_levels, device=dev)).view(self.n_levels, -1, 1, 1)
        v_embs = self.v_height_emb(torch.arange(self.n_levels, device=dev)).view(self.n_levels, -1, 1, 1)
        w_embs = self.w_height_emb(torch.arange(self.n_levels, device=dev)).view(self.n_levels, -1, 1, 1)
        k_embs = self.k_height_emb(torch.arange(self.n_levels, device=dev)).view(self.n_levels, -1, 1, 1)

        chunk_size = 9
        u_out, v_out, w_out, k_out = [], [], [], []
        for start in range(0, self.n_levels, chunk_size):
            end = min(start + chunk_size, self.n_levels)
            n = end - start

            u_f = (feat.unsqueeze(1) + u_embs[start:end].unsqueeze(0)).view(B * n, -1, H, W)
            v_f = (feat.unsqueeze(1) + v_embs[start:end].unsqueeze(0)).view(B * n, -1, H, W)
            w_f = (feat.unsqueeze(1) + w_embs[start:end].unsqueeze(0)).view(B * n, -1, H, W)
            k_f = (feat.unsqueeze(1) + k_embs[start:end].unsqueeze(0)).view(B * n, -1, H, W)

            u_out.append(self.u_head(u_f).view(B, n, 1, H, W))
            v_out.append(self.v_head(v_f).view(B, n, 1, H, W))
            w_out.append(self.w_head(w_f).view(B, n, 1, H, W))
            k_out.append(self.k_head(k_f).view(B, n, 1, H, W))

        output = torch.cat([
            torch.cat(u_out, dim=1),
            torch.cat(v_out, dim=1),
            torch.cat(w_out, dim=1),
            torch.cat(k_out, dim=1),
        ], dim=2)  # (B, 27, 4, H, W)
        return output


# ============================================================
# k Diffusion Denoiser
# ============================================================

class KDiffusionDenoiser(nn.Module):
    """
    Lightweight U-Net for denoising k field.

    Takes noisy k at timestep t, plus condition features from the main model,
    and predicts the noise ε that was added.

    Architecture: small U-Net with time embedding and condition injection.
    """
    def __init__(self, cond_channels: int = 48, k_channels: int = 27, base_ch: int = 32):
        super().__init__()
        self.k_channels = k_channels

        # Time embedding (sinusoidal)
        time_dim = base_ch * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Input: noisy k (27 channels) + condition (cond_channels)
        in_ch = k_channels + cond_channels
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1, bias=False),
            RMSGroupNorm(base_ch), nn.GELU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, 3, padding=1, stride=2, bias=False),
            RMSGroupNorm(base_ch * 2), nn.GELU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, padding=1, stride=2, bias=False),
            RMSGroupNorm(base_ch * 4), nn.GELU(),
        )

        # Bottleneck with time injection
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_ch * 4 + time_dim, base_ch * 4, 3, padding=1, bias=False),
            RMSGroupNorm(base_ch * 4), nn.GELU(),
            nn.Conv2d(base_ch * 4, base_ch * 4, 3, padding=1, bias=False),
            RMSGroupNorm(base_ch * 4), nn.GELU(),
        )

        # Decoder
        self.dec3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_ch * 4, base_ch * 2, 3, padding=1, bias=False),
            RMSGroupNorm(base_ch * 2), nn.GELU(),
        )
        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_ch * 2, base_ch, 3, padding=1, bias=False),
            RMSGroupNorm(base_ch), nn.GELU(),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1, bias=False),
            RMSGroupNorm(base_ch), nn.GELU(),
            nn.Conv2d(base_ch, k_channels, 1),  # output: predicted noise (27 channels)
        )

    def forward(self, noisy_k: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            noisy_k: (B, 27, H, W) noisy k field
            t: (B,) timestep
            condition: (B, cond_channels, H, W) condition features from main model
        Returns:
            (B, 27, H, W) predicted noise ε
        """
        B = noisy_k.shape[0]
        H, W = noisy_k.shape[-2], noisy_k.shape[-1]

        # Time embedding
        t_emb = self.time_mlp(t.view(B, 1))  # (B, time_dim)

        # Concatenate noisy k with condition
        x = torch.cat([noisy_k, condition], dim=1)  # (B, 27+cond, H, W)

        # Encoder
        e1 = self.enc1(x)       # (B, base_ch, H, W)
        e2 = self.enc2(e1)      # (B, base_ch*2, H/2, W/2)
        e3 = self.enc3(e2)      # (B, base_ch*4, H/4, W/4)

        # Inject time into bottleneck
        t_map = t_emb.view(B, -1, 1, 1).expand(B, -1, H // 4, W // 4)
        bot_in = torch.cat([e3, t_map], dim=1)
        bot = self.bottleneck(bot_in)  # (B, base_ch*4, H/4, W/4)

        # Decoder with skip connections
        d3 = self.dec3(bot) + e2  # (B, base_ch*2, H/2, W/2)
        d2 = self.dec2(d3) + e1   # (B, base_ch, H, W)
        d1 = self.dec1(d2)        # (B, 27, H, W)

        return d1


class CosineNoiseScheduler:
    """Cosine noise schedule for k diffusion."""
    def __init__(self, n_timesteps: int = 1000, s: float = 0.008):
        self.n_timesteps = n_timesteps
        steps = torch.arange(n_timesteps + 1, dtype=torch.float64)
        f = torch.cos((steps / n_timesteps + s) / (1 + s) * math.pi / 2) ** 2
        f = f / f[0]
        alpha_bar = f
        # Ensure monotonic decreasing
        alpha_bar = torch.clamp(alpha_bar, min=1e-10)

        self.alpha = alpha_bar[1:] / alpha_bar[:-1]
        self.alpha_bar = alpha_bar[1:]
        self.sigma = torch.sqrt(1 - self.alpha_bar)

    def add_noise(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Add noise to x_0 at timestep t."""
        B = x_0.shape[0]
        alpha_bar = self.alpha_bar.to(t.device)
        sigma = self.sigma.to(t.device)
        alpha_bar_t = alpha_bar[t].view(B, 1, 1, 1).to(x_0.dtype)
        sigma_t = sigma[t].view(B, 1, 1, 1).to(x_0.dtype)
        return alpha_bar_t.sqrt() * x_0 + sigma_t * noise

    def step_ddim(self, x_t: torch.Tensor, t: int, pred_noise: torch.Tensor, t_prev: int) -> torch.Tensor:
        """DDIM sampling step from t to t_prev."""
        alpha_bar_t = self.alpha_bar[t].to(x_t.device).to(x_t.dtype)
        alpha_bar_prev = self.alpha_bar[t_prev].to(x_t.device).to(x_t.dtype) if t_prev >= 0 else torch.tensor(1.0, device=x_t.device, dtype=x_t.dtype)

        # Predict x_0
        x_0_pred = (x_t - (1 - alpha_bar_t).sqrt() * pred_noise) / alpha_bar_t.sqrt()

        # DDIM step
        dir_xt = (1 - alpha_bar_prev).sqrt() * pred_noise
        x_prev = alpha_bar_prev.sqrt() * x_0_pred + dir_xt
        return x_prev


# ============================================================
# Main Model: Hybrid Swin-UNet + k Diffusion
# ============================================================

class HybridSwinUNetDiffusion(nn.Module):
    """
    Hybrid model: Swin-UNet for u/v/w (regression) + Diffusion for k.

    Architecture:
    - Shared encoder (Swin-UNet) processes input → condition features
    - Decoupled decoder: 4 independent heads for u/v/w/k_regression
    - k Diffusion: separate denoiser network that takes condition + noisy k

    Training:
    - u/v/w: standard regression loss (MSE/L1)
    - k_regression: provides a "prior" for k, used as x_0 in diffusion
    - k_diffusion: noise prediction loss (ε-prediction)

    Inference:
    - u/v/w: direct regression output
    - k: DDIM sampling from k_regression prior (or from pure noise)
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
        k_diffusion_steps: int = 1000,
        k_ddim_steps: int = 20,
        output_mean: Optional[torch.Tensor] = None,
        output_std: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.n_levels = n_levels
        self.base_channels = base_channels
        self.k_diffusion_steps = k_diffusion_steps
        self.k_ddim_steps = k_ddim_steps

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

        # Decoupled decoder for u/v/w/k_regression
        self.vertical_decoder = DecoupledVerticalDecoder(
            in_channels=base_channels,
            n_levels=n_levels,
            hidden_dim=base_channels,
        )

        # k Diffusion denoiser
        self.k_denoiser = KDiffusionDenoiser(
            cond_channels=base_channels,
            k_channels=n_levels,
            base_ch=32,
        )

        # Noise scheduler
        self.noise_scheduler = CosineNoiseScheduler(n_timesteps=k_diffusion_steps)

        self._initialize_weights()

        # Initialize k_head bias to k mean
        if hasattr(self.vertical_decoder, 'k_head'):
            last_layer = self.vertical_decoder.k_head[-1]
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

        return x  # condition features at (300, 300)

    def forward_train(self, x: torch.Tensor, k_target: torch.Tensor) -> dict:
        """
        Training forward pass.

        Args:
            x: input (B, 6, 300, 300)
            k_target: ground truth k (B, 27, 300, 300) — extracted from target[:, :, 3]

        Returns:
            dict with 'uvw_pred', 'k_regression', 'k_pred_noise', 't'
        """
        B = x.shape[0]
        cond = self.encode(x)

        # Regression output (u/v/w/k_regression)
        reg_output = self.vertical_decoder(cond)  # (B, 27, 4, 300, 300)
        uvw_pred = reg_output[:, :, :3]  # (B, 27, 3, 300, 300)
        k_regression = reg_output[:, :, 3]  # (B, 27, 300, 300)

        # k Diffusion: predict noise
        t = torch.randint(0, self.k_diffusion_steps, (B,), device=x.device)
        noise = torch.randn_like(k_target)
        noisy_k = self.noise_scheduler.add_noise(k_target, t, noise)

        pred_noise = self.k_denoiser(noisy_k, t.float(), cond)

        return {
            'uvw_pred': uvw_pred,
            'k_regression': k_regression,
            'k_pred_noise': pred_noise,
            'k_true_noise': noise,
            't': t,
        }

    def forward_inference(self, x: torch.Tensor, use_diffusion: bool = True) -> torch.Tensor:
        """
        Inference forward pass.

        Args:
            x: input (B, 6, 300, 300)
            use_diffusion: if True, use DDIM sampling for k; if False, use regression k

        Returns:
            (B, 27, 4, 300, 300)
        """
        B = x.shape[0]
        cond = self.encode(x)

        reg_output = self.vertical_decoder(cond)
        uvw_pred = reg_output[:, :, :3]
        k_regression = reg_output[:, :, 3]

        if not use_diffusion:
            output = torch.cat([uvw_pred, k_regression.unsqueeze(2)], dim=2)
            return output

        # DDIM sampling for k
        # Start from k_regression as x_0 prior (not pure noise)
        # This is "guided diffusion": start closer to the target
        k_pred = self._ddim_sample(cond, k_regression, B, x.device)
        output = torch.cat([uvw_pred, k_pred.unsqueeze(2)], dim=2)
        return output

    def _ddim_sample(self, cond: torch.Tensor, k_prior: torch.Tensor, B: int, device: torch.device) -> torch.Tensor:
        """DDIM sampling for k, starting from k_prior."""
        # Map DDIM steps to diffusion steps
        ddim_timesteps = torch.linspace(0, self.k_diffusion_steps - 1, self.k_ddim_steps + 1).long()

        # Start: add moderate noise to k_prior (not pure noise)
        # This gives the diffusion a "warm start" instead of starting from random
        start_t_idx = len(ddim_timesteps) // 2  # Start from midpoint noise level
        start_t = ddim_timesteps[start_t_idx]
        noise = torch.randn_like(k_prior)
        x_t = self.noise_scheduler.add_noise(k_prior, start_t.unsqueeze(0).expand(B), noise)

        for i in range(start_t_idx, len(ddim_timesteps) - 1):
            t = ddim_timesteps[i].item()
            t_prev = ddim_timesteps[i + 1].item()

            t_tensor = torch.tensor([t] * B, device=device, dtype=torch.float32)
            pred_noise = self.k_denoiser(x_t, t_tensor, cond)

            x_t = self.noise_scheduler.step_ddim(x_t, t, pred_noise, t_prev)

        return x_t

    def forward(self, x: torch.Tensor, k_target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Unified forward pass.

        If k_target is provided (training mode), calls forward_train which
        returns a dict with uvw_pred, k_regression, k_pred_noise, k_true_noise, t.
        If k_target is None (inference mode), calls forward_inference with diffusion.
        """
        if k_target is not None:
            return self.forward_train(x, k_target)
        return self.forward_inference(x, use_diffusion=True)

    def get_num_params(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        denoiser_params = sum(p.numel() for p in self.k_denoiser.parameters())
        return {
            'total': total,
            'trainable': trainable,
            'total_mb': total * 4 / (1024 ** 2),
            'denoiser_params': denoiser_params,
        }


def create_hybrid_model(config: dict = None) -> HybridSwinUNetDiffusion:
    """Factory function for hybrid model."""
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
        'k_diffusion_steps': 1000,
        'k_ddim_steps': 20,
        'output_mean': None,
        'output_std': None,
    }
    default_config.update(config)
    return HybridSwinUNetDiffusion(**default_config)


if __name__ == "__main__":
    model = create_hybrid_model()
    params = model.get_num_params()

    print(f"Model created!")
    print(f"   Total params: {params['total']:,} ({params['total_mb']:.1f} MB)")
    print(f"   Denoiser params: {params['denoiser_params']:,}")

    x = torch.randn(2, 6, 300, 300)

    # Test inference
    with torch.no_grad():
        y = model.forward_inference(x, use_diffusion=False)
    print(f"   Input: {x.shape}, Output (regression): {y.shape}")

    # Test training
    k_target = torch.randn(2, 27, 300, 300)
    result = model.forward_train(x, k_target)
    print(f"   uvw_pred: {result['uvw_pred'].shape}")
    print(f"   k_regression: {result['k_regression'].shape}")
    print(f"   k_pred_noise: {result['k_pred_noise'].shape}")