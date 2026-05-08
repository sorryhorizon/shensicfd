import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List


class EnhancedPhysicsLoss(nn.Module):
    """
    增强版物理约束损失函数 (Enhanced Physics-Informed Loss)
    
    包含完整的CFD物理约束：
    
    1. 数据保真损失
       - MSE Loss: 像素级重建精度
       - L1 Loss: 鲁棒性辅助
    
    2. 质量守恒约束 (Mass Conservation)
       - 连续性方程: ∂u/∂x + ∂v/∂y + ∂w/∂z = 0
       - 保证风场的物理一致性
    
    3. 边界层相似性理论 (Boundary Layer Similarity Theory)
       - 对数律分布: u(z) = (u*/κ) * ln(z/z0)
       - 近地面风速应遵循对数廓线
    
    4. 地形约束 (Terrain Constraint)
       - 垂直速度在地形表面应为零
       - 风不能穿透地形
    
    5. 湍流能量正则化 (TKE Regularization)
       - k ≥ 0 everywhere
       - k应在合理范围内
    
    6. 梯度平滑损失 (Gradient Smoothness)
       - 空间梯度连续性
       - 避免不合理的空间振荡
    
    权重配置基于CFD理论和经验值，可根据实验调整
    """
    def __init__(
        self,
        mse_weight: float = 1.0,
        l1_weight: float = 0.5,
        mass_conservation_weight: float = 0.1,
        boundary_layer_weight: float = 0.05,
        terrain_penalty_weight: float = 0.1,
        k_positive_weight: float = 0.05,
        gradient_smoothness_weight: float = 0.1,
        level_weights: bool = True,
        dx: float = 30.0,
        dy: float = 30.0,
        dz: float = 10.0,
        von_karman: float = 0.4,
        k_specialized_weight: float = 0.5,
        use_k_transform: bool = True,
        use_k_height_profile: bool = False,
    ):
        super().__init__()

        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.mass_conservation_weight = mass_conservation_weight
        self.boundary_layer_weight = boundary_layer_weight
        self.terrain_penalty_weight = terrain_penalty_weight
        self.k_positive_weight = k_positive_weight
        self.gradient_smoothness_weight = gradient_smoothness_weight

        self.k_specialized_weight = k_specialized_weight
        self.use_k_transform = use_k_transform
        self.use_k_height_profile = use_k_height_profile

        self.level_weights = level_weights
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.von_karman = von_karman
        
        if level_weights:
            self.register_buffer('level_importance', self._compute_level_weights())
        else:
            self.level_importance = None
        
        self.register_buffer('height_levels', self._get_height_levels())
        
        self.mse_loss_fn = nn.MSELoss(reduction='none')
        self.l1_loss_fn = nn.L1Loss(reduction='none')
    
    @staticmethod
    def _compute_level_weights() -> torch.Tensor:
        """
        计算垂直层级权重
        
        近地面层级更重要（影响人类活动、建筑载荷等）
        使用高度倒数作为权重基础
        """
        levels = torch.tensor([5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
                               55, 60, 65, 70, 75, 80, 85, 90, 95, 100,
                               106.5, 114.95, 125.94, 140.22, 158.78,
                               182.91, 214.29], dtype=torch.float32)
        
        weights = 1.0 / (levels / levels[0])
        weights = weights / weights.mean()
        
        return weights
    
    @staticmethod
    def _get_height_levels() -> torch.Tensor:
        """获取27个垂直层的高度值（米）"""
        return torch.tensor([5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
                            55, 60, 65, 70, 75, 80, 85, 90, 95, 100,
                            106.5, 114.95, 125.94, 140.22, 158.78,
                            182.91, 214.29], dtype=torch.float32)
    
    def compute_mass_conservation_loss(self, output: torch.Tensor) -> torch.Tensor:
        """
        计算质量守恒损失
        
        连续性方程: ∂u/∂x + ∂v/∂y + ∂w/∂z = 0
        
        使用中心差分计算偏导数
        """
        B, L, C, H, W = output.shape
        
        u = output[:, :, 0]
        v = output[:, :, 1]
        w = output[:, :, 2]
        
        dudx = (u[:, 1:-1, 1:-1, 2:] - u[:, 1:-1, 1:-1, :-2]) / (2 * self.dx)
        dvdy = (v[:, 1:-1, 2:, 1:-1] - v[:, 1:-1, :-2, 1:-1]) / (2 * self.dy)
        dwdz = (w[:, 2:, 1:-1, 1:-1] - w[:, :-2, 1:-1, 1:-1]) / (2 * self.dz)
        
        divergence = dudx + dvdy + dwdz
        divergence = torch.clamp(divergence, min=-100.0, max=100.0)
        
        loss = (divergence ** 2).mean()
        
        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(0.0, device=output.device)
        
        return loss
    
    def compute_boundary_layer_loss(
        self,
        output: torch.Tensor,
        roughness: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算边界层相似性理论损失 (向量化版本)

        根据对数律，近地面风速应满足：
        u(z) / u(z_ref) ≈ ln(z/z0) / ln(z_ref/z0)
        """
        B, L, C, H, W = output.shape
        u = output[:, :, 0]

        n_bl = min(L, 10)
        if n_bl <= 1:
            return torch.tensor(0.0, device=output.device)

        u_surface = u[:, 0:1]  # (B, 1, H, W)
        u_bl = u[:, 1:n_bl]    # (B, n_bl-1, H, W)

        z_ref = self.height_levels[0]
        z_levels = self.height_levels[1:n_bl].view(1, n_bl - 1, 1, 1).to(output.device)

        if roughness is not None:
            z0 = roughness.squeeze(1).unsqueeze(1)  # (B, 1, H, W)
            z0 = torch.clamp(z0, min=0.01, max=5.0)
            log_zi_z0 = torch.log(torch.clamp(z_levels / z0, min=0.01, max=100.0) + 1e-8)
            log_zref_z0 = torch.log(torch.clamp(z_ref / z0, min=0.01, max=100.0) + 1e-8)
            expected_ratio = log_zi_z0 / (log_zref_z0 + 1e-6)
        else:
            expected_ratio = torch.log(torch.clamp(z_levels / z_ref, min=0.1, max=10.0) + 1e-8)

        # Expand expected_ratio to match actual_ratio shape
        expected_ratio = expected_ratio.expand_as(u_bl)
        actual_ratio = u_bl / (torch.abs(u_surface) + 1e-3)

        loss = F.mse_loss(actual_ratio, expected_ratio)

        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(0.0, device=output.device)

        return loss
    
    def compute_terrain_penalty(
        self,
        output: torch.Tensor,
        dem: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        地形约束损失 (法向边界条件版本)

        CFD 不可穿透条件: u·n = 0
        即: w = u * dz/dx + v * dz/dy

        约束:
        1. 地面法向速度为零 (u·n = 0)
        2. 高层按高度衰减
        """
        B, L, C, H, W = output.shape
        u = output[:, :, 0]
        v = output[:, :, 1]
        w = output[:, :, 2]

        if dem is not None:
            # 计算 DEM 梯度 (Sobel 算子)
            dz_dx, dz_dy = self._compute_dem_gradient(dem)
            # dz_dx, dz_dy: (B, H, W)

            # 地面层 (level 0): 法向速度应为零
            u_g = u[:, 0]
            v_g = v[:, 0]
            w_g = w[:, 0]
            w_expected_g = u_g * dz_dx + v_g * dz_dy
            surface_penalty = F.mse_loss(w_g, w_expected_g)

            # 高层: 法向约束按高度衰减
            decay = torch.linspace(1.0, 0.1, L, device=w.device).view(1, L, 1, 1)
            # 计算每层的期望 w
            # u, v, w: (B, L, H, W)
            # dz_dx, dz_dy: (B, H, W) -> (B, 1, H, W)
            dz_dx_l = dz_dx.unsqueeze(1)
            dz_dy_l = dz_dy.unsqueeze(1)
            w_expected = u * dz_dx_l + v * dz_dy_l
            weighted_penalty = (decay * (w - w_expected) ** 2).mean()
        else:
            # 无 DEM 时回退到简单 w=0
            surface_penalty = (w[:, 0] ** 2).mean()
            decay = torch.linspace(1.0, 0.3, L, device=w.device).view(1, L, 1, 1)
            weighted_penalty = (decay * w ** 2).mean()

        loss = surface_penalty + weighted_penalty
        return loss

    @staticmethod
    def _compute_dem_gradient(dem: torch.Tensor) -> tuple:
        """计算 DEM 梯度 (Sobel 算子)"""
        # Sobel 核
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                dtype=dem.dtype, device=dem.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                dtype=dem.dtype, device=dem.device).view(1, 1, 3, 3)
        # dem: (B, 1, H, W)
        dz_dx = F.conv2d(dem, sobel_x, padding=1)
        dz_dy = F.conv2d(dem, sobel_y, padding=1)
        # 归一化到物理坡度 (除以 2 * pixel_distance)
        # 这里 pixel_distance = 1 (像素单位), 实际 dx=30m 在损失权重中考虑
        return dz_dx.squeeze(1), dz_dy.squeeze(1)
    
    def compute_tke_regularization(self, output: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算湍流能量(TKE)正则化

        物理约束：
        1. k ≥ 0 everywhere（TKE非负）
        2. k应在合理范围内（通常0.01 ~ 5.0 m²/s²）
        """
        B, L, C, H, W = output.shape
        k = output[:, :, 3]

        positive_penalty = F.relu(-k).mean()

        k_mean = k.mean()
        k_std = k.std()

        reasonable_range_penalty = (
            F.relu(-k_mean + 0.01) ** 2 +
            F.relu(k_mean - 5.0) ** 2 +
            F.relu(-k_std + 0.001) ** 2
        )

        losses = {
            'k_positive': positive_penalty,
            'k_reasonable': reasonable_range_penalty,
        }

        return losses

    def compute_k_terrain_constraint(self, output: torch.Tensor, dem: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        k 物理约束损失

        物理依据：
        1. 地面层 TKE 与地形坡度正相关: k_ground ~ slope_magnitude * 0.01
        2. TKE 与速度梯度平方正相关: k ~ (du/dz)^2 + (dv/dz)^2

        Args:
            output: (B, L, 4, H, W)
            dem: (B, 1, H, W) 地形高程

        Returns:
            k 物理约束损失标量
        """
        B, L, C, H, W = output.shape
        k = output[:, :, 3]
        loss = torch.tensor(0.0, device=output.device)

        if dem is not None and L > 0:
            dz_dx, dz_dy = self._compute_dem_gradient(dem)
            slope = torch.sqrt(dz_dx ** 2 + dz_dy ** 2 + 1e-6)

            # Ground level correlation: k_ground ~ slope * 0.01
            k_ground = k[:, 0]
            expected_k_ground = slope.squeeze(1) * 0.01
            loss = loss + F.mse_loss(k_ground, expected_k_ground)

        # Velocity gradient correlation: k ~ (du/dz)^2 + (dv/dz)^2
        if L > 1:
            dudz = (output[:, 1:, 0] - output[:, :-1, 0]) / self.dz
            dvdz = (output[:, 1:, 1] - output[:, :-1, 1]) / self.dz
            vel_grad_sq = dudz ** 2 + dvdz ** 2

            k_levels = k[:, 1:]
            expected_k = vel_grad_sq * 0.1
            loss = loss + F.mse_loss(k_levels, expected_k)

        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(0.0, device=output.device)

        return loss
    
    def compute_gradient_smoothness_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        计算梯度平滑损失

        惩罚预测场和真实场之间的梯度差异
        保证空间连续性和物理合理性
        """
        pred_dx = torch.diff(pred, dim=-1)
        pred_dy = torch.diff(pred, dim=-2)
        target_dx = torch.diff(target, dim=-1)
        target_dy = torch.diff(target, dim=-2)

        loss_x = F.mse_loss(pred_dx, target_dx)
        loss_y = F.mse_loss(pred_dy, target_dy)

        loss = (loss_x + loss_y) * 0.5

        return loss

    def compute_log_k_mse(self, pred_k: torch.Tensor, target_k: torch.Tensor) -> torch.Tensor:
        """
        计算k分量的对数MSE损失（处理长尾分布）

        对k值进行log(k + 0.01)变换，解决k值分布的长尾问题，
        使得模型能够更好地学习k的空间分布特征。

        Args:
            pred_k: 预测的k值张量 (B, L, H, W)
            target_k: 真实的k值张量 (B, L, H, W)

        Returns:
            对数变换后的MSE损失
        """
        clamped_pred = torch.clamp(pred_k, min=1e-6)
        clamped_target = torch.clamp(target_k, min=1e-6)

        log_pred = torch.log(clamped_pred + 0.01)
        log_target = torch.log(clamped_target + 0.01)

        loss = F.mse_loss(log_pred, log_target)

        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(0.0, device=pred_k.device)

        return loss

    def compute_k_variance_preservation(self, pred_k: torch.Tensor, target_k: torch.Tensor) -> torch.Tensor:
        """
        计算k分量方差保持损失

        确保预测k的空间异质性（空间方差）与真实值匹配，
        防止模型预测出过于均匀或过于分散的k场。

        Args:
            pred_k: 预测的k值张量 (B, L, H, W)
            target_k: 真实的k值张量 (B, L, H, W)

        Returns:
            方差差异的MSE损失
        """
        pred_var = pred_k.var(dim=[2, 3], keepdim=True)
        target_var = target_k.var(dim=[2, 3], keepdim=True)

        loss = F.mse_loss(pred_var, target_var)

        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(0.0, device=pred_k.device)

        return loss

    def compute_k_height_profile(self, output: torch.Tensor) -> torch.Tensor:
        """
        计算k分量高度剖面损失

        约束k随高度的变化符合物理规律：k应随高度增加而衰减。
        使用线性衰减作为期望剖面（从地面1.0到高层0.3）。

        Args:
            output: 完整输出张量 (B, L, C, H, W)，k在第4个通道(index=3)

        Returns:
            高度剖面匹配损失
        """
        k = output[:, :, 3]
        B, L, H, W = k.shape
        k_mean_per_level = k.mean(dim=[2, 3])
        
        expected_decay = torch.linspace(1.0, 0.3, L, device=k.device)
        actual_ratio = k_mean_per_level / (k_mean_per_level[:, 0:1] + 1e-6)
        
        expected_decay_batch = expected_decay.unsqueeze(0).expand(B, -1)
        loss = F.mse_loss(actual_ratio, expected_decay_batch)

        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(0.0, device=k.device)

        return loss
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        dem: Optional[torch.Tensor] = None,
        roughness: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> dict:
        """
        计算复合损失
        
        Args:
            pred: (B, 27, 4, H, W) 模型预测
            target: (B, 27, 4, H, W) 真实值
            dem: (B, 1, H, W) 可选的地形高程
            roughness: (B, 1, H, W) 可选的地表粗糙度
            return_dict: 是否返回详细损失字典
        
        Returns:
            如果return_dict=True，返回包含各项损失的字典，'total'为总损失
            否则只返回总损失标量
        """
        B, L, C, H, W = pred.shape
        
        if self.level_importance is not None:
            lw = self.level_importance.view(1, L, 1, 1, 1).to(pred.device)
            mse_per_element = self.mse_loss_fn(pred, target) * lw
            l1_per_element = self.l1_loss_fn(pred, target) * lw
        else:
            mse_per_element = self.mse_loss_fn(pred, target)
            l1_per_element = self.l1_loss_fn(pred, target)
        
        loss_mse = mse_per_element.mean() * self.mse_weight
        loss_l1 = l1_per_element.mean() * self.l1_weight
        
        loss_mass_conservation = self.compute_mass_conservation_loss(pred) * self.mass_conservation_weight
        
        loss_boundary_layer = self.compute_boundary_layer_loss(pred, roughness) * self.boundary_layer_weight
        
        loss_terrain = self.compute_terrain_penalty(pred, dem) * self.terrain_penalty_weight

        tke_losses = self.compute_tke_regularization(pred)
        loss_k_positive = tke_losses['k_positive'] * self.k_positive_weight
        loss_k_reasonable = tke_losses['k_reasonable'] * 0.02

        loss_k_terrain = self.compute_k_terrain_constraint(pred, dem) * (self.k_specialized_weight * 0.2)

        loss_gradient = self.compute_gradient_smoothness_loss(pred, target) * self.gradient_smoothness_weight

        # K-specialized loss (分离处理)
        if self.use_k_transform:
            pred_k = pred[:, :, 3]
            target_k = target[:, :, 3]

            loss_log_k_mse = self.compute_log_k_mse(pred_k, target_k) * self.k_specialized_weight
            loss_k_var = self.compute_k_variance_preservation(pred_k, target_k) * (self.k_specialized_weight * 0.3)
            loss_k_height = self.compute_k_height_profile(pred) * (self.k_specialized_weight * 0.2) if self.use_k_height_profile else torch.tensor(0.0, device=pred.device)

            # 将k从统一MSE中移除，使用专用loss替代
            loss_mse_uvw = mse_per_element[:, :, :3].mean() * self.mse_weight
            loss_l1_uvw = l1_per_element[:, :, :3].mean() * self.l1_weight

            clamped_pred_k = torch.clamp(pred_k, min=1e-6)
            clamped_target_k = torch.clamp(target_k, min=1e-6)
            loss_log_k_l1 = F.l1_loss(torch.log(clamped_pred_k + 0.01), torch.log(clamped_target_k + 0.01)) * self.k_specialized_weight * 0.3

            total_loss = (
                loss_mse_uvw +  # 只包含u,v,w的MSE
                loss_log_k_mse +  # k的log-MSE
                loss_l1_uvw +  # 只包含u,v,w的L1
                loss_log_k_l1 +  # k的log-L1
                loss_mass_conservation +
                loss_boundary_layer +
                loss_terrain +
                loss_k_positive * 0.2 +  # 增强k正性约束
                loss_k_reasonable * 0.05 +
                loss_k_var +  # 方差保持
                loss_k_height +  # 高度剖面
                loss_k_terrain +  # k地形/梯度物理约束
                loss_gradient
            )
        else:
            total_loss = (
                loss_mse +
                loss_l1 +
                loss_mass_conservation +
                loss_boundary_layer +
                loss_terrain +
                loss_k_positive +
                loss_k_reasonable +
                loss_gradient
            )
        
        if return_dict:
            if self.use_k_transform:
                logged_mse = loss_mse_uvw.detach()
                logged_l1 = loss_l1_uvw.detach()
            else:
                logged_mse = loss_mse.detach()
                logged_l1 = loss_l1.detach()

            losses = {
                'total': total_loss,
                'mse': logged_mse,
                'l1': logged_l1,
                'mass_conservation': loss_mass_conservation.detach(),
                'boundary_layer': loss_boundary_layer.detach(),
                'terrain': loss_terrain.detach(),
                'k_positive': loss_k_positive.detach(),
                'gradient': loss_gradient.detach(),
            }

            if self.use_k_transform:
                losses['log_k_mse'] = loss_log_k_mse.detach()
                losses['k_variance'] = loss_k_var.detach()
                losses['k_height_profile'] = loss_k_height.detach()
                losses['log_k_l1'] = loss_log_k_l1.detach()
                losses['k_terrain'] = loss_k_terrain.detach()
            
            var_names = ['u', 'v', 'w', 'k']
            for i, name in enumerate(var_names):
                var_mse = mse_per_element[:, :, i].mean().detach()
                losses[f'mse_{name}'] = var_mse
            
            sample_levels = [0, 9, 18, 26]
            for l in sample_levels:
                if l < L:
                    level_mse = mse_per_element[:, l].mean().detach()
                    losses[f'mse_level_{l}'] = level_mse
            
            return losses
        else:
            return total_loss


class ProgressiveLossScheduler(nn.Module):
    """
    渐进式损失调度器
    
    用于两阶段/多阶段训练策略：
    - 阶段1：主要优化数据保真损失
    - 阶段2：逐渐引入物理约束
    - 阶段3：平衡数据和物理约束
    
    通过调整各项损失的权重实现渐进式训练
    """
    def __init__(
        self,
        base_loss: EnhancedPhysicsLoss,
        n_stages: int = 3,
        stage_epochs: List[int] = [30, 40, 30],
    ):
        super().__init__()
        
        self.base_loss = base_loss
        self.n_stages = n_stages
        self.stage_epochs = stage_epochs
        
        self.stage_configs = [
            {
                'mse_weight': 1.0,
                'l1_weight': 0.5,
                'mass_conservation_weight': 0.01,
                'boundary_layer_weight': 0.0,
                'terrain_penalty_weight': 0.01,
                'k_positive_weight': 0.2,
                'gradient_smoothness_weight': 0.05,
                'k_specialized_weight': 0.5,
            },
            {
                'mse_weight': 1.0,
                'l1_weight': 0.5,
                'mass_conservation_weight': 0.05,
                'boundary_layer_weight': 0.02,
                'terrain_penalty_weight': 0.05,
                'k_positive_weight': 0.3,
                'gradient_smoothness_weight': 0.08,
                'k_specialized_weight': 0.7,
            },
            {
                'mse_weight': 1.0,
                'l1_weight': 0.5,
                'mass_conservation_weight': 0.1,
                'boundary_layer_weight': 0.05,
                'terrain_penalty_weight': 0.1,
                'k_positive_weight': 0.4,
                'gradient_smoothness_weight': 0.1,
                'k_specialized_weight': 1.0,
            },
        ]
        
        assert len(self.stage_configs) >= n_stages
    
    def get_current_stage(self, epoch: int) -> int:
        """根据当前epoch确定训练阶段"""
        cumulative_epochs = 0
        for stage in range(self.n_stages):
            cumulative_epochs += self.stage_epochs[stage]
            if epoch < cumulative_epochs:
                return stage
        return self.n_stages - 1
    
    def update_weights_for_stage(self, stage: int):
        """更新当前阶段的损失权重"""
        config = self.stage_configs[min(stage, len(self.stage_configs) - 1)]

        self.base_loss.mse_weight = config['mse_weight']
        self.base_loss.l1_weight = config['l1_weight']
        self.base_loss.mass_conservation_weight = config['mass_conservation_weight']
        self.base_loss.boundary_layer_weight = config['boundary_layer_weight']
        self.base_loss.terrain_penalty_weight = config['terrain_penalty_weight']
        self.base_loss.k_positive_weight = config['k_positive_weight']
        self.base_loss.gradient_smoothness_weight = config['gradient_smoothness_weight']

        if 'k_specialized_weight' in config:
            self.base_loss.k_specialized_weight = config['k_specialized_weight']
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        epoch: int,
        dem: Optional[torch.Tensor] = None,
        roughness: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> dict:
        """
        计算当前阶段的损失

        Args:
            pred: 模型预测
            target: 真实值
            epoch: 当前训练轮次
            dem: 可选的地形高程
            roughness: 可选的地表粗糙度
            return_dict: 是否返回详细损失字典

        Returns:
            损失字典
        """
        current_stage = self.get_current_stage(epoch)
        self.update_weights_for_stage(current_stage)

        losses = self.base_loss(pred, target, dem=dem, roughness=roughness, return_dict=return_dict)
        losses['current_stage'] = current_stage

        return losses
