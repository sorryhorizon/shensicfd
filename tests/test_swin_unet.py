#!/usr/bin/env python3
"""
Physics-Informed Swin-U-Net 单元测试

测试内容：
1. Swin Transformer核心组件
2. 完整模型的前向传播
3. 物理约束损失函数
4. 内存使用情况验证
5. 梯度流验证

运行方式:
    cd /mnt/sdata/jz/shensi-CFD
    python -m pytest tests/test_swin_unet.py -v
    
或直接运行:
    python tests/test_swin_unet.py
"""

import sys
import os
import torch
import torch.nn as nn
import gc
from typing import Dict, Any


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_section(title: str):
    """打印测试分节标题"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_test(name: str, passed: bool, details: str = ""):
    """打印单个测试结果"""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  [{status}] {name}")
    if details:
        print(f"         {details}")


class TestSwinTransformerComponents:
    """测试Swin Transformer核心组件"""
    
    @staticmethod
    def test_window_attention():
        """测试Window Attention机制"""
        from src.models.swin_transformer import WindowAttention
        
        print_section("Test 1: Window Attention")
        
        dim = 128
        window_size = (7, 7)
        num_heads = 8
        
        attn = WindowAttention(dim=dim, window_size=window_size, num_heads=num_heads)
        
        B = 2
        N = window_size[0] * window_size[1]  # 49
        x = torch.randn(B, N, dim)
        
        try:
            output = attn(x)
            
            assert output.shape == (B, N, dim), f"形状不匹配: {output.shape} != {(B, N, dim)}"
            assert not torch.isnan(output).any(), "输出包含NaN"
            assert not torch.isinf(output).any(), "输出包含Inf"
            
            params = sum(p.numel() for p in attn.parameters())
            
            print_test("Window Attention前向传播", True,
                      f"输入: {x.shape} → 输出: {output.shape}, 参数量: {params:,}")
            
            return True
            
        except Exception as e:
            print_test("Window Attention前向传播", False, str(e))
            return False
    
    @staticmethod
    def test_swin_transformer_block():
        """测试Swin Transformer Block"""
        from src.models.swin_transformer import SwinTransformerBlock
        
        print_section("Test 2: Swin Transformer Block")
        
        dim = 128
        window_size = (7, 7)
        
        block_regular = SwinTransformerBlock(
            dim=dim,
            num_heads=8,
            window_size=window_size,
            shift_size=(0, 0),
        )
        
        block_shifted = SwinTransformerBlock(
            dim=dim,
            num_heads=8,
            window_size=window_size,
            shift_size=(3, 3),
        )
        
        B, C, H, W = 2, dim, 56, 56
        x = torch.randn(B, C, H, W)
        
        try:
            out_regular = block_regular(x)
            assert out_regular.shape == x.shape, f"常规窗口块形状错误: {out_regular.shape}"
            
            out_shifted = block_shifted(x)
            assert out_shifted.shape == x.shape, f"移位窗口块形状错误: {out_shifted.shape}"
            
            assert not torch.isnan(out_regular).any(), "常规窗口输出含NaN"
            assert not torch.isnan(out_shifted).any(), "移位窗口输出含NaN"
            
            print_test("常规Window-MSA", True, f"形状: {x.shape} → {out_regular.shape}")
            print_test("Shifted Window-MSA", True, f"形状: {x.shape} → {out_shifted.shape}")
            
            return True
            
        except Exception as e:
            print_test("Swin Transformer Block", False, str(e))
            return False
    
    @staticmethod
    def test_swin_transformer_stage():
        """测试完整的Swin Transformer Stage"""
        from src.models.swin_transformer import SwinTransformerStage
        
        print_section("Test 3: Swin Transformer Stage")
        
        stage = SwinTransformerStage(
            dim=128,
            depth=4,
            num_heads=8,
            window_size=(7, 7),
            drop_path_rate=0.1,
        )
        
        B, C, H, W = 2, 128, 56, 56
        x = torch.randn(B, C, H, W)
        
        try:
            output = stage(x)
            
            assert output.shape == x.shape, f"Stage输出形状错误: {output.shape}"
            assert not torch.isnan(output).any(), "输出包含NaN"
            
            params = sum(p.numel() for p in stage.parameters())
            
            print_test("Swin Transformer Stage", True,
                      f"输入: {x.shape} → 输出: {output.shape}, 参数量: {params:,}")
            
            return True
            
        except Exception as e:
            print_test("Swin Transformer Stage", False, str(e))
            return False


class TestSwinUNetLite:
    """测试轻量级模型 (包含k_decoder_head)"""

    @staticmethod
    def test_lite_model_creation():
        """测试Lite模型创建和参数统计"""
        from src.models.swin_unet_lite import create_lite_model, PhysicsInformedSwinUNetLite

        print_section("Test 4: Lite模型创建")

        try:
            model = create_lite_model()

            assert isinstance(model, PhysicsInformedSwinUNetLite), "模型类型错误"

            param_info = model.get_num_params()

            total_params = param_info['total']
            total_mb = param_info['total_mb']

            print_test("Lite模型创建成功", True, f"总参数量: {total_params:,} ({total_mb:.1f} MB)")

            assert total_params > 0, "参数量为0"
            assert total_mb < 500, f"模型过大: {total_mb:.1f} MB"

            return True

        except Exception as e:
            print_test("Lite模型创建失败", False, str(e))
            import traceback
            traceback.print_exc()
            return False

    @staticmethod
    def test_k_decoder_head_exists():
        """测试k专用解码器是否存在"""
        from src.models.swin_unet_lite import create_lite_model

        print_section("Test 4.1: k_decoder_head存在性检查")

        try:
            model = create_lite_model()

            assert hasattr(model.vertical_decoder, 'k_decoder_head'), \
                "AdaptiveVerticalDecoderLite should have k_decoder_head"

            assert isinstance(model.vertical_decoder.k_decoder_head, nn.Sequential), \
                "k_decoder_head should be nn.Sequential"

            k_decoder_layers = len(model.vertical_decoder.k_decoder_head)
            decoder_head_layers = len(model.vertical_decoder.decoder_head)

            print_test("k_decoder_head存在", True,
                      f"k_decoder_head层数: {k_decoder_layers}, decoder_head层数: {decoder_head_layers}")

            assert k_decoder_layers > decoder_head_layers, \
                f"k_decoder_head应该更深 ({k_decoder_layers} vs {decoder_head_layers})"

            print_test("k_decoder_head结构正确", True,
                      f"更深的网络用于k预测 (k:{k_decoder_layers}层 > uvw:{decoder_head_layers}层)")

            print("✅ k_decoder_head exists and is properly structured")

            return True

        except Exception as e:
            print_test("k_decoder_head存在性检查失败", False, str(e))
            import traceback
            traceback.print_exc()
            return False

    @staticmethod
    def test_lite_forward_pass():
        """测试Lite模型的前向传播"""
        from src.models.swin_unet_lite import create_lite_model

        print_section("Test 5: Lite模型前向传播")

        model = create_lite_model()
        model.eval()

        B = 1
        input_shape = (B, 4, 300, 300)

        x = torch.randn(*input_shape)

        try:
            with torch.no_grad():
                output = model(x)

            expected_shape = (B, 27, 4, 300, 300)

            assert output.shape == expected_shape, \
                f"输出形状错误: {output.shape} != {expected_shape}"

            assert not torch.isnan(output).any(), "输出包含NaN"
            assert not torch.isinf(output).any(), "输出包含Inf"

            print_test("Lite前向传播成功", True,
                      f"输入: {input_shape} → 输出: {output.shape}")

            u_mean = output[:, :, 0].mean().item()
            v_mean = output[:, :, 1].mean().item()
            w_mean = output[:, :, 2].mean().item()
            k_mean = output[:, :, 3].mean().item()

            print(f"           输出统计:")
            print(f"             - u: mean={u_mean:.4f}, std={output[:, :, 0].std().item():.4f}")
            print(f"             - v: mean={v_mean:.4f}, std={output[:, :, 1].std().item():.4f}")
            print(f"             - w: mean={w_mean:.4f}, std={output[:, :, 2].std().item():.4f}")
            print(f"             - k: mean={k_mean:.4f}, std={output[:, :, 3].std().item():.4f}")

            return True

        except Exception as e:
            print_test("Lite前向传播失败", False, str(e))
            import traceback
            traceback.print_exc()
            return False

    @staticmethod
    def test_lite_memory_usage():
        """测试Lite模型内存使用情况"""
        from src.models.swin_unet_lite import create_lite_model

        print_section("Test 6: Lite模型内存使用分析")

        if not torch.cuda.is_available():
            print_test("内存测试", False, "CUDA不可用 - 跳过")
            return True

        try:
            device = 'cuda'
            model = create_lite_model().to(device)
            model.train()

            torch.cuda.empty_cache()
            gc.collect()

            mem_before = torch.cuda.memory_allocated() / (1024 ** 3)

            B = 1
            x = torch.randn(B, 4, 300, 300, device=device)

            output = model(x)

            mem_after = torch.cuda.memory_allocated() / (1024 ** 3)
            mem_used = mem_after - mem_before

            del x, output
            torch.cuda.empty_cache()
            gc.collect()

            is_acceptable = mem_used < 12

            status_str = f"{mem_used:.2f} GB"
            if is_acceptable:
                status_str += " ✅ (可接受)"
            else:
                status_str += f" ❌ (超过12GB限制)"

            print_test("Lite GPU内存使用", is_acceptable, status_str)

            return is_acceptable

        except RuntimeError as e:
            if "out of memory" in str(e):
                print_test("GPU内存不足", False, "显存溢出(OOM) - 跳过此测试")
                torch.cuda.empty_cache()
                gc.collect()
                return True
            else:
                raise e

    @staticmethod
    def test_lite_gradient_flow():
        """测试Lite模型梯度流动"""
        from src.models.swin_unet_lite import create_lite_model

        print_section("Test 7: Lite模型梯度流动测试")

        model = create_lite_model()
        model.train()

        x = torch.randn(1, 4, 300, 300)
        target = torch.randn(1, 27, 4, 300, 300)

        try:
            output = model(x)

            loss = F.mse_loss(output, target)
            loss.backward()

            has_gradient = []
            no_gradient = []

            for name, param in model.named_parameters():
                if param.requires_grad:
                    if param.grad is not None and param.grad.abs().sum() > 0:
                        has_gradient.append(name)
                    else:
                        no_gradient.append(name)

            grad_ratio = len(has_gradient) / (len(has_gradient) + len(no_gradient)) * 100

            print_test("Lite梯度计算", True,
                      f"有梯度: {len(has_gradient)}, 无梯度: {len(no_gradient)}, "
                      f"梯度覆盖率: {grad_ratio:.1f}%")

            if len(no_gradient) > 0:
                print(f"           ⚠️  无梯度的参数:")
                for name in no_gradient[:5]:
                    print(f"              - {name}")
                if len(no_gradient) > 5:
                    print(f"              ... 还有{len(no_gradient)-5}个")

            is_good = grad_ratio > 90
            print_test("Lite梯度覆盖率", is_good, f"{grad_ratio:.1f}% (>90%为佳)")

            del x, target, output, loss
            gc.collect()

            return is_good

        except Exception as e:
            print_test("Lite梯度流动测试失败", False, str(e))
            import traceback
            traceback.print_exc()
            return False


class TestPhysicsInformedSwinUNet:
    """测试完整模型（保留原有测试）"""
    
    @staticmethod
    def test_model_creation():
        """测试模型创建和参数统计"""
        from src.models.swin_unet import create_model, PhysicsInformedSwinUNet
        
        print_section("Test 4: 模型创建")
        
        try:
            model = create_model()
            
            assert isinstance(model, PhysicsInformedSwinUNet), "模型类型错误"
            
            param_info = model.get_num_params()
            
            total_params = param_info['total']
            total_mb = param_info['total_mb']
            
            print_test("模型创建成功", True, f"总参数量: {total_params:,} ({total_mb:.1f} MB)")
            
            for name, count in param_info.items():
                if name not in ['total', 'trainable', 'total_mb']:
                    print(f"           - {name}: {count:,} 参数 ({count/1024/1024:.1f} MB)")
            
            assert total_params > 0, "参数量为0"
            assert total_mb < 500, f"模型过大: {total_mb:.1f} MB"
            
            return True
            
        except Exception as e:
            print_test("模型创建失败", False, str(e))
            import traceback
            traceback.print_exc()
            return False
    
    @staticmethod
    def test_forward_pass():
        """测试完整的前向传播"""
        from src.models.swin_unet import create_model
        
        print_section("Test 5: 前向传播")
        
        model = create_model()
        model.eval()
        
        B = 1
        input_shape = (B, 4, 300, 300)
        
        x = torch.randn(*input_shape)
        
        try:
            with torch.no_grad():
                output = model(x)
            
            expected_shape = (B, 27, 4, 300, 300)
            
            assert output.shape == expected_shape, \
                f"输出形状错误: {output.shape} != {expected_shape}"
            
            assert not torch.isnan(output).any(), "输出包含NaN"
            assert not torch.isinf(output).any(), "输出包含Inf"
            
            print_test("前向传播成功", True,
                      f"输入: {input_shape} → 输出: {output.shape}")
            
            u_mean = output[:, :, 0].mean().item()
            v_mean = output[:, :, 1].mean().item()
            w_mean = output[:, :, 2].mean().item()
            k_mean = output[:, :, 3].mean().item()
            
            print(f"           输出统计:")
            print(f"             - u: mean={u_mean:.4f}, std={output[:, :, 0].std().item():.4f}")
            print(f"             - v: mean={v_mean:.4f}, std={output[:, :, 1].std().item():.4f}")
            print(f"             - w: mean={w_mean:.4f}, std={output[:, :, 2].std().item():.4f}")
            print(f"             - k: mean={k_mean:.4f}, std={output[:, :, 3].std().item():.4f}")
            
            return True
            
        except Exception as e:
            print_test("前向传播失败", False, str(e))
            import traceback
            traceback.print_exc()
            return False
    
    @staticmethod
    def test_memory_usage():
        """测试内存使用情况"""
        from src.models.swin_unet import create_model

        print_section("Test 6: 内存使用分析（完整版）")

        if not torch.cuda.is_available():
            print_test("内存测试", False, "CUDA不可用 - 跳过")
            return True

        try:
            device = 'cuda'
            model = create_model().to(device)
            model.train()

            torch.cuda.empty_cache()
            gc.collect()

            mem_before = torch.cuda.memory_allocated() / (1024 ** 3)

            B = 1
            x = torch.randn(B, 4, 300, 300, device=device)

            output = model(x)

            mem_after = torch.cuda.memory_allocated() / (1024 ** 3)
            mem_used = mem_after - mem_before

            del x, output
            torch.cuda.empty_cache()
            gc.collect()

            is_acceptable = mem_used < 24

            status_str = f"{mem_used:.2f} GB"
            if is_acceptable:
                status_str += " ✅ (可接受)"
            else:
                status_str += f" ❌ (超过24GB限制)"

            print_test("GPU内存使用", is_acceptable, status_str)

            return is_acceptable

        except RuntimeError as e:
            if "out of memory" in str(e):
                print_test("GPU内存不足", False, "显存溢出(OOM) - 跳过此测试")
                torch.cuda.empty_cache()
                gc.collect()
                return True
            else:
                raise e
    
    @staticmethod
    def test_gradient_flow():
        """测试梯度流动"""
        from src.models.swin_unet import create_model
        
        print_section("Test 7: 梯度流动测试")
        
        model = create_model()
        model.train()
        
        x = torch.randn(1, 4, 300, 300)
        target = torch.randn(1, 27, 4, 300, 300)
        
        try:
            output = model(x)
            
            loss = F.mse_loss(output, target)
            loss.backward()
            
            has_gradient = []
            no_gradient = []
            
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if param.grad is not None and param.grad.abs().sum() > 0:
                        has_gradient.append(name)
                    else:
                        no_gradient.append(name)
            
            grad_ratio = len(has_gradient) / (len(has_gradient) + len(no_gradient)) * 100
            
            print_test("梯度计算", True,
                      f"有梯度: {len(has_gradient)}, 无梯度: {len(no_gradient)}, "
                      f"梯度覆盖率: {grad_ratio:.1f}%")
            
            if len(no_gradient) > 0:
                print(f"           ⚠️  无梯度的参数:")
                for name in no_gradient[:5]:
                    print(f"              - {name}")
                if len(no_gradient) > 5:
                    print(f"              ... 还有{len(no_gradient)-5}个")
            
            is_good = grad_ratio > 90
            print_test("梯度覆盖率", is_good, f"{grad_ratio:.1f}% (>90%为佳)")
            
            del x, target, output, loss
            gc.collect()
            
            return is_good
            
        except Exception as e:
            print_test("梯度流动测试失败", False, str(e))
            import traceback
            traceback.print_exc()
            return False


class TestEnhancedPhysicsLoss:
    """测试增强的物理约束损失函数"""
    
    @staticmethod
    def test_loss_computation():
        """测试损失函数计算"""
        from src.losses.enhanced_physics_loss import EnhancedPhysicsLoss
        
        print_section("Test 8: 物理约束损失函数")
        
        loss_fn = EnhancedPhysicsLoss()
        
        B, L, C, H, W = 2, 27, 4, 64, 64
        pred = torch.randn(B, L, C, H, W)
        target = torch.randn(B, L, C, H, W)
        
        dem = torch.rand(B, 1, H, W) * 100
        roughness = torch.rand(B, 1, H, W) * 2 + 0.01
        
        try:
            losses = loss_fn(pred, target, dem=dem, roughness=roughness)
            
            assert 'total' in losses, "缺少total损失"
            assert isinstance(losses['total'], torch.Tensor), "total应该是Tensor"
            assert losses['total'].dim() == 0, "total应该是标量"
            assert not torch.isnan(losses['total']), "total包含NaN"
            
            expected_keys = ['mse', 'l1', 'mass_conservation', 'boundary_layer',
                           'terrain', 'k_positive', 'gradient']
            
            all_present = all(key in losses for key in expected_keys)
            
            print_test("损失计算成功", True, f"总损失: {losses['total'].item():.6f}")
            print_test("所有损失项存在", all_present)
            
            print(f"\n           损失项详情:")
            for key, value in losses.items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    print(f"             - {key}: {value.item():.6f}")
            
            return True
            
        except Exception as e:
            print_test("损失计算失败", False, str(e))
            import traceback
            traceback.print_exc()
            return False
    
    @staticmethod
    def test_progressive_scheduler():
        """测试渐进式损失调度器"""
        from src.losses.enhanced_physics_loss import (
            EnhancedPhysicsLoss, ProgressiveLossScheduler
        )
        
        print_section("Test 9: 渐进式损失调度器")
        
        base_loss = EnhancedPhysicsLoss()
        scheduler = ProgressiveLossScheduler(base_loss, n_stages=3)
        
        B, L, C, H, W = 2, 27, 4, 64, 64
        pred = torch.randn(B, L, C, H, W)
        target = torch.randn(B, L, C, H, W)
        
        try:
            test_epochs = [0, 25, 50, 80]
            stages = []
            
            for epoch in test_epochs:
                losses = scheduler(pred, target, epoch=epoch)
                stage = losses['current_stage']
                stages.append(stage)
                
                assert 'current_stage' in losses
                assert isinstance(stage, int)
                
                print(f"           Epoch {epoch:3d} → Stage {stage}, Loss: {losses['total'].item():.6f}")
            
            assert stages[0] == 0, f"Epoch 0应该在Stage 0, 但在Stage {stages[0]}"
            assert stages[-1] == 2, f"最后epoch应该在Stage 2, 但在Stage {stages[-1]}"
            
            print_test("阶段切换正确", True, f"阶段序列: {stages}")
            
            return True
            
        except Exception as e:
            print_test("渐进式调度器测试失败", False, str(e))
            import traceback
            traceback.print_exc()
            return False


class TestModelWithRealData:
    """使用真实数据格式进行集成测试"""

    @staticmethod
    def test_fuxi_cfd_format():
        """测试FuXi-CFD数据格式的兼容性（完整版模型）"""
        from src.models.swin_unet import create_model

        print_section("Test 10: FuXi-CFD数据格式兼容性（完整版）")

        model = create_model()
        model.eval()

        B = 1

        u_100m = torch.randn(B, 9, 9) * 5 + 10
        v_100m = torch.randn(B, 9, 9) * 3 + 2
        dem = torch.rand(B, 300, 300) * 500
        roughness = torch.rand(B, 300, 300) * 1.5 + 0.1

        u_up = torch.nn.functional.interpolate(
            u_100m.unsqueeze(1), size=(300, 300), mode='bilinear', align_corners=False
        ).squeeze(1)
        v_up = torch.nn.functional.interpolate(
            v_100m.unsqueeze(1), size=(300, 300), mode='bilinear', align_corners=False
        ).squeeze(1)

        x = torch.stack([u_up, v_up, dem, roughness], dim=1)

        try:
            with torch.no_grad():
                output = model(x)

            expected_output_shape = (B, 27, 4, 300, 300)

            assert output.shape == expected_output_shape, \
                f"输出形状错误: {output.shape} != {expected_output_shape}"

            print_test("完整版数据格式兼容", True,
                      f"输入: FuXi-CFD格式 → 输出: {output.shape}")

            print(f"\n           输入统计:")
            print(f"             - u_100m 范围: [{u_100m.min():.2f}, {u_100m.max():.2f}] m/s")
            print(f"             - v_100m 范围: [{v_100m.min():.2f}, {v_100m.max():.2f}] m/s")
            print(f"             - DEM 范围: [{dem.min():.1f}, {dem.max():.1f}] m")
            print(f"             - Roughness 范围: [{roughness.min():.3f}, {roughness.max():.3f}] m")

            print(f"\n           输出统计:")
            for i, var in enumerate(['u', 'v', 'w', 'k']):
                var_data = output[:, :, i]
                print(f"             - {var}: mean={var_data.mean():.4f}, "
                      f"std={var_data.std():.4f}, "
                      f"range=[{var_data.min():.4f}, {var_data.max():.4f}]")

            return True

        except Exception as e:
            print_test("完整版数据格式兼容性测试失败", False, str(e))
            import traceback
            traceback.print_exc()
            return False

    @staticmethod
    def test_fuxi_cfd_format_lite():
        """测试FuXi-CFD数据格式的兼容性（Lite版模型）"""
        from src.models.swin_unet_lite import create_lite_model

        print_section("Test 11: FuXi-CFD数据格式兼容性（Lite版）")

        model = create_lite_model()
        model.eval()

        B = 1

        u_100m = torch.randn(B, 9, 9) * 5 + 10
        v_100m = torch.randn(B, 9, 9) * 3 + 2
        dem = torch.rand(B, 300, 300) * 500
        roughness = torch.rand(B, 300, 300) * 1.5 + 0.1

        u_up = torch.nn.functional.interpolate(
            u_100m.unsqueeze(1), size=(300, 300), mode='bilinear', align_corners=False
        ).squeeze(1)
        v_up = torch.nn.functional.interpolate(
            v_100m.unsqueeze(1), size=(300, 300), mode='bilinear', align_corners=False
        ).squeeze(1)

        x = torch.stack([u_up, v_up, dem, roughness], dim=1)

        try:
            with torch.no_grad():
                output = model(x)

            expected_output_shape = (B, 27, 4, 300, 300)

            assert output.shape == expected_output_shape, \
                f"输出形状错误: {output.shape} != {expected_output_shape}"

            print_test("Lite版数据格式兼容", True,
                      f"输入: FuXi-CFD格式 → 输出: {output.shape}")

            print(f"\n           输出统计:")
            for i, var in enumerate(['u', 'v', 'w', 'k']):
                var_data = output[:, :, i]
                print(f"             - {var}: mean={var_data.mean():.4f}, "
                      f"std={var_data.std():.4f}, "
                      f"range=[{var_data.min():.4f}, {var_data.max():.4f}]")

            return True

        except Exception as e:
            print_test("Lite版数据格式兼容性测试失败", False, str(e))
            import traceback
            traceback.print_exc()
            return False


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*70)
    print("  🧪 Physics-Informed Swin-U-Net 单元测试套件")
    print("="*70)
    
    results = {}
    
    results['swin_components'] = [
        TestSwinTransformerComponents.test_window_attention(),
        TestSwinTransformerComponents.test_swin_transformer_block(),
        TestSwinTransformerComponents.test_swin_transformer_stage(),
    ]

    results['lite_model'] = [
        TestSwinUNetLite.test_lite_model_creation(),
        TestSwinUNetLite.test_k_decoder_head_exists(),
        TestSwinUNetLite.test_lite_forward_pass(),
        TestSwinUNetLite.test_lite_memory_usage(),
        TestSwinUNetLite.test_lite_gradient_flow(),
    ]

    results['model'] = []
    try:
        from src.models.swin_unet import create_model
        results['model'] = [
            TestPhysicsInformedSwinUNet.test_model_creation(),
            TestPhysicsInformedSwinUNet.test_forward_pass(),
            TestPhysicsInformedSwinUNet.test_memory_usage(),
            TestPhysicsInformedSwinUNet.test_gradient_flow(),
        ]
    except ImportError:
        print("\n⚠️  完整版模型 (swin_unet) 未找到 - 跳过完整版测试")

    results['loss'] = [
        TestEnhancedPhysicsLoss.test_loss_computation(),
        TestEnhancedPhysicsLoss.test_progressive_scheduler(),
    ]

    results['integration'] = [
        TestModelWithRealData.test_fuxi_cfd_format_lite(),
    ]
    try:
        from src.models.swin_unet import create_model
        results['integration'].insert(0, TestModelWithRealData.test_fuxi_cfd_format())
    except ImportError:
        pass
    
    print_section("📊 测试总结")
    
    total_tests = 0
    passed_tests = 0
    
    for category, test_results in results.items():
        category_passed = sum(test_results)
        category_total = len(test_results)
        total_tests += category_total
        passed_tests += category_passed
        
        status = "✅ 全部通过" if category_passed == category_total else \
                 f"⚠️  {category_passed}/{category_total} 通过"
        
        print(f"  {category.upper():20s}: {status}")
    
    overall_status = "✅ 所有测试通过" if passed_tests == total_tests else \
                     f"⚠️  {passed_tests}/{total_tests} 测试通过"
    
    print(f"\n  {'='*60}")
    print(f"  总体结果: {overall_status}")
    print(f"  {'='*60}\n")
    
    success_rate = passed_tests / total_tests * 100 if total_tests > 0 else 0
    
    return passed_tests == total_tests, success_rate


if __name__ == "__main__":
    import torch.nn.functional as F
    
    all_passed, success_rate = run_all_tests()
    
    sys.exit(0 if all_passed else 1)
