"""
FuXi-CFD 数据集加载器

适配数据集格式：
- 12,532个案例
- 输入: inputs.npz (dem, roughness, u_100m, v_100m)
- 输出: outputs.npz (u, v, w, k) - float16需要转换
"""

import os
import glob
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple
import random
import time


class FuXiCFDDataset(Dataset):
    """
    FuXi-CFD 数据集
    
    数据格式:
    - 每个案例是一个目录: case_XXXXXX/
    - inputs.npz: {dem(300,300), roughness(300,300), u_100m(9,9), v_100m(9,9)}
    - outputs.npz: {u(27,300,300), v(27,300,300), w(27,300,300), k(27,300,300)}
    """
    
    STATS_FILE = 'normalization_stats.npz'
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        normalize: bool = True,
        transform=None,
        seed: int = 42,
        prefetch_to_memory: bool = False,
    ):
        self.data_dir = data_dir
        self.split = split
        self.normalize = normalize
        self.transform = transform
        self.prefetch_to_memory = prefetch_to_memory
        
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            f"比例总和必须为1.0 (当前: {train_ratio+val_ratio+test_ratio})"
        
        all_cases = sorted([
            d for d in os.listdir(data_dir)
            if d.startswith('case_') and 
            os.path.isdir(os.path.join(data_dir, d)) and
            os.path.exists(os.path.join(data_dir, d, 'inputs.npz')) and
            os.path.exists(os.path.join(data_dir, d, 'outputs.npz'))
        ])
        
        total_cases = len(all_cases)
        
        random.seed(seed)
        random.shuffle(all_cases)
        
        n_train = int(total_cases * train_ratio)
        n_val = int(total_cases * val_ratio)
        
        if split == 'train':
            self.cases = all_cases[:n_train]
        elif split == 'val':
            self.cases = all_cases[n_train:n_train+n_val]
        elif split == 'test':
            self.cases = all_cases[n_train+n_val:]
        else:
            raise ValueError(f"未知的split: {split}")
        
        if normalize:
            self.stats = self._compute_or_load_statistics()
        else:
            self.stats = None
        
        print(f'✅ {split}集: {len(self.cases)} 样本 (总计{total_cases})')
        
        if self.prefetch_to_memory:
            self._prefetch_to_memory()
        
        if getattr(self, 'verbose_init', False):
            self.statistics = self.compute_dataset_statistics(self.data_dir, verbose=True)
        else:
            self.statistics = None
    
    def _compute_or_load_statistics(self) -> dict:
        """计算或加载归一化统计信息"""
        stats_path = os.path.join(self.data_dir, '.data_cache', self.STATS_FILE)
        
        if os.path.exists(stats_path):
            print(f'   📊 加载已有归一化统计: {stats_path}')
            np_stats = np.load(stats_path)
            stats = {
                'input_mean': np_stats['input_mean'].astype(np.float32),
                'input_std': np_stats['input_std'].astype(np.float32),
                'output_mean': np_stats['output_mean'].astype(np.float32),
                'output_std': np_stats['output_std'].astype(np.float32),
            }
            print(f'      input_mean:  {stats["input_mean"]}')
            print(f'      input_std:   {stats["input_std"]}')
            print(f'      output_mean: {stats["output_mean"]}')
            print(f'      output_std:  {stats["output_std"]}')
            return stats
        
        print(f'   📊 计算归一化统计信息 (采样500个案例)...')
        n_sample = min(500, len(self.cases))
        sample_indices = random.sample(range(len(self.cases)), n_sample)
        
        input_sum = np.zeros(6, dtype=np.float64)
        input_sq_sum = np.zeros(6, dtype=np.float64)
        output_sum = np.zeros(4, dtype=np.float64)
        output_sq_sum = np.zeros(4, dtype=np.float64)
        input_count = 0
        output_count = 0

        for i, idx in enumerate(sample_indices):
            if i % 100 == 0:
                print(f'      统计进度: {i}/{n_sample}')

            case_name = self.cases[idx]
            case_dir = os.path.join(self.data_dir, case_name)

            try:
                inputs = np.load(os.path.join(case_dir, 'inputs.npz'))
                outputs = np.load(os.path.join(case_dir, 'outputs.npz'))

                dem = inputs['dem'].astype(np.float64)
                roughness = inputs['roughness'].astype(np.float64)
                u_100m = inputs['u_100m'].astype(np.float64)
                v_100m = inputs['v_100m'].astype(np.float64)

                u_out = outputs['u'].astype(np.float64)
                v_out = outputs['v'].astype(np.float64)
                w_out = outputs['w'].astype(np.float64)
                k_out = outputs['k'].astype(np.float64)

                u_100m_up = np.array(torch.nn.functional.interpolate(
                    torch.from_numpy(u_100m[np.newaxis, np.newaxis, :, :]),
                    size=(300, 300), mode='bilinear', align_corners=False
                ).squeeze().numpy(), dtype=np.float64)
                v_100m_up = np.array(torch.nn.functional.interpolate(
                    torch.from_numpy(v_100m[np.newaxis, np.newaxis, :, :]),
                    size=(300, 300), mode='bilinear', align_corners=False
                ).squeeze().numpy(), dtype=np.float64)

                # Compute DEM gradients (Sobel via torch)
                dem_tensor = torch.from_numpy(dem).unsqueeze(0).unsqueeze(0)
                sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float64).view(1, 1, 3, 3)
                sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float64).view(1, 1, 3, 3)
                dz_dx = torch.nn.functional.conv2d(dem_tensor, sobel_x, padding=1).squeeze().numpy()
                dz_dy = torch.nn.functional.conv2d(dem_tensor, sobel_y, padding=1).squeeze().numpy()

                input_data = np.stack([u_100m_up, v_100m_up, dem, roughness, dz_dx, dz_dy], axis=0)
                output_data = np.stack([u_out, v_out, w_out, k_out], axis=0)
                
                n_in = dem.size
                n_out = u_out.size
                
                for ch in range(6):
                    input_sum[ch] += input_data[ch].sum()
                    input_sq_sum[ch] += (input_data[ch] ** 2).sum()
                for ch in range(4):
                    output_sum[ch] += output_data[ch].sum()
                    output_sq_sum[ch] += (output_data[ch] ** 2).sum()
                
                input_count += n_in
                output_count += n_out
                
                del inputs, outputs
            except Exception as e:
                print(f'      ⚠️ 统计跳过 [{case_name}]: {e}')
                continue
        
        input_mean = (input_sum / input_count).astype(np.float32)
        input_std = (np.sqrt(input_sq_sum / input_count - (input_sum / input_count) ** 2)).astype(np.float32)
        output_mean = (output_sum / output_count).astype(np.float32)
        output_std = (np.sqrt(output_sq_sum / output_count - (output_sum / output_count) ** 2)).astype(np.float32)
        
        input_std = np.maximum(input_std, 1e-6)
        output_std = np.maximum(output_std, 1e-6)
        
        stats = {
            'input_mean': input_mean,
            'input_std': input_std,
            'output_mean': output_mean,
            'output_std': output_std,
        }
        
        print(f'      input_mean:  {stats["input_mean"]}')
        print(f'      input_std:   {stats["input_std"]}')
        print(f'      output_mean: {stats["output_mean"]}')
        print(f'      output_std:  {stats["output_std"]}')
        
        cache_dir = os.path.join(self.data_dir, '.data_cache')
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(stats_path,
                 input_mean=input_mean, input_std=input_std,
                 output_mean=output_mean, output_std=output_std)
        print(f'   ✅ 归一化统计已保存: {stats_path}')
        
        return stats
    
    def normalize_input(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """归一化输入: (x - mean) / std"""
        if not self.normalize or self.stats is None:
            return input_tensor
        mean = torch.from_numpy(self.stats['input_mean']).view(6, 1, 1)
        std = torch.from_numpy(self.stats['input_std']).view(6, 1, 1)
        return (input_tensor - mean) / std
    
    def normalize_output(self, output_tensor: torch.Tensor) -> torch.Tensor:
        """归一化输出: (x - mean) / std, output shape: (27, 4, 300, 300)"""
        if not self.normalize or self.stats is None:
            return output_tensor
        mean = torch.from_numpy(self.stats['output_mean']).view(1, 4, 1, 1)
        std = torch.from_numpy(self.stats['output_std']).view(1, 4, 1, 1)
        return (output_tensor - mean) / std
    
    def denormalize_output(self, output_tensor: torch.Tensor) -> torch.Tensor:
        """反归一化输出: x * std + mean"""
        if not self.normalize or self.stats is None:
            return output_tensor
        mean = torch.from_numpy(self.stats['output_mean']).view(1, 4, 1, 1)
        std = torch.from_numpy(self.stats['output_std']).view(1, 4, 1, 1)
        return output_tensor * std + mean
    
    def _get_cache_path(self) -> str:
        """生成缓存文件路径"""
        case_hash = hashlib.md5(
            '|'.join(self.cases).encode()
        ).hexdigest()[:12]
        norm_suffix = '_norm' if self.normalize else '_raw'
        cache_dir = os.path.join(self.data_dir, '.data_cache')
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f'prefetch_{self.split}_{case_hash}{norm_suffix}.pt')

    def _prefetch_to_memory(self):
        """预加载数据集到内存（支持磁盘缓存）"""
        cache_path = self._get_cache_path()

        if os.path.exists(cache_path):
            print(f'\n💾 发现磁盘缓存: {os.path.basename(cache_path)}')
            start_time = time.time()
            cache_data = torch.load(cache_path, map_location='cpu', weights_only=False)
            self.cached_data = cache_data['data']
            load_time = time.time() - start_time
            print(f'   ✅ 从缓存加载完成！耗时: {load_time:.2f}s ({len(self.cached_data)} 样本)')
            return

        print(f'\n💾 预加载数据集到内存并完成预处理...')
        print(f'   数据集大小: {len(self.cases)} 样本')
        
        start_time = time.time()
        
        self.cached_data = []
        for i, case_name in enumerate(self.cases):
            if i % 100 == 0:
                print(f'   进度: {i}/{len(self.cases)} ({i/len(self.cases)*100:.1f}%)')
            
            case_dir = os.path.join(self.data_dir, case_name)
            inputs_path = os.path.join(case_dir, 'inputs.npz')
            outputs_path = os.path.join(case_dir, 'outputs.npz')
            
            inputs = np.load(inputs_path)
            outputs = np.load(outputs_path)
            
            dem = inputs['dem'].astype(np.float32)
            roughness = inputs['roughness'].astype(np.float32)
            u_100m = inputs['u_100m'].astype(np.float32)
            v_100m = inputs['v_100m'].astype(np.float32)
            
            u_out = outputs['u'].astype(np.float32)
            v_out = outputs['v'].astype(np.float32)
            w_out = outputs['w'].astype(np.float32)
            k_out = outputs['k'].astype(np.float32)
            
            # Compute DEM gradients
            dem_tensor = torch.from_numpy(dem).unsqueeze(0).unsqueeze(0)
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
            dz_dx = torch.nn.functional.conv2d(dem_tensor, sobel_x, padding=1).squeeze()
            dz_dy = torch.nn.functional.conv2d(dem_tensor, sobel_y, padding=1).squeeze()

            input_tensor = torch.zeros((6, 300, 300), dtype=torch.float32)
            input_tensor[0] = torch.nn.functional.interpolate(
                torch.from_numpy(u_100m).unsqueeze(0).unsqueeze(0),
                size=(300, 300), mode='bilinear', align_corners=False
            ).squeeze()
            input_tensor[1] = torch.nn.functional.interpolate(
                torch.from_numpy(v_100m).unsqueeze(0).unsqueeze(0),
                size=(300, 300), mode='bilinear', align_corners=False
            ).squeeze()
            input_tensor[2] = torch.from_numpy(dem)
            input_tensor[3] = torch.from_numpy(roughness)
            input_tensor[4] = dz_dx
            input_tensor[5] = dz_dy

            output_tensor = torch.stack([
                torch.from_numpy(u_out),
                torch.from_numpy(v_out),
                torch.from_numpy(w_out),
                torch.from_numpy(k_out),
            ], dim=0).permute(1, 0, 2, 3)
            
            if self.normalize and self.stats is not None:
                input_tensor = self.normalize_input(input_tensor)
                output_tensor = self.normalize_output(output_tensor)
            
            self.cached_data.append({
                'input': input_tensor,
                'target': output_tensor,
                'case_id': case_name,
            })
            
            del inputs, outputs

        save_start = time.time()
        torch.save({'data': self.cached_data, 'cases': self.cases}, cache_path)
        save_time = time.time() - save_start
        cache_size_mb = os.path.getsize(cache_path) / 1024 / 1024
        
        load_time = time.time() - start_time
        print(f'   ✅ 预加载+预处理完成！耗时: {load_time:.2f}s')
        print(f'   💿 缓存已保存: {cache_path} ({cache_size_mb:.1f} MB, 写入耗时 {save_time:.2f}s)')
    
    @staticmethod
    def compute_dataset_statistics(data_dir: str, n_samples: int = 100, verbose: bool = True) -> dict:
        """计算数据集统计信息"""
        cases = sorted([d for d in os.listdir(data_dir) if d.startswith('case_')])[:n_samples]
        
        if len(cases) == 0:
            return {}
        
        all_u, all_v, all_w, all_k = [], [], [], []
        
        for c in cases:
            try:
                out = np.load(os.path.join(data_dir, c, 'outputs.npz'))
                all_u.append(out['u'].flatten())
                all_v.append(out['v'].flatten())
                all_w.append(out['w'].flatten())
                all_k.append(out['k'].flatten())
            except:
                continue
        
        stats = {}
        for name, values in [('u', all_u), ('v', all_v), ('w', all_w), ('k', all_k)]:
            arr = np.concatenate(values)
            stats[name] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
            }
        
        if verbose:
            print("\n" + "="*70)
            print("📊 FuXi-CFD Dataset Statistics")
            print("="*70)
            for name in ['u', 'v', 'w', 'k']:
                s = stats[name]
                print(f"  {name}: mean={s['mean']:.4f}, std={s['std']:.4f}, "
                      f"min={s['min']:.4f}, max={s['max']:.4f}")
            print("="*70 + "\n")
        
        return stats
    
    @staticmethod
    def apply_k_log_transform(k_values: np.ndarray, epsilon: float = 0.01) -> np.ndarray:
        return np.log(np.clip(k_values, a_min=1e-6, a_max=None) + epsilon)
    
    def __len__(self):
        return len(self.cases)
    
    def _get_fallback_item(self, idx, retries=0):
        if retries >= 5:
            raise RuntimeError(f'数据加载连续5次失败，起始idx={idx}')
        next_idx = (idx + 1) % len(self)
        return self.__getitem__(next_idx, retries=retries + 1)

    def __getitem__(self, idx, retries=0):
        if hasattr(self, 'cached_data') and self.cached_data:
            cached = self.cached_data[idx]
            return {
                'input': cached['input'],
                'target': cached['target'],
                'case_id': cached['case_id'],
            }
        else:
            case_name = self.cases[idx]
            case_dir = os.path.join(self.data_dir, case_name)

            try:
                inputs = np.load(os.path.join(case_dir, 'inputs.npz'))
                outputs = np.load(os.path.join(case_dir, 'outputs.npz'))
            except Exception as e:
                print(f'⚠️ 加载失败 [{case_name}]: {e}')
                return self._get_fallback_item(idx, retries)

            try:
                dem = inputs['dem'].astype(np.float32)
                roughness = inputs['roughness'].astype(np.float32)
                u_100m = inputs['u_100m'].astype(np.float32)
                v_100m = inputs['v_100m'].astype(np.float32)

                u_out = outputs['u'].astype(np.float32)
                v_out = outputs['v'].astype(np.float32)
                w_out = outputs['w'].astype(np.float32)
                k_out = outputs['k'].astype(np.float32)

                # 计算 DEM 梯度 (Sobel 算子)
                dem_tensor = torch.from_numpy(dem).unsqueeze(0).unsqueeze(0)
                sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
                sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
                dz_dx = torch.nn.functional.conv2d(dem_tensor, sobel_x, padding=1).squeeze()
                dz_dy = torch.nn.functional.conv2d(dem_tensor, sobel_y, padding=1).squeeze()

                input_tensor = torch.zeros((6, 300, 300), dtype=torch.float32)
                input_tensor[0] = torch.nn.functional.interpolate(
                    torch.from_numpy(u_100m).unsqueeze(0).unsqueeze(0),
                    size=(300, 300), mode='bilinear', align_corners=False
                ).squeeze()
                input_tensor[1] = torch.nn.functional.interpolate(
                    torch.from_numpy(v_100m).unsqueeze(0).unsqueeze(0),
                    size=(300, 300), mode='bilinear', align_corners=False
                ).squeeze()
                input_tensor[2] = torch.from_numpy(dem)
                input_tensor[3] = torch.from_numpy(roughness)
                input_tensor[4] = dz_dx
                input_tensor[5] = dz_dy

                output_tensor = torch.stack([
                    torch.from_numpy(u_out),
                    torch.from_numpy(v_out),
                    torch.from_numpy(w_out),
                    torch.from_numpy(k_out),
                ], dim=0).permute(1, 0, 2, 3)

                if self.normalize and self.stats is not None:
                    input_tensor = self.normalize_input(input_tensor)
                    output_tensor = self.normalize_output(output_tensor)

                if self.transform:
                    input_tensor, output_tensor = self.transform(input_tensor, output_tensor)

                return {
                    'input': input_tensor,
                    'target': output_tensor,
                    'case_id': case_name,
                }

            except Exception as e:
                print(f'⚠️ 预处理失败 [{case_name}]: {e}')
                return self._get_fallback_item(idx, retries)


def create_dataloaders(
    data_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    pin_memory: bool = True,
    prefetch_to_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建训练/验证/测试数据加载器"""
    train_dataset = FuXiCFDDataset(
        data_dir=data_dir,
        split='train',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        normalize=True,
        prefetch_to_memory=prefetch_to_memory,
    )
    
    val_dataset = FuXiCFDDataset(
        data_dir=data_dir,
        split='val',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        normalize=True,
        prefetch_to_memory=False,
    )
    
    test_dataset = FuXiCFDDataset(
        data_dir=data_dir,
        split='test',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        normalize=True,
        prefetch_to_memory=False,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        prefetch_factor=2,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    import time
    
    data_dir = '/mnt/sdata/jz/fuxi_cfd/dataset'
    
    print('='*70)
    print('🧪 测试 FuXi-CFD 数据集加载器')
    print('='*70)
    
    start_time = time.time()
    
    train_ds = FuXiCFDDataset(data_dir, split='train', normalize=True)
    val_ds = FuXiCFDDataset(data_dir, split='val', normalize=True)
    
    load_time = time.time() - start_time
    print(f'\n⏱️ 数据集初始化耗时: {load_time:.2f}s')
    
    print('\n📦 测试单个样本加载:')
    sample = train_ds[0]
    
    print(f'   Input shape: {sample["input"].shape}')
    print(f'   Target shape: {sample["target"].shape}')
    print(f'   Case ID: {sample["case_id"]}')
    
    for i, name in enumerate(['u_100m', 'v_100m', 'dem', 'roughness', 'dem_dx', 'dem_dy']):
        data = sample['input'][i]
        print(f'   {name}: mean={data.mean():.4f}, std={data.std():.4f}, range=[{data.min():.4f}, {data.max():.4f}]')
    
    for i, name in enumerate(['u', 'v', 'w', 'k']):
        data = sample['target'][:, i, :, :].mean()
        print(f'   {name} (27层平均): mean={data:.4f}')
