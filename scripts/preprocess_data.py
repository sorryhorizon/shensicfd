#!/usr/bin/env python3
"""
数据预处理脚本 - 生成磁盘缓存
在后台运行，不占用GPU
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.fuxi_cfd_dataset import FuXiCFDDataset

def main():
    data_dir = '/mnt/sdata/jz/fuxi_cfd/dataset'
    
    print('='*70)
    print('🔄 预处理数据集 - 生成磁盘缓存')
    print('='*70)
    
    for split in ['train', 'val', 'test']:
        print(f'\n📂 处理 {split} 集...')
        dataset = FuXiCFDDataset(
            data_dir=data_dir,
            split=split,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            normalize=True,
            prefetch_to_memory=True,
        )
        print(f'✅ {split} 集缓存完成: {len(dataset)} 样本')
    
    print('\n' + '='*70)
    print('✅ 所有数据集预处理完成！')
    print('='*70)

if __name__ == '__main__':
    main()
