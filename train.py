"""
主训练脚本
"""

import os
import sys
import argparse
import yaml
import torch
import torch.distributed as dist
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models.complex_mtass import ComplexMTASS, count_parameters
from data.online_dataset import get_dataloader, OnlineMixtureDataset, STFT
from trainer.trainer import Trainer


def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    return rank, world_size, local_rank


def main():
    parser = argparse.ArgumentParser(description='Complex-MTASS Training')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--local_rank', type=int, default=0,
                       help='本地GPU编号')
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='批次大小')
    args = parser.parse_args()
    
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    
    rank, world_size, local_rank = setup_distributed()
    
    seed = config.get('training', {}).get('seed', 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)
    
    is_main = (rank == 0)
    
    if is_main:
        print("=" * 50)
        print("Complex-MTASS 训练")
        print("=" * 50)
        print(f"配置: {args.config}")
        print(f"分布式: {world_size > 1}")
        print(f"设备数: {world_size}")
        print()
    
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    model = ComplexMTASS(config['model'])
    
    if is_main:
        params = count_parameters(model)
        print(f"模型参数量:")
        print(f"  总参数: {params['total']:,}")
        print(f"  可训练: {params['trainable']:,}")
        print()
    
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[local_rank],
            find_unused_parameters=config.get('distributed', {}).get('find_unused_parameters', False)
        )
    
    model = model.to(device)
    
    data_cfg = config.get('data', {})
    
    train_loader = None
    val_loader = None
    
    try:
        train_loader = get_dataloader(
            config, 
            split='train',
            batch_size=config['training'].get('batch_size', 8),
            num_workers=data_cfg.get('num_workers', 4)
        )
        
        if is_main:
            print("数据加载器创建成功!")
            print(f"训练数据: 在线混合生成")
            print()
    except Exception as e:
        if is_main:
            print(f"警告: 无法创建数据加载器: {e}")
            print("将使用模拟数据进行测试...")
        
        class MockDataset:
            def __iter__(self):
                while True:
                    yield {
                        'mixture': torch.randn(2, 514, 100),
                        'sources': torch.randn(2, 3, 514, 100),
                        'source_types': [['speech', 'noise'], ['speech', 'music']]
                    }
        
        class MockLoader:
            def __init__(self):
                self.dataset = MockDataset()
            
            def __iter__(self):
                return iter(self.dataset)
            
            def __len__(self):
                return 100
        
        train_loader = MockLoader()
    
    trainer = Trainer(config)
    
    if world_size > 1:
        trainer.setup_distributed(local_rank)
    
    trainer.build_model(model)
    trainer.build_optimizer()
    trainer.build_scheduler()
    trainer.build_dataloaders(train_loader, val_loader)
    
    if args.resume:
        if is_main:
            print(f"恢复检查点: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    if is_main:
        print("开始训练...")
        print()
    
    trainer.train()
    
    if is_main:
        print()
        print("训练完成!")
        print(f"最佳损失: {trainer.best_loss:.4f}")
        print(f"检查点保存位置: {trainer.checkpoint_dir}")


if __name__ == "__main__":
    main()
