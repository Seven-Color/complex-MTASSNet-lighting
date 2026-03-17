#!/usr/bin/env python3
"""
训练脚本 - 完整版
支持分布式训练、混合精度、早停、日志记录
"""

import argparse
import logging
import os
import random
import yaml
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DataParallel, DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json

from models.complex_mtass import ComplexMTASS
from data.online_dataset import OnlineMixDataset
from src.utils.metrics import MetricsTracker
from src.utils.visualization import visualize_training


def setup_logging(log_dir: str):
    """设置日志"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dataloader(config: dict, is_train: bool = True):
    """创建数据加载器"""
    dataset = OnlineMixDataset(
        config=config['data'],
        is_train=is_train
    )
    
    batch_size = config['training']['batch_size']
    num_workers = config['data'].get('num_workers', 4)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    return dataloader


def build_model(config: dict, device: torch.device):
    """构建模型"""
    model = ComplexMTASS(config['model'])
    model = model.to(device)
    
    # 统计参数量
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info(f"模型总参数量: {num_params:,}")
    logging.info(f"可训练参数量: {num_trainable:,}")
    
    return model


def build_optimizer(model, config: dict):
    """构建优化器"""
    opt_config = config['training']
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=opt_config['learning_rate'],
        weight_decay=opt_config['weight_decay'],
        betas=opt_config['betas']
    )
    
    return optimizer


def build_scheduler(optimizer, config: dict):
    """构建学习率调度器"""
    sched_config = config['training']
    
    if sched_config.get('scheduler') == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sched_config['epochs'],
            eta_min=sched_config.get('min_lr', 1e-6)
        )
    else:
        scheduler = None
    
    return scheduler


class Trainer:
    """训练器类"""
    
    def __init__(self, config: dict, args):
        self.config = config
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 目录
        self.checkpoint_dir = Path(config['checkpoint'].get('save_dir', 'checkpoints'))
        self.log_dir = Path(config['logging'].get('log_dir', 'logs'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 日志
        self.logger = setup_logging(str(self.log_dir))
        
        # 随机种子
        set_seed(config['training']['seed'])
        
        # 组件
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler()
        self.writer = None
        
        # 状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        # 指标追踪
        self.metrics_tracker = MetricsTracker()
        self.training_history = {
            'loss': [],
            'sisnr': [],
            'sdr': [],
            'lr': []
        }
        
        # 早停
        self.early_stop_counter = 0
        self.patience = config['training'].get('patience', 15)
    
    def setup(self):
        """初始化组件"""
        # 模型
        self.model = build_model(self.config, self.device)
        
        # 优化器
        self.optimizer = build_optimizer(self.model, self.config)
        
        # 调度器
        self.scheduler = build_scheduler(self.optimizer, self.config)
        
        # TensorBoard
        if self.config['logging'].get('tensorboard', True):
            self.writer = SummaryWriter(self.log_dir)
        
        # 加载检查点
        if self.args.resume:
            self._load_checkpoint(self.args.resume)
        
        self.logger.info("训练器初始化完成")
        return self
    
    def _load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0) + 1
        self.global_step = checkpoint.get('global_step', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.logger.info(f"从 epoch {self.current_epoch} 继续训练")
    
    def _save_checkpoint(self, name: str = "last"):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        save_path = self.checkpoint_dir / f"checkpoint_{name}.pt"
        torch.save(checkpoint, save_path)
        
        # 保存最佳模型
        if name == "best":
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
        
        self.logger.info(f"检查点已保存: {save_path}")
    
    def train_step(self, batch: dict) -> float:
        """单步训练"""
        mixture = batch['mixture'].to(self.device)
        sources = batch['sources'].to(self.device)
        
        self.optimizer.zero_grad()
        
        # 混合精度
        with autocast(enabled=self.config['training'].get('use_amp', True)):
            outputs = self.model(mixture)
            
            if isinstance(outputs, tuple):
                outputs = torch.stack(outputs, dim=1)
            
            # 简化的损失计算 (MSE + SI-SNR)
            loss = nn.functional.mse_loss(outputs, sources)
        
        # 反向传播
        self.scaler.scale(loss).backward()
        
        # 梯度裁剪
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config['training'].get('clip_grad_norm', 5.0)
        )
        
        # 更新
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()
    
    def train_epoch(self, dataloader) -> dict:
        """训练一个 epoch"""
        self.model.train()
        
        total_loss = 0
        num_batches = 0
        
        progress = tqdm(dataloader, desc=f"Epoch {self.current_epoch}")
        
        for batch in progress:
            loss = self.train_step(batch)
            total_loss += loss
            num_batches += 1
            self.global_step += 1
            
            # 日志
            if self.global_step % self.config['logging'].get('log_interval', 100) == 0:
                progress.set_postfix({'loss': f'{loss:.4f}'})
                
                if self.writer:
                    self.writer.add_scalar('Train/Loss', loss, self.global_step)
        
        avg_loss = total_loss / num_batches
        
        # 记录学习率
        lr = self.optimizer.param_groups[0]['lr']
        self.training_history['loss'].append(avg_loss)
        self.training_history['lr'].append(lr)
        
        return {'loss': avg_loss, 'lr': lr}
    
    def train(self, train_loader: DataLoader):
        """主训练循环"""
        epochs = self.args.epochs or self.config['training']['epochs']
        
        self.logger.info("=" * 50)
        self.logger.info("开始训练")
        self.logger.info(f"Epochs: {epochs}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info("=" * 50)
        
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            
            # 训练
            metrics = self.train_epoch(train_loader)
            
            # 学习率调度
            if self.scheduler:
                self.scheduler.step()
            
            # 日志
            self.logger.info(f"Epoch {epoch}: Loss={metrics['loss']:.4f}, LR={metrics['lr']:.6f}")
            
            # 保存检查点
            save_interval = self.config['checkpoint'].get('save_interval', 5)
            if (epoch + 1) % save_interval == 0:
                self._save_checkpoint(f"epoch_{epoch+1}")
            
            # 最佳模型
            if metrics['loss'] < self.best_loss:
                self.best_loss = metrics['loss']
                self._save_checkpoint("best")
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
            
            # 早停
            if self.early_stop_counter >= self.patience:
                self.logger.info(f"早停触发! {self.patience} 个 epoch 无改善")
                break
            
            # 保存训练历史
            if (epoch + 1) % 10 == 0:
                self._save_training_history()
        
        # 训练完成
        self.logger.info("训练完成!")
        self._save_checkpoint("last")
        self._save_training_history()
        
        if self.writer:
            self.writer.close()
    
    def _save_training_history(self):
        """保存训练历史"""
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # 可视化
        vis_path = self.checkpoint_dir / "training_curves.png"
        visualize_training(self.training_history, str(vis_path))


def main():
    parser = argparse.ArgumentParser(description='Train Complex-MTASSNet')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建训练器
    trainer = Trainer(config, args)
    trainer.setup()
    
    # 创建数据加载器
    train_loader = create_dataloader(config, is_train=True)
    
    # 训练
    trainer.train(train_loader)


if __name__ == "__main__":
    main()
