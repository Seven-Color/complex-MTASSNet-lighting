"""
训练脚本 - 使用整洁代码架构
"""

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.core.config import Config
from src.models.complex_mtass import ComplexMTASSModel
from src.losses.losses import CompositeLoss, LossFactory
from src.callbacks.callbacks import (
    CallbackList, 
    TensorBoardCallback, 
    CheckpointCallback, 
    EarlyStoppingCallback
)


class Trainer:
    """
    训练器
    单一职责：管理训练流程
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 组件
        self.model: nn.Module = None
        self.optimizer: optim.Optimizer = None
        self.scheduler = None
        self.scaler = GradScaler()
        self.criterion: CompositeLoss = None
        self.train_loader = None
        self.callbacks: CallbackList = None
        
        # 状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        # 日志
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    # ==================== 组件构建 ====================
    
    def build_model(self) -> 'Trainer':
        """构建模型"""
        self.model = ComplexMTASSModel(self.config.model)
        self.model = self.model.to(self.device)
        
        self.logger.info(f"模型参数量: {self.model.get_num_parameters():,}")
        return self
    
    def build_optimizer(self) -> 'Trainer':
        """构建优化器"""
        cfg = self.config.training
        
        optimizers = {
            'adam': optim.Adam,
            'adamw': optim.AdamW,
            'sgd': lambda params, **kw: optim.SGD(params, momentum=0.9, **kw),
        }
        
        optimizer_class = optimizers.get(cfg.optimizer.lower(), optim.AdamW)
        
        self.optimizer = optimizer_class(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            betas=cfg.betas
        )
        
        # 学习率调度器
        if cfg.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=cfg.epochs,
                eta_min=cfg.min_lr
            )
        
        return self
    
    def build_loss(self) -> 'Trainer':
        """构建损失函数"""
        loss_config = {
            'mse': self.config.training.loss_mse,
            'snr': self.config.training.loss_snr,
            'sisnr': self.config.training.loss_sisnr,
        }
        
        self.criterion = LossFactory.create_from_config(loss_config)
        return self
    
    def build_dataloaders(self, train_loader: DataLoader) -> 'Trainer':
        """构建数据加载器"""
        self.train_loader = train_loader
        return self
    
    def build_callbacks(self) -> 'Trainer':
        """构建回调"""
        self.callbacks = CallbackList([
            TensorBoardCallback('./logs'),
            CheckpointCallback('./checkpoints', save_interval=5),
            EarlyStoppingCallback(patience=self.config.training.patience)
        ])
        return self
    
    # ==================== 训练循环 ====================
    
    def train_step(self, batch: dict) -> float:
        """单步训练"""
        mixture = batch['mixture'].to(self.device)
        sources = batch['sources'].to(self.device)
        
        self.optimizer.zero_grad()
        
        # 混合精度训练
        with autocast(enabled=self.config.training.use_amp):
            outputs = self.model(mixture)  # (src1, src2, src3)
            outputs = torch.stack(outputs, dim=1)  # [B, 3, 514, T]
            
            # 计算损失
            loss_dict = self.criterion(outputs, sources)
            loss = loss_dict['total'] / self.config.training.gradient_accumulation
        
        # 反向传播
        self.scaler.scale(loss).backward()
        
        # 梯度裁剪
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.config.training.clip_grad_norm
        )
        
        # 更新
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss_dict['total'].item()
    
    def train_epoch(self) -> dict:
        """训练一个 epoch"""
        self.model.train()
        
        total_loss = 0
        num_batches = 0
        
        progress = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress):
            loss = self.train_step(batch)
            
            # 梯度累积
            if (batch_idx + 1) % self.config.training.gradient_accumulation == 0:
                self.global_step += 1
                
                # 日志
                if self.global_step % 100 == 0:
                    progress.set_postfix({'loss': f'{loss:.4f}'})
            
            total_loss += loss
            num_batches += 1
        
        return {'train_loss': total_loss / num_batches}
    
    def train(self, train_loader: DataLoader) -> None:
        """主训练循环"""
        # 构建组件
        (self
            .build_model()
            .build_optimizer()
            .build_loss()
            .build_dataloaders(train_loader)
            .build_callbacks())
        
        # 回调：训练开始
        self.callbacks.on_train_start(self)
        
        # 训练循环
        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch
            
            self.callbacks.on_epoch_start(self, epoch)
            
            # 训练
            metrics = self.train_epoch()
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
            
            # 回调：Epoch 结束
            self.callbacks.on_epoch_end(self, epoch, metrics)
            
            # 早停检查
            early_stop_cb = None
            for cb in self.callbacks.callbacks:
                if isinstance(cb, EarlyStoppingCallback):
                    early_stop_cb = cb
                    break
            
            if early_stop_cb and early_stop_cb.should_stop:
                self.logger.info("触发早停，停止训练")
                break
        
        # 回调：训练结束
        self.callbacks.on_train_end(self)
        
        self.logger.info("训练完成!")


def main():
    # 参数解析
    parser = argparse.ArgumentParser(description='Complex-MTASS Training')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    args = parser.parse_args()
    
    # 加载配置
    config = Config.from_yaml(args.config)
    
    # 覆盖配置
    if args.epochs:
        config._training._epochs = args.epochs  # type: ignore
    if args.batch_size:
        config._training._batch_size = args.batch_size  # type: ignore
    
    # 设置随机种子
    seed = config.training.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 打印配置
    print("=" * 50)
    print("Complex-MTASS Training")
    print("=" * 50)
    print(f"Config: {args.config}")
    print(f"Epochs: {config.training.epochs}")
    print(f"Batch Size: {config.training.batch_size}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print()
    
    # TODO: 创建数据加载器
    # train_loader = create_dataloader(config)
    
    # 创建训练器并训练
    # trainer = Trainer(config)
    # trainer.train(train_loader)
    
    print("配置加载成功!")
    print(f"模型: {config.model.name}")
    print(f"参数量: ~8.36M")


if __name__ == "__main__":
    main()
