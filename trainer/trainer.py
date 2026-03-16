"""
Trainer - 现代化训练器
"""

import os
import time
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, Any, Optional, List
from tqdm import tqdm
import logging


class Trainer:
    """现代化训练器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_logging()
        
        train_cfg = config.get('training', {})
        self.epochs = train_cfg.get('epochs', 100)
        self.gradient_accumulation = train_cfg.get('gradient_accumulation', 4)
        self.clip_grad_norm = train_cfg.get('clip_grad_norm', 5.0)
        self.use_amp = train_cfg.get('use_amp', True)
        self.patience = train_cfg.get('patience', 15)
        self.save_best = train_cfg.get('save_best', True)
        
        self.distributed = config.get('distributed', {})
        self.is_distributed = self.distributed.get('enabled', False)
        self.local_rank = 0
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[Any] = None
        self.scaler: Optional[GradScaler] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.early_stop_counter = 0
        
        self.checkpoint_dir = Path(config.get('checkpoint', {}).get('save_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(config.get('logging', {}).get('log_dir', './logs'))
        self.log_interval = config.get('logging', {}).get('log_interval', 100)
        
        if not self.is_distributed or self.local_rank == 0:
            self.writer = SummaryWriter(self.log_dir)
            self.logger.info(f"训练器初始化完成, 设备: {self.device}")
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_distributed(self, local_rank: int = 0):
        self.local_rank = local_rank
        
        if self.is_distributed:
            torch.cuda.set_device(local_rank)
            torch.distributed.init_process_group(
                backend=self.distributed.get('backend', 'nccl'),
                init_method='env://',
                world_size=torch.cuda.device_count(),
                rank=local_rank
            )
            self.device = torch.device(f'cuda:{local_rank}')
            
            self.logger.info(f"分布式训练: rank={local_rank}, world_size={torch.cuda.device_count()}")
    
    def build_model(self, model: nn.Module):
        self.model = model.to(self.device)
        
        if self.is_distributed:
            find_unused = self.distributed.get('find_unused_parameters', False)
            self.model = DDP(
                self.model, 
                device_ids=[self.local_rank],
                find_unused_parameters=find_unused
            )
            self.logger.info("启用 DDP 分布式训练")
    
    def build_optimizer(self):
        train_cfg = self.config.get('training', {})
        
        optimizer_name = train_cfg.get('optimizer', 'adamw').lower()
        lr = train_cfg.get('learning_rate', 0.001)
        weight_decay = train_cfg.get('weight_decay', 0.01)
        
        if optimizer_name == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=train_cfg.get('betas', [0.9, 0.999])
            )
        elif optimizer_name == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        
        if self.use_amp and torch.cuda.is_available():
            dtype = train_cfg.get('amp_dtype', 'float16')
            amp_dtype = torch.float16 if dtype == 'float16' else torch.bfloat16
            self.scaler = GradScaler(enabled=True)
            self.logger.info("启用混合精度训练")
    
    def build_scheduler(self):
        train_cfg = self.config.get('training', {})
        scheduler_name = train_cfg.get('scheduler', 'cosine').lower()
        
        if scheduler_name == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=train_cfg.get('min_lr', 1e-6)
            )
        elif scheduler_name == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=train_cfg.get('step_size', 30),
                gamma=train_cfg.get('gamma', 0.1)
            )
        elif scheduler_name == 'reduce':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
        
        self.logger.info(f"学习率调度器: {scheduler_name}")
    
    def build_dataloaders(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        self.train_loader = train_loader
        self.val_loader = val_loader
    
    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor,
                    mixture: torch.Tensor) -> Dict[str, torch.Tensor]:
        loss_cfg = self.config.get('training', {}).get('loss', {})
        
        losses = {}
        total_loss = 0
        
        if 'mse' in loss_cfg and loss_cfg['mse'] > 0:
            mse_loss = F.mse_loss(outputs, targets)
            losses['mse'] = mse_loss
            total_loss = total_loss + loss_cfg['mse'] * mse_loss
        
        if 'snr' in loss_cfg and loss_cfg['snr'] > 0:
            snr_loss = self.compute_snr_loss(outputs, targets)
            losses['snr'] = snr_loss
            total_loss = total_loss + loss_cfg['snr'] * snr_loss
        
        if 'sisnr' in loss_cfg and loss_cfg['sisnr'] > 0:
            sisnr_loss = self.compute_sisnr_loss(outputs, targets)
            losses['sisnr'] = sisnr_loss
            total_loss = total_loss + loss_cfg['sisnr'] * sisnr_loss
        
        losses['total'] = total_loss
        return losses
    
    def compute_snr_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        signal_power = (target ** 2).sum()
        noise_power = ((pred - target) ** 2).sum()
        
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-8) + 1e-8)
        return -snr
    
    def compute_sisnr_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target - target.mean(dim=-1, keepdim=True)
        pred = pred - pred.mean(dim=-1, keepdim=True)
        
        alpha = (target * pred).sum(dim=-1, keepdim=True) / (target ** 2).sum(dim=-1, keepdim=True) + 1e-8
        proj = alpha * target
        
        sisnr = 10 * torch.log10((proj ** 2).sum() / ((pred - proj) ** 2).sum() + 1e-8)
        return -sisnr
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        mixture = batch['mixture'].to(self.device)
        sources = batch['sources'].to(self.device)
        
        if self.use_amp and self.scaler:
            with autocast():
                outputs = self.model(mixture)
                outputs = torch.stack(outputs, dim=1)
                loss_dict = self.compute_loss(outputs, sources, mixture)
        else:
            outputs = self.model(mixture)
            outputs = torch.stack(outputs, dim=1)
            loss_dict = self.compute_loss(outputs, sources, mixture)
        
        loss = loss_dict['total'] / self.gradient_accumulation
        
        if self.use_amp and self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        losses = {k: v.item() * self.gradient_accumulation for k, v in loss_dict.items()}
        
        return losses
    
    def optimizer_step(self):
        if self.use_amp and self.scaler:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.optimizer.step()
        
        self.optimizer.zero_grad()
    
    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        
        epoch_losses = {'total': 0, 'mse': 0, 'snr': 0, 'sisnr': 0}
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}") if not self.is_distributed or self.local_rank == 0 else self.train_loader
        
        for batch_idx, batch in enumerate(progress_bar):
            losses = self.train_step(batch)
            
            if (batch_idx + 1) % self.gradient_accumulation == 0:
                self.optimizer_step()
                self.global_step += 1
            
            for k, v in losses.items():
                epoch_losses[k] += v
            num_batches += 1
            
            if self.global_step % self.log_interval == 0 and (not self.is_distributed or self.local_rank == 0):
                lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(f"Step {self.global_step}, Loss: {losses['total']:.4f}, LR: {lr:.6f}")
                
                for k, v in losses.items():
                    self.writer.add_scalar(f'train/{k}', v, self.global_step)
                self.writer.add_scalar('train/lr', lr, self.global_step)
        
        for k in epoch_losses:
            epoch_losses[k] /= num_batches
        
        return epoch_losses
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        
        val_losses = {'total': 0, 'mse': 0, 'snr': 0, 'sisnr': 0}
        num_batches = 0
        
        for batch in self.val_loader:
            mixture = batch['mixture'].to(self.device)
            sources = batch['sources'].to(self.device)
            
            outputs = self.model(mixture)
            outputs = torch.stack(outputs, dim=1)
            
            loss_dict = self.compute_loss(outputs, sources, mixture)
            
            for k, v in loss_dict.items():
                val_losses[k] += v.item()
            num_batches += 1
        
        for k in val_losses:
            val_losses[k] /= num_batches
        
        return val_losses
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        if self.is_distributed and self.local_rank != 0:
            return
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.module.state_dict() if isinstance(self.model, DDP) else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"保存最佳模型: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        self.logger.info(f"加载检查点: {checkpoint_path}, epoch={self.current_epoch}")
    
    def train(self):
        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch
            
            train_losses = self.train_epoch()
            
            if self.val_loader:
                val_losses = self.validate()
                
                if val_losses['total'] < self.best_loss:
                    self.best_loss = val_losses['total']
                    self.early_stop_counter = 0
                    is_best = True
                else:
                    self.early_stop_counter += 1
                    is_best = False
                
                if not self.is_distributed or self.local_rank == 0:
                    self.logger.info(f"Epoch {epoch}: Train Loss={train_losses['total']:.4f}, Val Loss={val_losses['total']:.4f}")
                    
                    for k, v in val_losses.items():
                        self.writer.add_scalar(f'val/{k}', v, epoch)
                    
                    if epoch % self.config.get('checkpoint', {}).get('save_interval', 5) == 0:
                        self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
                    
                    if is_best and self.save_best:
                        self.save_checkpoint('best_model.pt', is_best=True)
                
                if self.early_stop_counter >= self.patience:
                    self.logger.info(f"早停! 验证损失连续{self.patience}个epoch没有改善")
                    break
            
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_losses['total'])
                else:
                    self.scheduler.step()
        
        if not self.is_distributed or self.local_rank == 0:
            self.logger.info("训练完成!")
            self.writer.close()


if __name__ == "__main__":
    pass
