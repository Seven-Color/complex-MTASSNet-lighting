"""
训练回调模块 - Observer Pattern
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import torch

# TensorBoard is optional
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


class Callback:
    """回调基类"""
    
    def on_train_start(self, trainer: 'Trainer') -> None:
        pass
    
    def on_train_end(self, trainer: 'Trainer') -> None:
        pass
    
    def on_epoch_start(self, trainer: 'Trainer', epoch: int) -> None:
        pass
    
    def on_epoch_end(self, trainer: 'Trainer', epoch: int, metrics: Dict[str, float]) -> None:
        pass
    
    def on_batch_start(self, trainer: 'Trainer', batch_idx: int) -> None:
        pass
    
    def on_batch_end(self, trainer: 'Trainer', batch_idx: int, loss: float) -> None:
        pass


class TensorBoardCallback(Callback):
    """TensorBoard 日志回调"""
    
    def __init__(self, log_dir: str = "./logs"):
        if not HAS_TENSORBOARD:
            self.logger = logging.getLogger(__name__)
            self.logger.warning("TensorBoard not installed, skipping TensorBoardCallback")
            self.enabled = False
            return
            
        self.writer = SummaryWriter(log_dir)
        self.enabled = True
        self.logger = logging.getLogger(__name__)
    
    def on_train_start(self, trainer: 'Trainer') -> None:
        if not self.enabled:
            return
        self.logger.info("训练开始")
    
    def on_train_end(self, trainer: 'Trainer') -> None:
        if not self.enabled:
            return
        self.writer.close()
        self.logger.info("训练结束")
    
    def on_epoch_end(self, trainer: 'Trainer', epoch: int, metrics: Dict[str, float]) -> None:
        if not self.enabled:
            return
        # 记录指标
        for name, value in metrics.items():
            self.writer.add_scalar(f"epoch/{name}", value, epoch)
        
        # 记录学习率
        lr = trainer.optimizer.param_groups[0]['lr']
        self.writer.add_scalar("epoch/lr", lr, epoch)
    
    def on_batch_end(self, trainer: 'Trainer', batch_idx: int, loss: float) -> None:
        if not self.enabled:
            return
        global_step = trainer.global_step
        self.writer.add_scalar("batch/loss", loss, global_step)


class CheckpointCallback(Callback):
    """检查点保存回调"""
    
    def __init__(
        self,
        save_dir: str = "./checkpoints",
        save_interval: int = 5,
        save_best: bool = True,
        metric: str = "val_loss"
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_interval = save_interval
        self.save_best = save_best
        self.metric = metric
        self.best_value = float('inf')
        
        self.logger = logging.getLogger(__name__)
    
    def on_epoch_end(self, trainer: 'Trainer', epoch: int, metrics: Dict[str, float]) -> None:
        # 定期保存
        if epoch % self.save_interval == 0:
            self._save_checkpoint(trainer, f"checkpoint_epoch_{epoch}.pt")
        
        # 保存最佳
        if self.save_best and self.metric in metrics:
            current_value = metrics[self.metric]
            if current_value < self.best_value:
                self.best_value = current_value
                self._save_checkpoint(trainer, "best_model.pt")
                self.logger.info(f"保存最佳模型: {self.metric}={current_value:.4f}")
    
    def _save_checkpoint(self, trainer: 'Trainer', filename: str) -> None:
        path = self.save_dir / filename
        
        checkpoint = {
            'epoch': trainer.current_epoch,
            'global_step': trainer.global_step,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'config': trainer.config
        }
        
        if trainer.scheduler:
            checkpoint['scheduler_state_dict'] = trainer.scheduler.state_dict()
        
        torch.save(checkpoint, path)


class EarlyStoppingCallback(Callback):
    """早停回调"""
    
    def __init__(
        self,
        patience: int = 10,
        metric: str = "val_loss",
        mode: str = "min",
        min_delta: float = 0.0
    ):
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.min_delta = min_delta
        
        self.best_value = float('inf') if mode == "min" else float('-inf')
        self.counter = 0
        self.should_stop = False
        
        self.logger = logging.getLogger(__name__)
    
    def on_epoch_end(self, trainer: 'Trainer', epoch: int, metrics: Dict[str, float]) -> None:
        if self.metric not in metrics:
            return
        
        current_value = metrics[self.metric]
        
        # 判断是否改善
        if self.mode == "min":
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            self.logger.info(f"早停计数器: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.should_stop = True
                self.logger.info(f"触发早停! {self.patience} 个 epoch 没有改善")


class CallbackList:
    """回调列表 - 管理多个回调"""
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []
    
    def add(self, callback: Callback) -> None:
        self.callbacks.append(callback)
    
    def on_train_start(self, trainer: 'Trainer') -> None:
        for cb in self.callbacks:
            cb.on_train_start(trainer)
    
    def on_train_end(self, trainer: 'Trainer') -> None:
        for cb in self.callbacks:
            cb.on_train_end(trainer)
    
    def on_epoch_start(self, trainer: 'Trainer', epoch: int) -> None:
        for cb in self.callbacks:
            cb.on_epoch_start(trainer, epoch)
    
    def on_epoch_end(self, trainer: 'Trainer', epoch: int, metrics: Dict[str, float]) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(trainer, epoch, metrics)
    
    def on_batch_start(self, trainer: 'Trainer', batch_idx: int) -> None:
        for cb in self.callbacks:
            cb.on_batch_start(trainer, batch_idx)
    
    def on_batch_end(self, trainer: 'Trainer', batch_idx: int, loss: float) -> None:
        for cb in self.callbacks:
            cb.on_batch_end(trainer, batch_idx, loss)
