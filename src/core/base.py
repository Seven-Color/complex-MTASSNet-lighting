"""
核心模块 - 基础抽象类
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn


class BaseModule(nn.Module):
    """模型基类"""
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        pass
    
    def count_parameters(self) -> int:
        """统计参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BaseLoss(nn.Module):
    """损失函数基类"""
    
    @abstractmethod
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算损失"""
        pass


class BaseCallback(ABC):
    """回调基类"""
    
    @abstractmethod
    def on_train_start(self, trainer: 'Trainer') -> None:
        """训练开始"""
        pass
    
    @abstractmethod
    def on_train_end(self, trainer: 'Trainer') -> None:
        """训练结束"""
        pass
    
    @abstractmethod
    def on_epoch_start(self, trainer: 'Trainer', epoch: int) -> None:
        """Epoch 开始"""
        pass
    
    @abstractmethod
    def on_epoch_end(self, trainer: 'Trainer', epoch: int, metrics: Dict[str, float]) -> None:
        """Epoch 结束"""
        pass
    
    @abstractmethod
    def on_batch_start(self, trainer: 'Trainer', batch_idx: int) -> None:
        """Batch 开始"""
        pass
    
    @abstractmethod
    def on_batch_end(self, trainer: 'Trainer', batch_idx: int, loss: float) -> None:
        """Batch 结束"""
        pass


class BaseDataProcessor(ABC):
    """数据处理器基类"""
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        """处理数据"""
        pass
    
    @abstractmethod
    def __call__(self, data: Any) -> Any:
        """可调用"""
        pass
