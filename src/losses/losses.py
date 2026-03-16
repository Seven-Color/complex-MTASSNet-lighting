"""
损失函数 - Strategy Pattern
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from src.core.base import BaseLoss


class MSELoss(BaseLoss):
    """均方误差损失"""
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.mse(pred, target)


class SNRLoss(BaseLoss):
    """
    信噪比损失
    Signal-to-Noise Ratio
    """
    
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        signal_power = (target ** 2).sum(dim=-1)
        noise_power = ((pred - target) ** 2).sum(dim=-1)
        
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-8) + 1e-8)
        
        if self.reduction == "mean":
            return -snr.mean()
        elif self.reduction == "sum":
            return -snr.sum()
        return -snr


class SISNRLoss(BaseLoss):
    """
    尺度不变信噪比损失
    Scale-Invariant SI-SNR
    """
    
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 去除均值
        target = target - target.mean(dim=-1, keepdim=True)
        pred = pred - pred.mean(dim=-1, keepdim=True)
        
        # 投影到目标方向
        alpha = ((target * pred).sum(dim=-1, keepdim=True) / 
                 (target ** 2).sum(dim=-1, keepdim=True) + 1e-8)
        proj = alpha * target
        
        # 计算 SI-SNR
        sisnr = 10 * torch.log10(
            (proj ** 2).sum(dim=-1) / 
            ((pred - proj) ** 2).sum(dim=-1) + 1e-8
        )
        
        if self.reduction == "mean":
            return -sisnr.mean()
        elif self.reduction == "sum":
            return -sisnr.sum()
        return -sisnr


class CompositeLoss(BaseLoss):
    """
    组合损失
    Strategy Pattern: 组合多个损失函数
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None
    ):
        super().__init__()
        
        # 默认权重
        if weights is None:
            weights = {"mse": 1.0, "snr": 0.5, "sisnr": 0.5}
        
        self.weights = weights
        self.losses = {}
        
        # 初始化损失函数
        if "mse" in weights:
            self.losses["mse"] = MSELoss()
        if "snr" in weights:
            self.losses["snr"] = SNRLoss()
        if "sisnr" in weights:
            self.losses["sisnr"] = SISNRLoss()
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        计算所有损失
        Returns:
            dict: 各损失值和总损失
        """
        total_loss = 0
        loss_dict = {}
        
        for name, loss_fn in self.losses.items():
            loss_value = loss_fn(pred, target)
            loss_dict[name] = loss_value
            total_loss = total_loss + self.weights.get(name, 0) * loss_value
        
        loss_dict["total"] = total_loss
        
        return loss_dict
    
    def get_weights(self) -> Dict[str, float]:
        return self.weights.copy()
    
    def set_weight(self, name: str, weight: float) -> None:
        """动态调整权重"""
        if name in self.losses:
            self.weights[name] = weight


class LossFactory:
    """
    损失函数工厂 - Factory Pattern
    """
    
    _loss_registry = {
        "mse": MSELoss,
        "snr": SNRLoss,
        "sisnr": SISNRLoss,
        "composite": CompositeLoss,
    }
    
    @classmethod
    def create(cls, name: str, **kwargs) -> BaseLoss:
        """创建损失函数"""
        if name not in cls._loss_registry:
            raise ValueError(f"Unknown loss: {name}. Available: {list(cls._loss_registry.keys())}")
        
        return cls._loss_registry[name](**kwargs)
    
    @classmethod
    def create_from_config(cls, config: dict) -> CompositeLoss:
        """从配置创建组合损失"""
        weights = {
            "mse": config.get("mse", 1.0),
            "snr": config.get("snr", 0.5),
            "sisnr": config.get("sisnr", 0.5),
        }
        return CompositeLoss(weights)
    
    @classmethod
    def register(cls, name: str, loss_class: type) -> None:
        """注册新的损失函数"""
        cls._loss_registry[name] = loss_class
