"""
卷积基础模块
遵循 DRY 原则，避免重复代码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class Conv1DBlock(nn.Module):
    """
    1D 卷积块
    Conv -> BatchNorm -> Activation -> Dropout
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        dilation: int = 1,
        dropout: float = 0.2,
        activation: str = "relu"
    ):
        super().__init__()
        
        # 计算 padding 保持长度不变
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = self._get_activation(activation)
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "leaky_relu": nn.LeakyReLU(),
            "prelu": nn.PReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
        }
        return activations.get(name.lower(), nn.ReLU())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class GatedConv1DBlock(nn.Module):
    """
    门控卷积块
    两条分支：一条卷积，一条门控，element-wise 相乘
    """
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation // 2
        
        # 特征分支
        self.conv_feat = nn.Conv1d(
            channels, channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.bn_feat = nn.BatchNorm1d(channels)
        
        # 门控分支
        self.conv_gate = nn.Conv1d(
            channels, channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.bn_gate = nn.BatchNorm1d(channels)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 特征
        feat = self.conv_feat(x)
        feat = self.bn_feat(feat)
        feat = F.relu(feat)
        
        # 门控
        gate = self.conv_gate(x)
        gate = self.bn_gate(gate)
        gate = torch.sigmoid(gate)
        
        # 门控相乘
        out = feat * gate
        out = self.dropout(out)
        
        return out + x  # 残差连接


class ResidualBlock(nn.Module):
    """
    残差块
    包含两个门控卷积，支持残差连接
    """
    
    def __init__(
        self,
        channels: int,
        hidden_ratio: int = 4,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # 通道压缩
        hidden_channels = channels // hidden_ratio
        
        self.conv_down = nn.Conv1d(channels, hidden_channels, kernel_size=1)
        self.bn_down = nn.BatchNorm1d(hidden_channels)
        
        # 门控卷积
        self.gated_conv = GatedConv1DBlock(
            hidden_channels, kernel_size, dilation, dropout
        )
        
        # 通道扩展
        self.conv_up = nn.Conv1d(hidden_channels, channels, kernel_size=1)
        self.bn_up = nn.BatchNorm1d(channels)
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        # 压缩
        h = self.conv_down(x)
        h = self.bn_down(h)
        h = self.activation(h)
        h = self.dropout(h)
        
        # 门控卷积
        h = self.gated_conv(h)
        
        # 扩展 + 残差
        h = self.conv_up(h)
        h = self.bn_up(h)
        
        return residual + h


class ChannelAttention(nn.Module):
    """
    通道注意力机制
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        
        # Avg pooling
        avg_out = self.avg_pool(x).view(b, c)
        avg_out = self.fc(avg_out).view(b, c, 1)
        
        # Max pooling
        max_out = self.max_pool(x).view(b, c)
        max_out = self.fc(max_out).view(b, c, 1)
        
        # 注意力权重
        attention = avg_out + max_out
        
        return x * attention


class MultiScaleBlock(nn.Module):
    """
    多尺度块
    支持不同 dilation 的并行卷积
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilations: list = None,
        dropout: float = 0.2
    ):
        super().__init__()
        
        if dilations is None:
            dilations = [1, 2, 4, 8]
        
        # 并行分支
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels // len(dilations),
                    kernel_size, padding=d * (kernel_size - 1) // 2,
                    dilation=d
                ),
                nn.BatchNorm1d(out_channels // len(dilations)),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for d in dilations
        ])
        
        # 融合
        self.fusion = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()
        
        # 残差连接
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        
        # 并行卷积
        branch_outputs = [branch(x) for branch in self.branches]
        out = torch.cat(branch_outputs, dim=1)
        
        # 融合
        out = self.fusion(out)
        out = self.bn(out)
        out = self.activation(out)
        
        return out + residual
