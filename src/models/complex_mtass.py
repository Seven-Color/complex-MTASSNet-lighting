"""
模型构建器 - Builder Pattern
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core.base import BaseModule
from src.core.config import ModelConfig


class SeparationHead(nn.Module):
    """
    分离头
    为每个源生成 mask
    """
    
    def __init__(self, in_channels: int, num_sources: int):
        super().__init__()
        
        self.masks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, in_channels // 2, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(in_channels // 2, 257, kernel_size=1)
            )
            for _ in range(num_sources)
        ])
    
    def forward(self, x: torch.Tensor, input_complex: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: 特征 [B, C, T]
            input_complex: 输入复数 [B, 514, T]
        Returns:
            sources: 分离后的源 [B, 514, T] * num_sources
        """
        sources = []
        
        # 提取幅度和相位
        x_real = input_complex[:, :257, :]
        x_imag = input_complex[:, 257:, :]
        
        for mask_conv in self.masks:
            mask = mask_conv(x)
            mask = F.softmax(mask, dim=1)
            
            # 应用 mask 到复数域
            src_real = mask * x_real
            src_imag = mask * x_imag
            sources.append(torch.cat([src_real, src_imag], dim=1))
        
        return sources


class Encoder(nn.Module):
    """编码器 - 将幅度转换为隐藏特征"""
    
    def __init__(self, in_channels: int, hidden_channels: int, dropout: float = 0.2):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Decoder(nn.Module):
    """解码器 - 将隐藏特征转换为特征"""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class MultiScaleResidualStack(nn.Module):
    """
    多尺度残差堆叠
    包含多个不同 dilation 的残差块
    """
    
    def __init__(
        self,
        channels: int,
        dilations: List[int],
        dropout: float = 0.2
    ):
        super().__init__()
        
        # 从配置创建残差块
        from src.models.blocks.conv_blocks import ResidualBlock
        
        self.blocks = nn.ModuleList([
            ResidualBlock(channels, 4, 3, d, dropout)
            for d in dilations
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class ResidualRefinement(nn.Module):
    """
    残差 refinement 模块
    用于 Stage 2 的残差修复
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_blocks: int,
        dropout: float = 0.2
    ):
        super().__init__()
        
        from src.models.blocks.conv_blocks import GatedConv1DBlock
        
        # 输入投影
        self.conv_in = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.bn_in = nn.BatchNorm1d(hidden_channels)
        
        # TCM 块
        self.tcm_blocks = nn.ModuleList([
            GatedConv1DBlock(hidden_channels, 3, 2 ** i, dropout)
            for i in range(num_blocks)
        ])
        
        # 输出投影
        self.conv_out = nn.Conv1d(hidden_channels, in_channels, kernel_size=1)
        self.bn_out = nn.BatchNorm1d(in_channels)
        
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        h = self.bn_in(h)
        h = self.activation(h)
        
        for block in self.tcm_blocks:
            h = block(h)
        
        h = self.conv_out(h)
        h = self.bn_out(h)
        
        return x + h  # 残差连接


class ComplexMTASSModel(nn.Module):
    """
    Complex-domain Multi-Task Audio Source Separation
    整洁的模块化架构
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.config = config
        self.num_sources = config.num_sources
        
        # ==================== Stage 1: 分离 ====================
        
        # Encoder
        self.encoder = Encoder(257, config.hidden_channels, config.dropout)
        
        # 多尺度残差块
        self.ms_res_blocks = MultiScaleResidualStack(
            config.hidden_channels,
            config.ms_resblock_dilations,
            config.dropout
        )
        
        # Decoder
        self.decoder = Decoder(config.hidden_channels, config.hidden_channels, config.dropout)
        
        # 分离头
        self.separation_head = SeparationHead(
            config.hidden_channels,
            config.num_sources
        )
        
        # ==================== Stage 2: 残差修复 ====================
        
        if config.stage2_enabled:
            self.residual_refinements = nn.ModuleList([
                ResidualRefinement(
                    514,
                    config.stage2_hidden,
                    config.stage2_blocks,
                    config.dropout
                )
                for _ in range(config.num_sources)
            ])
    
    def forward(
        self,
        x: torch.Tensor,
        return_masks: bool = False
    ) -> tuple:
        """
        Args:
            x: 输入 [B, 514, T] (real + imag 拼接)
            return_masks: 是否返回 mask
        Returns:
            sources: 分离后的源
            masks: (可选) 生成的 mask
        """
        # 提取幅度
        x_real = x[:, :257, :]
        x_imag = x[:, 257:, :]
        x_mag = torch.sqrt(x_real ** 2 + x_imag ** 2 + 1e-8)
        
        # Stage 1: 编码 -> 多尺度 -> 解码
        h = self.encoder(x_mag)
        h = self.ms_res_blocks(h)
        h = self.decoder(h)
        
        # 生成源
        sources = self.separation_head(h, x)
        
        # Stage 2: 残差 refinement
        if self.config.stage2_enabled and self.residual_refinements is not None:
            refined_sources = []
            for i, src in enumerate(sources):
                residual_input = x - src
                refined = self.residual_refinements[i](residual_input)
                refined_sources.append(src + refined)  # 原始 + 残差
            sources = refined_sources
        
        if return_masks:
            # 返回中间结果用于调试
            return tuple(sources), None
        
        return tuple(sources)
    
    def get_num_parameters(self) -> int:
        """获取参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ModelBuilder:
    """
    模型构建器 - Builder Pattern
    """
    
    def __init__(self):
        self._config: Optional[ModelConfig] = None
    
    def set_config(self, config: ModelConfig) -> 'ModelBuilder':
        self._config = config
        return self
    
    def build(self) -> ComplexMTASSModel:
        if self._config is None:
            raise ValueError("Config must be set before building model")
        return ComplexMTASSModel(self._config)
    
    @staticmethod
    def from_yaml(path: str) -> ComplexMTASSModel:
        """从 YAML 构建模型"""
        from src.core.config import Config
        
        config = Config.from_yaml(path)
        return ComplexMTASSModel(config.model)
