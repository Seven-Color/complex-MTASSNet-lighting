"""
Complex-MTASS 简化重构版
- 减少冗余的MS_ResBlock (15 -> 6)
- 变量参数化
- 支持配置
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Dict, Any


class ComplexMTASS(nn.Module):
    """
    Complex-domain Multi-Task Audio Source Separation
    简化版：减少冗余block，支持配置参数化
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        stage1_cfg = config.get('stage1', {})
        stage2_cfg = config.get('stage2', {})
        
        self.fft_size = config.get('fft_size', 512)
        self.hidden_channels = stage1_cfg.get('hidden_channels', 1024)
        self.num_sources = config.get('num_sources', 3)
        self.dropout = stage1_cfg.get('dropout', 0.2)
        
        # Stage 1: Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(257, self.hidden_channels, kernel_size=1),
            nn.BatchNorm1d(self.hidden_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Multi-Scale ResBlocks (简化版: 15 -> 6)
        dilations = stage1_cfg.get('ms_resblock_dilations', [1, 3, 5, 7, 11, 13])
        self.ms_resblocks = nn.ModuleList([
            MSResBlock(self.hidden_channels, dilation, self.dropout)
            for dilation in dilations
        ])
        
        # Stage 1: Decoder
        self.decoder = nn.Sequential(
            nn.Conv1d(self.hidden_channels, self.hidden_channels, kernel_size=1),
            nn.BatchNorm1d(self.hidden_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # 输出层: 为每个源输出mask
        self.source_masks = nn.ModuleList([
            nn.Conv1d(self.hidden_channels, 257, kernel_size=1)
            for _ in range(self.num_sources)
        ])
        
        # Stage 2: Residual Repair
        self.stage2_enabled = stage2_cfg.get('enabled', True)
        if self.stage2_enabled:
            repeats = stage2_cfg.get('repeats', 3)
            num_blocks = stage2_cfg.get('num_blocks', 8)
            res_hidden = stage2_cfg.get('hidden_channels', 256)
            
            self.residual_blocks = nn.ModuleList([
                GTCNBlock(repeats, num_blocks, res_hidden, self.dropout)
                for _ in range(self.num_sources)
            ])
    
    def forward(self, x: torch.Tensor, return_masks: bool = False):
        batch_size = x.shape[0]
        
        # 提取幅度和相位
        x_real = x[:, :257, :]
        x_imag = x[:, 257:, :]
        x_mag = torch.sqrt(x_real ** 2 + x_imag ** 2 + 1e-8)
        
        # Stage 1: 编码 + 多尺度处理
        h = self.encoder(x_mag)
        
        for resblock in self.ms_resblocks:
            h = resblock(h)
        
        h = self.decoder(h)
        
        # 生成源mask
        masks = [mask(h) for mask in self.source_masks]
        masks = [F.softmax(m, dim=1) for m in masks]
        
        # 应用mask到输入
        sources = []
        for mask in masks:
            src_real = mask * x_real
            src_imag = mask * x_imag
            sources.append(torch.cat([src_real, src_imag], dim=1))
        
        # Stage 2: 残差修复
        if self.stage2_enabled:
            refined_sources = []
            for i, src in enumerate(sources):
                residual_input = x - src
                residual = self.residual_blocks[i](residual_input)
                refined = src + residual
                refined_sources.append(refined)
            sources = refined_sources
        
        if return_masks:
            return tuple(sources), masks
        return tuple(sources)
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MSResBlock(nn.Module):
    """多尺度残差块"""
    
    def __init__(self, channels: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        
        self.channels = channels
        self.dilation = dilation
        
        self.conv_down = nn.Conv1d(channels, channels // 4, kernel_size=1)
        self.bn_down = nn.BatchNorm1d(channels // 4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.conv_gate = nn.Conv1d(
            channels // 4, channels // 4, 
            kernel_size=3, dilation=dilation, padding=dilation
        )
        self.bn_gate = nn.BatchNorm1d(channels // 4)
        
        self.conv_up = nn.Conv1d(channels // 4, channels, kernel_size=1)
        self.bn_up = nn.BatchNorm1d(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        h = self.conv_down(x)
        h = self.bn_down(h)
        h = self.relu(h)
        h = self.dropout(h)
        
        gate = self.conv_gate(h)
        gate = self.bn_gate(gate)
        h = torch.sigmoid(gate) * torch.tanh(gate)
        h = self.dropout(h)
        
        h = self.conv_up(h)
        h = self.bn_up(h)
        
        return residual + h


class GTCNBlock(nn.Module):
    """门控时序卷积块"""
    
    def __init__(self, repeats: int, num_blocks: int, hidden_channels: int, dropout: float = 0.2):
        super().__init__()
        
        self.conv_in = nn.Conv1d(514, hidden_channels, kernel_size=1)
        
        self.tcm_blocks = nn.ModuleList([
            TCMBlock(hidden_channels, 2 ** i, dropout)
            for i in range(num_blocks)
        ])
        
        self.conv_out = nn.Conv1d(hidden_channels, 514, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        
        for tcm in self.tcm_blocks:
            h = tcm(h)
        
        h = self.conv_out(h)
        return h


class TCMBlock(nn.Module):
    """时序卷积模块"""
    
    def __init__(self, channels: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        
        self.channels = channels
        self.dilation = dilation
        
        self.conv_in = nn.Conv1d(channels, channels // 4, kernel_size=1)
        self.bn_in = nn.BatchNorm1d(channels // 4)
        
        self.conv_gate = nn.Conv1d(
            channels // 4, channels // 4,
            kernel_size=3, dilation=dilation, padding=dilation
        )
        self.bn_gate = nn.BatchNorm1d(channels // 4)
        
        self.conv_out = nn.Conv1d(channels // 4, channels, kernel_size=1)
        self.bn_out = nn.BatchNorm1d(channels)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        h = self.conv_in(x)
        h = self.bn_in(h)
        h = F.relu(h)
        
        conv_out = self.conv_gate(h)
        conv_out = self.bn_gate(conv_out)
        
        gate = torch.sigmoid(conv_out)
        h = conv_out * gate
        h = self.dropout(h)
        
        h = self.conv_out(h)
        h = self.bn_out(h)
        
        return residual + h


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable
    }


if __name__ == "__main__":
    import yaml
    
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    model = ComplexMTASS(config['model'])
    x = torch.randn(2, 514, 1000)
    sources = model(x)
    
    print('Input:', x.shape)
    print('Sources:', len(sources), [s.shape for s in sources])
    print('Params:', count_parameters(model))
