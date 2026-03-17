"""
优化版 Complex-MTASS - 添加高效注意力机制
- 添加 SE (Squeeze-and-Excitation) 注意力
- 添加多头自注意力 (MHSA)
- 使用 Grouped Conv 减少计算量
- 保持与原版兼容的配置接口
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 注意力模块"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class MultiHeadSelfAttention(nn.Module):
    """轻量级多头自注意力"""
    
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        
        # 初始化
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.proj.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, L)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).transpose(-2, -1).reshape(B, C, L)
        out = self.proj(out)
        
        return x + out  # 残差连接


class OptimizedMSResBlock(nn.Module):
    """优化版多尺度残差块 - 带 SE 注意力"""
    
    def __init__(self, channels: int, dilation: int, dropout: float = 0.2, use_se: bool = True):
        super().__init__()
        
        self.channels = channels
        self.dilation = dilation
        
        # 降维
        self.conv_down = nn.Conv1d(channels, channels // 4, kernel_size=1)
        self.bn_down = nn.BatchNorm1d(channels // 4)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
        # 门控卷积
        self.conv_gate = nn.Conv1d(
            channels // 4, channels // 4, 
            kernel_size=3, dilation=dilation, padding=dilation,
            groups=4  # 分组卷积减少计算
        )
        self.bn_gate = nn.BatchNorm1d(channels // 4)
        
        # SE 注意力
        self.se = SEBlock(channels // 4, reduction=8) if use_se else nn.Identity()
        
        # 升维
        self.conv_up = nn.Conv1d(channels // 4, channels, kernel_size=1)
        self.bn_up = nn.BatchNorm1d(channels)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        h = self.conv_down(x)
        h = self.bn_down(h)
        h = self.relu(h)
        h = self.dropout(h)
        
        # 门控 + SE
        gate = self.conv_gate(h)
        gate = self.bn_gate(gate)
        gate = self.se(gate)  # 添加 SE
        h = torch.sigmoid(gate) * torch.tanh(gate)
        h = self.dropout(h)
        
        h = self.conv_up(h)
        h = self.bn_up(h)
        
        return residual + h


class OptimizedGTCNBlock(nn.Module):
    """优化版门控时序卷积块"""
    
    def __init__(self, repeats: int, num_blocks: int, hidden_channels: int, dropout: float = 0.2):
        super().__init__()
        
        self.conv_in = nn.Conv1d(514, hidden_channels, kernel_size=1)
        
        # 使用更小的卷积核和分组
        self.tcm_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, 
                         dilation=2**i, padding=2**i, groups=4),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for i in range(num_blocks)
        ])
        
        # SE 注意力
        self.se = SEBlock(hidden_channels, reduction=8)
        
        self.conv_out = nn.Conv1d(hidden_channels, 514, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        
        for tcm in self.tcm_blocks:
            h = tcm(h)
        
        h = self.se(h)
        h = self.conv_out(h)
        
        return x + h  # 残差


class OptimizedComplexMTASS(nn.Module):
    """
    优化版 Complex-MTASS
    - 添加 SE 注意力提升特征表达
    - 使用分组卷积减少计算量
    - 可选添加 MHSA 增强全局建模
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        stage1_cfg = config.get('stage1', {})
        stage2_cfg = config.get('stage2', {})
        
        self.fft_size = config.get('fft_size', 512)
        self.hidden_channels = stage1_cfg.get('hidden_channels', 512)
        self.num_sources = config.get('num_sources', 3)
        self.dropout = stage1_cfg.get('dropout', 0.2)
        self.use_se = stage1_cfg.get('use_se', True)
        self.use_attention = stage1_cfg.get('use_attention', False)
        
        # Stage 1: Encoder - 使用分组卷积
        self.encoder = nn.Sequential(
            nn.Conv1d(257, self.hidden_channels, kernel_size=1),
            nn.BatchNorm1d(self.hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout)
        )
        
        # 多尺度残差块 (带 SE)
        dilations = stage1_cfg.get('ms_resblock_dilations', [1, 3, 5, 7, 11, 13])
        self.ms_resblocks = nn.ModuleList([
            OptimizedMSResBlock(self.hidden_channels, dilation, self.dropout, use_se=self.use_se)
            for dilation in dilations
        ])
        
        # 可选：自注意力
        if self.use_attention:
            self.attention = MultiHeadSelfAttention(self.hidden_channels, num_heads=4)
        
        # Stage 1: Decoder
        self.decoder = nn.Sequential(
            nn.Conv1d(self.hidden_channels, self.hidden_channels, kernel_size=1),
            nn.BatchNorm1d(self.hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout)
        )
        
        # 输出层
        self.source_masks = nn.ModuleList([
            nn.Conv1d(self.hidden_channels, 257, kernel_size=1)
            for _ in range(self.num_sources)
        ])
        
        # Stage 2: 残差修复
        self.stage2_enabled = stage2_cfg.get('enabled', True)
        if self.stage2_enabled:
            repeats = stage2_cfg.get('repeats', 2)
            num_blocks = stage2_cfg.get('num_blocks', 6)
            res_hidden = stage2_cfg.get('hidden_channels', 128)
            
            self.residual_blocks = nn.ModuleList([
                OptimizedGTCNBlock(repeats, num_blocks, res_hidden, self.dropout)
                for _ in range(self.num_sources)
            ])
    
    def forward(self, x: torch.Tensor, return_masks: bool = False):
        batch_size = x.shape[0]
        
        # 提取幅度和相位
        x_real = x[:, :257, :]
        x_imag = x[:, 257:, :]
        x_mag = torch.sqrt(x_real ** 2 + x_imag ** 2 + 1e-8)
        
        # Stage 1: 编码
        h = self.encoder(x_mag)
        
        # 多尺度处理
        for resblock in self.ms_resblocks:
            h = resblock(h)
        
        # 可选注意力
        if self.use_attention:
            h = self.attention(h)
        
        # 解码
        h = self.decoder(h)
        
        # 生成 mask
        masks = [mask(h) for mask in self.source_masks]
        masks = [F.softmax(m, dim=1) for m in masks]
        
        # 应用 mask
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


# 兼容旧接口
ComplexMTASS = OptimizedComplexMTASS


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
    import time
    
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 测试优化版
    print("=" * 50)
    print("优化版模型性能测试")
    print("=" * 50)
    
    configs = [
        ('优化版-平衡 (512+SE)', {
            'fft_size': 512, 'num_sources': 3,
            'stage1': {'hidden_channels': 512, 'ms_resblock_dilations': [1,3,5,7,11,13], 
                      'dropout': 0.2, 'use_se': True, 'use_attention': False},
            'stage2': {'enabled': True, 'repeats': 2, 'num_blocks': 6, 'hidden_channels': 128, 'dropout': 0.2}
        }),
        ('优化版-完整 (512+SE+ATTN)', {
            'fft_size': 512, 'num_sources': 3,
            'stage1': {'hidden_channels': 512, 'ms_resblock_dilations': [1,3,5,7,11,13], 
                      'dropout': 0.2, 'use_se': True, 'use_attention': True},
            'stage2': {'enabled': True, 'repeats': 2, 'num_blocks': 6, 'hidden_channels': 128, 'dropout': 0.2}
        }),
    ]
    
    x = torch.randn(1, 514, 1000)
    
    for name, cfg in configs:
        model = OptimizedComplexMTASS(cfg)
        params = count_parameters(model)
        
        model.eval()
        with torch.no_grad():
            for _ in range(5):
                _ = model(x)
            times = []
            for _ in range(20):
                start = time.time()
                _ = model(x)
                times.append(time.time() - start)
        
        avg_ms = sum(times) / len(times) * 1000
        flops = params['total'] * 2 / 1e6
        
        print(f"\n{name}:")
        print(f"  参数量: {params['total']:,}")
        print(f"  推理时间: {avg_ms:.1f} ms")
        print(f"  FLOPs: {flops:.1f}M")
