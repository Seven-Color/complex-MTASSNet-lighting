"""
数据处理模块 - STFT 变换
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from src.core.config import DataConfig


class STFT(nn.Module):
    """
    短时傅里叶变换
    可训练的 STFT/iSTFT
    """
    
    def __init__(
        self,
        fft_size: int = 512,
        hop_size: int = 256,
        window: str = "hann",
        center: bool = True
    ):
        super().__init__()
        
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.center = center
        
        # 创建窗口
        if window == "hann":
            self.register_buffer('window', torch.hann_window(fft_size))
        elif window == "hamming":
            self.register_buffer('window', torch.hamming_window(fft_size))
        else:
            self.register_buffer('window', torch.ones(fft_size))
    
    def forward(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            audio: 音频 [B, T] 或 [B, 1, T]
        Returns:
            real: 实部 [B, F, T]
            imag: 虚部 [B, F, T]
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        
        # STFT
        complex_spec = torch.stft(
            audio.squeeze(1),  # [B, T]
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.fft_size,
            window=self.window,
            center=self.center,
            return_complex=True
        )
        
        # 分解为实部和虚部
        real = complex_spec.real
        imag = complex_spec.imag
        
        return real, imag
    
    def inverse(
        self,
        real: torch.Tensor,
        imag: torch.Tensor,
        length: Optional[int] = None
    ) -> torch.Tensor:
        """
        逆 STFT
        """
        complex_spec = torch.complex(real, imag)
        
        audio = torch.istft(
            complex_spec,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.fft_size,
            window=self.window,
            center=self.center,
            length=length
        )
        
        return audio
    
    def forward_with_magnitude(
        self,
        audio: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回幅度和相位
        """
        real, imag = self.forward(audio)
        magnitude = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        phase = torch.atan2(imag, real)
        
        return magnitude, phase, real  # real 作为相位信息
    
    def complex_to_features(
        self,
        real: torch.Tensor,
        imag: torch.Tensor
    ) -> torch.Tensor:
        """
        将复数转换为模型输入特征
        拼接 real 和 imag
        """
        return torch.cat([real, imag], dim=1)
    
    def features_to_complex(
        self,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将特征转换回 real 和 imag
        """
        half = features.shape[1] // 2
        real = features[:, :half, :]
        imag = features[:, half:, :]
        return real, imag


class STFTFactory:
    """
    STFT 工厂类 - Factory Pattern
    """
    
    @staticmethod
    def create(config: dict) -> STFT:
        """从配置创建 STFT"""
        stft_cfg = config.get('stft', {})
        
        return STFT(
            fft_size=stft_cfg.get('fft_size', 512),
            hop_size=stft_cfg.get('hop_size', 256),
            window=stft_cfg.get('window', 'hann'),
            center=stft_cfg.get('center', True)
        )
    
    @staticmethod
    def create_default() -> STFT:
        """创建默认 STFT"""
        return STFT()
