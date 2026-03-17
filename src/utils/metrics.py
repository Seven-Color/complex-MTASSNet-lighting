"""
评估指标模块 - SI-SNR, SAR, SDR, PESQ
"""

import torch
import numpy as np
from typing import Tuple, Optional


def sisnr(s_est: torch.Tensor, s_ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    计算 SI-SNR (Scale-Invariant Signal-to-Noise Ratio)
    
    Args:
        s_est: 估计信号 [B, T] 或 [T]
        s_ref: 参考信号 [B, T] 或 [T]
    
    Returns:
        SI-SNR 值
    """
    if s_est.dim() == 2:
        s_est = s_est.squeeze(1)
    if s_ref.dim() == 2:
        s_ref = s_ref.squeeze(1)
    
    # 能量
    s_ref_energy = torch.sqrt(torch.sum(s_ref ** 2, dim=-1, keepdim=True) + eps)
    
    # 投影
    s_proj = (torch.sum(s_est * s_ref, dim=-1, keepdim=True) / (s_ref_energy ** 2 + eps)) * s_ref
    
    # 噪声
    s_noise = s_est - s_proj
    
    # SI-SNR
    snr = 10 * torch.log10(torch.sum(s_proj ** 2, dim=-1) / (torch.sum(s_noise ** 2, dim=-1) + eps) + eps)
    
    return snr.mean()


def sdr(s_est: torch.Tensor, s_ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    计算 SDR (Signal-to-Distortion Ratio)
    """
    if s_est.dim() == 2:
        s_est = s_est.squeeze(1)
    if s_ref.dim() == 2:
        s_ref = s_ref.squeeze(1)
    
    s_ref_energy = torch.sum(s_ref ** 2, dim=-1) + eps
    error = s_est - s_ref
    error_energy = torch.sum(error ** 2, dim=-1) + eps
    
    sdr = 10 * torch.log10(s_ref_energy / error_energy)
    return sdr.mean()


def sar(s_est: torch.Tensor, s_ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    计算 SAR (Signal-to-Artifact Ratio)
    """
    if s_est.dim() == 2:
        s_est = s_est.squeeze(1)
    if s_ref.dim() == 2:
        s_ref = s_ref.squeeze(1)
    
    s_ref_energy = torch.sum(s_ref ** 2, dim=-1) + eps
    artifacts = s_est - s_ref
    artifact_energy = torch.sum(artifacts ** 2, dim=-1) + eps
    
    sar = 10 * torch.log10(s_ref_energy / artifact_energy)
    return sar.mean()


class MetricsTracker:
    """指标追踪器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.metrics = {
            'sisnr': [],
            'sdr': [],
            'sar': [],
        }
    
    def update(self, s_est: torch.Tensor, s_ref: torch.Tensor):
        """更新指标"""
        self.metrics['sisnr'].append(sisnr(s_est, s_ref).item())
        self.metrics['sdr'].append(sdr(s_est, s_ref).item())
        self.metrics['sar'].append(sar(s_est, s_ref).item())
    
    def get_avg(self) -> dict:
        """获取平均值"""
        return {
            'sisnr': np.mean(self.metrics['sisnr']),
            'sdr': np.mean(self.metrics['sdr']),
            'sar': np.mean(self.metrics['sar']),
        }
    
    def __repr__(self) -> str:
        avg = self.get_avg()
        return f"SI-SNR: {avg['sisnr']:.2f} dB | SDR: {avg['sdr']:.2f} dB | SAR: {avg['sar']:.2f} dB"


def compute_metrics_batch(est_sources: torch.Tensor, ref_sources: torch.Tensor) -> dict:
    """
    批量计算指标
    
    Args:
        est_sources: 估计信号 [B, N, T]
        ref_sources: 参考信号 [B, N, T]
    
    Returns:
        指标字典
    """
    batch_size = est_sources.shape[0]
    num_sources = est_sources.shape[1]
    
    sisnr_scores = []
    sdr_scores = []
    sar_scores = []
    
    for b in range(batch_size):
        for n in range(num_sources):
            sisnr_scores.append(sisnr(est_sources[b, n], ref_sources[b, n]).item())
            sdr_scores.append(sdr(est_sources[b, n], ref_sources[b, n]).item())
            sar_scores.append(sar(est_sources[b, n], ref_sources[b, n]).item())
    
    return {
        'sisnr': np.mean(sisnr_scores),
        'sdr': np.mean(sdr_scores),
        'sar': np.mean(sar_scores),
    }


if __name__ == "__main__":
    # 测试
    torch.manual_seed(42)
    s_ref = torch.randn(2, 16000)
    s_est = s_ref + torch.randn(2, 16000) * 0.1
    
    print(f"SI-SNR: {sisnr(s_est, s_ref):.2f} dB")
    print(f"SDR: {sdr(s_est, s_ref):.2f} dB")
    print(f"SAR: {sar(s_est, s_ref):.2f} dB")
