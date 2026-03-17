"""
可视化模块 - 频谱图、波形图、热力图
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Tuple


def plot_waveform(waveform: torch.Tensor, save_path: Optional[str] = None, 
                  title: str = "Waveform", sample_rate: int = 16000):
    """
    绘制波形图
    
    Args:
        waveform: 音频信号 [T] 或 [B, T]
        save_path: 保存路径
        title: 标题
        sample_rate: 采样率
    """
    if waveform.dim() == 2:
        waveform = waveform[0].cpu().numpy()
    else:
        waveform = waveform.cpu().numpy()
    
    time = np.arange(len(waveform)) / sample_rate
    
    plt.figure(figsize=(12, 4))
    plt.plot(time, waveform, linewidth=0.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_spectrogram(stft_tensor: torch.Tensor, save_path: Optional[str] = None,
                     title: str = "Spectrogram", sample_rate: int = 16000,
                     hop_size: int = 256, cmap: str = 'magma'):
    """
    绘制频谱图
    
    Args:
        stft_tensor: STFT 结果 [F, T] 或 [B, F, T]
        save_path: 保存路径
        title: 标题
        sample_rate: 采样率
        hop_size: 帧移
        cmap: 颜色映射
    """
    if stft_tensor.dim() == 3:
        stft_tensor = stft_tensor[0]
    
    # 计算幅度谱
    if stft_tensor.shape[0] * 2 == stft_tensor.shape[1]:
        # 复数域 [real, imag]
        real = stft_tensor[:stft_tensor.shape[0]//2]
        imag = stft_tensor[stft_tensor.shape[0]//2:]
        magnitude = torch.sqrt(real ** 2 + imag ** 2)
    else:
        magnitude = stft_tensor
    
    magnitude = torch.log(magnitude + 1e-8).cpu().numpy()
    
    freq = np.arange(magnitude.shape[0]) * sample_rate / (magnitude.shape[0] * 2)
    time = np.arange(magnitude.shape[1]) * hop_size / sample_rate
    
    plt.figure(figsize=(12, 6))
    plt.imshow(magnitude, aspect='auto', origin='lower', 
               extent=[time[0], time[-1], freq[0], freq[-1]], cmap=cmap)
    plt.colorbar(label='Magnitude (dB)')
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_mask(mask: torch.Tensor, save_path: Optional[str] = None,
              title: str = "Mask", sample_rate: int = 16000, hop_size: int = 256):
    """
    绘制 mask 热力图
    """
    if mask.dim() == 3:
        mask = mask[0]
    
    mask = mask.cpu().numpy()
    
    freq = np.arange(mask.shape[0]) * sample_rate / (mask.shape[0] * 2)
    time = np.arange(mask.shape[1]) * hop_size / sample_rate
    
    plt.figure(figsize=(12, 4))
    plt.imshow(mask, aspect='auto', origin='lower',
               extent=[time[0], time[-1], freq[0], freq[-1]], cmap='viridis')
    plt.colorbar(label='Mask Value')
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_sources_comparison(mixture: torch.Tensor, 
                           est_sources: List[torch.Tensor],
                           ref_sources: List[torch.Tensor],
                           save_dir: str,
                           sample_rate: int = 16000):
    """
    绘制源分离对比图
    
    Args:
        mixture: 混合信号 [T]
        est_sources: 估计的源列表
        ref_sources: 参考源列表
        save_dir: 保存目录
        sample_rate: 采样率
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 混合信号
    plot_waveform(mixture, str(save_dir / "mixture.png"), "Mixture", sample_rate)
    
    # 各源对比
    for i, (est, ref) in enumerate(zip(est_sources, ref_sources)):
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        
        # 估计
        est_np = est.cpu().numpy()
        time = np.arange(len(est_np)) / sample_rate
        axes[0].plot(time, est_np, linewidth=0.5)
        axes[0].set_title(f"Source {i+1} - Estimated")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Amplitude")
        axes[0].grid(True, alpha=0.3)
        
        # 参考
        ref_np = ref.cpu().numpy()
        axes[1].plot(time, ref_np, linewidth=0.5)
        axes[1].set_title(f"Source {i+1} - Reference")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Amplitude")
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / f"source_{i+1}_comparison.png", dpi=150)
        plt.close()


def visualize_training(metrics_history: dict, save_path: str):
    """
    可视化训练过程指标
    
    Args:
        metrics_history: {'loss': [...], 'sisnr': [...], ...}
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    if 'loss' in metrics_history:
        axes[0, 0].plot(metrics_history['loss'])
        axes[0, 0].set_title("Training Loss")
        axes[0, 0].set_xlabel("Step")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].grid(True, alpha=0.3)
    
    # SI-SNR
    if 'sisnr' in metrics_history:
        axes[0, 1].plot(metrics_history['sisnr'])
        axes[0, 1].set_title("SI-SNR")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("dB")
        axes[0, 1].grid(True, alpha=0.3)
    
    # SDR
    if 'sdr' in metrics_history:
        axes[1, 0].plot(metrics_history['sdr'])
        axes[1, 0].set_title("SDR")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("dB")
        axes[1, 0].grid(True, alpha=0.3)
    
    # Learning Rate
    if 'lr' in metrics_history:
        axes[1, 1].plot(metrics_history['lr'])
        axes[1, 1].set_title("Learning Rate")
        axes[1, 1].set_xlabel("Step")
        axes[1, 1].set_ylabel("LR")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    # 测试
    waveform = torch.randn(16000)
    plot_waveform(waveform, "test_waveform.png", "Test")
    print("可视化测试完成")
