"""
推理模块 - 模型推理和音频重构
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union
import yaml


class InferenceEngine:
    """
    推理引擎
    加载模型并进行音频源分离
    """
    
    def __init__(self, config_path: str, checkpoint_path: str, device: str = 'cuda'):
        """
        初始化推理引擎
        
        Args:
            config_path: 配置文件路径
            checkpoint_path: 模型检查点路径
            device: 设备 ('cuda' 或 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.config = self._load_config(config_path)
        
        # 加载模型
        self.model = self._build_model()
        self._load_checkpoint(checkpoint_path)
        self.model.eval()
        self.model.to(self.device)
        
        # STFT 参数
        self.fft_size = self.config['model'].get('fft_size', 512)
        self.hop_size = self.config['model'].get('hop_size', 256)
        self.window = torch.hann_window(self.fft_size)
        
    def _load_config(self, config_path: str) -> dict:
        """加载配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _build_model(self):
        """构建模型"""
        from models.complex_mtass import ComplexMTASS
        model = ComplexMTASS(self.config['model'])
        return model
    
    def _load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    def _apply_stft(self, waveform: torch.Tensor) -> torch.Tensor:
        """应用 STFT"""
        # 确保在正确设备上
        window = self.window.to(waveform.device)
        
        stft = torch.stft(
            waveform,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            window=window,
            center=True,
            return_complex=True
        )
        
        # 转换为复数域表示 [F, T] -> [2F, T]
        real = stft.real
        imag = stft.imag
        
        # 拼接实部和虚部 (取前半部分频率)
        half_freq = self.fft_size // 2 + 1
        complex_tensor = torch.cat([real[:half_freq], imag[:half_freq]], dim=0)
        
        return complex_tensor
    
    def _apply_istft(self, complex_tensor: torch.Tensor, length: int) -> torch.Tensor:
        """应用 ISTFT"""
        window = self.window.to(complex_tensor.device)
        
        half_freq = self.fft_size // 2 + 1
        real = complex_tensor[:half_freq]
        imag = complex_tensor[half_freq:]
        
        stft = torch.complex(real, imag)
        
        waveform = torch.istft(
            stft,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            window=window,
            center=True,
            length=length
        )
        
        return waveform
    
    @torch.no_grad()
    def separate(self, audio_path: str, output_dir: Optional[str] = None) -> List[str]:
        """
        分离音频
        
        Args:
            audio_path: 输入音频路径
            output_dir: 输出目录
        
        Returns:
            输出的源音频路径列表
        """
        # 加载音频
        waveform, sr = torchaudio.load(audio_path)
        
        # 重采样
        if sr != self.config['data'].get('sample_rate', 16000):
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
            sr = 16000
        
        # 单声道
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # 转换为模型输入
        waveform = waveform.to(self.device)
        complex_input = self._apply_stft(waveform.squeeze(0))
        complex_input = complex_input.unsqueeze(0)  # [B, 2F, T]
        
        # 推理
        sources = self.model(complex_input)
        
        # 保存分离的音频
        if output_dir is None:
            output_dir = Path(audio_path).parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        output_paths = []
        source_names = ['speech', 'noise', 'music'][:len(sources)]
        
        for i, src in enumerate(sources):
            # 转换回时域
            src_waveform = self._apply_istft(src.squeeze(0), waveform.shape[-1])
            
            # 保存
            output_path = output_dir / f"{Path(audio_path).stem}_{source_names[i]}.wav"
            torchaudio.save(str(output_path), src_waveform.cpu().unsqueeze(0), 16000)
            output_paths.append(str(output_path))
        
        return output_paths
    
    @torch.no_grad()
    def separate_batch(self, audio_paths: List[str], output_dir: str) -> List[List[str]]:
        """批量分离"""
        results = []
        for audio_path in audio_paths:
            outputs = self.separate(audio_path, output_dir)
            results.append(outputs)
        return results


def quick_inference(audio_path: str, config_path: str = "configs/config.yaml", 
                    checkpoint: str = "checkpoints/best.pt", output_dir: str = "outputs"):
    """
    快速推理接口
    
    Args:
        audio_path: 输入音频路径
        config_path: 配置文件路径
        checkpoint: 检查点路径
        output_dir: 输出目录
    """
    engine = InferenceEngine(config_path, checkpoint)
    output_paths = engine.separate(audio_path, output_dir)
    
    print(f"\n分离完成! 输出文件:")
    for path in output_paths:
        print(f"  - {path}")
    
    return output_paths


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Audio Source Separation Inference')
    parser.add_argument('--audio', type=str, required=True, help='Input audio file')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pt')
    parser.add_argument('--output', type=str, default='outputs')
    args = parser.parse_args()
    
    quick_inference(args.audio, args.config, args.checkpoint, args.output)
