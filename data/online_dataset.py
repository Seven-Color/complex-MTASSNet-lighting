"""
在线数据加载器
- 边训练边生成混合音频
- 支持实时STFT变换
- 数据增强
"""

import os
import random
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import Dict, List, Optional, Tuple, Any
import yaml
import scipy.signal as signal
from pathlib import Path


class STFT:
    """短时傅里叶变换"""
    
    def __init__(self, config: Dict[str, Any]):
        stft_cfg = config.get('stft', {})
        
        self.fft_size = stft_cfg.get('fft_size', 512)
        self.hop_size = stft_cfg.get('hop_size', 256)
        self.window = stft_cfg.get('window', 'hann')
        self.center = stft_cfg.get('center', True)
        self.normalized = stft_cfg.get('normalized', False)
        self.onesided = stft_cfg.get('onesided', True)
        
        if self.window == 'hann':
            self.win = torch.hann_window(self.fft_size)
        elif self.window == 'hamming':
            self.win = torch.hamming_window(self.fft_size)
        else:
            self.win = torch.ones(self.fft_size)
    
    def __call__(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        stft_result = torch.stft(
            audio,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.fft_size,
            window=self.win,
            center=self.center,
            normalized=self.normalized,
            onesided=self.onesided,
            return_complex=True
        )
        
        real = stft_result.real
        imag = stft_result.imag
        
        return real, imag
    
    def istft(self, real: torch.Tensor, imag: torch.Tensor, 
              length: Optional[int] = None) -> torch.Tensor:
        complex_spec = torch.complex(real, imag)
        
        audio = torch.istft(
            complex_spec,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.fft_size,
            window=self.win,
            center=self.center,
            normalized=self.normalized,
            onesided=self.onesided,
            length=length
        )
        
        return audio


class AudioSourceDataset(Dataset):
    """音频源数据集"""
    
    def __init__(self, audio_dir: str, sample_rate: int = 16000, 
                 audio_length: Optional[float] = None):
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.audio_length = audio_length
        self.audio_length_samples = int(audio_length * sample_rate) if audio_length else None
        
        self.file_paths = list(self.audio_dir.glob("**/*.wav"))
        self.file_paths.extend(list(self.audio_dir.glob("**/*.flac")))
        self.file_paths.extend(list(self.audio_dir.glob("**/*.mp3")))
        
        if len(self.file_paths) == 0:
            raise ValueError(f"没有找到音频文件: {audio_dir}")
        
        print(f"加载 {len(self.file_paths)} 个音频文件 from {audio_dir}")
    
    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        file_path = self.file_paths[idx]
        
        try:
            waveform, sr = torchaudio.load(file_path)
        except:
            return torch.zeros(self.audio_length_samples) if self.audio_length_samples else torch.zeros(16000)
        
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        waveform = waveform.squeeze(0)
        
        if self.audio_length_samples:
            if waveform.shape[0] > self.audio_length_samples:
                start = random.randint(0, waveform.shape[0] - self.audio_length_samples)
                waveform = waveform[start:start + self.audio_length_samples]
            elif waveform.shape[0] < self.audio_length_samples:
                padding = self.audio_length_samples - waveform.shape[0]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        return waveform


class OnlineMixtureDataset(IterableDataset):
    """在线混合数据集"""
    
    def __init__(self, config: Dict[str, Any], split: str = 'train'):
        self.config = config
        self.split = split
        
        data_cfg = config.get('data', {})
        aug_cfg = data_cfg.get('augmentation', {})
        mix_cfg = data_cfg.get('mixing', {})
        
        self.sample_rate = data_cfg.get('sample_rate', 16000)
        self.audio_length = data_cfg.get('audio_length', 4.0)
        self.audio_length_samples = int(self.audio_length * self.sample_rate)
        
        self.speech_dataset = None
        self.noise_dataset = None
        self.music_dataset = None
        
        speech_dir = data_cfg.get('speech_dir', '')
        noise_dir = data_cfg.get('noise_dir', '')
        music_dir = data_cfg.get('music_dir', '')
        
        if speech_dir and os.path.exists(speech_dir):
            self.speech_dataset = AudioSourceDataset(speech_dir, self.sample_rate, self.audio_length)
        
        if noise_dir and os.path.exists(noise_dir):
            self.noise_dataset = AudioSourceDataset(noise_dir, self.sample_rate, self.audio_length)
        
        if music_dir and os.path.exists(music_dir):
            self.music_dataset = AudioSourceDataset(music_dir, self.sample_rate, self.audio_length)
        
        if not any([self.speech_dataset, self.noise_dataset, self.music_dataset]):
            raise ValueError("没有可用的音频源数据!")
        
        self.augmentation_enabled = aug_cfg.get('enabled', True)
        self.pitch_shift_range = aug_cfg.get('pitch_shift', [-2, 2])
        self.time_stretch_range = aug_cfg.get('time_stretch', [0.9, 1.1])
        self.add_noise = aug_cfg.get('add_noise', True)
        self.noise_level_range = aug_cfg.get('noise_level', [-40, -20])
        
        self.snr_range = mix_cfg.get('snr_range', [-5, 20])
        
        self.stft = STFT(config)
        
        self.is_train = (split == 'train')
    
    def __iter__(self):
        while True:
            sources = []
            source_types = []
            
            if self.speech_dataset:
                speech_idx = random.randint(0, len(self.speech_dataset) - 1)
                speech = self.speech_dataset[speech_idx]
                if self.augmentation_enabled and self.is_train:
                    speech = self.augment_audio(speech)
                sources.append(speech)
                source_types.append('speech')
            
            available_sources = []
            if self.noise_dataset:
                available_sources.append(('noise', self.noise_dataset))
            if self.music_dataset:
                available_sources.append(('music', self.music_dataset))
            
            if available_sources and self.is_train:
                num_extra = random.randint(1, min(len(available_sources), 2))
                selected = random.sample(available_sources, num_extra)
                
                for source_type, dataset in selected:
                    idx = random.randint(0, len(dataset) - 1)
                    source = dataset[idx]
                    if self.augmentation_enabled:
                        source = self.augment_audio(source)
                    sources.append(source)
                    source_types.append(source_type)
            
            mixture, source_ri = self.mix_sources(sources, source_types)
            
            mix_real, mix_imag = self.stft(mixture)
            mix_features = torch.cat([mix_real, mix_imag], dim=0)
            
            sources_ri = []
            for src in sources:
                src_real, src_imag = self.stft(src)
                src_feat = torch.cat([src_real, src_imag], dim=0)
                sources_ri.append(src_feat)
            
            while len(sources_ri) < 3:
                sources_ri.append(torch.zeros_like(sources_ri[0]))
            
            yield {
                'mixture': mix_features,
                'sources': torch.stack(sources_ri),
                'source_types': source_types
            }
    
    def mix_sources(self, sources: List[torch.Tensor], 
                    source_types: List[str]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if not sources:
            return torch.zeros(self.audio_length_samples), []
        
        normalized_sources = []
        for src in sources:
            src = src - src.mean()
            if src.std() > 1e-5:
                src = src / src.std()
            normalized_sources.append(src)
        
        snr_db = random.uniform(*self.snr_range)
        snr_linear = 10 ** (snr_db / 20)
        
        if len(normalized_sources) == 1:
            mixture = normalized_sources[0]
        else:
            ref = normalized_sources[0]
            mixture = ref.clone()
            
            for i in range(1, len(normalized_sources)):
                other = normalized_sources[i]
                
                ref_power = (ref ** 2).mean()
                other_power = (other ** 2).mean()
                
                if other_power > 1e-8:
                    scale = torch.sqrt(ref_power / other_power) / snr_linear
                else:
                    scale = 0
                
                mixture = mixture + other * scale
        
        mixture = mixture / (mixture.abs().max() + 1e-8)
        
        return mixture, normalized_sources
    
    def augment_audio(self, audio: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.5:
            scale = random.uniform(0.5, 1.5)
            audio = audio * scale
        
        if self.add_noise and random.random() < 0.3:
            noise_level_db = random.uniform(*self.noise_level_range)
            noise_level = 10 ** (noise_level_db / 20)
            noise = torch.randn_like(audio) * noise_level
            audio = audio + noise
        
        return audio


def get_dataloader(config: Dict[str, Any], split: str = 'train',
                   batch_size: int = 8, num_workers: int = 4) -> DataLoader:
    dataset = OnlineMixtureDataset(config, split=split)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=config.get('data', {}).get('pin_memory', True),
        prefetch_factor=config.get('data', {}).get('prefetch_factor', 2),
        persistent_workers=config.get('data', {}).get('persistent_workers', True) if num_workers > 0 else False
    )
    
    return loader


def collate_fn(batch):
    max_len = max(x['mixture'].shape[-1] for x in batch)
    
    for item in batch:
        diff = max_len - item['mixture'].shape[-1]
        if diff > 0:
            item['mixture'] = torch.nn.functional.pad(item['mixture'], (0, diff))
            for i in range(len(item['sources'])):
                item['sources'][i] = torch.nn.functional.pad(item['sources'][i], (0, diff))
    
    return {
        'mixture': torch.stack([x['mixture'] for x in batch]),
        'sources': torch.stack([x['sources'] for x in batch]),
        'source_types': [x['source_types'] for x in batch]
    }


if __name__ == "__main__":
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    stft = STFT(config)
    audio = torch.randn(16000)
    real, imag = stft(audio)
    print(f"STFT输出: real={real.shape}, imag={imag.shape}")
