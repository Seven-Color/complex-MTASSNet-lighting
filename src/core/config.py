"""
配置管理模块
使用单例模式 + 不可变配置
"""

from typing import Any, Dict, Optional
from pathlib import Path
import yaml
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    """模型配置 - 不可变"""
    name: str = "ComplexMTASS"
    fft_size: int = 512
    hop_size: int = 256
    window: str = "hann"
    num_sources: int = 3
    
    # Stage 1
    hidden_channels: int = 1024
    ms_resblock_dilations: list = field(default_factory=lambda: [1, 3, 5, 7, 11, 13])
    dropout: float = 0.2
    
    # Stage 2
    stage2_enabled: bool = True
    stage2_repeats: int = 3
    stage2_blocks: int = 8
    stage2_hidden: int = 256


@dataclass(frozen=True)
class TrainingConfig:
    """训练配置 - 不可变"""
    seed: int = 42
    epochs: int = 100
    batch_size: int = 8
    gradient_accumulation: int = 4
    
    # Optimizer
    optimizer: str = "adamw"
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    
    # Scheduler
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Loss weights
    loss_mse: float = 1.0
    loss_snr: float = 0.5
    loss_sisnr: float = 0.5
    
    # Training strategy
    use_amp: bool = True
    clip_grad_norm: float = 5.0
    patience: int = 15
    save_best: bool = True


@dataclass(frozen=True)
class DataConfig:
    """数据配置 - 不可变"""
    sample_rate: int = 16000
    audio_length: float = 4.0
    
    # Directories
    speech_dir: str = "./dataset/speech"
    noise_dir: str = "./dataset/noise"
    music_dir: str = "./dataset/music"
    
    # Augmentation
    aug_enabled: bool = True
    aug_pitch_shift: tuple = (-2, 2)
    aug_time_stretch: tuple = (0.9, 1.1)
    aug_add_noise: bool = True
    aug_noise_level: tuple = (-40, -20)
    
    # Mixing
    mix_snr_range: tuple = (-5, 20)
    
    # DataLoader
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True
    pin_memory: bool = True


class Config:
    """
    全局配置管理器
    工厂方法 + 单例模式
    """
    
    _instance: Optional['Config'] = None
    _model: Optional[ModelConfig] = None
    _training: Optional[TrainingConfig] = None
    _data: Optional[DataConfig] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """从 YAML 文件加载配置"""
        config = cls()
        
        with open(path, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
        
        # 解析模型配置
        model_cfg = yaml_config.get('model', {})
        config._model = ModelConfig(
            name=model_cfg.get('name', 'ComplexMTASS'),
            fft_size=model_cfg.get('fft_size', 512),
            hop_size=model_cfg.get('hop_size', 256),
            window=model_cfg.get('window', 'hann'),
            num_sources=model_cfg.get('num_sources', 3),
            hidden_channels=model_cfg.get('stage1', {}).get('hidden_channels', 1024),
            ms_resblock_dilations=model_cfg.get('stage1', {}).get('ms_resblock_dilations', [1, 3, 5, 7, 11, 13]),
            dropout=model_cfg.get('stage1', {}).get('dropout', 0.2),
            stage2_enabled=model_cfg.get('stage2', {}).get('enabled', True),
            stage2_repeats=model_cfg.get('stage2', {}).get('repeats', 3),
            stage2_blocks=model_cfg.get('stage2', {}).get('num_blocks', 8),
            stage2_hidden=model_cfg.get('stage2', {}).get('hidden_channels', 256),
        )
        
        # 解析训练配置
        train_cfg = yaml_config.get('training', {})
        config._training = TrainingConfig(
            seed=train_cfg.get('seed', 42),
            epochs=train_cfg.get('epochs', 100),
            batch_size=train_cfg.get('batch_size', 8),
            gradient_accumulation=train_cfg.get('gradient_accumulation', 4),
            optimizer=train_cfg.get('optimizer', 'adamw'),
            learning_rate=train_cfg.get('learning_rate', 0.001),
            weight_decay=train_cfg.get('weight_decay', 0.01),
            betas=tuple(train_cfg.get('betas', [0.9, 0.999])),
            scheduler=train_cfg.get('scheduler', 'cosine'),
            warmup_epochs=train_cfg.get('warmup_epochs', 5),
            min_lr=train_cfg.get('min_lr', 1e-6),
            loss_mse=train_cfg.get('loss', {}).get('mse', 1.0),
            loss_snr=train_cfg.get('loss', {}).get('snr', 0.5),
            loss_sisnr=train_cfg.get('loss', {}).get('sisnr', 0.5),
            use_amp=train_cfg.get('use_amp', True),
            clip_grad_norm=train_cfg.get('clip_grad_norm', 5.0),
            patience=train_cfg.get('patience', 15),
            save_best=train_cfg.get('save_best', True),
        )
        
        # 解析数据配置
        data_cfg = yaml_config.get('data', {})
        aug_cfg = data_cfg.get('augmentation', {})
        mix_cfg = data_cfg.get('mixing', {})
        config._data = DataConfig(
            sample_rate=data_cfg.get('sample_rate', 16000),
            audio_length=data_cfg.get('audio_length', 4.0),
            speech_dir=data_cfg.get('speech_dir', './dataset/speech'),
            noise_dir=data_cfg.get('noise_dir', './dataset/noise'),
            music_dir=data_cfg.get('music_dir', './dataset/music'),
            aug_enabled=aug_cfg.get('enabled', True),
            aug_pitch_shift=tuple(aug_cfg.get('pitch_shift', [-2, 2])),
            aug_time_stretch=tuple(aug_cfg.get('time_stretch', [0.9, 1.1])),
            aug_add_noise=aug_cfg.get('add_noise', True),
            aug_noise_level=tuple(aug_cfg.get('noise_level', [-40, -20])),
            mix_snr_range=tuple(mix_cfg.get('snr_range', [-5, 20])),
            num_workers=data_cfg.get('num_workers', 4),
            prefetch_factor=data_cfg.get('prefetch_factor', 2),
            persistent_workers=data_cfg.get('persistent_workers', True),
            pin_memory=data_cfg.get('pin_memory', True),
        )
        
        return config
    
    @property
    def model(self) -> ModelConfig:
        return self._model
    
    @property
    def training(self) -> TrainingConfig:
        return self._training
    
    @property
    def data(self) -> DataConfig:
        return self._data
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'model': self._model.__dict__ if self._model else {},
            'training': self._training.__dict__ if self._training else {},
            'data': self._data.__dict__ if self._data else {},
        }
