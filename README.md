# Complex-MTASSNet-lighting

Complex-domain Multi-Task Audio Source Separation - 简化重构版

## 功能特性

- **简化模型架构**: MS_ResBlock 从 15 个减少到 6 个
- **变量参数化**: 通过 config.yaml 统一配置
- **在线数据生成**: 边训练边混合音频，无需预先制作数据集
- **STFT 内置**: 在训练中实时计算复数域特征
- **现代训练器**: 支持 DDP 分布式、混合精度、早停

## 模型架构

```
输入 (514, T)
    │
    ▼
Stage 1: Encoder → MS_ResBlocks(6) → Decoder
    │
    ▼
Mask 生成 (3个源)
    │
    ▼
Stage 2: 残差修复 (可选)
    │
    ▼
输出: speech, noise, music
```

## 快速开始

```bash
# 安装依赖
pip install torch torchaudio pyyaml tqdm tensorboard

# 准备数据
# 在 dataset/speech, dataset/noise, dataset/music 目录下放置原始音频文件

# 训练
python train.py --config configs/config.yaml

# 或使用自定义参数
python train.py --config configs/config.yaml --epochs 50 --batch_size 4
```

## 配置说明

```yaml
# configs/config.yaml
model:
  stage1:
    hidden_channels: 1024
    ms_resblock_dilations: [1, 3, 5, 7, 11, 13]
  stage2:
    enabled: true
    repeats: 3

training:
  epochs: 100
  batch_size: 8
  learning_rate: 0.001
  use_amp: true

data:
  sample_rate: 16000
  audio_length: 4.0
  speech_dir: "./dataset/speech"
  noise_dir: "./dataset/noise"
  music_dir: "./dataset/music"
```

## 参数量

约 8.36M 参数

## 原始论文

[Complex-MTASSNet: Complex-domain Multi-Task Audio Source Separation](https://github.com/Windstudent/Complex-MTASSNet)

## License

MIT
