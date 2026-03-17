"""
评估模块 - 模型评估和指标计算
"""

import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
import json

from src.utils.metrics import compute_metrics_batch, MetricsTracker
from src.utils.visualization import plot_sources_comparison


class Evaluator:
    """
    模型评估器
    """
    
    def __init__(self, model, config: dict, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()
        self.config = config
        
        # STFT 参数
        self.fft_size = config['model'].get('fft_size', 512)
        self.hop_size = config['model'].get('hop_size', 256)
        
        # 评估指标
        self.metrics_tracker = MetricsTracker()
    
    def _apply_stft(self, waveform: torch.Tensor) -> torch.Tensor:
        """STFT 变换"""
        window = torch.hann_window(self.fft_size).to(waveform.device)
        
        stft = torch.stft(
            waveform,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            window=window,
            center=True,
            return_complex=True
        )
        
        half_freq = self.fft_size // 2 + 1
        real = stft.real[:, :half_freq, :]
        imag = stft.imag[:, :half_freq, :]
        
        return torch.cat([real, imag], dim=1)
    
    @torch.no_grad()
    def evaluate_dataloader(self, dataloader, save_dir: str = None):
        """
        评估数据加载器
        
        Args:
            dataloader: 数据加载器
            save_dir: 可视化保存目录
        
        Returns:
            评估指标字典
        """
        self.metrics_tracker.reset()
        
        all_sources = []
        all_refs = []
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            mixture = batch['mixture'].to(self.device)
            sources = batch['sources'].to(self.device)
            
            # 转换到频域
            mix_stft = self._apply_stft(mixture)
            src_stft = self._apply_stft(sources)
            
            # 推理
            est_sources = self.model(mix_stft)
            
            # 堆叠
            if isinstance(est_sources, tuple):
                est_sources = torch.stack(est_sources, dim=1)
            
            # 计算指标
            # 需要从频域转回时域计算 SI-SNR
            for b in range(est_sources.shape[0]):
                for n in range(est_sources.shape[1]):
                    est = est_sources[b, n]
                    ref = sources[b, n]
                    
                    # 使用频域近似计算
                    est_mag = torch.sqrt(est[:257] ** 2 + est[257:] ** 2 + 1e-8)
                    ref_mag = torch.sqrt(ref[:257] ** 2 + ref[257:] ** 2 + 1e-8)
                    
                    self.metrics_tracker.update(est_mag, ref_mag)
        
        metrics = self.metrics_tracker.get_avg()
        
        print("\n" + "=" * 50)
        print("Evaluation Results")
        print("=" * 50)
        for k, v in metrics.items():
            print(f"{k.upper()}: {v:.2f} dB")
        print("=" * 50)
        
        return metrics
    
    @torch.no_grad()
    def evaluate_on_audio(self, audio_path: str, save_dir: str = None):
        """
        在单个音频上评估
        
        Args:
            audio_path: 音频路径
            save_dir: 保存目录
        """
        waveform, sr = torchaudio.load(audio_path)
        
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # 这里需要真实参考信号，实际使用中可能需要数据集
        # 暂时返回频域表示
        stft = self._apply_stft(waveform)
        sources = self.model(stft)
        
        return sources


def evaluate_checkpoint(checkpoint_path: str, config_path: str, dataloader, 
                       save_dir: str = "eval_results"):
    """
    评估检查点
    
    Args:
        checkpoint_path: 检查点路径
        config_path: 配置文件路径
        dataloader: 数据加载器
        save_dir: 保存目录
    """
    import yaml
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 加载模型
    from models.complex_mtass import ComplexMTASS
    model = ComplexMTASS(config['model'])
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # 评估
    evaluator = Evaluator(model, config)
    metrics = evaluator.evaluate_dataloader(dataloader, save_dir)
    
    # 保存结果
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = save_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Model')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--save_dir', type=str, default='eval_results')
    args = parser.parse_args()
    
    # TODO: 创建实际的评估数据加载器
    print("评估模块就绪，请提供数据加载器进行完整评估")
