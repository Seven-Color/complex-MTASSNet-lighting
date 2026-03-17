"""
单元测试模块
"""

import unittest
import torch
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.complex_mtass import ComplexMTASS, MSResBlock, GTCNBlock, count_parameters
from src.utils.metrics import sisnr, sdr, sar, compute_metrics_batch
from src.utils.visualization import plot_waveform, plot_spectrogram


class TestModel(unittest.TestCase):
    """模型测试"""
    
    @classmethod
    def setUpClass(cls):
        cls.config = {
            'fft_size': 512,
            'num_sources': 3,
            'stage1': {
                'hidden_channels': 256,  # 小一点用于测试
                'ms_resblock_dilations': [1, 3, 5],
                'dropout': 0.2
            },
            'stage2': {
                'enabled': True,
                'repeats': 2,
                'num_blocks': 4,
                'hidden_channels': 128,
                'dropout': 0.2
            }
        }
    
    def test_model_forward(self):
        """测试模型前向传播"""
        model = ComplexMTASS(self.config)
        x = torch.randn(2, 514, 100)  # [B, 2F, T]
        
        with torch.no_grad():
            outputs = model(x)
        
        self.assertEqual(len(outputs), 3)
        for out in outputs:
            self.assertEqual(out.shape, (2, 514, 100))
    
    def test_model_with_stages(self):
        """测试不同阶段的模型"""
        # Stage 2 关闭
        config = self.config.copy()
        config['stage2']['enabled'] = False
        
        model = ComplexMTASS(config)
        x = torch.randn(1, 514, 50)
        
        with torch.no_grad():
            outputs = model(x)
        
        self.assertEqual(len(outputs), 3)
    
    def test_get_num_params(self):
        """测试参数计数"""
        model = ComplexMTASS(self.config)
        num_params = count_parameters(model)
        
        self.assertGreater(num_params['total'], 0)
        self.assertEqual(num_params['total'], num_params['trainable'])


class TestMSResBlock(unittest.TestCase):
    """MSResBlock 测试"""
    
    def test_forward(self):
        """测试前向传播"""
        block = MSResBlock(channels=128, dilation=2, dropout=0.2)
        x = torch.randn(2, 128, 50)
        
        out = block(x)
        
        self.assertEqual(out.shape, x.shape)
    
    def test_dilation(self):
        """测试不同膨胀率"""
        for dilation in [1, 3, 5, 7]:
            block = MSResBlock(channels=64, dilation=dilation)
            x = torch.randn(1, 64, 100)
            out = block(x)
            self.assertEqual(out.shape, x.shape)


class TestGTCNBlock(unittest.TestCase):
    """GTCNBlock 测试"""
    
    def test_forward(self):
        """测试前向传播"""
        block = GTCNBlock(repeats=2, num_blocks=4, hidden_channels=64)
        x = torch.randn(2, 514, 50)
        
        out = block(x)
        
        self.assertEqual(out.shape, x.shape)


class TestMetrics(unittest.TestCase):
    """指标测试"""
    
    def test_sisnr(self):
        """测试 SI-SNR"""
        s_ref = torch.randn(2, 16000)
        s_est = s_ref + torch.randn(2, 16000) * 0.1
        
        score = sisnr(s_est, s_ref)
        
        self.assertIsInstance(score.item(), float)
        self.assertGreater(score.item(), 0)
    
    def test_sdr(self):
        """测试 SDR"""
        s_ref = torch.randn(16000)
        s_est = s_ref + torch.randn(16000) * 0.1
        
        score = sdr(s_est, s_ref)
        
        self.assertIsInstance(score.item(), float)
    
    def test_sar(self):
        """测试 SAR"""
        s_ref = torch.randn(16000)
        s_est = s_ref + torch.randn(16000) * 0.1
        
        score = sar(s_est, s_ref)
        
        self.assertIsInstance(score.item(), float)
    
    def test_batch_metrics(self):
        """测试批量指标"""
        est = torch.randn(2, 3, 16000)
        ref = torch.randn(2, 3, 16000)
        
        metrics = compute_metrics_batch(est, ref)
        
        self.assertIn('sisnr', metrics)
        self.assertIn('sdr', metrics)
        self.assertIn('sar', metrics)


class TestVisualization(unittest.TestCase):
    """可视化测试"""
    
    def test_plot_waveform(self):
        """测试波形图"""
        waveform = torch.randn(16000)
        
        # 只测试不报错
        try:
            plot_waveform(waveform, save_path=None)
        except Exception as e:
            self.fail(f"plot_waveform raised {e}")
    
    def test_plot_spectrogram(self):
        """测试频谱图"""
        # 复数域输入
        stft = torch.randn(257, 100)
        
        try:
            plot_spectrogram(stft, save_path=None)
        except Exception as e:
            self.fail(f"plot_spectrogram raised {e}")


def run_tests():
    """运行所有测试"""
    # 打印测试信息
    print("=" * 50)
    print("Running Unit Tests")
    print("=" * 50)
    
    # 发现并运行测试
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 打印结果
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✓ All tests passed!")
    else:
        print(f"✗ {len(result.failures)} failures, {len(result.errors)} errors")
    print("=" * 50)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_tests()
