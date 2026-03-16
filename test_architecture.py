"""Test the new architecture"""
import torch
import sys
sys.path.insert(0, '.')

from src.core.config import Config
from src.models.complex_mtass import ComplexMTASSModel

# Load config
config = Config.from_yaml('configs/config.yaml')

# Create model
model = ComplexMTASSModel(config.model)

# Test forward
x = torch.randn(2, 514, 100)
outputs = model(x)

print("Input:", x.shape)
print("Num outputs:", len(outputs))
for i, out in enumerate(outputs):
    print(f"Output {i+1}:", out.shape)

print("Parameters:", model.get_num_parameters())
