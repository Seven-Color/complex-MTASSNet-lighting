"""Complex-MTASS 模型模块"""
from .complex_mtass_optimized import OptimizedComplexMTASS as ComplexMTASS
from .complex_mtass_optimized import count_parameters, OptimizedMSResBlock, SEBlock, MultiHeadSelfAttention

# 兼容旧版
from .complex_mtass import MSResBlock, GTCNBlock

__all__ = ['ComplexMTASS', 'count_parameters', 'MSResBlock', 'GTCNBlock', 'SEBlock', 'MultiHeadSelfAttention']
