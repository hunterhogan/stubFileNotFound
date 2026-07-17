import torch
import torch.nn as nn
from _typeshed import Incomplete

class CatBnAct(nn.Module):
    bn: Incomplete
    act: Incomplete
    def __init__(self, in_chs, activation_fn=...) -> None: ...
    def forward(self, x): ...

class BnActConv2d(nn.Module):
    bn: Incomplete
    act: Incomplete
    conv: Incomplete
    def __init__(self, in_chs, out_chs, kernel_size, stride, padding: int = 0, groups: int = 1, activation_fn=...) -> None: ...
    def forward(self, x): ...

class InputBlock(nn.Module):
    conv: Incomplete
    bn: Incomplete
    act: Incomplete
    pool: Incomplete
    def __init__(self, num_init_features, kernel_size: int = 7, padding: int = 3, activation_fn=...) -> None: ...
    def forward(self, x): ...

class DualPathBlock(nn.Module):
    num_1x1_c: Incomplete
    inc: Incomplete
    b: Incomplete
    key_stride: int
    has_proj: bool
    c1x1_w_s2: Incomplete
    c1x1_w_s1: Incomplete
    c1x1_a: Incomplete
    c3x3_b: Incomplete
    c1x1_c: Incomplete
    c1x1_c1: Incomplete
    c1x1_c2: Incomplete
    def __init__(self, in_chs, num_1x1_a, num_3x3_b, num_1x1_c, inc, groups, block_type: str = 'normal', b: bool = False) -> None: ...
    def forward(self, x): ...

class DPN(nn.Module):
    test_time_pool: Incomplete
    b: Incomplete
    features: Incomplete
    last_linear: Incomplete
    def __init__(self, small: bool = False, num_init_features: int = 64, k_r: int = 96, groups: int = 32, b: bool = False, k_sec=(3, 4, 20, 3), inc_sec=(16, 32, 24, 128), num_classes: int = 1000, test_time_pool: bool = False) -> None: ...
    def logits(self, features): ...
    def forward(self, input): ...

def pooling_factor(pool_type: str = 'avg'): ...
def adaptive_avgmax_pool2d(x, pool_type: str = 'avg', padding: int = 0, count_include_pad: bool = False):
    """Selectable global pooling function with dynamic input kernel size"""

class AdaptiveAvgMaxPool2d(torch.nn.Module):
    """Selectable global pooling layer with dynamic input kernel size"""
    output_size: Incomplete
    pool_type: Incomplete
    pool: Incomplete
    def __init__(self, output_size: int = 1, pool_type: str = 'avg') -> None: ...
    def forward(self, x): ...
    def factor(self): ...
