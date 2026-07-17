import torch
import torch.nn as nn
from _typeshed import Incomplete
from typing import Literal

class Conv3x3GNReLU(nn.Module):
    upsample: Incomplete
    block: Incomplete
    def __init__(self, in_channels: int, out_channels: int, upsample: bool = False) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class FPNBlock(nn.Module):
    skip_conv: Incomplete
    interpolation_mode: Incomplete
    def __init__(self, pyramid_channels: int, skip_channels: int, interpolation_mode: str = 'nearest') -> None: ...
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor: ...

class SegmentationBlock(nn.Module):
    block: Incomplete
    def __init__(self, in_channels: int, out_channels: int, n_upsamples: int = 0) -> None: ...
    def forward(self, x): ...

class MergeBlock(nn.Module):
    policy: Incomplete
    def __init__(self, policy: Literal['add', 'cat']) -> None: ...
    def forward(self, x: list[torch.Tensor]) -> torch.Tensor: ...

class FPNDecoder(nn.Module):
    out_channels: Incomplete
    p5: Incomplete
    p4: Incomplete
    p3: Incomplete
    p2: Incomplete
    seg_blocks: Incomplete
    merge: Incomplete
    dropout: Incomplete
    def __init__(self, encoder_channels: list[int], encoder_depth: int = 5, pyramid_channels: int = 256, segmentation_channels: int = 128, dropout: float = 0.2, merge_policy: Literal['add', 'cat'] = 'add', interpolation_mode: str = 'nearest') -> None: ...
    def forward(self, features: list[torch.Tensor]) -> torch.Tensor: ...
