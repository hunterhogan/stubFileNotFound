import torch.nn as nn
from _typeshed import Incomplete

class Conv3x3GNReLU(nn.Module):
    upsample: Incomplete
    block: Incomplete
    def __init__(self, in_channels, out_channels, upsample: bool = False) -> None: ...
    def forward(self, x): ...

class FPNBlock(nn.Module):
    skip_conv: Incomplete
    def __init__(self, pyramid_channels, skip_channels) -> None: ...
    def forward(self, x, skip=None): ...

class SegmentationBlock(nn.Module):
    block: Incomplete
    def __init__(self, in_channels, out_channels, n_upsamples: int = 0) -> None: ...
    def forward(self, x): ...

class MergeBlock(nn.Module):
    policy: Incomplete
    def __init__(self, policy) -> None: ...
    def forward(self, x): ...

class FPNDecoder(nn.Module):
    out_channels: Incomplete
    p5: Incomplete
    p4: Incomplete
    p3: Incomplete
    p2: Incomplete
    seg_blocks: Incomplete
    merge: Incomplete
    dropout: Incomplete
    def __init__(self, encoder_channels, encoder_depth: int = 5, pyramid_channels: int = 256, segmentation_channels: int = 128, dropout: float = 0.2, merge_policy: str = 'add') -> None: ...
    def forward(self, *features): ...
