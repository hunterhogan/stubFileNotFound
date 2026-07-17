import torch
import torch.nn as nn
from _typeshed import Incomplete
from collections.abc import Sequence
from typing import Literal

class ConvBnRelu(nn.Module):
    conv: Incomplete
    activation: Incomplete
    bn: Incomplete
    add_relu: Incomplete
    interpolate: Incomplete
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True, add_relu: bool = True, interpolate: bool = False) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class FPABlock(nn.Module):
    upscale_mode: Incomplete
    align_corners: bool
    branch1: Incomplete
    mid: Incomplete
    down1: Incomplete
    down2: Incomplete
    down3: Incomplete
    conv2: Incomplete
    conv1: Incomplete
    def __init__(self, in_channels: int, out_channels: int, upscale_mode: str = 'bilinear') -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class GAUBlock(nn.Module):
    interpolation_mode: Incomplete
    align_corners: Incomplete
    conv1: Incomplete
    conv2: Incomplete
    def __init__(self, in_channels: int, out_channels: int, interpolation_mode: str = 'bilinear') -> None: ...
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: low level feature
            y: high level feature
        """

class PANDecoder(nn.Module):
    fpa: Incomplete
    gau3: Incomplete
    gau2: Incomplete
    gau1: Incomplete
    def __init__(self, encoder_channels: Sequence[int], encoder_depth: Literal[3, 4, 5], decoder_channels: int, interpolation_mode: str = 'bilinear') -> None: ...
    def forward(self, features: list[torch.Tensor]) -> torch.Tensor: ...
