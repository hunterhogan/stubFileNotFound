import torch
from _typeshed import Incomplete
from collections.abc import Iterable, Sequence
from torch import nn
from typing import Literal

__all__ = ['DeepLabV3Decoder', 'DeepLabV3PlusDecoder']

class DeepLabV3Decoder(nn.Module):
    aspp: Incomplete
    conv: Incomplete
    bn: Incomplete
    relu: Incomplete
    def __init__(self, in_channels: int, out_channels: int, atrous_rates: Iterable[int], aspp_separable: bool, aspp_dropout: float) -> None: ...
    def forward(self, features: list[torch.Tensor]) -> torch.Tensor: ...

class DeepLabV3PlusDecoder(nn.Module):
    aspp: Incomplete
    up: Incomplete
    block1: Incomplete
    block2: Incomplete
    def __init__(self, encoder_channels: Sequence[int], encoder_depth: Literal[3, 4, 5], out_channels: int, atrous_rates: Iterable[int], output_stride: Literal[8, 16], aspp_separable: bool, aspp_dropout: float) -> None: ...
    def forward(self, features: list[torch.Tensor]) -> torch.Tensor: ...

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None: ...

class ASPPSeparableConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None: ...

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class ASPP(nn.Module):
    convs: Incomplete
    project: Incomplete
    def __init__(self, in_channels: int, out_channels: int, atrous_rates: Iterable[int], separable: bool, dropout: float) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class SeparableConv2d(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = True) -> None: ...
