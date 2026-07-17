import torch
import torch.nn as nn
from _typeshed import Incomplete
from typing import Any

class PABBlock(nn.Module):
    pab_channels: Incomplete
    in_channels: Incomplete
    top_conv: Incomplete
    center_conv: Incomplete
    bottom_conv: Incomplete
    map_softmax: Incomplete
    out_conv: Incomplete
    def __init__(self, in_channels: int, pab_channels: int = 64) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class MFABBlock(nn.Module):
    hl_conv: Incomplete
    SE_ll: Incomplete
    SE_hl: Incomplete
    conv1: Incomplete
    conv2: Incomplete
    interpolation_mode: Incomplete
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, interpolation_mode: str = 'nearest', use_norm: bool | str | dict[str, Any] = 'batchnorm', reduction: int = 16) -> None: ...
    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor: ...

class DecoderBlock(nn.Module):
    conv1: Incomplete
    conv2: Incomplete
    interpolation_mode: Incomplete
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, interpolation_mode: str = 'nearest', use_norm: bool | str | dict[str, Any] = 'batchnorm') -> None: ...
    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor: ...

class MAnetDecoder(nn.Module):
    center: Incomplete
    blocks: Incomplete
    def __init__(self, encoder_channels: list[int], decoder_channels: list[int], n_blocks: int = 5, reduction: int = 16, use_norm: bool | str | dict[str, Any] = 'batchnorm', pab_channels: int = 64, interpolation_mode: str = 'nearest') -> None: ...
    def forward(self, features: list[torch.Tensor]) -> torch.Tensor: ...
