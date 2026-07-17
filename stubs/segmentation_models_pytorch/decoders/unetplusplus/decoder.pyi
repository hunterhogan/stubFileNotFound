import torch
import torch.nn as nn
from _typeshed import Incomplete
from typing import Any, Sequence

class DecoderBlock(nn.Module):
    conv1: Incomplete
    attention1: Incomplete
    conv2: Incomplete
    attention2: Incomplete
    interpolation_mode: Incomplete
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, use_norm: bool | str | dict[str, Any] = 'batchnorm', attention_type: str | None = None, interpolation_mode: str = 'nearest') -> None: ...
    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor: ...

class CenterBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, use_norm: bool | str | dict[str, Any] = 'batchnorm') -> None: ...

class UnetPlusPlusDecoder(nn.Module):
    in_channels: Incomplete
    skip_channels: Incomplete
    out_channels: Incomplete
    center: Incomplete
    blocks: Incomplete
    depth: Incomplete
    def __init__(self, encoder_channels: Sequence[int], decoder_channels: Sequence[int], n_blocks: int = 5, use_norm: bool | str | dict[str, Any] = 'batchnorm', attention_type: str | None = None, interpolation_mode: str = 'nearest', center: bool = False) -> None: ...
    def forward(self, features: list[torch.Tensor]) -> torch.Tensor: ...
