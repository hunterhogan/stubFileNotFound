import torch
import torch.nn as nn
from _typeshed import Incomplete
from segmentation_models_pytorch.base import modules as modules
from typing import Any

class TransposeX2(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, use_norm: bool | str | dict[str, Any] = 'batchnorm') -> None: ...

class DecoderBlock(nn.Module):
    block: Incomplete
    def __init__(self, in_channels: int, out_channels: int, use_norm: bool | str | dict[str, Any] = 'batchnorm') -> None: ...
    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor: ...

class LinknetDecoder(nn.Module):
    blocks: Incomplete
    def __init__(self, encoder_channels: list[int], prefinal_channels: int = 32, n_blocks: int = 5, use_norm: bool | str | dict[str, Any] = 'batchnorm') -> None: ...
    def forward(self, features: list[torch.Tensor]) -> torch.Tensor: ...
