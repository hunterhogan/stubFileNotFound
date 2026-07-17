import torch
import torch.nn as nn
from _typeshed import Incomplete
from segmentation_models_pytorch.base import modules as modules
from typing import Any

class PSPBlock(nn.Module):
    pool: Incomplete
    def __init__(self, in_channels: int, out_channels: int, pool_size: int, use_norm: bool | str | dict[str, Any] = 'batchnorm') -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class PSPModule(nn.Module):
    blocks: Incomplete
    def __init__(self, in_channels: int, sizes: tuple[int, ...] = (1, 2, 3, 6), use_norm: bool | str | dict[str, Any] = 'batchnorm') -> None: ...
    def forward(self, x): ...

class PSPDecoder(nn.Module):
    psp: Incomplete
    conv: Incomplete
    dropout: Incomplete
    def __init__(self, encoder_channels: list[int], use_norm: bool | str | dict[str, Any] = 'batchnorm', out_channels: int = 512, dropout: float = 0.2) -> None: ...
    def forward(self, features: list[torch.Tensor]) -> torch.Tensor: ...
