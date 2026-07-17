import torch.nn as nn
from ...base import modules as modules
from _typeshed import Incomplete

class PSPBlock(nn.Module):
    pool: Incomplete
    def __init__(self, in_channels, out_channels, pool_size, use_bathcnorm: bool = True) -> None: ...
    def forward(self, x): ...

class PSPModule(nn.Module):
    blocks: Incomplete
    def __init__(self, in_channels, sizes=(1, 2, 3, 6), use_bathcnorm: bool = True) -> None: ...
    def forward(self, x): ...

class PSPDecoder(nn.Module):
    psp: Incomplete
    conv: Incomplete
    dropout: Incomplete
    def __init__(self, encoder_channels, use_batchnorm: bool = True, out_channels: int = 512, dropout: float = 0.2) -> None: ...
    def forward(self, *features): ...
