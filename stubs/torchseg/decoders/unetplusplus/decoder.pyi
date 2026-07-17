import torch.nn as nn
from ...base import modules as modules
from _typeshed import Incomplete

class DecoderBlock(nn.Module):
    conv1: Incomplete
    attention1: Incomplete
    conv2: Incomplete
    attention2: Incomplete
    def __init__(self, in_channels, skip_channels, out_channels, use_batchnorm: bool = True, attention_type=None) -> None: ...
    def forward(self, x, skip=None): ...

class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm: bool = True) -> None: ...

class UnetPlusPlusDecoder(nn.Module):
    in_channels: Incomplete
    skip_channels: Incomplete
    out_channels: Incomplete
    center: Incomplete
    blocks: Incomplete
    depth: Incomplete
    def __init__(self, encoder_channels, decoder_channels, n_blocks: int = 5, use_batchnorm: bool = True, attention_type=None, center: bool = False) -> None: ...
    def forward(self, *features): ...
