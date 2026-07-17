import torch.nn as nn
from ...base import modules as modules
from _typeshed import Incomplete

class PAB(nn.Module):
    pab_channels: Incomplete
    in_channels: Incomplete
    top_conv: Incomplete
    center_conv: Incomplete
    bottom_conv: Incomplete
    map_softmax: Incomplete
    out_conv: Incomplete
    def __init__(self, in_channels, out_channels, pab_channels: int = 64) -> None: ...
    def forward(self, x): ...

class MFAB(nn.Module):
    hl_conv: Incomplete
    SE_ll: Incomplete
    SE_hl: Incomplete
    conv1: Incomplete
    conv2: Incomplete
    def __init__(self, in_channels, skip_channels, out_channels, use_batchnorm: bool = True, reduction: int = 16) -> None: ...
    def forward(self, x, skip=None): ...

class DecoderBlock(nn.Module):
    conv1: Incomplete
    conv2: Incomplete
    def __init__(self, in_channels, skip_channels, out_channels, use_batchnorm: bool = True) -> None: ...
    def forward(self, x, skip=None): ...

class MAnetDecoder(nn.Module):
    center: Incomplete
    blocks: Incomplete
    def __init__(self, encoder_channels, decoder_channels, n_blocks: int = 5, reduction: int = 16, use_batchnorm: bool = True, pab_channels: int = 64) -> None: ...
    def forward(self, *features): ...
