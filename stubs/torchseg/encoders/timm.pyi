import torch.nn as nn
from .supported import TIMM_ENCODERS as TIMM_ENCODERS
from _typeshed import Incomplete

class TimmEncoder(nn.Module):
    model: Incomplete
    in_channels: Incomplete
    indices: Incomplete
    depth: Incomplete
    output_stride: Incomplete
    out_channels: Incomplete
    reductions: Incomplete
    def __init__(self, name, pretrained: bool = True, in_channels: int = 3, depth=None, indices=None, output_stride=None, **kwargs) -> None: ...
    def fix_padding(self) -> None:
        """
        Some models like inceptionv4 or inceptionresnetv2 3x3 kernels with no padding
        resulting in odd numbered feature height/width dimensions. Update padding=1
        """
    def forward(self, x): ...

class TimmViTEncoder(nn.Module):
    model: Incomplete
    in_channels: Incomplete
    indices: Incomplete
    depth: Incomplete
    out_channels: Incomplete
    patch_size: Incomplete
    image_size: Incomplete
    num_tokens: Incomplete
    output_stride: Incomplete
    reductions: Incomplete
    norm: Incomplete
    scale_factors: Incomplete
    upsample: Incomplete
    def __init__(self, name, pretrained: bool = True, in_channels: int = 3, depth=None, indices=None, norm: bool = True, scale_factors=None, **kwargs) -> None: ...
    def forward(self, x): ...
