import torch
from typing import Sequence

class EncoderMixin:
    """Add encoder functionality such as:
    - output channels specification of feature tensors (produced by encoder)
    - patching first convolution for arbitrary input channels
    """
    def __init__(self) -> None: ...
    @property
    def out_channels(self):
        """Return channels dimensions for each tensor of forward output of encoder"""
    @property
    def output_stride(self): ...
    def set_in_channels(self, in_channels, pretrained: bool = True) -> None:
        """Change first convolution channels"""
    def get_stages(self) -> dict[int, Sequence[torch.nn.Module]]:
        """Override it in your implementation, should return a dictionary with keys as
        the output stride and values as the list of modules
        """
    def make_dilated(self, output_stride) -> None: ...
