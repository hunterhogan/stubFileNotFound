import torch
import torch.nn as nn
from _typeshed import Incomplete
from typing import Any, Sequence

class UnetDecoderBlock(nn.Module):
    """A decoder block in the U-Net architecture that performs upsampling and feature fusion."""
    interpolation_mode: Incomplete
    conv1: Incomplete
    attention1: Incomplete
    conv2: Incomplete
    attention2: Incomplete
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, use_norm: bool | str | dict[str, Any] = 'batchnorm', attention_type: str | None = None, interpolation_mode: str = 'nearest') -> None: ...
    def forward(self, feature_map: torch.Tensor, target_height: int, target_width: int, skip_connection: torch.Tensor | None = None) -> torch.Tensor: ...

class UnetCenterBlock(nn.Sequential):
    """Center block of the Unet decoder. Applied to the last feature map of the encoder."""
    def __init__(self, in_channels: int, out_channels: int, use_norm: bool | str | dict[str, Any] = 'batchnorm') -> None: ...

class UnetDecoder(nn.Module):
    """The decoder part of the U-Net architecture.

    Takes encoded features from different stages of the encoder and progressively upsamples them while
    combining with skip connections. This helps preserve fine-grained details in the final segmentation.
    """
    center: Incomplete
    blocks: Incomplete
    def __init__(self, encoder_channels: Sequence[int], decoder_channels: Sequence[int], n_blocks: int = 5, use_norm: bool | str | dict[str, Any] = 'batchnorm', attention_type: str | None = None, add_center_block: bool = False, interpolation_mode: str = 'nearest') -> None: ...
    def forward(self, features: list[torch.Tensor]) -> torch.Tensor: ...
