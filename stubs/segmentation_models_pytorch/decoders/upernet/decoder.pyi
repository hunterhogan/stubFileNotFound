import torch
import torch.nn as nn
from _typeshed import Incomplete
from typing import Any, Sequence

class PSPModule(nn.Module):
    blocks: Incomplete
    out_conv: Incomplete
    def __init__(self, in_channels: int, out_channels: int, sizes: Sequence[int] = (1, 2, 3, 6), use_norm: bool | str | dict[str, Any] = 'batchnorm') -> None: ...
    def forward(self, feature: torch.Tensor) -> torch.Tensor: ...

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class FPNLateralBlock(nn.Module):
    conv_norm_relu: Incomplete
    def __init__(self, lateral_channels: int, out_channels: int, use_norm: bool | str | dict[str, Any] = 'batchnorm') -> None: ...
    def forward(self, state_feature: torch.Tensor, lateral_feature: torch.Tensor) -> torch.Tensor: ...

class UPerNetDecoder(nn.Module):
    feature_norms: Incomplete
    psp: Incomplete
    fpn_lateral_blocks: Incomplete
    fpn_conv_blocks: Incomplete
    fusion_block: Incomplete
    def __init__(self, encoder_channels: Sequence[int], encoder_depth: int = 5, decoder_channels: int = 256, use_norm: bool | str | dict[str, Any] = 'batchnorm') -> None: ...
    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features (List[torch.Tensor]):
                features with: [1, 1/2, 1/4, 1/8, 1/16, ...] spatial resolutions,
                where the first feature is the highest resolution and the number
                of features is equal to encoder_depth + 1.
        """
