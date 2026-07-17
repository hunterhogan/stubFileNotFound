import torch
import torch.nn as nn
from _typeshed import Incomplete
from segmentation_models_pytorch.base.modules import Activation as Activation
from typing import Callable, Literal, Sequence

class ReadoutConcatBlock(nn.Module):
    """
    Concatenates the cls tokens with the features to make use of the global information aggregated in the prefix (cls) tokens.
    Projects the combined feature map to the original embedding dimension using a MLP.

    According to:
        https://github.com/isl-org/DPT/blob/cd3fe90bb4c48577535cc4d51b602acca688a2ee/dpt/vit.py#L79-L90
    """
    project: Incomplete
    def __init__(self, embed_dim: int, has_prefix_tokens: bool) -> None: ...
    def forward(self, features: torch.Tensor, prefix_tokens: torch.Tensor | None = None) -> torch.Tensor: ...

class ReadoutAddBlock(nn.Module):
    """
    Adds the prefix tokens to the features to make use of the global information aggregated in the prefix (cls) tokens.

    According to:
        https://github.com/isl-org/DPT/blob/cd3fe90bb4c48577535cc4d51b602acca688a2ee/dpt/vit.py#L71-L76
    """
    def forward(self, features: torch.Tensor, prefix_tokens: torch.Tensor | None = None) -> torch.Tensor: ...

class ReadoutIgnoreBlock(nn.Module):
    """
    Ignores the prefix tokens and returns the features as is.
    """
    def forward(self, features: torch.Tensor, *args, **kwargs) -> torch.Tensor: ...

class ReassembleBlock(nn.Module):
    """
    Processes the features such that they have progressively increasing embedding size and progressively decreasing
    spatial dimension
    """
    project_to_out_channel: Incomplete
    upsample: Incomplete
    project_to_feature_dim: Incomplete
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, upsample_factor: int) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class ResidualConvBlock(nn.Module):
    conv_1: Incomplete
    batch_norm_1: Incomplete
    conv_2: Incomplete
    batch_norm_2: Incomplete
    activation: Incomplete
    def __init__(self, feature_dim: int) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class FusionBlock(nn.Module):
    """
    Fuses the processed encoder features in a residual manner and upsamples them
    """
    residual_conv_block1: Incomplete
    residual_conv_block2: Incomplete
    project: Incomplete
    activation: Incomplete
    def __init__(self, feature_dim: int) -> None: ...
    def forward(self, feature: torch.Tensor, previous_feature: torch.Tensor | None = None) -> torch.Tensor: ...

class DPTDecoder(nn.Module):
    """
    Decoder part for DPT

    Processes the encoder features and class tokens (if encoder has class_tokens) to have spatial downsampling ratios of
    [1/4, 1/8, 1/16, 1/32, ...] relative to the input image spatial dimension.

    The decoder then fuses these features in a residual manner and progressively upsamples them by a factor of 2 so that the
    output has a downsampling ratio of 1/2 relative to the input image spatial dimension

    """
    projection_blocks: Incomplete
    reassemble_blocks: Incomplete
    fusion_blocks: Incomplete
    def __init__(self, encoder_out_channels: Sequence[int] = (756, 756, 756, 756), encoder_output_strides: Sequence[int] = (16, 16, 16, 16), encoder_has_prefix_tokens: bool = True, readout: Literal['cat', 'add', 'ignore'] = 'cat', intermediate_channels: Sequence[int] = (256, 512, 1024, 1024), fusion_channels: int = 256) -> None: ...
    def forward(self, features: list[torch.Tensor], prefix_tokens: list[torch.Tensor | None]) -> torch.Tensor: ...

class DPTSegmentationHead(nn.Module):
    head: Incomplete
    activation: Incomplete
    upsampling_factor: Incomplete
    def __init__(self, in_channels: int, out_channels: int, activation: str | Callable | None = None, kernel_size: int = 3, upsampling: float = 2.0) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
