import torch
import torch.nn as nn
from ._base import EncoderMixin
from _typeshed import Incomplete
from typing import Sequence

__all__ = ['MobileOne', 'reparameterize_model']

class SEBlock(nn.Module):
    """Squeeze and Excite module.

    Pytorch implementation of `Squeeze-and-Excitation Networks` -
    https://arxiv.org/pdf/1709.01507.pdf
    """
    reduce: Incomplete
    expand: Incomplete
    def __init__(self, in_channels: int, rd_ratio: float = 0.0625) -> None:
        """Construct a Squeeze and Excite Module.

        :param in_channels: Number of input channels.
        :param rd_ratio: Input channel reduction ratio.
        """
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply forward pass."""

class MobileOneBlock(nn.Module):
    """MobileOne building block.

    This block has a multi-branched architecture at train-time
    and plain-CNN style architecture at inference time
    For more details, please refer to our paper:
    `An Improved One millisecond Mobile Backbone` -
    https://arxiv.org/pdf/2206.04040.pdf
    """
    inference_mode: Incomplete
    groups: Incomplete
    stride: Incomplete
    kernel_size: Incomplete
    in_channels: Incomplete
    out_channels: Incomplete
    num_conv_branches: Incomplete
    se: Incomplete
    activation: Incomplete
    reparam_conv: Incomplete
    rbr_skip: Incomplete
    rbr_conv: Incomplete
    rbr_scale: Incomplete
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, inference_mode: bool = False, use_se: bool = False, num_conv_branches: int = 1) -> None:
        """Construct a MobileOneBlock module.

        :param in_channels: Number of channels in the input.
        :param out_channels: Number of channels produced by the block.
        :param kernel_size: Size of the convolution kernel.
        :param stride: Stride size.
        :param padding: Zero-padding size.
        :param dilation: Kernel dilation factor.
        :param groups: Group number.
        :param inference_mode: If True, instantiates model in inference mode.
        :param use_se: Whether to use SE-ReLU activations.
        :param num_conv_branches: Number of linear conv branches.
        """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward pass."""
    def reparameterize(self) -> None:
        """Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """

class MobileOne(nn.Module, EncoderMixin):
    """MobileOne Model

    Pytorch implementation of `An Improved One millisecond Mobile Backbone` -
    https://arxiv.org/pdf/2206.04040.pdf
    """
    inference_mode: Incomplete
    in_planes: Incomplete
    use_se: Incomplete
    num_conv_branches: Incomplete
    stage0: Incomplete
    cur_layer_idx: int
    stage1: Incomplete
    stage2: Incomplete
    stage3: Incomplete
    stage4: Incomplete
    def __init__(self, out_channels: list[int], num_blocks_per_stage: list[int] = [2, 8, 10, 1], width_multipliers: list[float] | None = None, inference_mode: bool = False, use_se: bool = False, depth: int = 5, in_channels: int = 3, output_stride: int = 32, num_conv_branches: int = 1) -> None:
        """Construct MobileOne model.

        :param num_blocks_per_stage: List of number of blocks per stage.
        :param num_classes: Number of classes in the dataset.
        :param width_multipliers: List of width multiplier for blocks in a stage.
        :param inference_mode: If True, instantiates model in inference mode.
        :param use_se: Whether to use SE-ReLU activations.
        :param num_conv_branches: Number of linear conv branches.
        """
    def get_stages(self) -> dict[int, Sequence[torch.nn.Module]]: ...
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Apply forward pass."""
    def load_state_dict(self, state_dict, **kwargs) -> None: ...
    def set_in_channels(self, in_channels, pretrained: bool = True) -> None:
        """Change first convolution channels"""

def reparameterize_model(model: torch.nn.Module) -> nn.Module:
    """Return a model where a multi-branched structure
        used in training is re-parameterized into a single branch
        for inference.

    :param model: MobileOne model in train mode.
    :return: MobileOne model in inference mode.
    """
