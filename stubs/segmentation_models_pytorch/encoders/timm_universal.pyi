import torch
import torch.nn as nn
from _typeshed import Incomplete
from typing import Any

class TimmUniversalEncoder(nn.Module):
    """
    A universal encoder leveraging the `timm` library for feature extraction from
    various model architectures, including traditional-style and transformer-style models.

    Features:
        - Supports configurable depth and output stride.
        - Ensures consistent multi-level feature extraction across diverse models.
        - Compatible with convolutional and transformer-like backbones.
    """
    name: Incomplete
    model: Incomplete
    def __init__(self, name: str, pretrained: bool = True, in_channels: int = 3, depth: int = 5, output_stride: int = 32, **kwargs: dict[str, Any]) -> None:
        """
        Initialize the encoder.

        Args:
            name (str): Model name to load from `timm`.
            pretrained (bool): Load pretrained weights (default: True).
            in_channels (int): Number of input channels (default: 3 for RGB).
            depth (int): Number of feature stages to extract (default: 5).
            output_stride (int): Desired output stride (default: 32).
            **kwargs: Additional arguments passed to `timm.create_model`.
        """
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward pass to extract multi-stage features.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            list[torch.Tensor]: List of feature maps at different scales.
        """
    @property
    def out_channels(self) -> list[int]:
        """
        Returns the number of output channels for each feature stage.

        Returns:
            list[int]: A list of channel dimensions at each scale.
        """
    @property
    def output_stride(self) -> int:
        """
        Returns the effective output stride based on the model depth.

        Returns:
            int: The effective output stride.
        """
    def load_state_dict(self, state_dict, **kwargs): ...
