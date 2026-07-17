import torch
import torch.nn as nn
from _typeshed import Incomplete
from typing import Any

def sample_block_indices_uniformly(n: int, total_num_blocks: int) -> list[int]:
    """
    Sample N block indices uniformly from the total number of blocks.
    """
def validate_output_indices(output_indices: list[int], model_num_blocks: int, depth: int):
    """
    Validate the output indices are within the valid range of the model and the
    length of the output indices is equal to the depth of the encoder.
    """
def preprocess_output_indices(output_indices: list[int] | None, model_num_blocks: int, depth: int) -> list[int]:
    """
    Preprocess the output indices for the encoder.
    """

class TimmViTEncoder(nn.Module):
    """
    A universal encoder leveraging the `timm` library for feature extraction from
    ViT style models

    Features:
        - Supports configurable depth.
        - Ensures consistent multi-level feature extraction across all ViT models.
    """
    name: Incomplete
    model: Incomplete
    output_strides: Incomplete
    output_stride: Incomplete
    out_channels: Incomplete
    has_prefix_tokens: Incomplete
    input_size: Incomplete
    is_fixed_input_size: Incomplete
    def __init__(self, name: str, pretrained: bool = True, in_channels: int = 3, depth: int = 4, output_indices: list[int] | None = None, **kwargs: dict[str, Any]) -> None:
        """
        Initialize the encoder.

        Args:
            name (str): ViT model name to load from `timm`.
            pretrained (bool): Load pretrained weights (default: True).
            in_channels (int): Number of input channels (default: 3 for RGB).
            depth (int): Number of feature stages to extract (default: 4).
            output_indices (Optional[list[int] | int]): Indices of blocks in the model to be used for feature extraction.
            **kwargs: Additional arguments passed to `timm.create_model`.
        """
    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
        """
        Forward pass to extract multi-stage features.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            tuple[list[torch.Tensor], list[torch.Tensor]]: Tuple of feature maps and cls tokens (if supported) at different scales.
        """
