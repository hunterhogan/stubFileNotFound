import torch
import torch.nn as nn
from _typeshed import Incomplete

class MLP(nn.Module):
    linear: Incomplete
    def __init__(self, skip_channels: int, segmentation_channels: int) -> None: ...
    def forward(self, x: torch.Tensor): ...

class SegformerDecoder(nn.Module):
    mlp_stage: Incomplete
    fuse_stage: Incomplete
    def __init__(self, encoder_channels: list[int], encoder_depth: int = 5, segmentation_channels: int = 256) -> None: ...
    def forward(self, features: list[torch.Tensor]) -> torch.Tensor: ...
