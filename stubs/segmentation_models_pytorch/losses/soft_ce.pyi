import torch
from _typeshed import Incomplete
from torch import nn

__all__ = ['SoftCrossEntropyLoss']

class SoftCrossEntropyLoss(nn.Module):
    __constants__: Incomplete
    smooth_factor: Incomplete
    ignore_index: Incomplete
    reduction: Incomplete
    dim: Incomplete
    def __init__(self, reduction: str = 'mean', smooth_factor: float | None = None, ignore_index: int | None = -100, dim: int = 1) -> None:
        """Drop-in replacement for torch.nn.CrossEntropyLoss with label_smoothing

        Args:
            smooth_factor: Factor to smooth target (e.g. if smooth_factor=0.1 then [1, 0, 0] -> [0.9, 0.05, 0.05])

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor: ...
