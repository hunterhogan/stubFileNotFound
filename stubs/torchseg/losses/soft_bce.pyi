import torch
import torch.nn as nn
from _typeshed import Incomplete

class SoftBCEWithLogitsLoss(nn.Module):
    __constants__: Incomplete
    ignore_index: Incomplete
    reduction: Incomplete
    smooth_factor: Incomplete
    def __init__(self, weight: torch.Tensor | None = None, ignore_index: int | None = -100, reduction: str = 'mean', smooth_factor: float | None = None, pos_weight: torch.Tensor | None = None) -> None:
        """
        Drop-in replacement for torch.nn.BCEWithLogitsLoss with few additions:
        ignore_index and label_smoothing

        Args:
            ignore_index: Specifies a target value that is ignored and does not
                contribute to the input gradient.
            smooth_factor: Factor to smooth target
                (e.g. if smooth_factor=0.1 then [1, 0, 1] -> [0.9, 0.1, 0.9])

        Shape
             - **y_pred** - torch.Tensor of shape NxCxHxW
             - **y_true** - torch.Tensor of shape NxHxW or Nx1xHxW

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt

        """
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: torch.Tensor of shape (N, C, H, W)
            y_true: torch.Tensor of shape (N, H, W)  or (N, 1, H, W)

        Returns:
            loss: torch.Tensor
        """
