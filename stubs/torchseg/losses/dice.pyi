import torch
import torch.nn as nn
from ._functional import soft_dice_score as soft_dice_score, to_tensor as to_tensor
from .constants import BINARY_MODE as BINARY_MODE, MULTICLASS_MODE as MULTICLASS_MODE, MULTILABEL_MODE as MULTILABEL_MODE
from _typeshed import Incomplete

class DiceLoss(nn.Module):
    mode: Incomplete
    classes: Incomplete
    from_logits: Incomplete
    smooth: Incomplete
    eps: Incomplete
    log_loss: Incomplete
    ignore_index: Incomplete
    def __init__(self, mode: str, classes: list[int] | None = None, log_loss: bool = False, from_logits: bool = True, smooth: float = 0.0, ignore_index: int | None = None, eps: float = 1e-07) -> None:
        """Dice loss for image segmentation task.
        It supports binary, multiclass and multilabel cases

        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            classes:  List of classes that contribute in loss computation.
                By default, all channels are included.
            log_loss: If True, loss computed as `- log(dice_coeff)`,
                otherwise `1 - dice_coeff`
            from_logits: If True, assumes input is raw logits
            smooth: Smoothness constant for dice coefficient (a)
            ignore_index: Label that indicates ignored pixels
                (does not contribute to loss)
            eps: A small epsilon for numerical stability to avoid zero division error
                (denominator will be always greater or equal to eps)

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor: ...
    def aggregate_loss(self, loss): ...
    def compute_score(self, output, target, smooth: float = 0.0, eps: float = 1e-07, dims=None) -> torch.Tensor: ...
