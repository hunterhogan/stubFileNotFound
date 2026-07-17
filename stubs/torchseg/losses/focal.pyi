import torch
import torch.nn as nn
from ._functional import focal_loss_with_logits as focal_loss_with_logits
from .constants import BINARY_MODE as BINARY_MODE, MULTICLASS_MODE as MULTICLASS_MODE, MULTILABEL_MODE as MULTILABEL_MODE
from _typeshed import Incomplete

class FocalLoss(nn.Module):
    mode: Incomplete
    ignore_index: Incomplete
    classes: Incomplete
    focal_loss_fn: Incomplete
    def __init__(self, mode: str, alpha: float | None = None, gamma: float | None = 2.0, ignore_index: int | None = None, reduction: str | None = 'mean', normalized: bool = False, reduced_threshold: float | None = None, classes: list[int] | None = None) -> None:
        '''Compute Focal loss

        Args:
            mode: Loss mode \'binary\', \'multiclass\' or \'multilabel\'
            alpha: Prior probability of having positive value in target.
            gamma: Power factor for dampening weight (focal strength).
            ignore_index: If not None, targets may contain values to be ignored. Target
                values equal to ignore_index will be ignored from loss computation.
            normalized: Use normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf)
            reduced_threshold: Switch to reduced focal loss.
                Note, when using this mode you should use `reduction="sum"`.
            classes:  List of classes that contribute in loss computation.
                By default, all channels are included. Only supported in multiclass mode

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt

        '''
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor: ...
