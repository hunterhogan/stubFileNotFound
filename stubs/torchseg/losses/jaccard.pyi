import torch
import torch.nn as nn
from ._functional import soft_jaccard_score as soft_jaccard_score, to_tensor as to_tensor
from .constants import BINARY_MODE as BINARY_MODE, MULTICLASS_MODE as MULTICLASS_MODE, MULTILABEL_MODE as MULTILABEL_MODE
from _typeshed import Incomplete

class JaccardLoss(nn.Module):
    mode: Incomplete
    classes: Incomplete
    from_logits: Incomplete
    smooth: Incomplete
    eps: Incomplete
    ignore_index: Incomplete
    log_loss: Incomplete
    def __init__(self, mode: str, classes: list[int] | None = None, log_loss: bool = False, from_logits: bool = True, smooth: float = 0.0, ignore_index: int | None = None, eps: float = 1e-07) -> None:
        """Jaccard loss for image segmentation task.
        It supports binary, multiclass and multilabel cases

        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            classes:  List of classes that contribute in loss computation.
                By default, all channels are included.
            log_loss: If True, loss computed as `- log(jaccard_coeff)`,
                otherwise `1 - jaccard_coeff`
            from_logits: If True, assumes input is raw logits
            smooth: Smoothness constant for dice coefficient
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
