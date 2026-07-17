import torch
from .dice import DiceLoss
from _typeshed import Incomplete

__all__ = ['TverskyLoss']

class TverskyLoss(DiceLoss):
    """Tversky loss for image segmentation task.
    Where FP and FN is weighted by alpha and beta params.
    With alpha == beta == 0.5, this loss becomes equal DiceLoss.
    It supports binary, multiclass and multilabel cases

    Args:
        mode: Metric mode {'binary', 'multiclass', 'multilabel'}
        classes: Optional list of classes that contribute in loss computation;
        By default, all channels are included.
        log_loss: If True, loss computed as ``-log(tversky)`` otherwise ``1 - tversky``
        from_logits: If True assumes input is raw logits
        smooth:
        ignore_index: Label that indicates ignored pixels (does not contribute to loss)
        eps: Small epsilon for numerical stability
        alpha: Weight constant that penalize model for FPs (False Positives)
        beta: Weight constant that penalize model for FNs (False Negatives)
        gamma: Constant that squares the error function. Defaults to ``1.0``

    Return:
        loss: torch.Tensor

    """
    alpha: Incomplete
    beta: Incomplete
    gamma: Incomplete
    def __init__(self, mode: str, classes: list[int] = None, log_loss: bool = False, from_logits: bool = True, smooth: float = 0.0, ignore_index: int | None = None, eps: float = 1e-07, alpha: float = 0.5, beta: float = 0.5, gamma: float = 1.0) -> None: ...
    def aggregate_loss(self, loss): ...
    def compute_score(self, output, target, smooth: float = 0.0, eps: float = 1e-07, dims=None) -> torch.Tensor: ...
