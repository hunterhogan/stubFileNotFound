from _typeshed import Incomplete
from torch.nn.modules.loss import _Loss

__all__ = ['LovaszLoss']

class LovaszLoss(_Loss):
    mode: Incomplete
    ignore_index: Incomplete
    per_image: Incomplete
    def __init__(self, mode: str, per_image: bool = False, ignore_index: int | None = None, from_logits: bool = True) -> None:
        """Lovasz loss for image segmentation task.
        It supports binary, multiclass and multilabel cases

        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            ignore_index: Label that indicates ignored pixels (does not contribute to loss)
            per_image: If True loss computed per each image and then averaged, else computed per whole batch

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
    def forward(self, y_pred, y_true): ...
