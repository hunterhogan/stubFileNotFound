import torch
import torch.nn as nn
from _typeshed import Incomplete

class MCCLoss(nn.Module):
    eps: Incomplete
    def __init__(self, eps: float = 1e-05) -> None:
        """Compute Matthews Correlation Coefficient Loss for image segmentation task.

        It only supports binary mode.

        Args:
            eps (float): Small epsilon to handle situations where all the
                samples in the dataset belong to one class

        Reference:
            https://github.com/kakumarabhishek/MCC-Loss
        """
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute MCC loss

        Args:
            y_pred (torch.Tensor): model prediction of shape (N, H, W) or (N, 1, H, W)
            y_true (torch.Tensor): labels of shape (N, H, W) or (N, 1, H, W)

        Returns:
            torch.Tensor: loss value (1 - mcc)
        """
