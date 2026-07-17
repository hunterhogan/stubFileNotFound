import torch

__all__ = ['focal_loss_with_logits', 'softmax_focal_loss_with_logits', 'soft_jaccard_score', 'soft_dice_score', 'wing_loss']

def focal_loss_with_logits(output: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, alpha: float | None = 0.25, reduction: str = 'mean', normalized: bool = False, reduced_threshold: float | None = None, eps: float = 1e-06) -> torch.Tensor:
    """Compute binary focal loss between target and output logits.
    See :class:`~pytorch_toolbelt.losses.FocalLoss` for details.

    Args:
        output: Tensor of arbitrary shape (predictions of the model)
        target: Tensor of the same shape as input
        gamma: Focal loss power factor
        alpha: Weight factor to balance positive and negative samples. Alpha must be in [0...1] range,
            high values will give more weight to positive class.
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum' | 'batchwise_mean'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`.
            'batchwise_mean' computes mean loss per sample in batch. Default: 'mean'
        normalized (bool): Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
        reduced_threshold (float, optional): Compute reduced focal loss (https://arxiv.org/abs/1903.01347).

    References:
        https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/loss/losses.py
    """
def softmax_focal_loss_with_logits(output: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, reduction: str = 'mean', normalized: bool = False, reduced_threshold: float | None = None, eps: float = 1e-06) -> torch.Tensor:
    """Softmax version of focal loss between target and output logits.
    See :class:`~pytorch_toolbelt.losses.FocalLoss` for details.

    Args:
        output: Tensor of shape [B, C, *] (Similar to nn.CrossEntropyLoss)
        target: Tensor of shape [B, *] (Similar to nn.CrossEntropyLoss)
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum' | 'batchwise_mean'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`.
            'batchwise_mean' computes mean loss per sample in batch. Default: 'mean'
        normalized (bool): Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
        reduced_threshold (float, optional): Compute reduced focal loss (https://arxiv.org/abs/1903.01347).
    """
def soft_jaccard_score(output: torch.Tensor, target: torch.Tensor, smooth: float = 0.0, eps: float = 1e-07, dims=None) -> torch.Tensor: ...
def soft_dice_score(output: torch.Tensor, target: torch.Tensor, smooth: float = 0.0, eps: float = 1e-07, dims=None) -> torch.Tensor: ...
def wing_loss(output: torch.Tensor, target: torch.Tensor, width: int = 5, curvature: float = 0.5, reduction: str = 'mean'):
    """Wing loss

    References:
        https://arxiv.org/pdf/1711.06753.pdf

    """
