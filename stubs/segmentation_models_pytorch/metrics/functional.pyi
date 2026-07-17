import torch

__all__ = ['get_stats', 'fbeta_score', 'f1_score', 'iou_score', 'accuracy', 'precision', 'recall', 'sensitivity', 'specificity', 'balanced_accuracy', 'positive_predictive_value', 'negative_predictive_value', 'false_negative_rate', 'false_positive_rate', 'false_discovery_rate', 'false_omission_rate', 'positive_likelihood_ratio', 'negative_likelihood_ratio']

def get_stats(output: torch.LongTensor | torch.FloatTensor, target: torch.LongTensor, mode: str, ignore_index: int | None = None, threshold: float | list[float] | None = None, num_classes: int | None = None) -> tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    """Compute true positive, false positive, false negative, true negative 'pixels'
    for each image and each class.

    Args:
        output (Union[torch.LongTensor, torch.FloatTensor]): Model output with following
            shapes and types depending on the specified ``mode``:

            'binary'
                shape (N, 1, ...) and ``torch.LongTensor`` or ``torch.FloatTensor``

            'multilabel'
                shape (N, C, ...) and ``torch.LongTensor`` or ``torch.FloatTensor``

            'multiclass'
                shape (N, ...) and ``torch.LongTensor``

        target (torch.LongTensor): Targets with following shapes depending on the specified ``mode``:

            'binary'
                shape (N, 1, ...)

            'multilabel'
                shape (N, C, ...)

            'multiclass'
                shape (N, ...)

        mode (str): One of ``'binary'`` | ``'multilabel'`` | ``'multiclass'``
        ignore_index (Optional[int]): Label to ignore on for metric computation.
            **Not** supported for ``'binary'`` and ``'multilabel'`` modes.  Defaults to None.
        threshold (Optional[float, List[float]]): Binarization threshold for
            ``output`` in case of ``'binary'`` or ``'multilabel'`` modes. Defaults to None.
        num_classes (Optional[int]): Number of classes, necessary attribute
            only for ``'multiclass'`` mode. Class values should be in range 0..(num_classes - 1).
            If ``ignore_index`` is specified it should be outside the classes range, e.g. ``-1`` or
            ``255``.

    Raises:
        ValueError: in case of misconfiguration.

    Returns:
        Tuple[torch.LongTensor]: true_positive, false_positive, false_negative,
            true_negative tensors (N, C) shape each.

    """
def fbeta_score(tp: torch.LongTensor, fp: torch.LongTensor, fn: torch.LongTensor, tn: torch.LongTensor, beta: float = 1.0, reduction: str | None = None, class_weights: list[float] | None = None, zero_division: str | float = 1.0) -> torch.Tensor:
    """F beta score"""
def f1_score(tp: torch.LongTensor, fp: torch.LongTensor, fn: torch.LongTensor, tn: torch.LongTensor, reduction: str | None = None, class_weights: list[float] | None = None, zero_division: str | float = 1.0) -> torch.Tensor:
    """F1 score"""
def iou_score(tp: torch.LongTensor, fp: torch.LongTensor, fn: torch.LongTensor, tn: torch.LongTensor, reduction: str | None = None, class_weights: list[float] | None = None, zero_division: str | float = 1.0) -> torch.Tensor:
    """IoU score or Jaccard index"""
def accuracy(tp: torch.LongTensor, fp: torch.LongTensor, fn: torch.LongTensor, tn: torch.LongTensor, reduction: str | None = None, class_weights: list[float] | None = None, zero_division: str | float = 1.0) -> torch.Tensor:
    """Accuracy"""
def sensitivity(tp: torch.LongTensor, fp: torch.LongTensor, fn: torch.LongTensor, tn: torch.LongTensor, reduction: str | None = None, class_weights: list[float] | None = None, zero_division: str | float = 1.0) -> torch.Tensor:
    """Sensitivity, recall, hit rate, or true positive rate (TPR)"""
def specificity(tp: torch.LongTensor, fp: torch.LongTensor, fn: torch.LongTensor, tn: torch.LongTensor, reduction: str | None = None, class_weights: list[float] | None = None, zero_division: str | float = 1.0) -> torch.Tensor:
    """Specificity, selectivity or true negative rate (TNR)"""
def balanced_accuracy(tp: torch.LongTensor, fp: torch.LongTensor, fn: torch.LongTensor, tn: torch.LongTensor, reduction: str | None = None, class_weights: list[float] | None = None, zero_division: str | float = 1.0) -> torch.Tensor:
    """Balanced accuracy"""
def positive_predictive_value(tp: torch.LongTensor, fp: torch.LongTensor, fn: torch.LongTensor, tn: torch.LongTensor, reduction: str | None = None, class_weights: list[float] | None = None, zero_division: str | float = 1.0) -> torch.Tensor:
    """Precision or positive predictive value (PPV)"""
def negative_predictive_value(tp: torch.LongTensor, fp: torch.LongTensor, fn: torch.LongTensor, tn: torch.LongTensor, reduction: str | None = None, class_weights: list[float] | None = None, zero_division: str | float = 1.0) -> torch.Tensor:
    """Negative predictive value (NPV)"""
def false_negative_rate(tp: torch.LongTensor, fp: torch.LongTensor, fn: torch.LongTensor, tn: torch.LongTensor, reduction: str | None = None, class_weights: list[float] | None = None, zero_division: str | float = 1.0) -> torch.Tensor:
    """Miss rate or false negative rate (FNR)"""
def false_positive_rate(tp: torch.LongTensor, fp: torch.LongTensor, fn: torch.LongTensor, tn: torch.LongTensor, reduction: str | None = None, class_weights: list[float] | None = None, zero_division: str | float = 1.0) -> torch.Tensor:
    """Fall-out or false positive rate (FPR)"""
def false_discovery_rate(tp: torch.LongTensor, fp: torch.LongTensor, fn: torch.LongTensor, tn: torch.LongTensor, reduction: str | None = None, class_weights: list[float] | None = None, zero_division: str | float = 1.0) -> torch.Tensor:
    """False discovery rate (FDR)"""
def false_omission_rate(tp: torch.LongTensor, fp: torch.LongTensor, fn: torch.LongTensor, tn: torch.LongTensor, reduction: str | None = None, class_weights: list[float] | None = None, zero_division: str | float = 1.0) -> torch.Tensor:
    """False omission rate (FOR)"""
def positive_likelihood_ratio(tp: torch.LongTensor, fp: torch.LongTensor, fn: torch.LongTensor, tn: torch.LongTensor, reduction: str | None = None, class_weights: list[float] | None = None, zero_division: str | float = 1.0) -> torch.Tensor:
    """Positive likelihood ratio (LR+)"""
def negative_likelihood_ratio(tp: torch.LongTensor, fp: torch.LongTensor, fn: torch.LongTensor, tn: torch.LongTensor, reduction: str | None = None, class_weights: list[float] | None = None, zero_division: str | float = 1.0) -> torch.Tensor:
    """Negative likelihood ratio (LR-)"""
precision = positive_predictive_value
recall = sensitivity
