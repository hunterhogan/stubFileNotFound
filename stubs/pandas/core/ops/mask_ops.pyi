import numpy as np

from pandas._libs import missing as libmissing
from typing import Any

def kleene_or(
    left: bool | np.ndarray[Any, Any],
    right: bool | np.ndarray[Any, Any],
    left_mask: np.ndarray[Any, Any] | None,
    right_mask: np.ndarray[Any, Any] | None,
): ...
def kleene_xor(
    left: bool | np.ndarray[Any, Any],
    right: bool | np.ndarray[Any, Any],
    left_mask: np.ndarray[Any, Any] | None,
    right_mask: np.ndarray[Any, Any] | None,
): ...
def kleene_and(
    left: bool | libmissing.NAType | np.ndarray[Any, Any],
    right: bool | libmissing.NAType | np.ndarray[Any, Any],
    left_mask: np.ndarray[Any, Any] | None,
    right_mask: np.ndarray[Any, Any] | None,
): ...
def raise_for_nan(value: Any, method: Any) -> None: ...
