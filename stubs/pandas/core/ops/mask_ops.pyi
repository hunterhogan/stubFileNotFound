import numpy as np

from pandas._libs import missing as libmissing
from typing import Any

def kleene_or(
    left: bool | np.ndarray,
    right: bool | np.ndarray,
    left_mask: np.ndarray | None,
    right_mask: np.ndarray | None,
): ...
def kleene_xor(
    left: bool | np.ndarray,
    right: bool | np.ndarray,
    left_mask: np.ndarray | None,
    right_mask: np.ndarray | None,
): ...
def kleene_and(
    left: bool | libmissing.NAType | np.ndarray,
    right: bool | libmissing.NAType | np.ndarray,
    left_mask: np.ndarray | None,
    right_mask: np.ndarray | None,
): ...
def raise_for_nan(value: Any, method: Any) -> None: ...
