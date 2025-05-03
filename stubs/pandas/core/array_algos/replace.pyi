import np
import npt
import re
from pandas._libs.lib import is_bool as is_bool
from pandas.core.dtypes.inference import is_re as is_re, is_re_compilable as is_re_compilable
from pandas.core.dtypes.missing import isna as isna
from re import Pattern
from typing import Any

TYPE_CHECKING: bool
def should_use_regex(regex: bool, to_replace: Any) -> bool:
    """
    Decide whether to treat `to_replace` as a regular expression.
    """
def compare_or_regex_search(a: ArrayLike, b: Scalar | Pattern, regex: bool, mask: npt.NDArray[np.bool_]) -> ArrayLike:
    """
    Compare two array-like inputs of the same shape or two scalar values

    Calls operator.eq or re.search, depending on regex argument. If regex is
    True, perform an element-wise regex matching.

    Parameters
    ----------
    a : array-like
    b : scalar or regex pattern
    regex : bool
    mask : np.ndarray[bool]

    Returns
    -------
    mask : array-like of bool
    """
def replace_regex(values: ArrayLike, rx: re.Pattern, value, mask: npt.NDArray[np.bool_] | None) -> None:
    """
    Parameters
    ----------
    values : ArrayLike
        Object dtype.
    rx : re.Pattern
    value : Any
    mask : np.ndarray[bool], optional

    Notes
    -----
    Alters values in-place.
    """
