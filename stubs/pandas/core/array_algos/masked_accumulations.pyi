import np
import npt
from typing import Callable

TYPE_CHECKING: bool
def _cum_func(func: Callable, values: np.ndarray, mask: npt.NDArray[np.bool_], *, skipna: bool = ...):
    """
    Accumulations for 1D masked array.

    We will modify values in place to replace NAs with the appropriate fill value.

    Parameters
    ----------
    func : np.cumsum, np.cumprod, np.maximum.accumulate, np.minimum.accumulate
    values : np.ndarray
        Numpy array with the values (can be of any dtype that support the
        operation).
    mask : np.ndarray
        Boolean numpy array (True values indicate missing values).
    skipna : bool, default True
        Whether to skip NA.
    """
def cumsum(values: np.ndarray, mask: npt.NDArray[np.bool_], *, skipna: bool = ...): ...
def cumprod(values: np.ndarray, mask: npt.NDArray[np.bool_], *, skipna: bool = ...): ...
def cummin(values: np.ndarray, mask: npt.NDArray[np.bool_], *, skipna: bool = ...): ...
def cummax(values: np.ndarray, mask: npt.NDArray[np.bool_], *, skipna: bool = ...): ...
