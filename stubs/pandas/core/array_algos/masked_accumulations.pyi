import numpy as np
from pandas._typing import npt as npt
from collections.abc import Callable

def _cum_func(func: Callable, values: np.ndarray, mask: npt.NDArray[np.bool_], *, skipna: bool = True):
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
def cumsum(values: np.ndarray, mask: npt.NDArray[np.bool_], *, skipna: bool = True): ...
def cumprod(values: np.ndarray, mask: npt.NDArray[np.bool_], *, skipna: bool = True): ...
def cummin(values: np.ndarray, mask: npt.NDArray[np.bool_], *, skipna: bool = True): ...
def cummax(values: np.ndarray, mask: npt.NDArray[np.bool_], *, skipna: bool = True): ...
