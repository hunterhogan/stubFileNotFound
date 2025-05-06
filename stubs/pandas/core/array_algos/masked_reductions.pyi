import numpy as np
from pandas._typing import AxisInt as AxisInt, npt as npt
from pandas.core.nanops import check_below_min_count as check_below_min_count
from collections.abc import Callable

def _reductions(func: Callable, values: np.ndarray, mask: npt.NDArray[np.bool_], *, skipna: bool = True, min_count: int = 0, axis: AxisInt | None = None, **kwargs):
    """
    Sum, mean or product for 1D masked array.

    Parameters
    ----------
    func : np.sum or np.prod
    values : np.ndarray
        Numpy array with the values (can be of any dtype that support the
        operation).
    mask : np.ndarray[bool]
        Boolean numpy array (True values indicate missing values).
    skipna : bool, default True
        Whether to skip NA.
    min_count : int, default 0
        The required number of valid values to perform the operation. If fewer than
        ``min_count`` non-NA values are present the result will be NA.
    axis : int, optional, default None
    """
def sum(values: np.ndarray, mask: npt.NDArray[np.bool_], *, skipna: bool = True, min_count: int = 0, axis: AxisInt | None = None): ...
def prod(values: np.ndarray, mask: npt.NDArray[np.bool_], *, skipna: bool = True, min_count: int = 0, axis: AxisInt | None = None): ...
def _minmax(func: Callable, values: np.ndarray, mask: npt.NDArray[np.bool_], *, skipna: bool = True, axis: AxisInt | None = None):
    """
    Reduction for 1D masked array.

    Parameters
    ----------
    func : np.min or np.max
    values : np.ndarray
        Numpy array with the values (can be of any dtype that support the
        operation).
    mask : np.ndarray[bool]
        Boolean numpy array (True values indicate missing values).
    skipna : bool, default True
        Whether to skip NA.
    axis : int, optional, default None
    """
def min(values: np.ndarray, mask: npt.NDArray[np.bool_], *, skipna: bool = True, axis: AxisInt | None = None): ...
def max(values: np.ndarray, mask: npt.NDArray[np.bool_], *, skipna: bool = True, axis: AxisInt | None = None): ...
def mean(values: np.ndarray, mask: npt.NDArray[np.bool_], *, skipna: bool = True, axis: AxisInt | None = None): ...
def var(values: np.ndarray, mask: npt.NDArray[np.bool_], *, skipna: bool = True, axis: AxisInt | None = None, ddof: int = 1): ...
def std(values: np.ndarray, mask: npt.NDArray[np.bool_], *, skipna: bool = True, axis: AxisInt | None = None, ddof: int = 1): ...
