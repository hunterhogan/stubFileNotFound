import numpy as np
from pandas._libs import iNaT as iNaT
from pandas.core.dtypes.missing import isna as isna
from collections.abc import Callable

def _cum_func(func: Callable, values: np.ndarray, *, skipna: bool = True):
    """
    Accumulations for 1D datetimelike arrays.

    Parameters
    ----------
    func : np.cumsum, np.maximum.accumulate, np.minimum.accumulate
    values : np.ndarray
        Numpy array with the values (can be of any dtype that support the
        operation). Values is changed is modified inplace.
    skipna : bool, default True
        Whether to skip NA.
    """
def cumsum(values: np.ndarray, *, skipna: bool = True) -> np.ndarray: ...
def cummin(values: np.ndarray, *, skipna: bool = True): ...
def cummax(values: np.ndarray, *, skipna: bool = True): ...
