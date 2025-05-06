import numpy as np
from _typeshed import Incomplete
from pandas._typing import ArrayLike as ArrayLike, AxisInt as AxisInt, npt as npt
from pandas.core.arrays.base import ExtensionArray as ExtensionArray
from pandas.core.dtypes.common import ensure_platform_int as ensure_platform_int, is_1d_only_ea_dtype as is_1d_only_ea_dtype
from typing import overload

@overload
def take_nd(arr: np.ndarray, indexer, axis: AxisInt = ..., fill_value=..., allow_fill: bool = ...) -> np.ndarray: ...
@overload
def take_nd(arr: ExtensionArray, indexer, axis: AxisInt = ..., fill_value=..., allow_fill: bool = ...) -> ArrayLike: ...
def _take_nd_ndarray(arr: np.ndarray, indexer: npt.NDArray[np.intp] | None, axis: AxisInt, fill_value, allow_fill: bool) -> np.ndarray: ...
def take_1d(arr: ArrayLike, indexer: npt.NDArray[np.intp], fill_value: Incomplete | None = None, allow_fill: bool = True, mask: npt.NDArray[np.bool_] | None = None) -> ArrayLike:
    """
    Specialized version for 1D arrays. Differences compared to `take_nd`:

    - Assumes input array has already been converted to numpy array / EA
    - Assumes indexer is already guaranteed to be intp dtype ndarray
    - Only works for 1D arrays

    To ensure the lowest possible overhead.

    Note: similarly to `take_nd`, this function assumes that the indexer is
    a valid(ated) indexer with no out of bound indices.

    Parameters
    ----------
    arr : np.ndarray or ExtensionArray
        Input array.
    indexer : ndarray
        1-D array of indices to take (validated indices, intp dtype).
    fill_value : any, default np.nan
        Fill value to replace -1 values with
    allow_fill : bool, default True
        If False, indexer is assumed to contain no -1 values so no filling
        will be done.  This short-circuits computation of a mask. Result is
        undefined if allow_fill == False and -1 is present in indexer.
    mask : np.ndarray, optional, default None
        If `allow_fill` is True, and the mask (where indexer == -1) is already
        known, it can be passed to avoid recomputation.
    """
def take_2d_multi(arr: np.ndarray, indexer: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]], fill_value=...) -> np.ndarray:
    """
    Specialized Cython take which sets NaN values in one pass.
    """
def _get_take_nd_function_cached(ndim: int, arr_dtype: np.dtype, out_dtype: np.dtype, axis: AxisInt):
    """
    Part of _get_take_nd_function below that doesn't need `mask_info` and thus
    can be cached (mask_info potentially contains a numpy ndarray which is not
    hashable and thus cannot be used as argument for cached function).
    """
def _get_take_nd_function(ndim: int, arr_dtype: np.dtype, out_dtype: np.dtype, axis: AxisInt = 0, mask_info: Incomplete | None = None):
    '''
    Get the appropriate "take" implementation for the given dimension, axis
    and dtypes.
    '''
def _view_wrapper(f, arr_dtype: Incomplete | None = None, out_dtype: Incomplete | None = None, fill_wrap: Incomplete | None = None): ...
def _convert_wrapper(f, conv_dtype): ...

_take_1d_dict: Incomplete
_take_2d_axis0_dict: Incomplete
_take_2d_axis1_dict: Incomplete
_take_2d_multi_dict: Incomplete

def _take_nd_object(arr: np.ndarray, indexer: npt.NDArray[np.intp], out: np.ndarray, axis: AxisInt, fill_value, mask_info) -> None: ...
def _take_2d_multi_object(arr: np.ndarray, indexer: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]], out: np.ndarray, fill_value, mask_info) -> None: ...
def _take_preprocess_indexer_and_fill_value(arr: np.ndarray, indexer: npt.NDArray[np.intp], fill_value, allow_fill: bool, mask: npt.NDArray[np.bool_] | None = None): ...
