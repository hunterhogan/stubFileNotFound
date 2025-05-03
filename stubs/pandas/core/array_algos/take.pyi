import functools
import np
import npt
import pandas._libs.algos as libalgos
import pandas._libs.lib
import pandas._libs.lib as lib
from pandas._libs.algos import ensure_platform_int as ensure_platform_int
from pandas.core.construction import ensure_wrapped_if_datetimelike as ensure_wrapped_if_datetimelike
from pandas.core.dtypes.cast import maybe_promote as maybe_promote
from pandas.core.dtypes.common import is_1d_only_ea_dtype as is_1d_only_ea_dtype
from pandas.core.dtypes.missing import na_value_for_dtype as na_value_for_dtype

TYPE_CHECKING: bool
def take_nd(arr: ArrayLike, indexer, axis: AxisInt = ..., fill_value: pandas._libs.lib._NoDefault = ..., allow_fill: bool = ...) -> ArrayLike:
    """
    Specialized Cython take which sets NaN values in one pass

    This dispatches to ``take`` defined on ExtensionArrays.

    Note: this function assumes that the indexer is a valid(ated) indexer with
    no out of bound indices.

    Parameters
    ----------
    arr : np.ndarray or ExtensionArray
        Input array.
    indexer : ndarray
        1-D array of indices to take, subarrays corresponding to -1 value
        indices are filed with fill_value
    axis : int, default 0
        Axis to take from
    fill_value : any, default np.nan
        Fill value to replace -1 values with
    allow_fill : bool, default True
        If False, indexer is assumed to contain no -1 values so no filling
        will be done.  This short-circuits computation of a mask.  Result is
        undefined if allow_fill == False and -1 is present in indexer.

    Returns
    -------
    subarray : np.ndarray or ExtensionArray
        May be the same type as the input, or cast to an ndarray.
    """
def _take_nd_ndarray(arr: np.ndarray, indexer: npt.NDArray[np.intp] | None, axis: AxisInt, fill_value, allow_fill: bool) -> np.ndarray: ...
def take_1d(arr: ArrayLike, indexer: npt.NDArray[np.intp], fill_value, allow_fill: bool = ..., mask: npt.NDArray[np.bool_] | None) -> ArrayLike:
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
def take_2d_multi(arr: np.ndarray, indexer: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]], fill_value: float = ...) -> np.ndarray:
    """
    Specialized Cython take which sets NaN values in one pass.
    """

_get_take_nd_function_cached: functools._lru_cache_wrapper
def _get_take_nd_function(ndim: int, arr_dtype: np.dtype, out_dtype: np.dtype, axis: AxisInt = ..., mask_info):
    '''
    Get the appropriate "take" implementation for the given dimension, axis
    and dtypes.
    '''
def _view_wrapper(f, arr_dtype, out_dtype, fill_wrap): ...
def _convert_wrapper(f, conv_dtype): ...

_take_1d_dict: dict
_take_2d_axis0_dict: dict
_take_2d_axis1_dict: dict
_take_2d_multi_dict: dict
def _take_nd_object(arr: np.ndarray, indexer: npt.NDArray[np.intp], out: np.ndarray, axis: AxisInt, fill_value, mask_info) -> None: ...
def _take_2d_multi_object(arr: np.ndarray, indexer: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]], out: np.ndarray, fill_value, mask_info) -> None: ...
def _take_preprocess_indexer_and_fill_value(arr: np.ndarray, indexer: npt.NDArray[np.intp], fill_value, allow_fill: bool, mask: npt.NDArray[np.bool_] | None): ...
