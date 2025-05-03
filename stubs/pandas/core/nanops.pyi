import np
import npt
import pandas._libs.lib as lib
from builtins import AxisInt, Shape
from pandas._config.config import get_option as get_option
from pandas._libs.lib import is_complex as is_complex, is_float as is_float, is_integer as is_integer
from pandas._libs.tslibs.nattype import NaT as NaT, NaTType as NaTType
from pandas._typing import F as F
from pandas.compat._optional import import_optional_dependency as import_optional_dependency
from pandas.core.dtypes.common import is_float_dtype as is_float_dtype, is_numeric_dtype as is_numeric_dtype, is_object_dtype as is_object_dtype, needs_i8_conversion as needs_i8_conversion, pandas_dtype as pandas_dtype
from pandas.core.dtypes.missing import isna as isna, na_value_for_dtype as na_value_for_dtype, notna as notna
from pandas.util._exceptions import find_stack_level as find_stack_level
from typing import Any, ArrayLike, Callable, CorrelationMethod, Dtype, DtypeObj, Scalar

iNaT: int
npt: None
bn: None
_BOTTLENECK_INSTALLED: bool
_USE_BOTTLENECK: bool
def set_use_bottleneck(v: bool = ...) -> None: ...

class disallow:
    def __init__(self, *dtypes: Dtype) -> None: ...
    def check(self, obj) -> bool: ...
    def __call__(self, f: F) -> F: ...

class bottleneck_switch:
    def __init__(self, name, **kwargs) -> None: ...
    def __call__(self, alt: F) -> F: ...
def _bn_ok_dtype(dtype: DtypeObj, name: str) -> bool: ...
def _has_infs(result) -> bool: ...
def _get_fill_value(dtype: DtypeObj, fill_value: Scalar | None, fill_value_typ):
    """return the correct fill value for the dtype of the values"""
def _maybe_get_mask(values: np.ndarray, skipna: bool, mask: npt.NDArray[np.bool_] | None) -> npt.NDArray[np.bool_] | None:
    """
    Compute a mask if and only if necessary.

    This function will compute a mask iff it is necessary. Otherwise,
    return the provided mask (potentially None) when a mask does not need to be
    computed.

    A mask is never necessary if the values array is of boolean or integer
    dtypes, as these are incapable of storing NaNs. If passing a NaN-capable
    dtype that is interpretable as either boolean or integer data (eg,
    timedelta64), a mask must be provided.

    If the skipna parameter is False, a new mask will not be computed.

    The mask is computed using isna() by default. Setting invert=True selects
    notna() as the masking function.

    Parameters
    ----------
    values : ndarray
        input array to potentially compute mask for
    skipna : bool
        boolean for whether NaNs should be skipped
    mask : Optional[ndarray]
        nan-mask if known

    Returns
    -------
    Optional[np.ndarray[bool]]
    """
def _get_values(values: np.ndarray, skipna: bool, fill_value: Any, fill_value_typ: str | None, mask: npt.NDArray[np.bool_] | None) -> tuple[np.ndarray, npt.NDArray[np.bool_] | None]:
    """
    Utility to get the values view, mask, dtype, dtype_max, and fill_value.

    If both mask and fill_value/fill_value_typ are not None and skipna is True,
    the values array will be copied.

    For input arrays of boolean or integer dtypes, copies will only occur if a
    precomputed mask, a fill_value/fill_value_typ, and skipna=True are
    provided.

    Parameters
    ----------
    values : ndarray
        input array to potentially compute mask for
    skipna : bool
        boolean for whether NaNs should be skipped
    fill_value : Any
        value to fill NaNs with
    fill_value_typ : str
        Set to '+inf' or '-inf' to handle dtype-specific infinities
    mask : Optional[np.ndarray[bool]]
        nan-mask if known

    Returns
    -------
    values : ndarray
        Potential copy of input value array
    mask : Optional[ndarray[bool]]
        Mask for values, if deemed necessary to compute
    """
def _get_dtype_max(dtype: np.dtype) -> np.dtype: ...
def _na_ok_dtype(dtype: DtypeObj) -> bool: ...
def _wrap_results(result, dtype: np.dtype, fill_value):
    """wrap our results if needed"""
def _datetimelike_compat(func: F) -> F:
    """
    If we have datetime64 or timedelta64 values, ensure we have a correct
    mask before calling the wrapped function, then cast back afterwards.
    """
def _na_for_min_count(values: np.ndarray, axis: AxisInt | None) -> Scalar | np.ndarray:
    """
    Return the missing value for `values`.

    Parameters
    ----------
    values : ndarray
    axis : int or None
        axis for the reduction, required if values.ndim > 1.

    Returns
    -------
    result : scalar or ndarray
        For 1-D values, returns a scalar of the correct missing type.
        For 2-D values, returns a 1-D array where each element is missing.
    """
def maybe_operate_rowwise(func: F) -> F:
    """
    NumPy operations on C-contiguous ndarrays with axis=1 can be
    very slow if axis 1 >> axis 0.
    Operate row-by-row and concatenate the results.
    """
def nanany(values: np.ndarray, *, axis: AxisInt | None, skipna: bool = ..., mask: npt.NDArray[np.bool_] | None) -> bool:
    """
    Check if any elements along an axis evaluate to True.

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : bool

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, 2])
    >>> nanops.nanany(s.values)
    True

    >>> from pandas.core import nanops
    >>> s = pd.Series([np.nan])
    >>> nanops.nanany(s.values)
    False
    """
def nanall(values: np.ndarray, *, axis: AxisInt | None, skipna: bool = ..., mask: npt.NDArray[np.bool_] | None) -> bool:
    """
    Check if all elements along an axis evaluate to True.

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : bool

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, 2, np.nan])
    >>> nanops.nanall(s.values)
    True

    >>> from pandas.core import nanops
    >>> s = pd.Series([1, 0])
    >>> nanops.nanall(s.values)
    False
    """
def nansum(*args, **kwargs) -> float:
    """
    Sum the elements along an axis ignoring NaNs

    Parameters
    ----------
    values : ndarray[dtype]
    axis : int, optional
    skipna : bool, default True
    min_count: int, default 0
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : dtype

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, 2, np.nan])
    >>> nanops.nansum(s.values)
    3.0
    """
def _mask_datetimelike_result(result: np.ndarray | np.datetime64 | np.timedelta64, axis: AxisInt | None, mask: npt.NDArray[np.bool_], orig_values: np.ndarray) -> np.ndarray | np.datetime64 | np.timedelta64 | NaTType: ...
def nanmean(values: np.ndarray, *, axis: AxisInt | None, skipna: bool = ..., **kwds) -> float:
    """
    Compute the mean of the element along an axis ignoring NaNs

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    float
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, 2, np.nan])
    >>> nanops.nanmean(s.values)
    1.5
    """
def nanmedian(values, *, axis: AxisInt | None, skipna: bool = ..., **kwds):
    """
    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : float
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, np.nan, 2, 2])
    >>> nanops.nanmedian(s.values)
    2.0
    """
def _get_empty_reduction_result(shape: Shape, axis: AxisInt) -> np.ndarray:
    """
    The result from a reduction on an empty ndarray.

    Parameters
    ----------
    shape : Tuple[int, ...]
    axis : int

    Returns
    -------
    np.ndarray
    """
def _get_counts_nanvar(values_shape: Shape, mask: npt.NDArray[np.bool_] | None, axis: AxisInt | None, ddof: int, dtype: np.dtype = ...) -> tuple[float | np.ndarray, float | np.ndarray]:
    """
    Get the count of non-null values along an axis, accounting
    for degrees of freedom.

    Parameters
    ----------
    values_shape : Tuple[int, ...]
        shape tuple from values ndarray, used if mask is None
    mask : Optional[ndarray[bool]]
        locations in values that should be considered missing
    axis : Optional[int]
        axis to count along
    ddof : int
        degrees of freedom
    dtype : type, optional
        type to use for count

    Returns
    -------
    count : int, np.nan or np.ndarray
    d : int, np.nan or np.ndarray
    """
def nanstd(values, *, axis: AxisInt | None, skipna: bool = ..., **kwds):
    """
    Compute the standard deviation along given axis while ignoring NaNs

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    ddof : int, default 1
        Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
        where N represents the number of elements.
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : float
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, np.nan, 2, 3])
    >>> nanops.nanstd(s.values)
    1.0
    """
def nanvar(*args, **kwargs):
    """
    Compute the variance along given axis while ignoring NaNs

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    ddof : int, default 1
        Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
        where N represents the number of elements.
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : float
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, np.nan, 2, 3])
    >>> nanops.nanvar(s.values)
    1.0
    """
def nansem(*args, **kwargs) -> float:
    """
    Compute the standard error in the mean along given axis while ignoring NaNs

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    ddof : int, default 1
        Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
        where N represents the number of elements.
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : float64
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, np.nan, 2, 3])
    >>> nanops.nansem(s.values)
     0.5773502691896258
    """
def _nanminmax(meth, fill_value_typ): ...
def nanmin(values: np.ndarray, *, axis: AxisInt | None, skipna: bool = ..., **kwds): ...
def nanmax(values: np.ndarray, *, axis: AxisInt | None, skipna: bool = ..., **kwds): ...
def nanargmax(values: np.ndarray, *, axis: AxisInt | None, skipna: bool = ..., mask: npt.NDArray[np.bool_] | None) -> int | np.ndarray:
    """
    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : int or ndarray[int]
        The index/indices  of max value in specified axis or -1 in the NA case

    Examples
    --------
    >>> from pandas.core import nanops
    >>> arr = np.array([1, 2, 3, np.nan, 4])
    >>> nanops.nanargmax(arr)
    4

    >>> arr = np.array(range(12), dtype=np.float64).reshape(4, 3)
    >>> arr[2:, 2] = np.nan
    >>> arr
    array([[ 0.,  1.,  2.],
           [ 3.,  4.,  5.],
           [ 6.,  7., nan],
           [ 9., 10., nan]])
    >>> nanops.nanargmax(arr, axis=1)
    array([2, 2, 1, 1])
    """
def nanargmin(values: np.ndarray, *, axis: AxisInt | None, skipna: bool = ..., mask: npt.NDArray[np.bool_] | None) -> int | np.ndarray:
    """
    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : int or ndarray[int]
        The index/indices of min value in specified axis or -1 in the NA case

    Examples
    --------
    >>> from pandas.core import nanops
    >>> arr = np.array([1, 2, 3, np.nan, 4])
    >>> nanops.nanargmin(arr)
    0

    >>> arr = np.array(range(12), dtype=np.float64).reshape(4, 3)
    >>> arr[2:, 0] = np.nan
    >>> arr
    array([[ 0.,  1.,  2.],
           [ 3.,  4.,  5.],
           [nan,  7.,  8.],
           [nan, 10., 11.]])
    >>> nanops.nanargmin(arr, axis=1)
    array([0, 0, 1, 1])
    """
def nanskew(*args, **kwargs) -> float:
    """
    Compute the sample skewness.

    The statistic computed here is the adjusted Fisher-Pearson standardized
    moment coefficient G1. The algorithm computes this coefficient directly
    from the second and third central moment.

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : float64
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, np.nan, 1, 2])
    >>> nanops.nanskew(s.values)
    1.7320508075688787
    """
def nankurt(*args, **kwargs) -> float:
    """
    Compute the sample excess kurtosis

    The statistic computed here is the adjusted Fisher-Pearson standardized
    moment coefficient G2, computed directly from the second and fourth
    central moment.

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : float64
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, np.nan, 1, 3, 2])
    >>> nanops.nankurt(s.values)
    -1.2892561983471076
    """
def nanprod(*args, **kwargs) -> float:
    """
    Parameters
    ----------
    values : ndarray[dtype]
    axis : int, optional
    skipna : bool, default True
    min_count: int, default 0
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    Dtype
        The product of all elements on a given axis. ( NaNs are treated as 1)

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, 2, 3, np.nan])
    >>> nanops.nanprod(s.values)
    6.0
    """
def _maybe_arg_null_out(result: np.ndarray, axis: AxisInt | None, mask: npt.NDArray[np.bool_] | None, skipna: bool) -> np.ndarray | int: ...
def _get_counts(values_shape: Shape, mask: npt.NDArray[np.bool_] | None, axis: AxisInt | None, dtype: np.dtype[np.floating] = ...) -> np.floating | npt.NDArray[np.floating]:
    """
    Get the count of non-null values along an axis

    Parameters
    ----------
    values_shape : tuple of int
        shape tuple from values ndarray, used if mask is None
    mask : Optional[ndarray[bool]]
        locations in values that should be considered missing
    axis : Optional[int]
        axis to count along
    dtype : type, optional
        type to use for count

    Returns
    -------
    count : scalar or array
    """
def _maybe_null_out(result: np.ndarray | float | NaTType, axis: AxisInt | None, mask: npt.NDArray[np.bool_] | None, shape: tuple[int, ...], min_count: int = ...) -> np.ndarray | float | NaTType:
    """
    Returns
    -------
    Dtype
        The product of all elements on a given axis. ( NaNs are treated as 1)
    """
def check_below_min_count(shape: tuple[int, ...], mask: npt.NDArray[np.bool_] | None, min_count: int) -> bool:
    """
    Check for the `min_count` keyword. Returns True if below `min_count` (when
    missing value should be returned from the reduction).

    Parameters
    ----------
    shape : tuple
        The shape of the values (`values.shape`).
    mask : ndarray[bool] or None
        Boolean numpy array (typically of same shape as `shape`) or None.
    min_count : int
        Keyword passed through from sum/prod call.

    Returns
    -------
    bool
    """
def _zero_out_fperr(arg): ...
def nancorr(*args, **kwargs) -> float:
    """
    a, b: ndarrays
    """
def get_corr_func(method: CorrelationMethod) -> Callable[[np.ndarray, np.ndarray], float]: ...
def nancov(*args, **kwargs) -> float: ...
def _ensure_numeric(x): ...
def na_accum_func(values: ArrayLike, accum_func, *, skipna: bool) -> ArrayLike:
    """
    Cumulative function with skipna support.

    Parameters
    ----------
    values : np.ndarray or ExtensionArray
    accum_func : {np.cumprod, np.maximum.accumulate, np.cumsum, np.minimum.accumulate}
    skipna : bool

    Returns
    -------
    np.ndarray or ExtensionArray
    """
