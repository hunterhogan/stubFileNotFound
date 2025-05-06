import numpy as np
from _typeshed import Incomplete
from pandas import Series as Series
from pandas._libs.tslibs import NaT as NaT, iNaT as iNaT
from pandas._typing import ArrayLike as ArrayLike, DtypeObj as DtypeObj, NDFrame as NDFrame, NDFrameT as NDFrameT, Scalar as Scalar, npt as npt
from pandas.core.dtypes.common import DT64NS_DTYPE as DT64NS_DTYPE, TD64NS_DTYPE as TD64NS_DTYPE, ensure_object as ensure_object, is_scalar as is_scalar, is_string_or_object_np_dtype as is_string_or_object_np_dtype
from pandas.core.dtypes.dtypes import CategoricalDtype as CategoricalDtype, DatetimeTZDtype as DatetimeTZDtype, ExtensionDtype as ExtensionDtype, IntervalDtype as IntervalDtype, PeriodDtype as PeriodDtype
from pandas.core.dtypes.generic import ABCDataFrame as ABCDataFrame, ABCExtensionArray as ABCExtensionArray, ABCIndex as ABCIndex, ABCMultiIndex as ABCMultiIndex, ABCSeries as ABCSeries
from pandas.core.indexes.base import Index as Index
from re import Pattern
from typing import overload

isposinf_scalar: Incomplete
isneginf_scalar: Incomplete
nan_checker: Incomplete
INF_AS_NA: bool
_dtype_object: Incomplete
_dtype_str: Incomplete

@overload
def isna(obj: Scalar | Pattern) -> bool: ...
@overload
def isna(obj: ArrayLike | Index | list) -> npt.NDArray[np.bool_]: ...
@overload
def isna(obj: NDFrameT) -> NDFrameT: ...
@overload
def isna(obj: NDFrameT | ArrayLike | Index | list) -> NDFrameT | npt.NDArray[np.bool_]: ...
@overload
def isna(obj: object) -> bool | npt.NDArray[np.bool_] | NDFrame: ...
isnull = isna

def _isna(obj, inf_as_na: bool = False):
    """
    Detect missing values, treating None, NaN or NA as null. Infinite
    values will also be treated as null if inf_as_na is True.

    Parameters
    ----------
    obj: ndarray or object value
        Input array or scalar value.
    inf_as_na: bool
        Whether to treat infinity as null.

    Returns
    -------
    boolean ndarray or boolean
    """
def _use_inf_as_na(key) -> None:
    """
    Option change callback for na/inf behaviour.

    Choose which replacement for numpy.isnan / -numpy.isfinite is used.

    Parameters
    ----------
    flag: bool
        True means treat None, NaN, INF, -INF as null (old way),
        False means None and NaN are null, but INF, -INF are not null
        (new way).

    Notes
    -----
    This approach to setting global module values is discussed and
    approved here:

    * https://stackoverflow.com/questions/4859217/
      programmatically-creating-variables-in-python/4859312#4859312
    """
def _isna_array(values: ArrayLike, inf_as_na: bool = False):
    """
    Return an array indicating which values of the input array are NaN / NA.

    Parameters
    ----------
    obj: ndarray or ExtensionArray
        The input array whose elements are to be checked.
    inf_as_na: bool
        Whether or not to treat infinite values as NA.

    Returns
    -------
    array-like
        Array of boolean values denoting the NA status of each element.
    """
def _isna_string_dtype(values: np.ndarray, inf_as_na: bool) -> npt.NDArray[np.bool_]: ...
def _has_record_inf_value(record_as_array: np.ndarray) -> np.bool_: ...
def _isna_recarray_dtype(values: np.rec.recarray, inf_as_na: bool) -> npt.NDArray[np.bool_]: ...
@overload
def notna(obj: Scalar) -> bool: ...
@overload
def notna(obj: ArrayLike | Index | list) -> npt.NDArray[np.bool_]: ...
@overload
def notna(obj: NDFrameT) -> NDFrameT: ...
@overload
def notna(obj: NDFrameT | ArrayLike | Index | list) -> NDFrameT | npt.NDArray[np.bool_]: ...
@overload
def notna(obj: object) -> bool | npt.NDArray[np.bool_] | NDFrame: ...
notnull = notna

def array_equivalent(left, right, strict_nan: bool = False, dtype_equal: bool = False) -> bool:
    """
    True if two arrays, left and right, have equal non-NaN elements, and NaNs
    in corresponding locations.  False otherwise. It is assumed that left and
    right are NumPy arrays of the same dtype. The behavior of this function
    (particularly with respect to NaNs) is not defined if the dtypes are
    different.

    Parameters
    ----------
    left, right : ndarrays
    strict_nan : bool, default False
        If True, consider NaN and None to be different.
    dtype_equal : bool, default False
        Whether `left` and `right` are known to have the same dtype
        according to `is_dtype_equal`. Some methods like `BlockManager.equals`.
        require that the dtypes match. Setting this to ``True`` can improve
        performance, but will give different results for arrays that are
        equal but different dtypes.

    Returns
    -------
    b : bool
        Returns True if the arrays are equivalent.

    Examples
    --------
    >>> array_equivalent(
    ...     np.array([1, 2, np.nan]),
    ...     np.array([1, 2, np.nan]))
    True
    >>> array_equivalent(
    ...     np.array([1, np.nan, 2]),
    ...     np.array([1, 2, np.nan]))
    False
    """
def _array_equivalent_float(left: np.ndarray, right: np.ndarray) -> bool: ...
def _array_equivalent_datetimelike(left: np.ndarray, right: np.ndarray): ...
def _array_equivalent_object(left: np.ndarray, right: np.ndarray, strict_nan: bool): ...
def array_equals(left: ArrayLike, right: ArrayLike) -> bool:
    """
    ExtensionArray-compatible implementation of array_equivalent.
    """
def infer_fill_value(val):
    """
    infer the fill value for the nan/NaT from the provided
    scalar/ndarray/list-like if we are a NaT, return the correct dtyped
    element to provide proper block construction
    """
def construct_1d_array_from_inferred_fill_value(value: object, length: int) -> ArrayLike: ...
def maybe_fill(arr: np.ndarray) -> np.ndarray:
    """
    Fill numpy.ndarray with NaN, unless we have a integer or boolean dtype.
    """
def na_value_for_dtype(dtype: DtypeObj, compat: bool = True):
    """
    Return a dtype compat na value

    Parameters
    ----------
    dtype : string / dtype
    compat : bool, default True

    Returns
    -------
    np.dtype or a pandas dtype

    Examples
    --------
    >>> na_value_for_dtype(np.dtype('int64'))
    0
    >>> na_value_for_dtype(np.dtype('int64'), compat=False)
    nan
    >>> na_value_for_dtype(np.dtype('float64'))
    nan
    >>> na_value_for_dtype(np.dtype('bool'))
    False
    >>> na_value_for_dtype(np.dtype('datetime64[ns]'))
    numpy.datetime64('NaT')
    """
def remove_na_arraylike(arr: Series | Index | np.ndarray):
    """
    Return array-like containing only true/non-NaN values, possibly empty.
    """
def is_valid_na_for_dtype(obj, dtype: DtypeObj) -> bool:
    """
    isna check that excludes incompatible dtypes

    Parameters
    ----------
    obj : object
    dtype : np.datetime64, np.timedelta64, DatetimeTZDtype, or PeriodDtype

    Returns
    -------
    bool
    """
def isna_all(arr: ArrayLike) -> bool:
    """
    Optimized equivalent to isna(arr).all()
    """
