import numpy as np
from _typeshed import Incomplete
from collections.abc import Sequence, Sized
from pandas import Index as Index
from pandas._libs import Interval as Interval, Period as Period, lib as lib
from pandas._libs.missing import NA as NA, NAType as NAType, checknull as checknull
from pandas._libs.tslibs import NaT as NaT, OutOfBoundsDatetime as OutOfBoundsDatetime, OutOfBoundsTimedelta as OutOfBoundsTimedelta, Timedelta as Timedelta, Timestamp as Timestamp, is_supported_dtype as is_supported_dtype
from pandas._typing import ArrayLike as ArrayLike, Dtype as Dtype, DtypeObj as DtypeObj, NumpyIndexT as NumpyIndexT, Scalar as Scalar, npt as npt
from pandas.core.arrays import Categorical as Categorical, DatetimeArray as DatetimeArray, ExtensionArray as ExtensionArray, IntervalArray as IntervalArray, PeriodArray as PeriodArray, TimedeltaArray as TimedeltaArray
from pandas.core.dtypes.common import ensure_int16 as ensure_int16, ensure_int32 as ensure_int32, ensure_int64 as ensure_int64, ensure_int8 as ensure_int8, ensure_object as ensure_object, ensure_str as ensure_str, is_bool as is_bool, is_complex as is_complex, is_float as is_float, is_integer as is_integer, is_object_dtype as is_object_dtype, is_scalar as is_scalar, is_string_dtype as is_string_dtype
from pandas.core.dtypes.dtypes import ArrowDtype as ArrowDtype, BaseMaskedDtype as BaseMaskedDtype, CategoricalDtype as CategoricalDtype, DatetimeTZDtype as DatetimeTZDtype, ExtensionDtype as ExtensionDtype, IntervalDtype as IntervalDtype, PandasExtensionDtype as PandasExtensionDtype, PeriodDtype as PeriodDtype
from pandas.core.dtypes.generic import ABCExtensionArray as ABCExtensionArray, ABCIndex as ABCIndex, ABCSeries as ABCSeries
from pandas.core.dtypes.missing import is_valid_na_for_dtype as is_valid_na_for_dtype, isna as isna, na_value_for_dtype as na_value_for_dtype, notna as notna
from pandas.errors import IntCastingNaNError as IntCastingNaNError, LossySetitemError as LossySetitemError
from typing import Any, Literal, TypeVar, overload

_int8_max: Incomplete
_int16_max: Incomplete
_int32_max: Incomplete
_dtype_obj: Incomplete
NumpyArrayT = TypeVar('NumpyArrayT', bound=np.ndarray)

def maybe_convert_platform(values: list | tuple | range | np.ndarray | ExtensionArray) -> ArrayLike:
    """try to do platform conversion, allow ndarray or list here"""
def is_nested_object(obj) -> bool:
    """
    return a boolean if we have a nested object, e.g. a Series with 1 or
    more Series elements

    This may not be necessarily be performant.

    """
def maybe_box_datetimelike(value: Scalar, dtype: Dtype | None = None) -> Scalar:
    """
    Cast scalar to Timestamp or Timedelta if scalar is datetime-like
    and dtype is not object.

    Parameters
    ----------
    value : scalar
    dtype : Dtype, optional

    Returns
    -------
    scalar
    """
def maybe_box_native(value: Scalar | None | NAType) -> Scalar | None | NAType:
    """
    If passed a scalar cast the scalar to a python native type.

    Parameters
    ----------
    value : scalar or Series

    Returns
    -------
    scalar or Series
    """
def _maybe_unbox_datetimelike(value: Scalar, dtype: DtypeObj) -> Scalar:
    '''
    Convert a Timedelta or Timestamp to timedelta64 or datetime64 for setting
    into a numpy array.  Failing to unbox would risk dropping nanoseconds.

    Notes
    -----
    Caller is responsible for checking dtype.kind in "mM"
    '''
def _disallow_mismatched_datetimelike(value, dtype: DtypeObj):
    '''
    numpy allows np.array(dt64values, dtype="timedelta64[ns]") and
    vice-versa, but we do not want to allow this, so we need to
    check explicitly
    '''
@overload
def maybe_downcast_to_dtype(result: np.ndarray, dtype: str | np.dtype) -> np.ndarray: ...
@overload
def maybe_downcast_to_dtype(result: ExtensionArray, dtype: str | np.dtype) -> ArrayLike: ...
@overload
def maybe_downcast_numeric(result: np.ndarray, dtype: np.dtype, do_round: bool = False) -> np.ndarray: ...
@overload
def maybe_downcast_numeric(result: ExtensionArray, dtype: DtypeObj, do_round: bool = False) -> ArrayLike: ...
def maybe_upcast_numeric_to_64bit(arr: NumpyIndexT) -> NumpyIndexT:
    """
    If array is a int/uint/float bit size lower than 64 bit, upcast it to 64 bit.

    Parameters
    ----------
    arr : ndarray or ExtensionArray

    Returns
    -------
    ndarray or ExtensionArray
    """
def maybe_cast_pointwise_result(result: ArrayLike, dtype: DtypeObj, numeric_only: bool = False, same_dtype: bool = True) -> ArrayLike:
    """
    Try casting result of a pointwise operation back to the original dtype if
    appropriate.

    Parameters
    ----------
    result : array-like
        Result to cast.
    dtype : np.dtype or ExtensionDtype
        Input Series from which result was calculated.
    numeric_only : bool, default False
        Whether to cast only numerics or datetimes as well.
    same_dtype : bool, default True
        Specify dtype when calling _from_sequence

    Returns
    -------
    result : array-like
        result maybe casted to the dtype.
    """
def _maybe_cast_to_extension_array(cls, obj: ArrayLike, dtype: ExtensionDtype | None = None) -> ArrayLike:
    """
    Call to `_from_sequence` that returns the object unchanged on Exception.

    Parameters
    ----------
    cls : class, subclass of ExtensionArray
    obj : arraylike
        Values to pass to cls._from_sequence
    dtype : ExtensionDtype, optional

    Returns
    -------
    ExtensionArray or obj
    """
@overload
def ensure_dtype_can_hold_na(dtype: np.dtype) -> np.dtype: ...
@overload
def ensure_dtype_can_hold_na(dtype: ExtensionDtype) -> ExtensionDtype: ...

_canonical_nans: Incomplete

def maybe_promote(dtype: np.dtype, fill_value=...):
    """
    Find the minimal dtype that can hold both the given dtype and fill_value.

    Parameters
    ----------
    dtype : np.dtype
    fill_value : scalar, default np.nan

    Returns
    -------
    dtype
        Upcasted from dtype argument if necessary.
    fill_value
        Upcasted from fill_value argument if necessary.

    Raises
    ------
    ValueError
        If fill_value is a non-scalar and dtype is not object.
    """
def _maybe_promote_cached(dtype, fill_value, fill_value_type): ...
def _maybe_promote(dtype: np.dtype, fill_value=...): ...
def _ensure_dtype_type(value, dtype: np.dtype):
    """
    Ensure that the given value is an instance of the given dtype.

    e.g. if out dtype is np.complex64_, we should have an instance of that
    as opposed to a python complex object.

    Parameters
    ----------
    value : object
    dtype : np.dtype

    Returns
    -------
    object
    """
def infer_dtype_from(val) -> tuple[DtypeObj, Any]:
    """
    Interpret the dtype from a scalar or array.

    Parameters
    ----------
    val : object
    """
def infer_dtype_from_scalar(val) -> tuple[DtypeObj, Any]:
    """
    Interpret the dtype from a scalar.

    Parameters
    ----------
    val : object
    """
def dict_compat(d: dict[Scalar, Scalar]) -> dict[Scalar, Scalar]:
    """
    Convert datetimelike-keyed dicts to a Timestamp-keyed dict.

    Parameters
    ----------
    d: dict-like object

    Returns
    -------
    dict
    """
def infer_dtype_from_array(arr) -> tuple[DtypeObj, ArrayLike]:
    """
    Infer the dtype from an array.

    Parameters
    ----------
    arr : array

    Returns
    -------
    tuple (pandas-compat dtype, array)


    Examples
    --------
    >>> np.asarray([1, '1'])
    array(['1', '1'], dtype='<U21')

    >>> infer_dtype_from_array([1, '1'])
    (dtype('O'), [1, '1'])
    """
def _maybe_infer_dtype_type(element):
    '''
    Try to infer an object\'s dtype, for use in arithmetic ops.

    Uses `element.dtype` if that\'s available.
    Objects implementing the iterator protocol are cast to a NumPy array,
    and from there the array\'s type is used.

    Parameters
    ----------
    element : object
        Possibly has a `.dtype` attribute, and possibly the iterator
        protocol.

    Returns
    -------
    tipo : type

    Examples
    --------
    >>> from collections import namedtuple
    >>> Foo = namedtuple("Foo", "dtype")
    >>> _maybe_infer_dtype_type(Foo(np.dtype("i8")))
    dtype(\'int64\')
    '''
def invalidate_string_dtypes(dtype_set: set[DtypeObj]) -> None:
    """
    Change string like dtypes to object for
    ``DataFrame.select_dtypes()``.
    """
def coerce_indexer_dtype(indexer, categories) -> np.ndarray:
    """coerce the indexer input array to the smallest dtype possible"""
def convert_dtypes(input_array: ArrayLike, convert_string: bool = True, convert_integer: bool = True, convert_boolean: bool = True, convert_floating: bool = True, infer_objects: bool = False, dtype_backend: Literal['numpy_nullable', 'pyarrow'] = 'numpy_nullable') -> DtypeObj:
    '''
    Convert objects to best possible type, and optionally,
    to types supporting ``pd.NA``.

    Parameters
    ----------
    input_array : ExtensionArray or np.ndarray
    convert_string : bool, default True
        Whether object dtypes should be converted to ``StringDtype()``.
    convert_integer : bool, default True
        Whether, if possible, conversion can be done to integer extension types.
    convert_boolean : bool, defaults True
        Whether object dtypes should be converted to ``BooleanDtypes()``.
    convert_floating : bool, defaults True
        Whether, if possible, conversion can be done to floating extension types.
        If `convert_integer` is also True, preference will be give to integer
        dtypes if the floats can be faithfully casted to integers.
    infer_objects : bool, defaults False
        Whether to also infer objects to float/int if possible. Is only hit if the
        object array contains pd.NA.
    dtype_backend : {\'numpy_nullable\', \'pyarrow\'}, default \'numpy_nullable\'
        Back-end data type applied to the resultant :class:`DataFrame`
        (still experimental). Behaviour is as follows:

        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
          (default).
        * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype`
          DataFrame.

        .. versionadded:: 2.0

    Returns
    -------
    np.dtype, or ExtensionDtype
    '''
def maybe_infer_to_datetimelike(value: npt.NDArray[np.object_]) -> np.ndarray | DatetimeArray | TimedeltaArray | PeriodArray | IntervalArray:
    """
    we might have a array (or single object) that is datetime like,
    and no dtype is passed don't change the value unless we find a
    datetime/timedelta set

    this is pretty strict in that a datetime/timedelta is REQUIRED
    in addition to possible nulls/string likes

    Parameters
    ----------
    value : np.ndarray[object]

    Returns
    -------
    np.ndarray, DatetimeArray, TimedeltaArray, PeriodArray, or IntervalArray

    """
def maybe_cast_to_datetime(value: np.ndarray | list, dtype: np.dtype) -> ExtensionArray | np.ndarray:
    """
    try to cast the array/value to a datetimelike dtype, converting float
    nan to iNaT

    Caller is responsible for handling ExtensionDtype cases and non dt64/td64
    cases.
    """
def _ensure_nanosecond_dtype(dtype: DtypeObj) -> None:
    '''
    Convert dtypes with granularity less than nanosecond to nanosecond

    >>> _ensure_nanosecond_dtype(np.dtype("M8[us]"))

    >>> _ensure_nanosecond_dtype(np.dtype("M8[D]"))
    Traceback (most recent call last):
        ...
    TypeError: dtype=datetime64[D] is not supported. Supported resolutions are \'s\', \'ms\', \'us\', and \'ns\'

    >>> _ensure_nanosecond_dtype(np.dtype("m8[ps]"))
    Traceback (most recent call last):
        ...
    TypeError: dtype=timedelta64[ps] is not supported. Supported resolutions are \'s\', \'ms\', \'us\', and \'ns\'
    '''
def find_result_type(left_dtype: DtypeObj, right: Any) -> DtypeObj:
    """
    Find the type/dtype for the result of an operation between objects.

    This is similar to find_common_type, but looks at the right object instead
    of just its dtype. This can be useful in particular when the right
    object does not have a `dtype`.

    Parameters
    ----------
    left_dtype : np.dtype or ExtensionDtype
    right : Any

    Returns
    -------
    np.dtype or ExtensionDtype

    See also
    --------
    find_common_type
    numpy.result_type
    """
def common_dtype_categorical_compat(objs: Sequence[Index | ArrayLike], dtype: DtypeObj) -> DtypeObj:
    """
    Update the result of find_common_type to account for NAs in a Categorical.

    Parameters
    ----------
    objs : list[np.ndarray | ExtensionArray | Index]
    dtype : np.dtype or ExtensionDtype

    Returns
    -------
    np.dtype or ExtensionDtype
    """
def np_find_common_type(*dtypes: np.dtype) -> np.dtype:
    """
    np.find_common_type implementation pre-1.25 deprecation using np.result_type
    https://github.com/pandas-dev/pandas/pull/49569#issuecomment-1308300065

    Parameters
    ----------
    dtypes : np.dtypes

    Returns
    -------
    np.dtype
    """
@overload
def find_common_type(types: list[np.dtype]) -> np.dtype: ...
@overload
def find_common_type(types: list[ExtensionDtype]) -> DtypeObj: ...
@overload
def find_common_type(types: list[DtypeObj]) -> DtypeObj: ...
def construct_2d_arraylike_from_scalar(value: Scalar, length: int, width: int, dtype: np.dtype, copy: bool) -> np.ndarray: ...
def construct_1d_arraylike_from_scalar(value: Scalar, length: int, dtype: DtypeObj | None) -> ArrayLike:
    """
    create a np.ndarray / pandas type of specified shape and dtype
    filled with values

    Parameters
    ----------
    value : scalar value
    length : int
    dtype : pandas_dtype or np.dtype

    Returns
    -------
    np.ndarray / pandas type of length, filled with value

    """
def _maybe_box_and_unbox_datetimelike(value: Scalar, dtype: DtypeObj): ...
def construct_1d_object_array_from_listlike(values: Sized) -> np.ndarray:
    """
    Transform any list-like object in a 1-dimensional numpy array of object
    dtype.

    Parameters
    ----------
    values : any iterable which has a len()

    Raises
    ------
    TypeError
        * If `values` does not have a len()

    Returns
    -------
    1-dimensional numpy array of dtype object
    """
def maybe_cast_to_integer_array(arr: list | np.ndarray, dtype: np.dtype) -> np.ndarray:
    '''
    Takes any dtype and returns the casted version, raising for when data is
    incompatible with integer/unsigned integer dtypes.

    Parameters
    ----------
    arr : np.ndarray or list
        The array to cast.
    dtype : np.dtype
        The integer dtype to cast the array to.

    Returns
    -------
    ndarray
        Array of integer or unsigned integer dtype.

    Raises
    ------
    OverflowError : the dtype is incompatible with the data
    ValueError : loss of precision has occurred during casting

    Examples
    --------
    If you try to coerce negative values to unsigned integers, it raises:

    >>> pd.Series([-1], dtype="uint64")
    Traceback (most recent call last):
        ...
    OverflowError: Trying to coerce negative values to unsigned integers

    Also, if you try to coerce float values to integers, it raises:

    >>> maybe_cast_to_integer_array([1, 2, 3.5], dtype=np.dtype("int64"))
    Traceback (most recent call last):
        ...
    ValueError: Trying to coerce float values to integers
    '''
def can_hold_element(arr: ArrayLike, element: Any) -> bool:
    """
    Can we do an inplace setitem with this element in an array with this dtype?

    Parameters
    ----------
    arr : np.ndarray or ExtensionArray
    element : Any

    Returns
    -------
    bool
    """
def np_can_hold_element(dtype: np.dtype, element: Any) -> Any:
    '''
    Raise if we cannot losslessly set this element into an ndarray with this dtype.

    Specifically about places where we disagree with numpy.  i.e. there are
    cases where numpy will raise in doing the setitem that we do not check
    for here, e.g. setting str "X" into a numeric ndarray.

    Returns
    -------
    Any
        The element, potentially cast to the dtype.

    Raises
    ------
    ValueError : If we cannot losslessly store this element with this dtype.
    '''
def _dtype_can_hold_range(rng: range, dtype: np.dtype) -> bool:
    """
    _maybe_infer_dtype_type infers to int64 (and float64 for very large endpoints),
    but in many cases a range can be held by a smaller integer dtype.
    Check if this is one of those cases.
    """
def np_can_cast_scalar(element: Scalar, dtype: np.dtype) -> bool:
    """
    np.can_cast pandas-equivalent for pre 2-0 behavior that allowed scalar
    inference

    Parameters
    ----------
    element : Scalar
    dtype : np.dtype

    Returns
    -------
    bool
    """
