import numpy as np
from _typeshed import Incomplete
from collections.abc import Sequence
from numpy import ma
from pandas import Index as Index, Series as Series
from pandas._config import using_pyarrow_string_dtype as using_pyarrow_string_dtype
from pandas._libs import lib as lib
from pandas._libs.tslibs import Period as Period, get_supported_dtype as get_supported_dtype, is_supported_dtype as is_supported_dtype
from pandas._typing import AnyArrayLike as AnyArrayLike, ArrayLike as ArrayLike, Dtype as Dtype, DtypeObj as DtypeObj, T as T
from pandas.core.arrays.base import ExtensionArray as ExtensionArray
from pandas.core.dtypes.base import ExtensionDtype as ExtensionDtype
from pandas.core.dtypes.cast import construct_1d_arraylike_from_scalar as construct_1d_arraylike_from_scalar, construct_1d_object_array_from_listlike as construct_1d_object_array_from_listlike, maybe_cast_to_datetime as maybe_cast_to_datetime, maybe_cast_to_integer_array as maybe_cast_to_integer_array, maybe_convert_platform as maybe_convert_platform, maybe_infer_to_datetimelike as maybe_infer_to_datetimelike, maybe_promote as maybe_promote
from pandas.core.dtypes.common import is_list_like as is_list_like, is_object_dtype as is_object_dtype, is_string_dtype as is_string_dtype, pandas_dtype as pandas_dtype
from pandas.core.dtypes.dtypes import NumpyEADtype as NumpyEADtype
from pandas.core.dtypes.generic import ABCDataFrame as ABCDataFrame, ABCExtensionArray as ABCExtensionArray, ABCIndex as ABCIndex, ABCSeries as ABCSeries
from pandas.core.dtypes.missing import isna as isna
from pandas.util._exceptions import find_stack_level as find_stack_level
from typing import overload

def array(data: Sequence[object] | AnyArrayLike, dtype: Dtype | None = None, copy: bool = True) -> ExtensionArray:
    '''
    Create an array.

    Parameters
    ----------
    data : Sequence of objects
        The scalars inside `data` should be instances of the
        scalar type for `dtype`. It\'s expected that `data`
        represents a 1-dimensional array of data.

        When `data` is an Index or Series, the underlying array
        will be extracted from `data`.

    dtype : str, np.dtype, or ExtensionDtype, optional
        The dtype to use for the array. This may be a NumPy
        dtype or an extension type registered with pandas using
        :meth:`pandas.api.extensions.register_extension_dtype`.

        If not specified, there are two possibilities:

        1. When `data` is a :class:`Series`, :class:`Index`, or
           :class:`ExtensionArray`, the `dtype` will be taken
           from the data.
        2. Otherwise, pandas will attempt to infer the `dtype`
           from the data.

        Note that when `data` is a NumPy array, ``data.dtype`` is
        *not* used for inferring the array type. This is because
        NumPy cannot represent all the types of data that can be
        held in extension arrays.

        Currently, pandas will infer an extension dtype for sequences of

        ============================== =======================================
        Scalar Type                    Array Type
        ============================== =======================================
        :class:`pandas.Interval`       :class:`pandas.arrays.IntervalArray`
        :class:`pandas.Period`         :class:`pandas.arrays.PeriodArray`
        :class:`datetime.datetime`     :class:`pandas.arrays.DatetimeArray`
        :class:`datetime.timedelta`    :class:`pandas.arrays.TimedeltaArray`
        :class:`int`                   :class:`pandas.arrays.IntegerArray`
        :class:`float`                 :class:`pandas.arrays.FloatingArray`
        :class:`str`                   :class:`pandas.arrays.StringArray` or
                                       :class:`pandas.arrays.ArrowStringArray`
        :class:`bool`                  :class:`pandas.arrays.BooleanArray`
        ============================== =======================================

        The ExtensionArray created when the scalar type is :class:`str` is determined by
        ``pd.options.mode.string_storage`` if the dtype is not explicitly given.

        For all other cases, NumPy\'s usual inference rules will be used.
    copy : bool, default True
        Whether to copy the data, even if not necessary. Depending
        on the type of `data`, creating the new array may require
        copying data, even if ``copy=False``.

    Returns
    -------
    ExtensionArray
        The newly created array.

    Raises
    ------
    ValueError
        When `data` is not 1-dimensional.

    See Also
    --------
    numpy.array : Construct a NumPy array.
    Series : Construct a pandas Series.
    Index : Construct a pandas Index.
    arrays.NumpyExtensionArray : ExtensionArray wrapping a NumPy array.
    Series.array : Extract the array stored within a Series.

    Notes
    -----
    Omitting the `dtype` argument means pandas will attempt to infer the
    best array type from the values in the data. As new array types are
    added by pandas and 3rd party libraries, the "best" array type may
    change. We recommend specifying `dtype` to ensure that

    1. the correct array type for the data is returned
    2. the returned array type doesn\'t change as new extension types
       are added by pandas and third-party libraries

    Additionally, if the underlying memory representation of the returned
    array matters, we recommend specifying the `dtype` as a concrete object
    rather than a string alias or allowing it to be inferred. For example,
    a future version of pandas or a 3rd-party library may include a
    dedicated ExtensionArray for string data. In this event, the following
    would no longer return a :class:`arrays.NumpyExtensionArray` backed by a
    NumPy array.

    >>> pd.array([\'a\', \'b\'], dtype=str)
    <NumpyExtensionArray>
    [\'a\', \'b\']
    Length: 2, dtype: str32

    This would instead return the new ExtensionArray dedicated for string
    data. If you really need the new array to be backed by a  NumPy array,
    specify that in the dtype.

    >>> pd.array([\'a\', \'b\'], dtype=np.dtype("<U1"))
    <NumpyExtensionArray>
    [\'a\', \'b\']
    Length: 2, dtype: str32

    Finally, Pandas has arrays that mostly overlap with NumPy

      * :class:`arrays.DatetimeArray`
      * :class:`arrays.TimedeltaArray`

    When data with a ``datetime64[ns]`` or ``timedelta64[ns]`` dtype is
    passed, pandas will always return a ``DatetimeArray`` or ``TimedeltaArray``
    rather than a ``NumpyExtensionArray``. This is for symmetry with the case of
    timezone-aware data, which NumPy does not natively support.

    >>> pd.array([\'2015\', \'2016\'], dtype=\'datetime64[ns]\')
    <DatetimeArray>
    [\'2015-01-01 00:00:00\', \'2016-01-01 00:00:00\']
    Length: 2, dtype: datetime64[ns]

    >>> pd.array(["1h", "2h"], dtype=\'timedelta64[ns]\')
    <TimedeltaArray>
    [\'0 days 01:00:00\', \'0 days 02:00:00\']
    Length: 2, dtype: timedelta64[ns]

    Examples
    --------
    If a dtype is not specified, pandas will infer the best dtype from the values.
    See the description of `dtype` for the types pandas infers for.

    >>> pd.array([1, 2])
    <IntegerArray>
    [1, 2]
    Length: 2, dtype: Int64

    >>> pd.array([1, 2, np.nan])
    <IntegerArray>
    [1, 2, <NA>]
    Length: 3, dtype: Int64

    >>> pd.array([1.1, 2.2])
    <FloatingArray>
    [1.1, 2.2]
    Length: 2, dtype: Float64

    >>> pd.array(["a", None, "c"])
    <StringArray>
    [\'a\', <NA>, \'c\']
    Length: 3, dtype: string

    >>> with pd.option_context("string_storage", "pyarrow"):
    ...     arr = pd.array(["a", None, "c"])
    ...
    >>> arr
    <ArrowStringArray>
    [\'a\', <NA>, \'c\']
    Length: 3, dtype: string

    >>> pd.array([pd.Period(\'2000\', freq="D"), pd.Period("2000", freq="D")])
    <PeriodArray>
    [\'2000-01-01\', \'2000-01-01\']
    Length: 2, dtype: period[D]

    You can use the string alias for `dtype`

    >>> pd.array([\'a\', \'b\', \'a\'], dtype=\'category\')
    [\'a\', \'b\', \'a\']
    Categories (2, object): [\'a\', \'b\']

    Or specify the actual dtype

    >>> pd.array([\'a\', \'b\', \'a\'],
    ...          dtype=pd.CategoricalDtype([\'a\', \'b\', \'c\'], ordered=True))
    [\'a\', \'b\', \'a\']
    Categories (3, object): [\'a\' < \'b\' < \'c\']

    If pandas does not infer a dedicated extension type a
    :class:`arrays.NumpyExtensionArray` is returned.

    >>> pd.array([1 + 1j, 3 + 2j])
    <NumpyExtensionArray>
    [(1+1j), (3+2j)]
    Length: 2, dtype: complex128

    As mentioned in the "Notes" section, new extension types may be added
    in the future (by pandas or 3rd party libraries), causing the return
    value to no longer be a :class:`arrays.NumpyExtensionArray`. Specify the
    `dtype` as a NumPy dtype if you need to ensure there\'s no future change in
    behavior.

    >>> pd.array([1, 2], dtype=np.dtype("int32"))
    <NumpyExtensionArray>
    [1, 2]
    Length: 2, dtype: int32

    `data` must be 1-dimensional. A ValueError is raised when the input
    has the wrong dimensionality.

    >>> pd.array(1)
    Traceback (most recent call last):
      ...
    ValueError: Cannot pass scalar \'1\' to \'pandas.array\'.
    '''

_typs: Incomplete

@overload
def extract_array(obj: Series | Index, extract_numpy: bool = ..., extract_range: bool = ...) -> ArrayLike: ...
@overload
def extract_array(obj: T, extract_numpy: bool = ..., extract_range: bool = ...) -> T | ArrayLike: ...
def ensure_wrapped_if_datetimelike(arr):
    """
    Wrap datetime64 and timedelta64 ndarrays in DatetimeArray/TimedeltaArray.
    """
def sanitize_masked_array(data: ma.MaskedArray) -> np.ndarray:
    """
    Convert numpy MaskedArray to ensure mask is softened.
    """
def sanitize_array(data, index: Index | None, dtype: DtypeObj | None = None, copy: bool = False, *, allow_2d: bool = False) -> ArrayLike:
    """
    Sanitize input data to an ndarray or ExtensionArray, copy if specified,
    coerce to the dtype if specified.

    Parameters
    ----------
    data : Any
    index : Index or None, default None
    dtype : np.dtype, ExtensionDtype, or None, default None
    copy : bool, default False
    allow_2d : bool, default False
        If False, raise if we have a 2D Arraylike.

    Returns
    -------
    np.ndarray or ExtensionArray
    """
def range_to_ndarray(rng: range) -> np.ndarray:
    """
    Cast a range object to ndarray.
    """
def _sanitize_non_ordered(data) -> None:
    """
    Raise only for unordered sets, e.g., not for dict_keys
    """
def _sanitize_ndim(result: ArrayLike, data, dtype: DtypeObj | None, index: Index | None, *, allow_2d: bool = False) -> ArrayLike:
    """
    Ensure we have a 1-dimensional result array.
    """
def _sanitize_str_dtypes(result: np.ndarray, data, dtype: np.dtype | None, copy: bool) -> np.ndarray:
    """
    Ensure we have a dtype that is supported by pandas.
    """
def _maybe_repeat(arr: ArrayLike, index: Index | None) -> ArrayLike:
    """
    If we have a length-1 array and an index describing how long we expect
    the result to be, repeat the array.
    """
def _try_cast(arr: list | np.ndarray, dtype: np.dtype, copy: bool) -> ArrayLike:
    """
    Convert input to numpy ndarray and optionally cast to a given dtype.

    Parameters
    ----------
    arr : ndarray or list
        Excludes: ExtensionArray, Series, Index.
    dtype : np.dtype
    copy : bool
        If False, don't copy the data if not needed.

    Returns
    -------
    np.ndarray or ExtensionArray
    """
