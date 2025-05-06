import numpy as np
import pyarrow as pa
import re
from _typeshed import Incomplete
from collections.abc import Sequence
from pandas import Series as Series
from pandas._libs import lib as lib
from pandas._libs.tslibs import NaT as NaT, Timedelta as Timedelta, Timestamp as Timestamp, timezones as timezones
from pandas._typing import ArrayLike as ArrayLike, AxisInt as AxisInt, Dtype as Dtype, FillnaOptions as FillnaOptions, InterpolateOptions as InterpolateOptions, Iterator as Iterator, NpDtype as NpDtype, NumpySorter as NumpySorter, NumpyValueArrayLike as NumpyValueArrayLike, PositionalIndexer as PositionalIndexer, Scalar as Scalar, Self as Self, SortKind as SortKind, TakeIndexer as TakeIndexer, TimeAmbiguous as TimeAmbiguous, TimeNonexistent as TimeNonexistent, npt as npt
from pandas.compat import pa_version_under10p1 as pa_version_under10p1, pa_version_under11p0 as pa_version_under11p0, pa_version_under13p0 as pa_version_under13p0
from pandas.core import missing as missing, ops as ops, roperator as roperator
from pandas.core.algorithms import map_array as map_array
from pandas.core.arraylike import OpsMixin as OpsMixin
from pandas.core.arrays._arrow_string_mixins import ArrowStringArrayMixin as ArrowStringArrayMixin
from pandas.core.arrays._utils import to_numpy_dtype_inference as to_numpy_dtype_inference
from pandas.core.arrays.base import ExtensionArray as ExtensionArray, ExtensionArraySupportsAnyAll as ExtensionArraySupportsAnyAll
from pandas.core.arrays.datetimes import DatetimeArray as DatetimeArray
from pandas.core.arrays.masked import BaseMaskedArray as BaseMaskedArray
from pandas.core.arrays.string_ import StringDtype as StringDtype
from pandas.core.arrays.timedeltas import TimedeltaArray as TimedeltaArray
from pandas.core.dtypes.cast import can_hold_element as can_hold_element, infer_dtype_from_scalar as infer_dtype_from_scalar
from pandas.core.dtypes.common import CategoricalDtype as CategoricalDtype, is_array_like as is_array_like, is_bool_dtype as is_bool_dtype, is_float_dtype as is_float_dtype, is_integer as is_integer, is_list_like as is_list_like, is_numeric_dtype as is_numeric_dtype, is_scalar as is_scalar
from pandas.core.dtypes.dtypes import ArrowDtype as ArrowDtype, DatetimeTZDtype as DatetimeTZDtype
from pandas.core.dtypes.missing import isna as isna
from pandas.core.indexers import check_array_indexer as check_array_indexer, unpack_tuple_and_ellipses as unpack_tuple_and_ellipses, validate_indices as validate_indices
from pandas.core.strings.base import BaseStringArrayMethods as BaseStringArrayMethods
from pandas.io._util import _arrow_dtype_mapping as _arrow_dtype_mapping
from pandas.tseries.frequencies import to_offset as to_offset
from pandas.util._decorators import doc as doc
from pandas.util._validators import validate_fillna_kwargs as validate_fillna_kwargs
from typing import Any, Literal

from collections.abc import Callable

ARROW_CMP_FUNCS: Incomplete
ARROW_LOGICAL_FUNCS: Incomplete
ARROW_BIT_WISE_FUNCS: Incomplete

def cast_for_truediv(arrow_array: pa.ChunkedArray, pa_object: pa.Array | pa.Scalar) -> tuple[pa.ChunkedArray, pa.Array | pa.Scalar]: ...
def floordiv_compat(left: pa.ChunkedArray | pa.Array | pa.Scalar, right: pa.ChunkedArray | pa.Array | pa.Scalar) -> pa.ChunkedArray: ...

ARROW_ARITHMETIC_FUNCS: Incomplete

def get_unit_from_pa_dtype(pa_dtype): ...
def to_pyarrow_type(dtype: ArrowDtype | pa.DataType | Dtype | None) -> pa.DataType | None:
    """
    Convert dtype to a pyarrow type instance.
    """

class ArrowExtensionArray(OpsMixin, ExtensionArraySupportsAnyAll, ArrowStringArrayMixin, BaseStringArrayMethods):
    '''
    Pandas ExtensionArray backed by a PyArrow ChunkedArray.

    .. warning::

       ArrowExtensionArray is considered experimental. The implementation and
       parts of the API may change without warning.

    Parameters
    ----------
    values : pyarrow.Array or pyarrow.ChunkedArray

    Attributes
    ----------
    None

    Methods
    -------
    None

    Returns
    -------
    ArrowExtensionArray

    Notes
    -----
    Most methods are implemented using `pyarrow compute functions. <https://arrow.apache.org/docs/python/api/compute.html>`__
    Some methods may either raise an exception or raise a ``PerformanceWarning`` if an
    associated compute function is not available based on the installed version of PyArrow.

    Please install the latest version of PyArrow to enable the best functionality and avoid
    potential bugs in prior versions of PyArrow.

    Examples
    --------
    Create an ArrowExtensionArray with :func:`pandas.array`:

    >>> pd.array([1, 1, None], dtype="int64[pyarrow]")
    <ArrowExtensionArray>
    [1, 1, <NA>]
    Length: 3, dtype: int64[pyarrow]
    '''
    _pa_array: pa.ChunkedArray
    _dtype: ArrowDtype
    def __init__(self, values: pa.Array | pa.ChunkedArray) -> None: ...
    @classmethod
    def _from_sequence(cls, scalars, *, dtype: Dtype | None = None, copy: bool = False):
        """
        Construct a new ExtensionArray from a sequence of scalars.
        """
    @classmethod
    def _from_sequence_of_strings(cls, strings, *, dtype: Dtype | None = None, copy: bool = False):
        """
        Construct a new ExtensionArray from a sequence of strings.
        """
    @classmethod
    def _box_pa(cls, value, pa_type: pa.DataType | None = None) -> pa.Array | pa.ChunkedArray | pa.Scalar:
        """
        Box value into a pyarrow Array, ChunkedArray or Scalar.

        Parameters
        ----------
        value : any
        pa_type : pa.DataType | None

        Returns
        -------
        pa.Array or pa.ChunkedArray or pa.Scalar
        """
    @classmethod
    def _box_pa_scalar(cls, value, pa_type: pa.DataType | None = None) -> pa.Scalar:
        """
        Box value into a pyarrow Scalar.

        Parameters
        ----------
        value : any
        pa_type : pa.DataType | None

        Returns
        -------
        pa.Scalar
        """
    @classmethod
    def _box_pa_array(cls, value, pa_type: pa.DataType | None = None, copy: bool = False) -> pa.Array | pa.ChunkedArray:
        """
        Box value into a pyarrow Array or ChunkedArray.

        Parameters
        ----------
        value : Sequence
        pa_type : pa.DataType | None

        Returns
        -------
        pa.Array or pa.ChunkedArray
        """
    def __getitem__(self, item: PositionalIndexer):
        """Select a subset of self.

        Parameters
        ----------
        item : int, slice, or ndarray
            * int: The position in 'self' to get.
            * slice: A slice object, where 'start', 'stop', and 'step' are
              integers or None
            * ndarray: A 1-d boolean NumPy ndarray the same length as 'self'

        Returns
        -------
        item : scalar or ExtensionArray

        Notes
        -----
        For scalar ``item``, return a scalar value suitable for the array's
        type. This should be an instance of ``self.dtype.type``.
        For slice ``key``, return an instance of ``ExtensionArray``, even
        if the slice is length 0 or 1.
        For a boolean mask, return an instance of ``ExtensionArray``, filtered
        to the values where ``item`` is True.
        """
    def __iter__(self) -> Iterator[Any]:
        """
        Iterate over elements of the array.
        """
    def __arrow_array__(self, type: Incomplete | None = None):
        """Convert myself to a pyarrow ChunkedArray."""
    def __array__(self, dtype: NpDtype | None = None, copy: bool | None = None) -> np.ndarray:
        """Correctly construct numpy arrays when passed to `np.asarray()`."""
    def __invert__(self) -> Self: ...
    def __neg__(self) -> Self: ...
    def __pos__(self) -> Self: ...
    def __abs__(self) -> Self: ...
    def __getstate__(self): ...
    def __setstate__(self, state) -> None: ...
    def _cmp_method(self, other, op): ...
    def _evaluate_op_method(self, other, op, arrow_funcs): ...
    def _logical_method(self, other, op): ...
    def _arith_method(self, other, op): ...
    def equals(self, other) -> bool: ...
    @property
    def dtype(self) -> ArrowDtype:
        """
        An instance of 'ExtensionDtype'.
        """
    @property
    def nbytes(self) -> int:
        """
        The number of bytes needed to store this object in memory.
        """
    def __len__(self) -> int:
        """
        Length of this array.

        Returns
        -------
        length : int
        """
    def __contains__(self, key) -> bool: ...
    @property
    def _hasna(self) -> bool: ...
    def isna(self) -> npt.NDArray[np.bool_]:
        """
        Boolean NumPy array indicating if each value is missing.

        This should return a 1-D array the same length as 'self'.
        """
    def any(self, *, skipna: bool = True, **kwargs):
        '''
        Return whether any element is truthy.

        Returns False unless there is at least one element that is truthy.
        By default, NAs are skipped. If ``skipna=False`` is specified and
        missing values are present, similar :ref:`Kleene logic <boolean.kleene>`
        is used as for logical operations.

        Parameters
        ----------
        skipna : bool, default True
            Exclude NA values. If the entire array is NA and `skipna` is
            True, then the result will be False, as for an empty array.
            If `skipna` is False, the result will still be True if there is
            at least one element that is truthy, otherwise NA will be returned
            if there are NA\'s present.

        Returns
        -------
        bool or :attr:`pandas.NA`

        See Also
        --------
        ArrowExtensionArray.all : Return whether all elements are truthy.

        Examples
        --------
        The result indicates whether any element is truthy (and by default
        skips NAs):

        >>> pd.array([True, False, True], dtype="boolean[pyarrow]").any()
        True
        >>> pd.array([True, False, pd.NA], dtype="boolean[pyarrow]").any()
        True
        >>> pd.array([False, False, pd.NA], dtype="boolean[pyarrow]").any()
        False
        >>> pd.array([], dtype="boolean[pyarrow]").any()
        False
        >>> pd.array([pd.NA], dtype="boolean[pyarrow]").any()
        False
        >>> pd.array([pd.NA], dtype="float64[pyarrow]").any()
        False

        With ``skipna=False``, the result can be NA if this is logically
        required (whether ``pd.NA`` is True or False influences the result):

        >>> pd.array([True, False, pd.NA], dtype="boolean[pyarrow]").any(skipna=False)
        True
        >>> pd.array([1, 0, pd.NA], dtype="boolean[pyarrow]").any(skipna=False)
        True
        >>> pd.array([False, False, pd.NA], dtype="boolean[pyarrow]").any(skipna=False)
        <NA>
        >>> pd.array([0, 0, pd.NA], dtype="boolean[pyarrow]").any(skipna=False)
        <NA>
        '''
    def all(self, *, skipna: bool = True, **kwargs):
        '''
        Return whether all elements are truthy.

        Returns True unless there is at least one element that is falsey.
        By default, NAs are skipped. If ``skipna=False`` is specified and
        missing values are present, similar :ref:`Kleene logic <boolean.kleene>`
        is used as for logical operations.

        Parameters
        ----------
        skipna : bool, default True
            Exclude NA values. If the entire array is NA and `skipna` is
            True, then the result will be True, as for an empty array.
            If `skipna` is False, the result will still be False if there is
            at least one element that is falsey, otherwise NA will be returned
            if there are NA\'s present.

        Returns
        -------
        bool or :attr:`pandas.NA`

        See Also
        --------
        ArrowExtensionArray.any : Return whether any element is truthy.

        Examples
        --------
        The result indicates whether all elements are truthy (and by default
        skips NAs):

        >>> pd.array([True, True, pd.NA], dtype="boolean[pyarrow]").all()
        True
        >>> pd.array([1, 1, pd.NA], dtype="boolean[pyarrow]").all()
        True
        >>> pd.array([True, False, pd.NA], dtype="boolean[pyarrow]").all()
        False
        >>> pd.array([], dtype="boolean[pyarrow]").all()
        True
        >>> pd.array([pd.NA], dtype="boolean[pyarrow]").all()
        True
        >>> pd.array([pd.NA], dtype="float64[pyarrow]").all()
        True

        With ``skipna=False``, the result can be NA if this is logically
        required (whether ``pd.NA`` is True or False influences the result):

        >>> pd.array([True, True, pd.NA], dtype="boolean[pyarrow]").all(skipna=False)
        <NA>
        >>> pd.array([1, 1, pd.NA], dtype="boolean[pyarrow]").all(skipna=False)
        <NA>
        >>> pd.array([True, False, pd.NA], dtype="boolean[pyarrow]").all(skipna=False)
        False
        >>> pd.array([1, 0, pd.NA], dtype="boolean[pyarrow]").all(skipna=False)
        False
        '''
    def argsort(self, *, ascending: bool = True, kind: SortKind = 'quicksort', na_position: str = 'last', **kwargs) -> np.ndarray: ...
    def _argmin_max(self, skipna: bool, method: str) -> int: ...
    def argmin(self, skipna: bool = True) -> int: ...
    def argmax(self, skipna: bool = True) -> int: ...
    def copy(self) -> Self:
        """
        Return a shallow copy of the array.

        Underlying ChunkedArray is immutable, so a deep copy is unnecessary.

        Returns
        -------
        type(self)
        """
    def dropna(self) -> Self:
        """
        Return ArrowExtensionArray without NA values.

        Returns
        -------
        ArrowExtensionArray
        """
    def _pad_or_backfill(self, *, method: FillnaOptions, limit: int | None = None, limit_area: Literal['inside', 'outside'] | None = None, copy: bool = True) -> Self: ...
    def fillna(self, value: object | ArrayLike | None = None, method: FillnaOptions | None = None, limit: int | None = None, copy: bool = True) -> Self: ...
    def isin(self, values: ArrayLike) -> npt.NDArray[np.bool_]: ...
    def _values_for_factorize(self) -> tuple[np.ndarray, Any]:
        """
        Return an array and missing value suitable for factorization.

        Returns
        -------
        values : ndarray
        na_value : pd.NA

        Notes
        -----
        The values returned by this method are also used in
        :func:`pandas.util.hash_pandas_object`.
        """
    def factorize(self, use_na_sentinel: bool = True) -> tuple[np.ndarray, ExtensionArray]: ...
    def reshape(self, *args, **kwargs) -> None: ...
    def round(self, decimals: int = 0, *args, **kwargs) -> Self:
        """
        Round each value in the array a to the given number of decimals.

        Parameters
        ----------
        decimals : int, default 0
            Number of decimal places to round to. If decimals is negative,
            it specifies the number of positions to the left of the decimal point.
        *args, **kwargs
            Additional arguments and keywords have no effect.

        Returns
        -------
        ArrowExtensionArray
            Rounded values of the ArrowExtensionArray.

        See Also
        --------
        DataFrame.round : Round values of a DataFrame.
        Series.round : Round values of a Series.
        """
    def searchsorted(self, value: NumpyValueArrayLike | ExtensionArray, side: Literal['left', 'right'] = 'left', sorter: NumpySorter | None = None) -> npt.NDArray[np.intp] | np.intp: ...
    def take(self, indices: TakeIndexer, allow_fill: bool = False, fill_value: Any = None) -> ArrowExtensionArray:
        '''
        Take elements from an array.

        Parameters
        ----------
        indices : sequence of int or one-dimensional np.ndarray of int
            Indices to be taken.
        allow_fill : bool, default False
            How to handle negative values in `indices`.

            * False: negative values in `indices` indicate positional indices
              from the right (the default). This is similar to
              :func:`numpy.take`.

            * True: negative values in `indices` indicate
              missing values. These values are set to `fill_value`. Any other
              other negative values raise a ``ValueError``.

        fill_value : any, optional
            Fill value to use for NA-indices when `allow_fill` is True.
            This may be ``None``, in which case the default NA value for
            the type, ``self.dtype.na_value``, is used.

            For many ExtensionArrays, there will be two representations of
            `fill_value`: a user-facing "boxed" scalar, and a low-level
            physical NA value. `fill_value` should be the user-facing version,
            and the implementation should handle translating that to the
            physical version for processing the take if necessary.

        Returns
        -------
        ExtensionArray

        Raises
        ------
        IndexError
            When the indices are out of bounds for the array.
        ValueError
            When `indices` contains negative values other than ``-1``
            and `allow_fill` is True.

        See Also
        --------
        numpy.take
        api.extensions.take

        Notes
        -----
        ExtensionArray.take is called by ``Series.__getitem__``, ``.loc``,
        ``iloc``, when `indices` is a sequence of values. Additionally,
        it\'s called by :meth:`Series.reindex`, or any other method
        that causes realignment, with a `fill_value`.
        '''
    def _maybe_convert_datelike_array(self):
        """Maybe convert to a datelike array."""
    def _to_datetimearray(self) -> DatetimeArray:
        """Convert a pyarrow timestamp typed array to a DatetimeArray."""
    def _to_timedeltaarray(self) -> TimedeltaArray:
        """Convert a pyarrow duration typed array to a TimedeltaArray."""
    def _values_for_json(self) -> np.ndarray: ...
    def to_numpy(self, dtype: npt.DTypeLike | None = None, copy: bool = False, na_value: object = ...) -> np.ndarray: ...
    def map(self, mapper, na_action: Incomplete | None = None): ...
    def duplicated(self, keep: Literal['first', 'last', False] = 'first') -> npt.NDArray[np.bool_]: ...
    def unique(self) -> Self:
        """
        Compute the ArrowExtensionArray of unique values.

        Returns
        -------
        ArrowExtensionArray
        """
    def value_counts(self, dropna: bool = True) -> Series:
        """
        Return a Series containing counts of each unique value.

        Parameters
        ----------
        dropna : bool, default True
            Don't include counts of missing values.

        Returns
        -------
        counts : Series

        See Also
        --------
        Series.value_counts
        """
    @classmethod
    def _concat_same_type(cls, to_concat) -> Self:
        """
        Concatenate multiple ArrowExtensionArrays.

        Parameters
        ----------
        to_concat : sequence of ArrowExtensionArrays

        Returns
        -------
        ArrowExtensionArray
        """
    def _accumulate(self, name: str, *, skipna: bool = True, **kwargs) -> ArrowExtensionArray | ExtensionArray:
        """
        Return an ExtensionArray performing an accumulation operation.

        The underlying data type might change.

        Parameters
        ----------
        name : str
            Name of the function, supported values are:
            - cummin
            - cummax
            - cumsum
            - cumprod
        skipna : bool, default True
            If True, skip NA values.
        **kwargs
            Additional keyword arguments passed to the accumulation function.
            Currently, there is no supported kwarg.

        Returns
        -------
        array

        Raises
        ------
        NotImplementedError : subclass does not define accumulations
        """
    def _reduce_pyarrow(self, name: str, *, skipna: bool = True, **kwargs) -> pa.Scalar:
        """
        Return a pyarrow scalar result of performing the reduction operation.

        Parameters
        ----------
        name : str
            Name of the function, supported values are:
            { any, all, min, max, sum, mean, median, prod,
            std, var, sem, kurt, skew }.
        skipna : bool, default True
            If True, skip NaN values.
        **kwargs
            Additional keyword arguments passed to the reduction function.
            Currently, `ddof` is the only supported kwarg.

        Returns
        -------
        pyarrow scalar

        Raises
        ------
        TypeError : subclass does not define reductions
        """
    def _reduce(self, name: str, *, skipna: bool = True, keepdims: bool = False, **kwargs):
        """
        Return a scalar result of performing the reduction operation.

        Parameters
        ----------
        name : str
            Name of the function, supported values are:
            { any, all, min, max, sum, mean, median, prod,
            std, var, sem, kurt, skew }.
        skipna : bool, default True
            If True, skip NaN values.
        **kwargs
            Additional keyword arguments passed to the reduction function.
            Currently, `ddof` is the only supported kwarg.

        Returns
        -------
        scalar

        Raises
        ------
        TypeError : subclass does not define reductions
        """
    def _reduce_calc(self, name: str, *, skipna: bool = True, keepdims: bool = False, **kwargs): ...
    def _explode(self):
        """
        See Series.explode.__doc__.
        """
    def __setitem__(self, key, value) -> None:
        """Set one or more values inplace.

        Parameters
        ----------
        key : int, ndarray, or slice
            When called from, e.g. ``Series.__setitem__``, ``key`` will be
            one of

            * scalar int
            * ndarray of integers.
            * boolean ndarray
            * slice object

        value : ExtensionDtype.type, Sequence[ExtensionDtype.type], or object
            value or values to be set of ``key``.

        Returns
        -------
        None
        """
    def _rank_calc(self, *, axis: AxisInt = 0, method: str = 'average', na_option: str = 'keep', ascending: bool = True, pct: bool = False): ...
    def _rank(self, *, axis: AxisInt = 0, method: str = 'average', na_option: str = 'keep', ascending: bool = True, pct: bool = False):
        """
        See Series.rank.__doc__.
        """
    def _quantile(self, qs: npt.NDArray[np.float64], interpolation: str) -> Self:
        """
        Compute the quantiles of self for each quantile in `qs`.

        Parameters
        ----------
        qs : np.ndarray[float64]
        interpolation: str

        Returns
        -------
        same type as self
        """
    def _mode(self, dropna: bool = True) -> Self:
        """
        Returns the mode(s) of the ExtensionArray.

        Always returns `ExtensionArray` even if only one value.

        Parameters
        ----------
        dropna : bool, default True
            Don't consider counts of NA values.

        Returns
        -------
        same type as self
            Sorted, if possible.
        """
    def _maybe_convert_setitem_value(self, value):
        """Maybe convert value to be pyarrow compatible."""
    def interpolate(self, *, method: InterpolateOptions, axis: int, index, limit, limit_direction, limit_area, copy: bool, **kwargs) -> Self:
        """
        See NDFrame.interpolate.__doc__.
        """
    @classmethod
    def _if_else(cls, cond: npt.NDArray[np.bool_] | bool, left: ArrayLike | Scalar, right: ArrayLike | Scalar):
        """
        Choose values based on a condition.

        Analogous to pyarrow.compute.if_else, with logic
        to fallback to numpy for unsupported types.

        Parameters
        ----------
        cond : npt.NDArray[np.bool_] or bool
        left : ArrayLike | Scalar
        right : ArrayLike | Scalar

        Returns
        -------
        pa.Array
        """
    @classmethod
    def _replace_with_mask(cls, values: pa.Array | pa.ChunkedArray, mask: npt.NDArray[np.bool_] | bool, replacements: ArrayLike | Scalar):
        """
        Replace items selected with a mask.

        Analogous to pyarrow.compute.replace_with_mask, with logic
        to fallback to numpy for unsupported types.

        Parameters
        ----------
        values : pa.Array or pa.ChunkedArray
        mask : npt.NDArray[np.bool_] or bool
        replacements : ArrayLike or Scalar
            Replacement value(s)

        Returns
        -------
        pa.Array or pa.ChunkedArray
        """
    def _to_masked(self): ...
    def _groupby_op(self, *, how: str, has_dropped_na: bool, min_count: int, ngroups: int, ids: npt.NDArray[np.intp], **kwargs): ...
    def _apply_elementwise(self, func: Callable) -> list[list[Any]]:
        """Apply a callable to each element while maintaining the chunking structure."""
    def _str_count(self, pat: str, flags: int = 0): ...
    def _str_contains(self, pat, case: bool = True, flags: int = 0, na: Incomplete | None = None, regex: bool = True): ...
    def _str_startswith(self, pat: str | tuple[str, ...], na: Incomplete | None = None): ...
    def _str_endswith(self, pat: str | tuple[str, ...], na: Incomplete | None = None): ...
    def _str_replace(self, pat: str | re.Pattern, repl: str | Callable, n: int = -1, case: bool = True, flags: int = 0, regex: bool = True): ...
    def _str_repeat(self, repeats: int | Sequence[int]): ...
    def _str_match(self, pat: str, case: bool = True, flags: int = 0, na: Scalar | None = None): ...
    def _str_fullmatch(self, pat, case: bool = True, flags: int = 0, na: Scalar | None = None): ...
    def _str_find(self, sub: str, start: int = 0, end: int | None = None): ...
    def _str_join(self, sep: str): ...
    def _str_partition(self, sep: str, expand: bool): ...
    def _str_rpartition(self, sep: str, expand: bool): ...
    def _str_slice(self, start: int | None = None, stop: int | None = None, step: int | None = None): ...
    def _str_isalnum(self): ...
    def _str_isalpha(self): ...
    def _str_isdecimal(self): ...
    def _str_isdigit(self): ...
    def _str_islower(self): ...
    def _str_isnumeric(self): ...
    def _str_isspace(self): ...
    def _str_istitle(self): ...
    def _str_isupper(self): ...
    def _str_len(self): ...
    def _str_lower(self): ...
    def _str_upper(self): ...
    def _str_strip(self, to_strip: Incomplete | None = None): ...
    def _str_lstrip(self, to_strip: Incomplete | None = None): ...
    def _str_rstrip(self, to_strip: Incomplete | None = None): ...
    def _str_removeprefix(self, prefix: str): ...
    def _str_casefold(self): ...
    def _str_encode(self, encoding: str, errors: str = 'strict'): ...
    def _str_extract(self, pat: str, flags: int = 0, expand: bool = True): ...
    def _str_findall(self, pat: str, flags: int = 0): ...
    def _str_get_dummies(self, sep: str = '|'): ...
    def _str_index(self, sub: str, start: int = 0, end: int | None = None): ...
    def _str_rindex(self, sub: str, start: int = 0, end: int | None = None): ...
    def _str_normalize(self, form: str): ...
    def _str_rfind(self, sub: str, start: int = 0, end: Incomplete | None = None): ...
    def _str_split(self, pat: str | None = None, n: int | None = -1, expand: bool = False, regex: bool | None = None): ...
    def _str_rsplit(self, pat: str | None = None, n: int | None = -1): ...
    def _str_translate(self, table: dict[int, str]): ...
    def _str_wrap(self, width: int, **kwargs): ...
    @property
    def _dt_days(self): ...
    @property
    def _dt_hours(self): ...
    @property
    def _dt_minutes(self): ...
    @property
    def _dt_seconds(self): ...
    @property
    def _dt_milliseconds(self): ...
    @property
    def _dt_microseconds(self): ...
    @property
    def _dt_nanoseconds(self): ...
    def _dt_to_pytimedelta(self): ...
    def _dt_total_seconds(self): ...
    def _dt_as_unit(self, unit: str): ...
    @property
    def _dt_year(self): ...
    @property
    def _dt_day(self): ...
    @property
    def _dt_day_of_week(self): ...
    _dt_dayofweek = _dt_day_of_week
    _dt_weekday = _dt_day_of_week
    @property
    def _dt_day_of_year(self): ...
    _dt_dayofyear = _dt_day_of_year
    @property
    def _dt_hour(self): ...
    def _dt_isocalendar(self): ...
    @property
    def _dt_is_leap_year(self): ...
    @property
    def _dt_is_month_start(self): ...
    @property
    def _dt_is_month_end(self): ...
    @property
    def _dt_is_year_start(self): ...
    @property
    def _dt_is_year_end(self): ...
    @property
    def _dt_is_quarter_start(self): ...
    @property
    def _dt_is_quarter_end(self): ...
    @property
    def _dt_days_in_month(self): ...
    _dt_daysinmonth = _dt_days_in_month
    @property
    def _dt_microsecond(self): ...
    @property
    def _dt_minute(self): ...
    @property
    def _dt_month(self): ...
    @property
    def _dt_nanosecond(self): ...
    @property
    def _dt_quarter(self): ...
    @property
    def _dt_second(self): ...
    @property
    def _dt_date(self): ...
    @property
    def _dt_time(self): ...
    @property
    def _dt_tz(self): ...
    @property
    def _dt_unit(self): ...
    def _dt_normalize(self): ...
    def _dt_strftime(self, format: str): ...
    def _round_temporally(self, method: Literal['ceil', 'floor', 'round'], freq, ambiguous: TimeAmbiguous = 'raise', nonexistent: TimeNonexistent = 'raise'): ...
    def _dt_ceil(self, freq, ambiguous: TimeAmbiguous = 'raise', nonexistent: TimeNonexistent = 'raise'): ...
    def _dt_floor(self, freq, ambiguous: TimeAmbiguous = 'raise', nonexistent: TimeNonexistent = 'raise'): ...
    def _dt_round(self, freq, ambiguous: TimeAmbiguous = 'raise', nonexistent: TimeNonexistent = 'raise'): ...
    def _dt_day_name(self, locale: str | None = None): ...
    def _dt_month_name(self, locale: str | None = None): ...
    def _dt_to_pydatetime(self): ...
    def _dt_tz_localize(self, tz, ambiguous: TimeAmbiguous = 'raise', nonexistent: TimeNonexistent = 'raise'): ...
    def _dt_tz_convert(self, tz): ...

def transpose_homogeneous_pyarrow(arrays: Sequence[ArrowExtensionArray]) -> list[ArrowExtensionArray]:
    """Transpose arrow extension arrays in a list, but faster.

    Input should be a list of arrays of equal length and all have the same
    dtype. The caller is responsible for ensuring validity of input data.
    """
