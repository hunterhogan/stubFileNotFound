import _abc
import np
import npt
import pa
import pandas._libs.lib as lib
import pandas._libs.tslibs.timezones as timezones
import pandas.core.algorithms as algos
import pandas.core.arraylike
import pandas.core.arrays._arrow_string_mixins
import pandas.core.arrays.base
import pandas.core.common as com
import pandas.core.missing as missing
import pandas.core.ops as ops
import pandas.core.roperator as roperator
import pandas.core.strings.base
import re
from pandas._libs.lib import is_integer as is_integer, is_list_like as is_list_like, is_scalar as is_scalar
from pandas._libs.tslibs.nattype import NaT as NaT
from pandas._libs.tslibs.offsets import to_offset as to_offset
from pandas._libs.tslibs.timedeltas import Timedelta as Timedelta
from pandas._libs.tslibs.timestamps import Timestamp as Timestamp
from pandas.core.algorithms import map_array as map_array
from pandas.core.arraylike import OpsMixin as OpsMixin
from pandas.core.arrays._arrow_string_mixins import ArrowStringArrayMixin as ArrowStringArrayMixin
from pandas.core.arrays._utils import to_numpy_dtype_inference as to_numpy_dtype_inference
from pandas.core.arrays.base import ExtensionArray as ExtensionArray, ExtensionArraySupportsAnyAll as ExtensionArraySupportsAnyAll
from pandas.core.arrays.masked import BaseMaskedArray as BaseMaskedArray
from pandas.core.arrays.string_ import StringDtype as StringDtype
from pandas.core.dtypes.cast import can_hold_element as can_hold_element, infer_dtype_from_scalar as infer_dtype_from_scalar
from pandas.core.dtypes.common import is_bool_dtype as is_bool_dtype, is_float_dtype as is_float_dtype, is_numeric_dtype as is_numeric_dtype
from pandas.core.dtypes.dtypes import CategoricalDtype as CategoricalDtype, DatetimeTZDtype as DatetimeTZDtype
from pandas.core.dtypes.inference import is_array_like as is_array_like
from pandas.core.dtypes.missing import isna as isna
from pandas.core.indexers.utils import check_array_indexer as check_array_indexer, unpack_tuple_and_ellipses as unpack_tuple_and_ellipses, validate_indices as validate_indices
from pandas.core.strings.base import BaseStringArrayMethods as BaseStringArrayMethods
from pandas.io._util import _arrow_dtype_mapping as _arrow_dtype_mapping
from pandas.util._decorators import doc as doc
from pandas.util._validators import validate_fillna_kwargs as validate_fillna_kwargs
from typing import Any, Callable, ClassVar, Literal

TYPE_CHECKING: bool
pa_version_under10p1: bool
pa_version_under11p0: bool
pa_version_under13p0: bool
def get_unit_from_pa_dtype(pa_dtype): ...
def to_pyarrow_type(dtype: ArrowDtype | pa.DataType | Dtype | None) -> pa.DataType | None:
    """
    Convert dtype to a pyarrow type instance.
    """

class ArrowExtensionArray(pandas.core.arraylike.OpsMixin, pandas.core.arrays.base.ExtensionArraySupportsAnyAll, pandas.core.arrays._arrow_string_mixins.ArrowStringArrayMixin, pandas.core.strings.base.BaseStringArrayMethods):
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    def __init__(self, values: pa.Array | pa.ChunkedArray) -> None: ...
    @classmethod
    def _from_sequence(cls, scalars, *, dtype: Dtype | None, copy: bool = ...):
        """
        Construct a new ExtensionArray from a sequence of scalars.
        """
    @classmethod
    def _from_sequence_of_strings(cls, strings, *, dtype: Dtype | None, copy: bool = ...):
        """
        Construct a new ExtensionArray from a sequence of strings.
        """
    @classmethod
    def _box_pa(cls, value, pa_type: pa.DataType | None) -> pa.Array | pa.ChunkedArray | pa.Scalar:
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
    def _box_pa_scalar(cls, value, pa_type: pa.DataType | None) -> pa.Scalar:
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
    def _box_pa_array(cls, value, pa_type: pa.DataType | None, copy: bool = ...) -> pa.Array | pa.ChunkedArray:
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
    def __arrow_array__(self, type):
        """Convert myself to a pyarrow ChunkedArray."""
    def __array__(self, dtype: NpDtype | None, copy: bool | None) -> np.ndarray:
        """Correctly construct numpy arrays when passed to `np.asarray()`."""
    def __invert__(self) -> Self: ...
    def __neg__(self) -> Self: ...
    def __pos__(self) -> Self: ...
    def __abs__(self) -> Self: ...
    def _cmp_method(self, other, op): ...
    def _evaluate_op_method(self, other, op, arrow_funcs): ...
    def _logical_method(self, other, op): ...
    def _arith_method(self, other, op): ...
    def equals(self, other) -> bool: ...
    def __len__(self) -> int:
        """
        Length of this array.

        Returns
        -------
        length : int
        """
    def __contains__(self, key) -> bool: ...
    def isna(self) -> npt.NDArray[np.bool_]:
        """
        Boolean NumPy array indicating if each value is missing.

        This should return a 1-D array the same length as 'self'.
        """
    def any(self, *, skipna: bool = ..., **kwargs):
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
    def all(self, *, skipna: bool = ..., **kwargs):
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
    def argsort(self, *, ascending: bool = ..., kind: SortKind = ..., na_position: str = ..., **kwargs) -> np.ndarray: ...
    def _argmin_max(self, skipna: bool, method: str) -> int: ...
    def argmin(self, skipna: bool = ...) -> int: ...
    def argmax(self, skipna: bool = ...) -> int: ...
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
    def _pad_or_backfill(self, *, method: FillnaOptions, limit: int | None, limit_area: Literal['inside', 'outside'] | None, copy: bool = ...) -> Self: ...
    def fillna(self, value: object | ArrayLike | None, method: FillnaOptions | None, limit: int | None, copy: bool = ...) -> Self:
        '''
        Fill NA/NaN values using the specified method.

        Parameters
        ----------
        value : scalar, array-like
            If a scalar value is passed it is used to fill all missing values.
            Alternatively, an array-like "value" can be given. It\'s expected
            that the array-like have the same length as \'self\'.
        method : {\'backfill\', \'bfill\', \'pad\', \'ffill\', None}, default None
            Method to use for filling holes in reindexed Series:

            * pad / ffill: propagate last valid observation forward to next valid.
            * backfill / bfill: use NEXT valid observation to fill gap.

            .. deprecated:: 2.1.0

        limit : int, default None
            If method is specified, this is the maximum number of consecutive
            NaN values to forward/backward fill. In other words, if there is
            a gap with more than this number of consecutive NaNs, it will only
            be partially filled. If method is not specified, this is the
            maximum number of entries along the entire axis where NaNs will be
            filled.

            .. deprecated:: 2.1.0

        copy : bool, default True
            Whether to make a copy of the data before filling. If False, then
            the original should be modified and no new memory should be allocated.
            For ExtensionArray subclasses that cannot do this, it is at the
            author\'s discretion whether to ignore "copy=False" or to raise.
            The base class implementation ignores the keyword in pad/backfill
            cases.

        Returns
        -------
        ExtensionArray
            With NA/NaN filled.

        Examples
        --------
        >>> arr = pd.array([np.nan, np.nan, 2, 3, np.nan, np.nan])
        >>> arr.fillna(0)
        <IntegerArray>
        [0, 0, 2, 3, 0, 0]
        Length: 6, dtype: Int64
        '''
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
    def factorize(self, use_na_sentinel: bool = ...) -> tuple[np.ndarray, ExtensionArray]:
        '''
        Encode the extension array as an enumerated type.

        Parameters
        ----------
        use_na_sentinel : bool, default True
            If True, the sentinel -1 will be used for NaN values. If False,
            NaN values will be encoded as non-negative integers and will not drop the
            NaN from the uniques of the values.

            .. versionadded:: 1.5.0

        Returns
        -------
        codes : ndarray
            An integer NumPy array that\'s an indexer into the original
            ExtensionArray.
        uniques : ExtensionArray
            An ExtensionArray containing the unique values of `self`.

            .. note::

               uniques will *not* contain an entry for the NA value of
               the ExtensionArray if there are any missing values present
               in `self`.

        See Also
        --------
        factorize : Top-level factorize method that dispatches here.

        Notes
        -----
        :meth:`pandas.factorize` offers a `sort` keyword as well.

        Examples
        --------
        >>> idx1 = pd.PeriodIndex(["2014-01", "2014-01", "2014-02", "2014-02",
        ...                       "2014-03", "2014-03"], freq="M")
        >>> arr, idx = idx1.factorize()
        >>> arr
        array([0, 0, 1, 1, 2, 2])
        >>> idx
        PeriodIndex([\'2014-01\', \'2014-02\', \'2014-03\'], dtype=\'period[M]\')
        '''
    def reshape(self, *args, **kwargs): ...
    def round(self, decimals: int = ..., *args, **kwargs) -> Self:
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
    def searchsorted(self, value: NumpyValueArrayLike | ExtensionArray, side: Literal['left', 'right'] = ..., sorter: NumpySorter | None) -> npt.NDArray[np.intp] | np.intp:
        """
        Find indices where elements should be inserted to maintain order.

        Find the indices into a sorted array `self` (a) such that, if the
        corresponding elements in `value` were inserted before the indices,
        the order of `self` would be preserved.

        Assuming that `self` is sorted:

        ======  ================================
        `side`  returned index `i` satisfies
        ======  ================================
        left    ``self[i-1] < value <= self[i]``
        right   ``self[i-1] <= value < self[i]``
        ======  ================================

        Parameters
        ----------
        value : array-like, list or scalar
            Value(s) to insert into `self`.
        side : {'left', 'right'}, optional
            If 'left', the index of the first suitable location found is given.
            If 'right', return the last such index.  If there is no suitable
            index, return either 0 or N (where N is the length of `self`).
        sorter : 1-D array-like, optional
            Optional array of integer indices that sort array a into ascending
            order. They are typically the result of argsort.

        Returns
        -------
        array of ints or int
            If value is array-like, array of insertion points.
            If value is scalar, a single integer.

        See Also
        --------
        numpy.searchsorted : Similar method from NumPy.

        Examples
        --------
        >>> arr = pd.array([1, 2, 3, 5])
        >>> arr.searchsorted([4])
        array([3])
        """
    def take(self, indices: TakeIndexer, allow_fill: bool = ..., fill_value: Any) -> ArrowExtensionArray:
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
    def to_numpy(self, dtype: npt.DTypeLike | None, copy: bool = ..., na_value: object = ...) -> np.ndarray:
        """
        Convert to a NumPy ndarray.

        This is similar to :meth:`numpy.asarray`, but may provide additional control
        over how the conversion is done.

        Parameters
        ----------
        dtype : str or numpy.dtype, optional
            The dtype to pass to :meth:`numpy.asarray`.
        copy : bool, default False
            Whether to ensure that the returned value is a not a view on
            another array. Note that ``copy=False`` does not *ensure* that
            ``to_numpy()`` is no-copy. Rather, ``copy=True`` ensure that
            a copy is made, even if not strictly necessary.
        na_value : Any, optional
            The value to use for missing values. The default value depends
            on `dtype` and the type of the array.

        Returns
        -------
        numpy.ndarray
        """
    def map(self, mapper, na_action): ...
    def duplicated(self, keep: Literal['first', 'last', False] = ...) -> npt.NDArray[np.bool_]:
        '''
        Return boolean ndarray denoting duplicate values.

        Parameters
        ----------
        keep : {\'first\', \'last\', False}, default \'first\'
            - ``first`` : Mark duplicates as ``True`` except for the first occurrence.
            - ``last`` : Mark duplicates as ``True`` except for the last occurrence.
            - False : Mark all duplicates as ``True``.

        Returns
        -------
        ndarray[bool]

        Examples
        --------
        >>> pd.array([1, 1, 2, 3, 3], dtype="Int64").duplicated()
        array([False,  True, False, False,  True])
        '''
    def unique(self) -> Self:
        """
        Compute the ArrowExtensionArray of unique values.

        Returns
        -------
        ArrowExtensionArray
        """
    def value_counts(self, dropna: bool = ...) -> Series:
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
    def _accumulate(self, name: str, *, skipna: bool = ..., **kwargs) -> ArrowExtensionArray | ExtensionArray:
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
    def _reduce_pyarrow(self, name: str, *, skipna: bool = ..., **kwargs) -> pa.Scalar:
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
    def _reduce(self, name: str, *, skipna: bool = ..., keepdims: bool = ..., **kwargs):
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
    def _reduce_calc(self, name: str, *, skipna: bool = ..., keepdims: bool = ..., **kwargs): ...
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
    def _rank_calc(self, *, axis: AxisInt = ..., method: str = ..., na_option: str = ..., ascending: bool = ..., pct: bool = ...): ...
    def _rank(self, *, axis: AxisInt = ..., method: str = ..., na_option: str = ..., ascending: bool = ..., pct: bool = ...):
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
    def _mode(self, dropna: bool = ...) -> Self:
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
    def _str_count(self, pat: str, flags: int = ...): ...
    def _str_contains(self, pat, case: bool = ..., flags: int = ..., na, regex: bool = ...): ...
    def _str_startswith(self, pat: str | tuple[str, ...], na): ...
    def _str_endswith(self, pat: str | tuple[str, ...], na): ...
    def _str_replace(self, pat: str | re.Pattern, repl: str | Callable, n: int = ..., case: bool = ..., flags: int = ..., regex: bool = ...): ...
    def _str_repeat(self, repeats: int | Sequence[int]): ...
    def _str_match(self, pat: str, case: bool = ..., flags: int = ..., na: Scalar | None): ...
    def _str_fullmatch(self, pat, case: bool = ..., flags: int = ..., na: Scalar | None): ...
    def _str_find(self, sub: str, start: int = ..., end: int | None): ...
    def _str_join(self, sep: str): ...
    def _str_partition(self, sep: str, expand: bool): ...
    def _str_rpartition(self, sep: str, expand: bool): ...
    def _str_slice(self, start: int | None, stop: int | None, step: int | None): ...
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
    def _str_strip(self, to_strip): ...
    def _str_lstrip(self, to_strip): ...
    def _str_rstrip(self, to_strip): ...
    def _str_removeprefix(self, prefix: str): ...
    def _str_casefold(self): ...
    def _str_encode(self, encoding: str, errors: str = ...): ...
    def _str_extract(self, pat: str, flags: int = ..., expand: bool = ...): ...
    def _str_findall(self, pat: str, flags: int = ...): ...
    def _str_get_dummies(self, sep: str = ...): ...
    def _str_index(self, sub: str, start: int = ..., end: int | None): ...
    def _str_rindex(self, sub: str, start: int = ..., end: int | None): ...
    def _str_normalize(self, form: str): ...
    def _str_rfind(self, sub: str, start: int = ..., end): ...
    def _str_split(self, pat: str | None, n: int | None = ..., expand: bool = ..., regex: bool | None): ...
    def _str_rsplit(self, pat: str | None, n: int | None = ...): ...
    def _str_translate(self, table: dict[int, str]): ...
    def _str_wrap(self, width: int, **kwargs): ...
    def _dt_to_pytimedelta(self): ...
    def _dt_total_seconds(self): ...
    def _dt_as_unit(self, unit: str): ...
    def _dt_isocalendar(self): ...
    def _dt_normalize(self): ...
    def _dt_strftime(self, format: str): ...
    def _round_temporally(self, method: Literal['ceil', 'floor', 'round'], freq, ambiguous: TimeAmbiguous = ..., nonexistent: TimeNonexistent = ...): ...
    def _dt_ceil(self, freq, ambiguous: TimeAmbiguous = ..., nonexistent: TimeNonexistent = ...): ...
    def _dt_floor(self, freq, ambiguous: TimeAmbiguous = ..., nonexistent: TimeNonexistent = ...): ...
    def _dt_round(self, freq, ambiguous: TimeAmbiguous = ..., nonexistent: TimeNonexistent = ...): ...
    def _dt_day_name(self, locale: str | None): ...
    def _dt_month_name(self, locale: str | None): ...
    def _dt_to_pydatetime(self): ...
    def _dt_tz_localize(self, tz, ambiguous: TimeAmbiguous = ..., nonexistent: TimeNonexistent = ...): ...
    def _dt_tz_convert(self, tz): ...
    @property
    def dtype(self): ...
    @property
    def nbytes(self): ...
    @property
    def _hasna(self): ...
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
    @property
    def _dt_year(self): ...
    @property
    def _dt_day(self): ...
    @property
    def _dt_day_of_week(self): ...
    @property
    def _dt_dayofweek(self): ...
    @property
    def _dt_weekday(self): ...
    @property
    def _dt_day_of_year(self): ...
    @property
    def _dt_dayofyear(self): ...
    @property
    def _dt_hour(self): ...
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
    @property
    def _dt_daysinmonth(self): ...
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
def transpose_homogeneous_pyarrow(arrays: Sequence[ArrowExtensionArray]) -> list[ArrowExtensionArray]:
    """Transpose arrow extension arrays in a list, but faster.

    Input should be a list of arrays of equal length and all have the same
    dtype. The caller is responsible for ensuring validity of input data.
    """
