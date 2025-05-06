import numpy as np
from _typeshed import Incomplete
from collections.abc import Iterator, Sequence
from pandas import Index as Index, Series as Series
from pandas._libs import lib as lib
from pandas._libs.interval import Interval as Interval, IntervalMixin as IntervalMixin, VALID_CLOSED as VALID_CLOSED, intervals_to_interval_bounds as intervals_to_interval_bounds
from pandas._libs.missing import NA as NA
from pandas._typing import ArrayLike as ArrayLike, AxisInt as AxisInt, Dtype as Dtype, FillnaOptions as FillnaOptions, IntervalClosedType as IntervalClosedType, NpDtype as NpDtype, PositionalIndexer as PositionalIndexer, ScalarIndexer as ScalarIndexer, Self as Self, SequenceIndexer as SequenceIndexer, SortKind as SortKind, TimeArrayLike as TimeArrayLike, npt as npt
from pandas.core.algorithms import isin as isin, take as take, unique as unique
from pandas.core.arrays import ArrowExtensionArray as ArrowExtensionArray
from pandas.core.arrays.base import ExtensionArray as ExtensionArray, _extension_array_shared_docs as _extension_array_shared_docs
from pandas.core.arrays.datetimes import DatetimeArray as DatetimeArray
from pandas.core.arrays.timedeltas import TimedeltaArray as TimedeltaArray
from pandas.core.construction import ensure_wrapped_if_datetimelike as ensure_wrapped_if_datetimelike, extract_array as extract_array
from pandas.core.dtypes.cast import LossySetitemError as LossySetitemError, maybe_upcast_numeric_to_64bit as maybe_upcast_numeric_to_64bit
from pandas.core.dtypes.common import is_float_dtype as is_float_dtype, is_integer_dtype as is_integer_dtype, is_list_like as is_list_like, is_object_dtype as is_object_dtype, is_scalar as is_scalar, is_string_dtype as is_string_dtype, needs_i8_conversion as needs_i8_conversion, pandas_dtype as pandas_dtype
from pandas.core.dtypes.dtypes import CategoricalDtype as CategoricalDtype, IntervalDtype as IntervalDtype
from pandas.core.dtypes.generic import ABCDataFrame as ABCDataFrame, ABCDatetimeIndex as ABCDatetimeIndex, ABCIntervalIndex as ABCIntervalIndex, ABCPeriodIndex as ABCPeriodIndex
from pandas.core.dtypes.missing import is_valid_na_for_dtype as is_valid_na_for_dtype, isna as isna, notna as notna
from pandas.core.indexers import check_array_indexer as check_array_indexer
from pandas.core.ops import invalid_comparison as invalid_comparison, unpack_zerodim_and_defer as unpack_zerodim_and_defer
from pandas.errors import IntCastingNaNError as IntCastingNaNError
from pandas.util._decorators import Appender as Appender
from typing import Literal, overload

IntervalSide = TimeArrayLike | np.ndarray
IntervalOrNA = Interval | float
_interval_shared_docs: dict[str, str]
_shared_docs_kwargs: Incomplete

class IntervalArray(IntervalMixin, ExtensionArray):
    can_hold_na: bool
    _na_value: Incomplete
    _fill_value: Incomplete
    @property
    def ndim(self) -> Literal[1]: ...
    _left: IntervalSide
    _right: IntervalSide
    _dtype: IntervalDtype
    def __new__(cls, data, closed: IntervalClosedType | None = None, dtype: Dtype | None = None, copy: bool = False, verify_integrity: bool = True): ...
    @classmethod
    def _simple_new(cls, left: IntervalSide, right: IntervalSide, dtype: IntervalDtype) -> Self: ...
    @classmethod
    def _ensure_simple_new_inputs(cls, left, right, closed: IntervalClosedType | None = None, copy: bool = False, dtype: Dtype | None = None) -> tuple[IntervalSide, IntervalSide, IntervalDtype]:
        """Ensure correctness of input parameters for cls._simple_new."""
    @classmethod
    def _from_sequence(cls, scalars, *, dtype: Dtype | None = None, copy: bool = False) -> Self: ...
    @classmethod
    def _from_factorized(cls, values: np.ndarray, original: IntervalArray) -> Self: ...
    @classmethod
    def from_breaks(cls, breaks, closed: IntervalClosedType | None = 'right', copy: bool = False, dtype: Dtype | None = None) -> Self: ...
    @classmethod
    def from_arrays(cls, left, right, closed: IntervalClosedType | None = 'right', copy: bool = False, dtype: Dtype | None = None) -> Self: ...
    @classmethod
    def from_tuples(cls, data, closed: IntervalClosedType | None = 'right', copy: bool = False, dtype: Dtype | None = None) -> Self: ...
    @classmethod
    def _validate(cls, left, right, dtype: IntervalDtype) -> None:
        """
        Verify that the IntervalArray is valid.

        Checks that

        * dtype is correct
        * left and right match lengths
        * left and right have the same missing values
        * left is always below right
        """
    def _shallow_copy(self, left, right) -> Self:
        """
        Return a new IntervalArray with the replacement attributes

        Parameters
        ----------
        left : Index
            Values to be used for the left-side of the intervals.
        right : Index
            Values to be used for the right-side of the intervals.
        """
    @property
    def dtype(self) -> IntervalDtype: ...
    @property
    def nbytes(self) -> int: ...
    @property
    def size(self) -> int: ...
    def __iter__(self) -> Iterator: ...
    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, key: ScalarIndexer) -> IntervalOrNA: ...
    @overload
    def __getitem__(self, key: SequenceIndexer) -> Self: ...
    def __setitem__(self, key, value) -> None: ...
    def _cmp_method(self, other, op): ...
    def __eq__(self, other): ...
    def __ne__(self, other): ...
    def __gt__(self, other): ...
    def __ge__(self, other): ...
    def __lt__(self, other): ...
    def __le__(self, other): ...
    def argsort(self, *, ascending: bool = True, kind: SortKind = 'quicksort', na_position: str = 'last', **kwargs) -> np.ndarray: ...
    def min(self, *, axis: AxisInt | None = None, skipna: bool = True) -> IntervalOrNA: ...
    def max(self, *, axis: AxisInt | None = None, skipna: bool = True) -> IntervalOrNA: ...
    def _pad_or_backfill(self, *, method: FillnaOptions, limit: int | None = None, limit_area: Literal['inside', 'outside'] | None = None, copy: bool = True) -> Self: ...
    def fillna(self, value: Incomplete | None = None, method: Incomplete | None = None, limit: int | None = None, copy: bool = True) -> Self:
        '''
        Fill NA/NaN values using the specified method.

        Parameters
        ----------
        value : scalar, dict, Series
            If a scalar value is passed it is used to fill all missing values.
            Alternatively, a Series or dict can be used to fill in different
            values for each index. The value should not be a list. The
            value(s) passed should be either Interval objects or NA/NaN.
        method : {\'backfill\', \'bfill\', \'pad\', \'ffill\', None}, default None
            (Not implemented yet for IntervalArray)
            Method to use for filling holes in reindexed Series
        limit : int, default None
            (Not implemented yet for IntervalArray)
            If method is specified, this is the maximum number of consecutive
            NaN values to forward/backward fill. In other words, if there is
            a gap with more than this number of consecutive NaNs, it will only
            be partially filled. If method is not specified, this is the
            maximum number of entries along the entire axis where NaNs will be
            filled.
        copy : bool, default True
            Whether to make a copy of the data before filling. If False, then
            the original should be modified and no new memory should be allocated.
            For ExtensionArray subclasses that cannot do this, it is at the
            author\'s discretion whether to ignore "copy=False" or to raise.

        Returns
        -------
        filled : IntervalArray with NA/NaN filled
        '''
    def astype(self, dtype, copy: bool = True):
        """
        Cast to an ExtensionArray or NumPy array with dtype 'dtype'.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.

        copy : bool, default True
            Whether to copy the data, even if not necessary. If False,
            a copy is made only if the old dtype does not match the
            new dtype.

        Returns
        -------
        array : ExtensionArray or ndarray
            ExtensionArray or NumPy ndarray with 'dtype' for its dtype.
        """
    def equals(self, other) -> bool: ...
    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[IntervalArray]) -> Self:
        """
        Concatenate multiple IntervalArray

        Parameters
        ----------
        to_concat : sequence of IntervalArray

        Returns
        -------
        IntervalArray
        """
    def copy(self) -> Self:
        """
        Return a copy of the array.

        Returns
        -------
        IntervalArray
        """
    def isna(self) -> np.ndarray: ...
    def shift(self, periods: int = 1, fill_value: object = None) -> IntervalArray: ...
    def take(self, indices, *, allow_fill: bool = False, fill_value: Incomplete | None = None, axis: Incomplete | None = None, **kwargs) -> Self:
        '''
        Take elements from the IntervalArray.

        Parameters
        ----------
        indices : sequence of integers
            Indices to be taken.

        allow_fill : bool, default False
            How to handle negative values in `indices`.

            * False: negative values in `indices` indicate positional indices
              from the right (the default). This is similar to
              :func:`numpy.take`.

            * True: negative values in `indices` indicate
              missing values. These values are set to `fill_value`. Any other
              other negative values raise a ``ValueError``.

        fill_value : Interval or NA, optional
            Fill value to use for NA-indices when `allow_fill` is True.
            This may be ``None``, in which case the default NA value for
            the type, ``self.dtype.na_value``, is used.

            For many ExtensionArrays, there will be two representations of
            `fill_value`: a user-facing "boxed" scalar, and a low-level
            physical NA value. `fill_value` should be the user-facing version,
            and the implementation should handle translating that to the
            physical version for processing the take if necessary.

        axis : any, default None
            Present for compat with IntervalIndex; does nothing.

        Returns
        -------
        IntervalArray

        Raises
        ------
        IndexError
            When the indices are out of bounds for the array.
        ValueError
            When `indices` contains negative values other than ``-1``
            and `allow_fill` is True.
        '''
    def _validate_listlike(self, value): ...
    def _validate_scalar(self, value): ...
    def _validate_setitem_value(self, value): ...
    def value_counts(self, dropna: bool = True) -> Series:
        """
        Returns a Series containing counts of each interval.

        Parameters
        ----------
        dropna : bool, default True
            Don't include counts of NaN.

        Returns
        -------
        counts : Series

        See Also
        --------
        Series.value_counts
        """
    def _formatter(self, boxed: bool = False): ...
    @property
    def left(self) -> Index:
        """
        Return the left endpoints of each Interval in the IntervalArray as an Index.

        Examples
        --------

        >>> interv_arr = pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(2, 5)])
        >>> interv_arr
        <IntervalArray>
        [(0, 1], (2, 5]]
        Length: 2, dtype: interval[int64, right]
        >>> interv_arr.left
        Index([0, 2], dtype='int64')
        """
    @property
    def right(self) -> Index:
        """
        Return the right endpoints of each Interval in the IntervalArray as an Index.

        Examples
        --------

        >>> interv_arr = pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(2, 5)])
        >>> interv_arr
        <IntervalArray>
        [(0, 1], (2, 5]]
        Length: 2, dtype: interval[int64, right]
        >>> interv_arr.right
        Index([1, 5], dtype='int64')
        """
    @property
    def length(self) -> Index:
        """
        Return an Index with entries denoting the length of each Interval.

        Examples
        --------

        >>> interv_arr = pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(1, 5)])
        >>> interv_arr
        <IntervalArray>
        [(0, 1], (1, 5]]
        Length: 2, dtype: interval[int64, right]
        >>> interv_arr.length
        Index([1, 4], dtype='int64')
        """
    @property
    def mid(self) -> Index:
        """
        Return the midpoint of each Interval in the IntervalArray as an Index.

        Examples
        --------

        >>> interv_arr = pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(1, 5)])
        >>> interv_arr
        <IntervalArray>
        [(0, 1], (1, 5]]
        Length: 2, dtype: interval[int64, right]
        >>> interv_arr.mid
        Index([0.5, 3.0], dtype='float64')
        """
    def overlaps(self, other): ...
    @property
    def closed(self) -> IntervalClosedType:
        """
        String describing the inclusive side the intervals.

        Either ``left``, ``right``, ``both`` or ``neither``.

        Examples
        --------

        For arrays:

        >>> interv_arr = pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(1, 5)])
        >>> interv_arr
        <IntervalArray>
        [(0, 1], (1, 5]]
        Length: 2, dtype: interval[int64, right]
        >>> interv_arr.closed
        'right'

        For Interval Index:

        >>> interv_idx = pd.interval_range(start=0, end=2)
        >>> interv_idx
        IntervalIndex([(0, 1], (1, 2]], dtype='interval[int64, right]')
        >>> interv_idx.closed
        'right'
        """
    def set_closed(self, closed: IntervalClosedType) -> Self: ...
    @property
    def is_non_overlapping_monotonic(self) -> bool: ...
    def __array__(self, dtype: NpDtype | None = None, copy: bool | None = None) -> np.ndarray:
        """
        Return the IntervalArray's data as a numpy array of Interval
        objects (with dtype='object')
        """
    def __arrow_array__(self, type: Incomplete | None = None):
        """
        Convert myself into a pyarrow Array.
        """
    def to_tuples(self, na_tuple: bool = True) -> np.ndarray: ...
    def _putmask(self, mask: npt.NDArray[np.bool_], value) -> None: ...
    def insert(self, loc: int, item: Interval) -> Self:
        """
        Return a new IntervalArray inserting new item at location. Follows
        Python numpy.insert semantics for negative values.  Only Interval
        objects and NA can be inserted into an IntervalIndex

        Parameters
        ----------
        loc : int
        item : Interval

        Returns
        -------
        IntervalArray
        """
    def delete(self, loc) -> Self: ...
    def repeat(self, repeats: int | Sequence[int], axis: AxisInt | None = None) -> Self: ...
    def contains(self, other): ...
    def isin(self, values: ArrayLike) -> npt.NDArray[np.bool_]: ...
    @property
    def _combined(self) -> IntervalSide: ...
    def _from_combined(self, combined: np.ndarray) -> IntervalArray:
        """
        Create a new IntervalArray with our dtype from a 1D complex128 ndarray.
        """
    def unique(self) -> IntervalArray: ...

def _maybe_convert_platform_interval(values) -> ArrayLike:
    """
    Try to do platform conversion, with special casing for IntervalArray.
    Wrapper around maybe_convert_platform that alters the default return
    dtype in certain cases to be compatible with IntervalArray.  For example,
    empty lists return with integer dtype instead of object dtype, which is
    prohibited for IntervalArray.

    Parameters
    ----------
    values : array-like

    Returns
    -------
    array
    """
