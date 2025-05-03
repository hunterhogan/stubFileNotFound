import np
import npt
import pandas._libs.interval
import pandas._libs.lib as lib
import pandas.compat.numpy.function as nv
import pandas.core.arrays.base
import pandas.core.common as com
from builtins import AxisInt
from pandas._libs.interval import Interval as Interval, IntervalMixin as IntervalMixin, intervals_to_interval_bounds as intervals_to_interval_bounds
from pandas._libs.lib import is_list_like as is_list_like, is_scalar as is_scalar
from pandas._libs.missing import NA as NA
from pandas.core.algorithms import isin as isin, take as take, unique as unique, value_counts as value_counts
from pandas.core.arrays.arrow.array import ArrowExtensionArray as ArrowExtensionArray
from pandas.core.arrays.base import ExtensionArray as ExtensionArray
from pandas.core.arrays.datetimes import DatetimeArray as DatetimeArray
from pandas.core.arrays.timedeltas import TimedeltaArray as TimedeltaArray
from pandas.core.construction import ensure_wrapped_if_datetimelike as ensure_wrapped_if_datetimelike, extract_array as extract_array, pd_array as pd_array
from pandas.core.dtypes.cast import maybe_upcast_numeric_to_64bit as maybe_upcast_numeric_to_64bit
from pandas.core.dtypes.common import is_float_dtype as is_float_dtype, is_integer_dtype as is_integer_dtype, is_object_dtype as is_object_dtype, is_string_dtype as is_string_dtype, needs_i8_conversion as needs_i8_conversion, pandas_dtype as pandas_dtype
from pandas.core.dtypes.dtypes import CategoricalDtype as CategoricalDtype, IntervalDtype as IntervalDtype
from pandas.core.dtypes.generic import ABCDataFrame as ABCDataFrame, ABCDatetimeIndex as ABCDatetimeIndex, ABCIntervalIndex as ABCIntervalIndex, ABCPeriodIndex as ABCPeriodIndex
from pandas.core.dtypes.missing import is_valid_na_for_dtype as is_valid_na_for_dtype, isna as isna, notna as notna
from pandas.core.indexers.utils import check_array_indexer as check_array_indexer
from pandas.core.ops.common import unpack_zerodim_and_defer as unpack_zerodim_and_defer
from pandas.core.ops.invalid import invalid_comparison as invalid_comparison
from pandas.errors import IntCastingNaNError as IntCastingNaNError, LossySetitemError as LossySetitemError
from pandas.util._decorators import Appender as Appender
from typing import ArrayLike, ClassVar, Dtype, FillnaOptions, IntervalClosedType, IntervalOrNA, IntervalSide, Literal, NpDtype, PositionalIndexer, SortKind

TYPE_CHECKING: bool
VALID_CLOSED: frozenset
Self: None
npt: None
_extension_array_shared_docs: dict
_interval_shared_docs: dict
_shared_docs_kwargs: dict

class IntervalArray(pandas._libs.interval.IntervalMixin, pandas.core.arrays.base.ExtensionArray):
    can_hold_na: ClassVar[bool] = ...
    _na_value: ClassVar[float] = ...
    _fill_value: ClassVar[float] = ...
    @classmethod
    def __init__(cls, data, closed: IntervalClosedType | None, dtype: Dtype | None, copy: bool = ..., verify_integrity: bool = ...) -> None: ...
    @classmethod
    def _simple_new(cls, left: IntervalSide, right: IntervalSide, dtype: IntervalDtype) -> Self: ...
    @classmethod
    def _ensure_simple_new_inputs(cls, left, right, closed: IntervalClosedType | None, copy: bool = ..., dtype: Dtype | None) -> tuple[IntervalSide, IntervalSide, IntervalDtype]:
        """Ensure correctness of input parameters for cls._simple_new."""
    @classmethod
    def _from_sequence(cls, scalars, *, dtype: Dtype | None, copy: bool = ...) -> Self: ...
    @classmethod
    def _from_factorized(cls, values: np.ndarray, original: IntervalArray) -> Self: ...
    @classmethod
    def from_breaks(cls, breaks, closed: IntervalClosedType | None = ..., copy: bool = ..., dtype: Dtype | None) -> Self:
        """
        Construct an IntervalArray from an array of splits.

        Parameters
        ----------
        breaks : array-like (1-dimensional)
            Left and right bounds for each interval.
        closed : {'left', 'right', 'both', 'neither'}, default 'right'
            Whether the intervals are closed on the left-side, right-side, both
            or neither.        
        copy : bool, default False
            Copy the data.
        dtype : dtype or None, default None
            If None, dtype will be inferred.

        Returns
        -------
        IntervalArray

        See Also
        --------
        interval_range : Function to create a fixed frequency IntervalIndex.
        IntervalArray.from_arrays : Construct from a left and right array.
        IntervalArray.from_tuples : Construct from a sequence of tuples.

        Examples
        --------
        >>> pd.arrays.IntervalArray.from_breaks([0, 1, 2, 3])
        <IntervalArray>
        [(0, 1], (1, 2], (2, 3]]
        Length: 3, dtype: interval[int64, right]
        """
    @classmethod
    def from_arrays(cls, left, right, closed: IntervalClosedType | None = ..., copy: bool = ..., dtype: Dtype | None) -> Self:
        """
        Construct from two arrays defining the left and right bounds.

        Parameters
        ----------
        left : array-like (1-dimensional)
            Left bounds for each interval.
        right : array-like (1-dimensional)
            Right bounds for each interval.
        closed : {'left', 'right', 'both', 'neither'}, default 'right'
            Whether the intervals are closed on the left-side, right-side, both
            or neither.        
        copy : bool, default False
            Copy the data.
        dtype : dtype, optional
            If None, dtype will be inferred.

        Returns
        -------
        IntervalArray

        Raises
        ------
        ValueError
            When a value is missing in only one of `left` or `right`.
            When a value in `left` is greater than the corresponding value
            in `right`.

        See Also
        --------
        interval_range : Function to create a fixed frequency IntervalIndex.
        IntervalArray.from_breaks : Construct an IntervalArray from an array of
            splits.
        IntervalArray.from_tuples : Construct an IntervalArray from an
            array-like of tuples.

        Notes
        -----
        Each element of `left` must be less than or equal to the `right`
        element at the same position. If an element is missing, it must be
        missing in both `left` and `right`. A TypeError is raised when
        using an unsupported type for `left` or `right`. At the moment,
        'category', 'object', and 'string' subtypes are not supported.

        Examples
        --------
        >>> pd.arrays.IntervalArray.from_arrays([0, 1, 2], [1, 2, 3])
        <IntervalArray>
        [(0, 1], (1, 2], (2, 3]]
        Length: 3, dtype: interval[int64, right]
        """
    @classmethod
    def from_tuples(cls, data, closed: IntervalClosedType | None = ..., copy: bool = ..., dtype: Dtype | None) -> Self:
        """
        Construct an IntervalArray from an array-like of tuples.

        Parameters
        ----------
        data : array-like (1-dimensional)
            Array of tuples.
        closed : {'left', 'right', 'both', 'neither'}, default 'right'
            Whether the intervals are closed on the left-side, right-side, both
            or neither.        
        copy : bool, default False
            By-default copy the data, this is compat only and ignored.
        dtype : dtype or None, default None
            If None, dtype will be inferred.

        Returns
        -------
        IntervalArray

        See Also
        --------
        interval_range : Function to create a fixed frequency IntervalIndex.
        IntervalArray.from_arrays : Construct an IntervalArray from a left and
                                    right array.
        IntervalArray.from_breaks : Construct an IntervalArray from an array of
                                    splits.

        Examples
        --------
        >>> pd.arrays.IntervalArray.from_tuples([(0, 1), (1, 2)])
        <IntervalArray>
        [(0, 1], (1, 2]]
        Length: 2, dtype: interval[int64, right]
        """
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
    def __iter__(self) -> Iterator: ...
    def __len__(self) -> int: ...
    def __getitem__(self, key: PositionalIndexer) -> Self | IntervalOrNA: ...
    def __setitem__(self, key, value) -> None: ...
    def _cmp_method(self, other, op): ...
    def __eq__(self, other) -> bool: ...
    def __ne__(self, other) -> bool: ...
    def __gt__(self, other) -> bool: ...
    def __ge__(self, other) -> bool: ...
    def __lt__(self, other) -> bool: ...
    def __le__(self, other) -> bool: ...
    def argsort(self, *, ascending: bool = ..., kind: SortKind = ..., na_position: str = ..., **kwargs) -> np.ndarray: ...
    def min(self, *, axis: AxisInt | None, skipna: bool = ...) -> IntervalOrNA: ...
    def max(self, *, axis: AxisInt | None, skipna: bool = ...) -> IntervalOrNA: ...
    def _pad_or_backfill(self, *, method: FillnaOptions, limit: int | None, limit_area: Literal['inside', 'outside'] | None, copy: bool = ...) -> Self: ...
    def fillna(self, value, method, limit: int | None, copy: bool = ...) -> Self:
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
    def astype(self, dtype, copy: bool = ...):
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
    def shift(self, periods: int = ..., fill_value: object) -> IntervalArray: ...
    def take(self, indices, *, allow_fill: bool = ..., fill_value, axis, **kwargs) -> Self:
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
    def value_counts(self, dropna: bool = ...) -> Series:
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
    def _formatter(self, boxed: bool = ...): ...
    def overlaps(self, other):
        """
        Check elementwise if an Interval overlaps the values in the IntervalArray.

        Two intervals overlap if they share a common point, including closed
        endpoints. Intervals that only have an open endpoint in common do not
        overlap.

        Parameters
        ----------
        other : IntervalArray
            Interval to check against for an overlap.

        Returns
        -------
        ndarray
            Boolean array positionally indicating where an overlap occurs.

        See Also
        --------
        Interval.overlaps : Check whether two Interval objects overlap.

        Examples
        --------
        >>> data = [(0, 1), (1, 3), (2, 4)]
        >>> intervals = pd.arrays.IntervalArray.from_tuples(data)
        >>> intervals
        <IntervalArray>
        [(0, 1], (1, 3], (2, 4]]
        Length: 3, dtype: interval[int64, right]

        >>> intervals.overlaps(pd.Interval(0.5, 1.5))
        array([ True,  True, False])

        Intervals that share closed endpoints overlap:

        >>> intervals.overlaps(pd.Interval(1, 3, closed='left'))
        array([ True,  True, True])

        Intervals that only have an open endpoint in common do not overlap:

        >>> intervals.overlaps(pd.Interval(1, 2, closed='right'))
        array([False,  True, False])
        """
    def set_closed(self, closed: IntervalClosedType) -> Self:
        """
        Return an identical IntervalArray closed on the specified side.

        Parameters
        ----------
        closed : {'left', 'right', 'both', 'neither'}
            Whether the intervals are closed on the left-side, right-side, both
            or neither.

        Returns
        -------
        IntervalArray

        Examples
        --------
        >>> index = pd.arrays.IntervalArray.from_breaks(range(4))
        >>> index
        <IntervalArray>
        [(0, 1], (1, 2], (2, 3]]
        Length: 3, dtype: interval[int64, right]
        >>> index.set_closed('both')
        <IntervalArray>
        [[0, 1], [1, 2], [2, 3]]
        Length: 3, dtype: interval[int64, both]
        """
    def __array__(self, dtype: NpDtype | None, copy: bool | None) -> np.ndarray:
        """
        Return the IntervalArray's data as a numpy array of Interval
        objects (with dtype='object')
        """
    def __arrow_array__(self, type):
        """
        Convert myself into a pyarrow Array.
        """
    def to_tuples(self, na_tuple: bool = ...) -> np.ndarray:
        """
        Return an ndarray (if self is IntervalArray) or Index (if self is IntervalIndex) of tuples of the form (left, right).

        Parameters
        ----------
        na_tuple : bool, default True
            If ``True``, return ``NA`` as a tuple ``(nan, nan)``. If ``False``,
            just return ``NA`` as ``nan``.

        Returns
        -------
        tuples: ndarray (if self is IntervalArray) or Index (if self is IntervalIndex)

        Examples
        --------
        For :class:`pandas.IntervalArray`:

        >>> idx = pd.arrays.IntervalArray.from_tuples([(0, 1), (1, 2)])
        >>> idx
        <IntervalArray>
        [(0, 1], (1, 2]]
        Length: 2, dtype: interval[int64, right]
        >>> idx.to_tuples()
        array([(0, 1), (1, 2)], dtype=object)

        For :class:`pandas.IntervalIndex`:

        >>> idx = pd.interval_range(start=0, end=2)
        >>> idx
        IntervalIndex([(0, 1], (1, 2]], dtype='interval[int64, right]')
        >>> idx.to_tuples()
        Index([(0, 1), (1, 2)], dtype='object')
        """
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
    def repeat(self, repeats: int | Sequence[int], axis: AxisInt | None) -> Self:
        """
        Repeat elements of a IntervalArray.

        Returns a new IntervalArray where each element of the current IntervalArray
        is repeated consecutively a given number of times.

        Parameters
        ----------
        repeats : int or array of ints
            The number of repetitions for each element. This should be a
            non-negative integer. Repeating 0 times will return an empty
            IntervalArray.
        axis : None
            Must be ``None``. Has no effect but is accepted for compatibility
            with numpy.

        Returns
        -------
        IntervalArray
            Newly created IntervalArray with repeated elements.

        See Also
        --------
        Series.repeat : Equivalent function for Series.
        Index.repeat : Equivalent function for Index.
        numpy.repeat : Similar method for :class:`numpy.ndarray`.
        ExtensionArray.take : Take arbitrary positions.

        Examples
        --------
        >>> cat = pd.Categorical(['a', 'b', 'c'])
        >>> cat
        ['a', 'b', 'c']
        Categories (3, object): ['a', 'b', 'c']
        >>> cat.repeat(2)
        ['a', 'a', 'b', 'b', 'c', 'c']
        Categories (3, object): ['a', 'b', 'c']
        >>> cat.repeat([1, 2, 3])
        ['a', 'b', 'b', 'c', 'c', 'c']
        Categories (3, object): ['a', 'b', 'c']
        """
    def contains(self, other):
        """
        Check elementwise if the Intervals contain the value.

        Return a boolean mask whether the value is contained in the Intervals
        of the IntervalArray.

        Parameters
        ----------
        other : scalar
            The value to check whether it is contained in the Intervals.

        Returns
        -------
        boolean array

        See Also
        --------
        Interval.contains : Check whether Interval object contains value.
        IntervalArray.overlaps : Check if an Interval overlaps the values in the
            IntervalArray.

        Examples
        --------
        >>> intervals = pd.arrays.IntervalArray.from_tuples([(0, 1), (1, 3), (2, 4)])
        >>> intervals
        <IntervalArray>
        [(0, 1], (1, 3], (2, 4]]
        Length: 3, dtype: interval[int64, right]

        >>> intervals.contains(0.5)
        array([ True, False, False])
        """
    def isin(self, values: ArrayLike) -> npt.NDArray[np.bool_]: ...
    def _from_combined(self, combined: np.ndarray) -> IntervalArray:
        """
        Create a new IntervalArray with our dtype from a 1D complex128 ndarray.
        """
    def unique(self) -> IntervalArray: ...
    @property
    def ndim(self): ...
    @property
    def dtype(self): ...
    @property
    def nbytes(self): ...
    @property
    def size(self): ...
    @property
    def left(self): ...
    @property
    def right(self): ...
    @property
    def length(self): ...
    @property
    def mid(self): ...
    @property
    def closed(self): ...
    @property
    def is_non_overlapping_monotonic(self): ...
    @property
    def _combined(self): ...
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
