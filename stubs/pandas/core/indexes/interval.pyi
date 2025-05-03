import np
import npt
import pandas._libs.lib as lib
import pandas.core.arrays.interval
import pandas.core.common as com
import pandas.core.indexes.base as ibase
import pandas.core.indexes.extension
from _typeshed import Incomplete
from pandas._libs.algos import ensure_platform_int as ensure_platform_int
from pandas._libs.interval import Interval as Interval, IntervalMixin as IntervalMixin, IntervalTree as IntervalTree
from pandas._libs.lib import is_integer as is_integer, is_list_like as is_list_like, is_scalar as is_scalar
from pandas._libs.properties import cache_readonly as cache_readonly
from pandas._libs.tslibs.offsets import BaseOffset as BaseOffset, to_offset as to_offset
from pandas._libs.tslibs.period import Period as Period
from pandas._libs.tslibs.timedeltas import Timedelta as Timedelta
from pandas._libs.tslibs.timestamps import Timestamp as Timestamp
from pandas.core.algorithms import unique as unique
from pandas.core.arrays.datetimelike import validate_periods as validate_periods
from pandas.core.arrays.interval import IntervalArray as IntervalArray
from pandas.core.dtypes.cast import find_common_type as find_common_type, infer_dtype_from_scalar as infer_dtype_from_scalar, maybe_box_datetimelike as maybe_box_datetimelike, maybe_downcast_numeric as maybe_downcast_numeric, maybe_upcast_numeric_to_64bit as maybe_upcast_numeric_to_64bit
from pandas.core.dtypes.common import is_float_dtype as is_float_dtype, is_integer_dtype as is_integer_dtype, is_object_dtype as is_object_dtype, pandas_dtype as pandas_dtype
from pandas.core.dtypes.dtypes import DatetimeTZDtype as DatetimeTZDtype, IntervalDtype as IntervalDtype
from pandas.core.dtypes.inference import is_number as is_number
from pandas.core.dtypes.missing import is_valid_na_for_dtype as is_valid_na_for_dtype
from pandas.core.indexers.utils import is_valid_positional_slice as is_valid_positional_slice
from pandas.core.indexes.base import Index as Index, ensure_index as ensure_index, maybe_extract_name as maybe_extract_name
from pandas.core.indexes.datetimes import DatetimeIndex as DatetimeIndex, date_range as date_range
from pandas.core.indexes.extension import ExtensionIndex as ExtensionIndex, inherit_names as inherit_names
from pandas.core.indexes.multi import MultiIndex as MultiIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex as TimedeltaIndex, timedelta_range as timedelta_range
from pandas.errors import InvalidIndexError as InvalidIndexError
from pandas.util._decorators import Appender as Appender
from pandas.util._exceptions import rewrite_exception as rewrite_exception
from typing import Any, ClassVar, Literal

TYPE_CHECKING: bool
_interval_shared_docs: dict
_index_shared_docs: dict
_index_doc_kwargs: dict
def _get_next_label(label): ...
def _get_prev_label(label): ...
def _new_IntervalIndex(cls, d):
    """
    This is called upon unpickling, rather than the default which doesn't have
    arguments and breaks __new__.
    """

class IntervalIndex(pandas.core.indexes.extension.ExtensionIndex):
    _typ: ClassVar[str] = ...
    _can_hold_strings: ClassVar[bool] = ...
    _data_cls: ClassVar[type[pandas.core.arrays.interval.IntervalArray]] = ...
    _requires_unique_msg: ClassVar[str] = ...
    _engine: Incomplete
    _multiindex: Incomplete
    is_monotonic_decreasing: Incomplete
    is_unique: Incomplete
    _index_as_unique: Incomplete
    _should_fallback_to_positional: Incomplete
    left: Incomplete
    right: Incomplete
    mid: Incomplete
    is_non_overlapping_monotonic: Incomplete
    closed: Incomplete
    closed_left: Incomplete
    closed_right: Incomplete
    open_left: Incomplete
    open_right: Incomplete
    is_empty: Incomplete
    @classmethod
    def __init__(cls, data, closed: IntervalClosedType | None, dtype: Dtype | None, copy: bool = ..., name: Hashable | None, verify_integrity: bool = ...) -> Self: ...
    @classmethod
    def from_breaks(cls, breaks, closed: IntervalClosedType | None = ..., name: Hashable | None, copy: bool = ..., dtype: Dtype | None) -> IntervalIndex:
        """
        Construct an IntervalIndex from an array of splits.

        Parameters
        ----------
        breaks : array-like (1-dimensional)
            Left and right bounds for each interval.
        closed : {'left', 'right', 'both', 'neither'}, default 'right'
            Whether the intervals are closed on the left-side, right-side, both
            or neither.        
        name : str, optional
             Name of the resulting IntervalIndex.
        copy : bool, default False
            Copy the data.
        dtype : dtype or None, default None
            If None, dtype will be inferred.

        Returns
        -------
        IntervalIndex

        See Also
        --------
        interval_range : Function to create a fixed frequency IntervalIndex.
        IntervalIndex.from_arrays : Construct from a left and right array.
        IntervalIndex.from_tuples : Construct from a sequence of tuples.

        Examples
        --------
        >>> pd.IntervalIndex.from_breaks([0, 1, 2, 3])
        IntervalIndex([(0, 1], (1, 2], (2, 3]],
                      dtype='interval[int64, right]')
        """
    @classmethod
    def from_arrays(cls, left, right, closed: IntervalClosedType = ..., name: Hashable | None, copy: bool = ..., dtype: Dtype | None) -> IntervalIndex:
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
        name : str, optional
             Name of the resulting IntervalIndex.
        copy : bool, default False
            Copy the data.
        dtype : dtype, optional
            If None, dtype will be inferred.

        Returns
        -------
        IntervalIndex

        Raises
        ------
        ValueError
            When a value is missing in only one of `left` or `right`.
            When a value in `left` is greater than the corresponding value
            in `right`.

        See Also
        --------
        interval_range : Function to create a fixed frequency IntervalIndex.
        IntervalIndex.from_breaks : Construct an IntervalIndex from an array of
            splits.
        IntervalIndex.from_tuples : Construct an IntervalIndex from an
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
        >>> pd.IntervalIndex.from_arrays([0, 1, 2], [1, 2, 3])
        IntervalIndex([(0, 1], (1, 2], (2, 3]],
                      dtype='interval[int64, right]')
        """
    @classmethod
    def from_tuples(cls, data, closed: IntervalClosedType = ..., name: Hashable | None, copy: bool = ..., dtype: Dtype | None) -> IntervalIndex:
        """
        Construct an IntervalIndex from an array-like of tuples.

        Parameters
        ----------
        data : array-like (1-dimensional)
            Array of tuples.
        closed : {'left', 'right', 'both', 'neither'}, default 'right'
            Whether the intervals are closed on the left-side, right-side, both
            or neither.        
        name : str, optional
             Name of the resulting IntervalIndex.
        copy : bool, default False
            By-default copy the data, this is compat only and ignored.
        dtype : dtype or None, default None
            If None, dtype will be inferred.

        Returns
        -------
        IntervalIndex

        See Also
        --------
        interval_range : Function to create a fixed frequency IntervalIndex.
        IntervalIndex.from_arrays : Construct an IntervalIndex from a left and
                                    right array.
        IntervalIndex.from_breaks : Construct an IntervalIndex from an array of
                                    splits.

        Examples
        --------
        >>> pd.IntervalIndex.from_tuples([(0, 1), (1, 2)])
        IntervalIndex([(0, 1], (1, 2]],
                       dtype='interval[int64, right]')
        """
    def __contains__(self, key: Any) -> bool:
        """
        return a boolean if this key is IN the index
        We *only* accept an Interval

        Parameters
        ----------
        key : Interval

        Returns
        -------
        bool
        """
    def _getitem_slice(self, slobj: slice) -> IntervalIndex:
        """
        Fastpath for __getitem__ when we know we have a slice.
        """
    def __reduce__(self): ...
    def memory_usage(self, deep: bool = ...) -> int:
        """
        Memory usage of the values.

        Parameters
        ----------
        deep : bool, default False
            Introspect the data deeply, interrogate
            `object` dtypes for system-level memory consumption.

        Returns
        -------
        bytes used

        See Also
        --------
        numpy.ndarray.nbytes : Total bytes consumed by the elements of the
            array.

        Notes
        -----
        Memory usage does not include memory consumed by elements that
        are not components of the array if deep=False or if used on PyPy

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3])
        >>> idx.memory_usage()
        24
        """
    def _needs_i8_conversion(self, key) -> bool:
        """
        Check if a given key needs i8 conversion. Conversion is necessary for
        Timestamp, Timedelta, DatetimeIndex, and TimedeltaIndex keys. An
        Interval-like requires conversion if its endpoints are one of the
        aforementioned types.

        Assumes that any list-like data has already been cast to an Index.

        Parameters
        ----------
        key : scalar or Index-like
            The key that should be checked for i8 conversion

        Returns
        -------
        bool
        """
    def _maybe_convert_i8(self, key):
        """
        Maybe convert a given key to its equivalent i8 value(s). Used as a
        preprocessing step prior to IntervalTree queries (self._engine), which
        expects numeric data.

        Parameters
        ----------
        key : scalar or list-like
            The key that should maybe be converted to i8.

        Returns
        -------
        scalar or list-like
            The original key if no conversion occurred, int if converted scalar,
            Index with an int64 dtype if converted list-like.
        """
    def _searchsorted_monotonic(self, label, side: Literal['left', 'right'] = ...): ...
    def get_loc(self, key) -> int | slice | np.ndarray:
        """
        Get integer location, slice or boolean mask for requested label.

        Parameters
        ----------
        key : label

        Returns
        -------
        int if unique index, slice if monotonic index, else mask

        Examples
        --------
        >>> i1, i2 = pd.Interval(0, 1), pd.Interval(1, 2)
        >>> index = pd.IntervalIndex([i1, i2])
        >>> index.get_loc(1)
        0

        You can also supply a point inside an interval.

        >>> index.get_loc(1.5)
        1

        If a label is in several intervals, you get the locations of all the
        relevant intervals.

        >>> i3 = pd.Interval(0, 2)
        >>> overlapping_index = pd.IntervalIndex([i1, i2, i3])
        >>> overlapping_index.get_loc(0.5)
        array([ True, False,  True])

        Only exact matches will be returned if an interval is provided.

        >>> index.get_loc(pd.Interval(0, 1))
        0
        """
    def _get_indexer(self, target: Index, method: str | None, limit: int | None, tolerance: Any | None) -> npt.NDArray[np.intp]: ...
    def get_indexer_non_unique(self, target: Index) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        """
        Compute indexer and mask for new index given the current index.

        The indexer should be then used as an input to ndarray.take to align the
        current data to the new index.

        Parameters
        ----------
        target : IntervalIndex or list of Intervals

        Returns
        -------
        indexer : np.ndarray[np.intp]
            Integers from 0 to n - 1 indicating that the index at these
            positions matches the corresponding target values. Missing values
            in the target are marked by -1.
        missing : np.ndarray[np.intp]
            An indexer into the target of the values not found.
            These correspond to the -1 in the indexer array.

        Examples
        --------
        >>> index = pd.Index(['c', 'b', 'a', 'b', 'b'])
        >>> index.get_indexer_non_unique(['b', 'b'])
        (array([1, 3, 4, 1, 3, 4]), array([], dtype=int64))

        In the example below there are no matched values.

        >>> index = pd.Index(['c', 'b', 'a', 'b', 'b'])
        >>> index.get_indexer_non_unique(['q', 'r', 't'])
        (array([-1, -1, -1]), array([0, 1, 2]))

        For this reason, the returned ``indexer`` contains only integers equal to -1.
        It demonstrates that there's no match between the index and the ``target``
        values at these positions. The mask [0, 1, 2] in the return value shows that
        the first, second, and third elements are missing.

        Notice that the return value is a tuple contains two items. In the example
        below the first item is an array of locations in ``index``. The second
        item is a mask shows that the first and third elements are missing.

        >>> index = pd.Index(['c', 'b', 'a', 'b', 'b'])
        >>> index.get_indexer_non_unique(['f', 'b', 's'])
        (array([-1,  1,  3,  4, -1]), array([0, 2]))
        """
    def _get_indexer_unique_sides(self, target: IntervalIndex) -> npt.NDArray[np.intp]:
        """
        _get_indexer specialized to the case where both of our sides are unique.
        """
    def _get_indexer_pointwise(self, target: Index) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        """
        pointwise implementation for get_indexer and get_indexer_non_unique.
        """
    def _convert_slice_indexer(self, key: slice, kind: Literal['loc', 'getitem']): ...
    def _maybe_cast_slice_bound(self, label, side: str): ...
    def _is_comparable_dtype(self, dtype: DtypeObj) -> bool: ...
    def _intersection(self, other, sort):
        """
        intersection specialized to the case with matching dtypes.
        """
    def _intersection_unique(self, other: IntervalIndex) -> IntervalIndex:
        """
        Used when the IntervalIndex does not have any common endpoint,
        no matter left or right.
        Return the intersection with another IntervalIndex.
        Parameters
        ----------
        other : IntervalIndex
        Returns
        -------
        IntervalIndex
        """
    def _intersection_non_unique(self, other: IntervalIndex) -> IntervalIndex:
        """
        Used when the IntervalIndex does have some common endpoints,
        on either sides.
        Return the intersection with another IntervalIndex.

        Parameters
        ----------
        other : IntervalIndex

        Returns
        -------
        IntervalIndex
        """
    def _get_engine_target(self) -> np.ndarray: ...
    def _from_join_target(self, result): ...
    def __array__(self, *args, **kwargs):
        """
        Return the IntervalArray's data as a numpy array of Interval
        objects (with dtype='object')
        """
    def overlaps(self, *args, **kwargs):
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
    def contains(self, *args, **kwargs):
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
    def set_closed(self, *args, **kwargs):
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
    def to_tuples(self, *args, **kwargs):
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
    @property
    def inferred_type(self): ...
    @property
    def is_overlapping(self): ...
    @property
    def length(self): ...
def _is_valid_endpoint(endpoint) -> bool:
    """
    Helper for interval_range to check if start/end are valid types.
    """
def _is_type_compatible(a, b) -> bool:
    """
    Helper for interval_range to check type compat of start/end/freq.
    """
def interval_range(start, end, periods, freq, name: Hashable | None, closed: IntervalClosedType = ...) -> IntervalIndex:
    """
    Return a fixed frequency IntervalIndex.

    Parameters
    ----------
    start : numeric or datetime-like, default None
        Left bound for generating intervals.
    end : numeric or datetime-like, default None
        Right bound for generating intervals.
    periods : int, default None
        Number of periods to generate.
    freq : numeric, str, Timedelta, datetime.timedelta, or DateOffset, default None
        The length of each interval. Must be consistent with the type of start
        and end, e.g. 2 for numeric, or '5H' for datetime-like.  Default is 1
        for numeric and 'D' for datetime-like.
    name : str, default None
        Name of the resulting IntervalIndex.
    closed : {'left', 'right', 'both', 'neither'}, default 'right'
        Whether the intervals are closed on the left-side, right-side, both
        or neither.

    Returns
    -------
    IntervalIndex

    See Also
    --------
    IntervalIndex : An Index of intervals that are all closed on the same side.

    Notes
    -----
    Of the four parameters ``start``, ``end``, ``periods``, and ``freq``,
    exactly three must be specified. If ``freq`` is omitted, the resulting
    ``IntervalIndex`` will have ``periods`` linearly spaced elements between
    ``start`` and ``end``, inclusively.

    To learn more about datetime-like frequency strings, please see `this link
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__.

    Examples
    --------
    Numeric ``start`` and  ``end`` is supported.

    >>> pd.interval_range(start=0, end=5)
    IntervalIndex([(0, 1], (1, 2], (2, 3], (3, 4], (4, 5]],
                  dtype='interval[int64, right]')

    Additionally, datetime-like input is also supported.

    >>> pd.interval_range(start=pd.Timestamp('2017-01-01'),
    ...                   end=pd.Timestamp('2017-01-04'))
    IntervalIndex([(2017-01-01 00:00:00, 2017-01-02 00:00:00],
                   (2017-01-02 00:00:00, 2017-01-03 00:00:00],
                   (2017-01-03 00:00:00, 2017-01-04 00:00:00]],
                  dtype='interval[datetime64[ns], right]')

    The ``freq`` parameter specifies the frequency between the left and right.
    endpoints of the individual intervals within the ``IntervalIndex``.  For
    numeric ``start`` and ``end``, the frequency must also be numeric.

    >>> pd.interval_range(start=0, periods=4, freq=1.5)
    IntervalIndex([(0.0, 1.5], (1.5, 3.0], (3.0, 4.5], (4.5, 6.0]],
                  dtype='interval[float64, right]')

    Similarly, for datetime-like ``start`` and ``end``, the frequency must be
    convertible to a DateOffset.

    >>> pd.interval_range(start=pd.Timestamp('2017-01-01'),
    ...                   periods=3, freq='MS')
    IntervalIndex([(2017-01-01 00:00:00, 2017-02-01 00:00:00],
                   (2017-02-01 00:00:00, 2017-03-01 00:00:00],
                   (2017-03-01 00:00:00, 2017-04-01 00:00:00]],
                  dtype='interval[datetime64[ns], right]')

    Specify ``start``, ``end``, and ``periods``; the frequency is generated
    automatically (linearly spaced).

    >>> pd.interval_range(start=0, end=6, periods=4)
    IntervalIndex([(0.0, 1.5], (1.5, 3.0], (3.0, 4.5], (4.5, 6.0]],
              dtype='interval[float64, right]')

    The ``closed`` parameter specifies which endpoints of the individual
    intervals within the ``IntervalIndex`` are closed.

    >>> pd.interval_range(end=5, periods=4, closed='both')
    IntervalIndex([[1, 2], [2, 3], [3, 4], [4, 5]],
                  dtype='interval[int64, both]')
    """
