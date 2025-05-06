import numpy as np
from _typeshed import Incomplete
from collections.abc import Hashable
from pandas._libs import lib as lib
from pandas._libs.interval import Interval as Interval, IntervalMixin as IntervalMixin, IntervalTree as IntervalTree
from pandas._libs.tslibs import BaseOffset as BaseOffset, Period as Period, Timedelta as Timedelta, Timestamp as Timestamp, to_offset as to_offset
from pandas._typing import Dtype as Dtype, DtypeObj as DtypeObj, IntervalClosedType as IntervalClosedType, Self as Self, npt as npt
from pandas.core.algorithms import unique as unique
from pandas.core.arrays.datetimelike import validate_periods as validate_periods
from pandas.core.arrays.interval import IntervalArray as IntervalArray, _interval_shared_docs as _interval_shared_docs
from pandas.core.dtypes.cast import find_common_type as find_common_type, infer_dtype_from_scalar as infer_dtype_from_scalar, maybe_box_datetimelike as maybe_box_datetimelike, maybe_downcast_numeric as maybe_downcast_numeric, maybe_upcast_numeric_to_64bit as maybe_upcast_numeric_to_64bit
from pandas.core.dtypes.common import ensure_platform_int as ensure_platform_int, is_float_dtype as is_float_dtype, is_integer as is_integer, is_integer_dtype as is_integer_dtype, is_list_like as is_list_like, is_number as is_number, is_object_dtype as is_object_dtype, is_scalar as is_scalar, pandas_dtype as pandas_dtype
from pandas.core.dtypes.dtypes import DatetimeTZDtype as DatetimeTZDtype, IntervalDtype as IntervalDtype
from pandas.core.dtypes.missing import is_valid_na_for_dtype as is_valid_na_for_dtype
from pandas.core.indexers import is_valid_positional_slice as is_valid_positional_slice
from pandas.core.indexes.base import Index as Index, _index_shared_docs as _index_shared_docs, ensure_index as ensure_index, maybe_extract_name as maybe_extract_name
from pandas.core.indexes.datetimes import DatetimeIndex as DatetimeIndex, date_range as date_range
from pandas.core.indexes.extension import ExtensionIndex as ExtensionIndex, inherit_names as inherit_names
from pandas.core.indexes.multi import MultiIndex as MultiIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex as TimedeltaIndex, timedelta_range as timedelta_range
from pandas.errors import InvalidIndexError as InvalidIndexError
from pandas.util._decorators import Appender as Appender, cache_readonly as cache_readonly
from pandas.util._exceptions import rewrite_exception as rewrite_exception
from typing import Any, Literal

_index_doc_kwargs: Incomplete

def _get_next_label(label): ...
def _get_prev_label(label): ...
def _new_IntervalIndex(cls, d):
    """
    This is called upon unpickling, rather than the default which doesn't have
    arguments and breaks __new__.
    """

class IntervalIndex(ExtensionIndex):
    _typ: str
    closed: IntervalClosedType
    is_non_overlapping_monotonic: bool
    closed_left: bool
    closed_right: bool
    open_left: bool
    open_right: bool
    _data: IntervalArray
    _values: IntervalArray
    _can_hold_strings: bool
    _data_cls = IntervalArray
    def __new__(cls, data, closed: IntervalClosedType | None = None, dtype: Dtype | None = None, copy: bool = False, name: Hashable | None = None, verify_integrity: bool = True) -> Self: ...
    @classmethod
    def from_breaks(cls, breaks, closed: IntervalClosedType | None = 'right', name: Hashable | None = None, copy: bool = False, dtype: Dtype | None = None) -> IntervalIndex: ...
    @classmethod
    def from_arrays(cls, left, right, closed: IntervalClosedType = 'right', name: Hashable | None = None, copy: bool = False, dtype: Dtype | None = None) -> IntervalIndex: ...
    @classmethod
    def from_tuples(cls, data, closed: IntervalClosedType = 'right', name: Hashable | None = None, copy: bool = False, dtype: Dtype | None = None) -> IntervalIndex: ...
    def _engine(self) -> IntervalTree: ...
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
    def _multiindex(self) -> MultiIndex: ...
    def __reduce__(self): ...
    @property
    def inferred_type(self) -> str:
        """Return a string of the type inferred from the values"""
    def memory_usage(self, deep: bool = False) -> int: ...
    def is_monotonic_decreasing(self) -> bool:
        """
        Return True if the IntervalIndex is monotonic decreasing (only equal or
        decreasing values), else False
        """
    def is_unique(self) -> bool:
        """
        Return True if the IntervalIndex contains unique elements, else False.
        """
    @property
    def is_overlapping(self) -> bool:
        """
        Return True if the IntervalIndex has overlapping intervals, else False.

        Two intervals overlap if they share a common point, including closed
        endpoints. Intervals that only have an open endpoint in common do not
        overlap.

        Returns
        -------
        bool
            Boolean indicating if the IntervalIndex has overlapping intervals.

        See Also
        --------
        Interval.overlaps : Check whether two Interval objects overlap.
        IntervalIndex.overlaps : Check an IntervalIndex elementwise for
            overlaps.

        Examples
        --------
        >>> index = pd.IntervalIndex.from_tuples([(0, 2), (1, 3), (4, 5)])
        >>> index
        IntervalIndex([(0, 2], (1, 3], (4, 5]],
              dtype='interval[int64, right]')
        >>> index.is_overlapping
        True

        Intervals that share closed endpoints overlap:

        >>> index = pd.interval_range(0, 3, closed='both')
        >>> index
        IntervalIndex([[0, 1], [1, 2], [2, 3]],
              dtype='interval[int64, both]')
        >>> index.is_overlapping
        True

        Intervals that only have an open endpoint in common do not overlap:

        >>> index = pd.interval_range(0, 3, closed='left')
        >>> index
        IntervalIndex([[0, 1), [1, 2), [2, 3)],
              dtype='interval[int64, left]')
        >>> index.is_overlapping
        False
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
    def _searchsorted_monotonic(self, label, side: Literal['left', 'right'] = 'left'): ...
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
    def _get_indexer(self, target: Index, method: str | None = None, limit: int | None = None, tolerance: Any | None = None) -> npt.NDArray[np.intp]: ...
    def get_indexer_non_unique(self, target: Index) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]: ...
    def _get_indexer_unique_sides(self, target: IntervalIndex) -> npt.NDArray[np.intp]:
        """
        _get_indexer specialized to the case where both of our sides are unique.
        """
    def _get_indexer_pointwise(self, target: Index) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        """
        pointwise implementation for get_indexer and get_indexer_non_unique.
        """
    def _index_as_unique(self) -> bool: ...
    _requires_unique_msg: str
    def _convert_slice_indexer(self, key: slice, kind: Literal['loc', 'getitem']): ...
    def _should_fallback_to_positional(self) -> bool: ...
    def _maybe_cast_slice_bound(self, label, side: str): ...
    def _is_comparable_dtype(self, dtype: DtypeObj) -> bool: ...
    def left(self) -> Index: ...
    def right(self) -> Index: ...
    def mid(self) -> Index: ...
    @property
    def length(self) -> Index: ...
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
    def _from_join_target(self, result) -> None: ...

def _is_valid_endpoint(endpoint) -> bool:
    """
    Helper for interval_range to check if start/end are valid types.
    """
def _is_type_compatible(a, b) -> bool:
    """
    Helper for interval_range to check type compat of start/end/freq.
    """
def interval_range(start: Incomplete | None = None, end: Incomplete | None = None, periods: Incomplete | None = None, freq: Incomplete | None = None, name: Hashable | None = None, closed: IntervalClosedType = 'right') -> IntervalIndex:
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
