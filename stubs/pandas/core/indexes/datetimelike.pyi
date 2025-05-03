import _abc
import abc
import np
import npt
import pandas._libs.lib as lib
import pandas._libs.tslibs.parsing as parsing
import pandas.compat.numpy.function as nv
import pandas.core.common as com
import pandas.core.indexes.base as ibase
import pandas.core.indexes.extension
from _typeshed import Incomplete
from pandas._config import using_copy_on_write as using_copy_on_write
from pandas._libs.lib import is_integer as is_integer, is_list_like as is_list_like
from pandas._libs.properties import cache_readonly as cache_readonly
from pandas._libs.tslibs.dtypes import Resolution as Resolution, freq_to_period_freqstr as freq_to_period_freqstr
from pandas._libs.tslibs.nattype import NaT as NaT
from pandas._libs.tslibs.offsets import BaseOffset as BaseOffset, Tick as Tick, to_offset as to_offset
from pandas._libs.tslibs.timedeltas import Timedelta as Timedelta
from pandas.core.arrays.base import ExtensionArray as ExtensionArray
from pandas.core.arrays.datetimelike import DatetimeLikeArrayMixin as DatetimeLikeArrayMixin
from pandas.core.arrays.datetimes import DatetimeArray as DatetimeArray
from pandas.core.arrays.period import PeriodArray as PeriodArray
from pandas.core.arrays.timedeltas import TimedeltaArray as TimedeltaArray
from pandas.core.dtypes.concat import concat_compat as concat_compat
from pandas.core.dtypes.dtypes import CategoricalDtype as CategoricalDtype
from pandas.core.indexes.base import Index as Index
from pandas.core.indexes.extension import NDArrayBackedExtensionIndex as NDArrayBackedExtensionIndex
from pandas.core.indexes.range import RangeIndex as RangeIndex
from pandas.core.tools.timedeltas import to_timedelta as to_timedelta
from pandas.errors import InvalidIndexError as InvalidIndexError, NullFrequencyError as NullFrequencyError
from pandas.util._decorators import Appender as Appender, doc as doc
from pandas.util._exceptions import find_stack_level as find_stack_level
from typing import Any, Callable, ClassVar

TYPE_CHECKING: bool
_index_shared_docs: dict
_index_doc_kwargs: dict

class DatetimeIndexOpsMixin(pandas.core.indexes.extension.NDArrayBackedExtensionIndex, abc.ABC):
    _can_hold_strings: ClassVar[bool] = ...
    _default_na_rep: ClassVar[str] = ...
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    freq: Incomplete
    _resolution_obj: Incomplete
    resolution: Incomplete
    hasnans: Incomplete
    def mean(self, *, skipna: bool = ..., axis: int | None = ...):
        """
        Return the mean value of the Array.

        Parameters
        ----------
        skipna : bool, default True
            Whether to ignore any NaT elements.
        axis : int, optional, default 0

        Returns
        -------
        scalar
            Timestamp or Timedelta.

        See Also
        --------
        numpy.ndarray.mean : Returns the average of array elements along a given axis.
        Series.mean : Return the mean value in a Series.

        Notes
        -----
        mean is only defined for Datetime and Timedelta dtypes, not for Period.

        Examples
        --------
        For :class:`pandas.DatetimeIndex`:

        >>> idx = pd.date_range('2001-01-01 00:00', periods=3)
        >>> idx
        DatetimeIndex(['2001-01-01', '2001-01-02', '2001-01-03'],
                      dtype='datetime64[ns]', freq='D')
        >>> idx.mean()
        Timestamp('2001-01-02 00:00:00')

        For :class:`pandas.TimedeltaIndex`:

        >>> tdelta_idx = pd.to_timedelta([1, 2, 3], unit='D')
        >>> tdelta_idx
        TimedeltaIndex(['1 days', '2 days', '3 days'],
                        dtype='timedelta64[ns]', freq=None)
        >>> tdelta_idx.mean()
        Timedelta('2 days 00:00:00')
        """
    def equals(self, other: Any) -> bool:
        """
        Determines if two Index objects contain the same elements.
        """
    def __contains__(self, key: Any) -> bool:
        """
        Return a boolean indicating whether the provided key is in the index.

        Parameters
        ----------
        key : label
            The key to check if it is present in the index.

        Returns
        -------
        bool
            Whether the key search is in the index.

        Raises
        ------
        TypeError
            If the key is not hashable.

        See Also
        --------
        Index.isin : Returns an ndarray of boolean dtype indicating whether the
            list-like key is in the index.

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3, 4])
        >>> idx
        Index([1, 2, 3, 4], dtype='int64')

        >>> 2 in idx
        True
        >>> 6 in idx
        False
        """
    def _convert_tolerance(self, tolerance, target): ...
    def format(self, name: bool = ..., formatter: Callable | None, na_rep: str = ..., date_format: str | None) -> list[str]:
        """
        Render a string representation of the Index.
        """
    def _format_with_header(self, *, header: list[str], na_rep: str, date_format: str | None) -> list[str]: ...
    def _format_attrs(self):
        """
        Return a list of tuples of the (attr,formatted_value).
        """
    def _summary(self, name) -> str:
        """
        Return a summarized representation.

        Parameters
        ----------
        name : str
            name to use in the summary representation

        Returns
        -------
        String with a summarized representation of the index
        """
    def _can_partial_date_slice(self, reso: Resolution) -> bool: ...
    def _parsed_string_to_bounds(self, reso: Resolution, parsed): ...
    def _parse_with_reso(self, label: str): ...
    def _get_string_slice(self, key: str): ...
    def _partial_date_slice(self, reso: Resolution, parsed: datetime) -> slice | npt.NDArray[np.intp]:
        """
        Parameters
        ----------
        reso : Resolution
        parsed : datetime

        Returns
        -------
        slice or ndarray[intp]
        """
    def _maybe_cast_slice_bound(self, label, side: str):
        """
        If label is a string, cast it to scalar type according to resolution.

        Parameters
        ----------
        label : object
        side : {'left', 'right'}

        Returns
        -------
        label : object

        Notes
        -----
        Value of `side` parameter should be validated in caller.
        """
    def shift(self, periods: int = ..., freq) -> Self:
        """
        Shift index by desired number of time frequency increments.

        This method is for shifting the values of datetime-like indexes
        by a specified time increment a given number of times.

        Parameters
        ----------
        periods : int, default 1
            Number of periods (or increments) to shift by,
            can be positive or negative.
        freq : pandas.DateOffset, pandas.Timedelta or string, optional
            Frequency increment to shift by.
            If None, the index is shifted by its own `freq` attribute.
            Offset aliases are valid strings, e.g., 'D', 'W', 'M' etc.

        Returns
        -------
        pandas.DatetimeIndex
            Shifted index.

        See Also
        --------
        Index.shift : Shift values of Index.
        PeriodIndex.shift : Shift values of PeriodIndex.
        """
    def _maybe_cast_listlike_indexer(self, keyarr):
        """
        Analogue to maybe_cast_indexer for get_indexer instead of get_loc.
        """
    @property
    def asi8(self): ...
    @property
    def freqstr(self): ...
    @property
    def _formatter_func(self): ...

class DatetimeTimedeltaMixin(DatetimeIndexOpsMixin):
    _comparables: ClassVar[list] = ...
    _attributes: ClassVar[list] = ...
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    _is_unique: Incomplete
    inferred_freq: Incomplete
    _as_range_index: Incomplete
    def as_unit(self, unit: str) -> Self:
        """
        Convert to a dtype with the given unit resolution.

        Parameters
        ----------
        unit : {'s', 'ms', 'us', 'ns'}

        Returns
        -------
        same type as self

        Examples
        --------
        For :class:`pandas.DatetimeIndex`:

        >>> idx = pd.DatetimeIndex(['2020-01-02 01:02:03.004005006'])
        >>> idx
        DatetimeIndex(['2020-01-02 01:02:03.004005006'],
                      dtype='datetime64[ns]', freq=None)
        >>> idx.as_unit('s')
        DatetimeIndex(['2020-01-02 01:02:03'], dtype='datetime64[s]', freq=None)

        For :class:`pandas.TimedeltaIndex`:

        >>> tdelta_idx = pd.to_timedelta(['1 day 3 min 2 us 42 ns'])
        >>> tdelta_idx
        TimedeltaIndex(['1 days 00:03:00.000002042'],
                        dtype='timedelta64[ns]', freq=None)
        >>> tdelta_idx.as_unit('s')
        TimedeltaIndex(['1 days 00:03:00'], dtype='timedelta64[s]', freq=None)
        """
    def _with_freq(self, freq): ...
    def shift(self, periods: int = ..., freq) -> Self:
        """
        Shift index by desired number of time frequency increments.

        This method is for shifting the values of datetime-like indexes
        by a specified time increment a given number of times.

        Parameters
        ----------
        periods : int, default 1
            Number of periods (or increments) to shift by,
            can be positive or negative.
        freq : pandas.DateOffset, pandas.Timedelta or string, optional
            Frequency increment to shift by.
            If None, the index is shifted by its own `freq` attribute.
            Offset aliases are valid strings, e.g., 'D', 'W', 'M' etc.

        Returns
        -------
        pandas.DatetimeIndex
            Shifted index.

        See Also
        --------
        Index.shift : Shift values of Index.
        PeriodIndex.shift : Shift values of PeriodIndex.
        """
    def _can_range_setop(self, other) -> bool: ...
    def _wrap_range_setop(self, other, res_i8) -> Self: ...
    def _range_intersect(self, other, sort) -> Self: ...
    def _range_union(self, other, sort) -> Self: ...
    def _intersection(self, other: Index, sort: bool = ...) -> Index:
        """
        intersection specialized to the case with matching dtypes and both non-empty.
        """
    def _fast_intersect(self, other, sort): ...
    def _can_fast_intersect(self, other: Self) -> bool: ...
    def _can_fast_union(self, other: Self) -> bool: ...
    def _fast_union(self, other: Self, sort) -> Self: ...
    def _union(self, other, sort): ...
    def _get_join_freq(self, other):
        """
        Get the freq to attach to the result of a join operation.
        """
    def _wrap_joined_index(self, joined, other, lidx: npt.NDArray[np.intp], ridx: npt.NDArray[np.intp]): ...
    def _get_engine_target(self) -> np.ndarray: ...
    def _from_join_target(self, result: np.ndarray): ...
    def _get_delete_freq(self, loc: int | slice | Sequence[int]):
        """
        Find the `freq` for self.delete(loc).
        """
    def _get_insert_freq(self, loc: int, item):
        """
        Find the `freq` for self.insert(loc, item).
        """
    def delete(self, loc) -> Self:
        """
        Make new Index with passed location(-s) deleted.

        Parameters
        ----------
        loc : int or list of int
            Location of item(-s) which will be deleted.
            Use a list of locations to delete more than one value at the same time.

        Returns
        -------
        Index
            Will be same type as self, except for RangeIndex.

        See Also
        --------
        numpy.delete : Delete any rows and column from NumPy array (ndarray).

        Examples
        --------
        >>> idx = pd.Index(['a', 'b', 'c'])
        >>> idx.delete(1)
        Index(['a', 'c'], dtype='object')

        >>> idx = pd.Index(['a', 'b', 'c'])
        >>> idx.delete([0, 2])
        Index(['b'], dtype='object')
        """
    def insert(self, loc: int, item):
        """
        Make new Index inserting new item at location.

        Follows Python numpy.insert semantics for negative values.

        Parameters
        ----------
        loc : int
        item : object

        Returns
        -------
        Index

        Examples
        --------
        >>> idx = pd.Index(['a', 'b', 'c'])
        >>> idx.insert(1, 'x')
        Index(['a', 'x', 'b', 'c'], dtype='object')
        """
    def take(self, indices, axis: Axis = ..., allow_fill: bool = ..., fill_value, **kwargs) -> Self:
        """
        Return a new Index of the values selected by the indices.

        For internal compatibility with numpy arrays.

        Parameters
        ----------
        indices : array-like
            Indices to be taken.
        axis : int, optional
            The axis over which to select values, always 0.
        allow_fill : bool, default True
        fill_value : scalar, default None
            If allow_fill=True and fill_value is not None, indices specified by
            -1 are regarded as NA. If Index doesn't hold NA, raise ValueError.

        Returns
        -------
        Index
            An index formed of elements at the given indices. Will be the same
            type as self, except for RangeIndex.

        See Also
        --------
        numpy.ndarray.take: Return an array formed from the
            elements of a at the given indices.

        Examples
        --------
        >>> idx = pd.Index(['a', 'b', 'c'])
        >>> idx.take([2, 2, 1, 2])
        Index(['c', 'c', 'b', 'c'], dtype='object')
        """
    @property
    def _is_monotonic_increasing(self): ...
    @property
    def _is_monotonic_decreasing(self): ...
    @property
    def unit(self): ...
    @property
    def values(self): ...
