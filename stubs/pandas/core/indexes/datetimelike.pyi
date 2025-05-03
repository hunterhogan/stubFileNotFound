import abc
import numpy as np
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime
from pandas import CategoricalIndex as CategoricalIndex
from pandas._config import using_copy_on_write as using_copy_on_write
from pandas._libs import NaT as NaT, Timedelta as Timedelta, lib as lib
from pandas._libs.tslibs import BaseOffset as BaseOffset, Resolution as Resolution, Tick as Tick, parsing as parsing, to_offset as to_offset
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr as freq_to_period_freqstr
from pandas._typing import Axis as Axis, Self as Self, npt as npt
from pandas.core.arrays import DatetimeArray as DatetimeArray, ExtensionArray as ExtensionArray, PeriodArray as PeriodArray, TimedeltaArray as TimedeltaArray
from pandas.core.arrays.datetimelike import DatetimeLikeArrayMixin as DatetimeLikeArrayMixin
from pandas.core.dtypes.common import is_integer as is_integer, is_list_like as is_list_like
from pandas.core.dtypes.concat import concat_compat as concat_compat
from pandas.core.dtypes.dtypes import CategoricalDtype as CategoricalDtype
from pandas.core.indexes.base import Index as Index, _index_shared_docs as _index_shared_docs
from pandas.core.indexes.extension import NDArrayBackedExtensionIndex as NDArrayBackedExtensionIndex
from pandas.core.indexes.range import RangeIndex as RangeIndex
from pandas.core.tools.timedeltas import to_timedelta as to_timedelta
from pandas.errors import InvalidIndexError as InvalidIndexError, NullFrequencyError as NullFrequencyError
from pandas.util._decorators import Appender as Appender, cache_readonly as cache_readonly, doc as doc
from pandas.util._exceptions import find_stack_level as find_stack_level
from typing import Any, Callable

_index_doc_kwargs: Incomplete

class DatetimeIndexOpsMixin(NDArrayBackedExtensionIndex, ABC, metaclass=abc.ABCMeta):
    """
    Common ops mixin to support a unified interface datetimelike Index.
    """
    _can_hold_strings: bool
    _data: DatetimeArray | TimedeltaArray | PeriodArray
    def mean(self, *, skipna: bool = True, axis: int | None = 0): ...
    @property
    def freq(self) -> BaseOffset | None: ...
    @freq.setter
    def freq(self, value) -> None: ...
    @property
    def asi8(self) -> npt.NDArray[np.int64]: ...
    @property
    def freqstr(self) -> str: ...
    @abstractmethod
    def _resolution_obj(self) -> Resolution: ...
    def resolution(self) -> str: ...
    def hasnans(self) -> bool: ...
    def equals(self, other: Any) -> bool:
        """
        Determines if two Index objects contain the same elements.
        """
    def __contains__(self, key: Any) -> bool: ...
    def _convert_tolerance(self, tolerance, target): ...
    _default_na_rep: str
    def format(self, name: bool = False, formatter: Callable | None = None, na_rep: str = 'NaT', date_format: str | None = None) -> list[str]:
        """
        Render a string representation of the Index.
        """
    def _format_with_header(self, *, header: list[str], na_rep: str, date_format: str | None = None) -> list[str]: ...
    @property
    def _formatter_func(self): ...
    def _format_attrs(self):
        """
        Return a list of tuples of the (attr,formatted_value).
        """
    def _summary(self, name: Incomplete | None = None) -> str: ...
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
    def shift(self, periods: int = 1, freq: Incomplete | None = None) -> Self:
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
    def _maybe_cast_listlike_indexer(self, keyarr): ...

class DatetimeTimedeltaMixin(DatetimeIndexOpsMixin, ABC, metaclass=abc.ABCMeta):
    """
    Mixin class for methods shared by DatetimeIndex and TimedeltaIndex,
    but not PeriodIndex
    """
    _data: DatetimeArray | TimedeltaArray
    _comparables: Incomplete
    _attributes: Incomplete
    _is_monotonic_increasing: Incomplete
    _is_monotonic_decreasing: Incomplete
    _is_unique: Incomplete
    @property
    def unit(self) -> str: ...
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
    @property
    def values(self) -> np.ndarray: ...
    def shift(self, periods: int = 1, freq: Incomplete | None = None) -> Self: ...
    def inferred_freq(self) -> str | None: ...
    def _as_range_index(self) -> RangeIndex: ...
    def _can_range_setop(self, other) -> bool: ...
    def _wrap_range_setop(self, other, res_i8) -> Self: ...
    def _range_intersect(self, other, sort) -> Self: ...
    def _range_union(self, other, sort) -> Self: ...
    def _intersection(self, other: Index, sort: bool = False) -> Index:
        """
        intersection specialized to the case with matching dtypes and both non-empty.
        """
    def _fast_intersect(self, other, sort): ...
    def _can_fast_intersect(self, other: Self) -> bool: ...
    def _can_fast_union(self, other: Self) -> bool: ...
    def _fast_union(self, other: Self, sort: Incomplete | None = None) -> Self: ...
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
    def delete(self, loc) -> Self: ...
    def insert(self, loc: int, item): ...
    def take(self, indices, axis: Axis = 0, allow_fill: bool = True, fill_value: Incomplete | None = None, **kwargs) -> Self: ...
