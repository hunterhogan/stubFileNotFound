from collections.abc import Hashable, Sequence
from datetime import datetime, timedelta, tzinfo as _tzinfo
from pandas import DataFrame, Index, Timedelta, TimedeltaIndex, Timestamp
from pandas._libs.tslibs.offsets import DateOffset
from pandas._typing import (
	AxesData, DateAndDatetimeLike, Dtype, Frequency, IntervalClosedType, np_ndarray_dt, np_ndarray_td, TimeUnit, TimeZones)
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.indexes.accessors import DatetimeIndexProperties
from pandas.core.indexes.datetimelike import DatetimeTimedeltaMixin
from pandas.core.series import TimedeltaSeries, TimestampSeries
from pandas.tseries.offsets import BaseOffset
from typing import Any, final, overload
from typing_extensions import Self
import numpy as np

class DatetimeIndex(
    DatetimeTimedeltaMixin[Timestamp, np.datetime64], DatetimeIndexProperties
):
    def __new__(
        cls,
        data: AxesData[Any],
        freq: Frequency = ...,
        tz: TimeZones = ...,
        ambiguous: str = 'raise',
        dayfirst: bool = False,
        yearfirst: bool = False,
        dtype: Dtype | None = None,
        copy: bool = False,
        name: Hashable = None,
    ) -> Self: ...
    def __reduce__(self) -> Any: ...
    # various ignores needed for mypy, as we do want to restrict what can be used in
    # arithmetic for these types
    @overload  # type: ignore[override]
    def __add__(self, other: TimedeltaSeries) -> TimestampSeries: ...
    @overload
    def __add__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: timedelta | Timedelta | TimedeltaIndex | BaseOffset
    ) -> DatetimeIndex: ...
    @overload  # type: ignore[override]
    def __sub__(self, other: TimedeltaSeries) -> TimestampSeries: ...
    @overload
    def __sub__(
        self,
        other: timedelta | np.timedelta64 | np_ndarray_td | TimedeltaIndex | BaseOffset,
    ) -> DatetimeIndex: ...
    @overload
    def __sub__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: datetime | np.datetime64 | np_ndarray_dt | DatetimeIndex
    ) -> TimedeltaIndex: ...
    @final
    def to_series(self, index: Any=None, name: Hashable = None) -> TimestampSeries: ...
    def snap(self, freq: str = 'S') -> Any: ...
    def slice_indexer(self, start: Any=None, end: Any=None, step: Any=None) -> Any: ...
    def searchsorted(self, value: Any, side: str = 'left', sorter: Any=None) -> Any: ...
    @property
    def inferred_type(self) -> str: ...
    def indexer_at_time(self, time: Any, asof: bool = False) -> Any: ...
    def indexer_between_time(
        self,
        start_time: datetime | str,
        end_time: datetime | str,
        include_start: bool = True,
        include_end: bool = True,
    ) -> Any: ...
    def to_julian_date(self) -> Index[float]: ...
    def isocalendar(self) -> DataFrame: ...
    @property
    def tzinfo(self) -> _tzinfo | None: ...
    @property
    def dtype(self) -> np.dtype[Any] | DatetimeTZDtype: ...
    def shift(
        self, periods: int = 1, freq: DateOffset | Timedelta | str | None = None
    ) -> Self: ...

@overload
def date_range(
    start: str | DateAndDatetimeLike,
    end: str | DateAndDatetimeLike,
    freq: str | timedelta | Timedelta | BaseOffset | None = None,
    tz: TimeZones = None,
    normalize: bool = False,
    name: Hashable | None = None,
    inclusive: IntervalClosedType = "both",
    unit: TimeUnit | None = None,
) -> DatetimeIndex: ...
@overload
def date_range(
    start: str | DateAndDatetimeLike,
    end: str | DateAndDatetimeLike,
    periods: int,
    tz: TimeZones = None,
    normalize: bool = False,
    name: Hashable | None = None,
    inclusive: IntervalClosedType = "both",
    unit: TimeUnit | None = None,
) -> DatetimeIndex: ...
@overload
def date_range(
    start: str | DateAndDatetimeLike,
    *,
    periods: int,
    freq: str | timedelta | Timedelta | BaseOffset | None = None,
    tz: TimeZones = None,
    normalize: bool = False,
    name: Hashable | None = None,
    inclusive: IntervalClosedType = "both",
    unit: TimeUnit | None = None,
) -> DatetimeIndex: ...
@overload
def date_range(
    *,
    end: str | DateAndDatetimeLike,
    periods: int,
    freq: str | timedelta | Timedelta | BaseOffset | None = None,
    tz: TimeZones = None,
    normalize: bool = False,
    name: Hashable | None = None,
    inclusive: IntervalClosedType = "both",
    unit: TimeUnit | None = None,
) -> DatetimeIndex: ...
@overload
def bdate_range(
    start: str | DateAndDatetimeLike | None = None,
    end: str | DateAndDatetimeLike | None = None,
    periods: int | None = None,
    freq: str | timedelta | Timedelta | BaseOffset = 'B',
    tz: TimeZones = None,
    normalize: bool = True,
    name: Hashable | None = None,
    weekmask: str | None = None,
    holidays: None = None,
    inclusive: IntervalClosedType = 'both',
) -> DatetimeIndex: ...
@overload
def bdate_range(
    start: str | DateAndDatetimeLike | None = None,
    end: str | DateAndDatetimeLike | None = None,
    periods: int | None = None,
    *,
    freq: str | timedelta | Timedelta | BaseOffset,
    tz: TimeZones = None,
    normalize: bool = True,
    name: Hashable | None = None,
    weekmask: str | None = None,
    holidays: Sequence[str | DateAndDatetimeLike],
    inclusive: IntervalClosedType = 'both',
) -> DatetimeIndex: ...
