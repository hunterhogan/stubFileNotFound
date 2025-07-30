from collections.abc import (
    Hashable,
    Sequence,
)
from datetime import (
    datetime,
    timedelta,
    tzinfo as _tzinfo,
)
from typing import (
    final,
    overload,
)

import numpy as np
from pandas import (
    DataFrame,
    Index,
    Timedelta,
    TimedeltaIndex,
    Timestamp,
)
from pandas.core.indexes.accessors import DatetimeIndexProperties
from pandas.core.indexes.datetimelike import DatetimeTimedeltaMixin
from pandas.core.series import (
    TimedeltaSeries,
    TimestampSeries,
)
from typing_extensions import Self

from pandas._typing import (
    AxesData,
    DateAndDatetimeLike,
    Dtype,
    Frequency,
    IntervalClosedType,
    TimeUnit,
    TimeZones,
)

from pandas.core.dtypes.dtypes import DatetimeTZDtype

from pandas.tseries.offsets import BaseOffset
from typing import Any

class DatetimeIndex(DatetimeTimedeltaMixin[Timestamp], DatetimeIndexProperties):
    def __init__(
        self,
        data: AxesData[Any],
        freq: Frequency = ...,
        tz: TimeZones = ...,
        ambiguous: str = ...,
        dayfirst: bool = ...,
        yearfirst: bool = ...,
        dtype: Dtype = ...,
        copy: bool = ...,
        name: Hashable = ...,
    ) -> None: ...
    def __reduce__(self) -> Any: ...
    # various ignores needed for mypy, as we do want to restrict what can be used in
    # arithmetic for these types
    @overload
    def __add__(self, other: TimedeltaSeries) -> TimestampSeries: ...
    @overload
    def __add__(
        self, other: timedelta | Timedelta | TimedeltaIndex | BaseOffset
    ) -> DatetimeIndex: ...
    @overload
    def __sub__(self, other: TimedeltaSeries) -> TimestampSeries: ...
    @overload
    def __sub__(
        self, other: timedelta | Timedelta | TimedeltaIndex | BaseOffset
    ) -> DatetimeIndex: ...
    @overload
    def __sub__(
        self, other: datetime | Timestamp | DatetimeIndex
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
        self, start_time: Any, end_time: Any, include_start: bool = True, include_end: bool = True
    ) -> Any: ...
    def to_julian_date(self) -> Index[float]: ...
    def isocalendar(self) -> DataFrame: ...
    @property
    def tzinfo(self) -> _tzinfo | None: ...
    @property
    def dtype(self) -> np.dtype | DatetimeTZDtype: ...
    def shift(self, periods: int = 1, freq: Any=None) -> Self: ...

def date_range(
    start: str | DateAndDatetimeLike | None = None,
    end: str | DateAndDatetimeLike | None = None,
    periods: int | None = None,
    freq: str | timedelta | Timedelta | BaseOffset = None,
    tz: TimeZones = None,
    normalize: bool = False,
    name: Hashable | None = None,
    inclusive: IntervalClosedType = 'both',
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
