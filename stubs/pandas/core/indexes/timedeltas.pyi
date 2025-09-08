from collections.abc import Hashable, Sequence
from pandas import DateOffset, Index, Period
from pandas._libs import Timedelta, Timestamp
from pandas._libs.tslibs import BaseOffset
from pandas._typing import AxesData, np_ndarray_td, num, TimedeltaConvertibleTypes
from pandas.core.indexes.accessors import TimedeltaIndexProperties
from pandas.core.indexes.datetimelike import DatetimeTimedeltaMixin
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.period import PeriodIndex
from pandas.core.series import TimedeltaSeries
from typing import Any, final, Literal, overload
from typing_extensions import Self
import datetime as dt
import numpy as np

class TimedeltaIndex(
    DatetimeTimedeltaMixin[Timedelta, np.timedelta64], TimedeltaIndexProperties
):
    def __new__(
        cls,
        data: (
            Sequence[dt.timedelta | Timedelta | np.timedelta64 | float] | AxesData[Any]
        ) | None = None,
        freq: str | BaseOffset = ...,
        closed: object = ...,
        dtype: Literal["<m8[ns]"] = "<m8[ns]",
        copy: bool = False,
        name: str | None = None,
    ) -> Self: ...
    # various ignores needed for mypy, as we do want to restrict what can be used in
    # arithmetic for these types
    @overload  # type: ignore[override]
    def __add__(self, other: Period) -> PeriodIndex: ...
    @overload
    def __add__(self, other: DatetimeIndex) -> DatetimeIndex: ...
    @overload
    def __add__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: dt.timedelta | Timedelta | Self
    ) -> Self: ...
    def __radd__(self, other: dt.datetime | Timestamp | DatetimeIndex) -> DatetimeIndex: ...  # type: ignore[override]
    def __sub__(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: dt.timedelta | np.timedelta64 | np_ndarray_td | Self
    ) -> Self: ...
    def __mul__(self, other: num) -> Self: ...
    @overload  # type: ignore[override]
    def __truediv__(self, other: num | Sequence[float]) -> Self: ...
    @overload
    def __truediv__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: dt.timedelta | Sequence[dt.timedelta]
    ) -> Index[float]: ...
    def __rtruediv__(self, other: dt.timedelta | Sequence[dt.timedelta]) -> Index[float]: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    @overload  # type: ignore[override]
    def __floordiv__(self, other: num | Sequence[float]) -> Self: ...
    @overload
    def __floordiv__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: dt.timedelta | Sequence[dt.timedelta]
    ) -> Index[int]: ...
    def __rfloordiv__(self, other: dt.timedelta | Sequence[dt.timedelta]) -> Index[int]: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def searchsorted(self, value: Any, side: str = 'left', sorter: Any=None) -> Any: ...
    @property
    def inferred_type(self) -> str: ...
    @final
    def to_series(self, index: Any=None, name: Hashable = None) -> TimedeltaSeries: ...
    def shift(self, periods: int = 1, freq: Any=None) -> Self: ...

@overload
def timedelta_range(
    start: TimedeltaConvertibleTypes,
    end: TimedeltaConvertibleTypes,
    *,
    freq: str | DateOffset | Timedelta | dt.timedelta | None = None,
    name: Hashable | None = None,
    closed: Literal["left", "right"] | None = None,
    unit: None | str = None,
) -> TimedeltaIndex: ...
@overload
def timedelta_range(
    *,
    end: TimedeltaConvertibleTypes,
    periods: int,
    freq: str | DateOffset | Timedelta | dt.timedelta | None = None,
    name: Hashable | None = None,
    closed: Literal["left", "right"] | None = None,
    unit: None | str = None,
) -> TimedeltaIndex: ...
@overload
def timedelta_range(
    start: TimedeltaConvertibleTypes,
    *,
    periods: int,
    freq: str | DateOffset | Timedelta | dt.timedelta | None = None,
    name: Hashable | None = None,
    closed: Literal["left", "right"] | None = None,
    unit: None | str = None,
) -> TimedeltaIndex: ...
@overload
def timedelta_range(
    start: TimedeltaConvertibleTypes,
    end: TimedeltaConvertibleTypes,
    periods: int,
    *,
    name: Hashable | None = None,
    closed: Literal["left", "right"] | None = None,
    unit: None | str = None,
) -> TimedeltaIndex: ...
