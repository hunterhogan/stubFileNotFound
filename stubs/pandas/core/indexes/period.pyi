from collections.abc import Hashable
import datetime
from typing import overload

import numpy as np
import pandas as pd
from pandas import Index
from pandas.core.indexes.accessors import PeriodIndexFieldOps
from pandas.core.indexes.datetimelike import DatetimeIndexOpsMixin
from pandas.core.indexes.timedeltas import TimedeltaIndex
from typing_extensions import Self

from pandas._libs.tslibs import (
    BaseOffset,
    NaTType,
    Period,
)
from pandas._libs.tslibs.period import _PeriodAddSub
from typing import Any

class PeriodIndex(DatetimeIndexOpsMixin[pd.Period], PeriodIndexFieldOps):
    def __new__(
        cls,
        data: Any=None,
        ordinal: Any=None,
        freq: Any=None,
        tz: Any=...,
        dtype: Any=None,
        copy: bool = False,
        name: Hashable = None,
        **fields,
    ): ...
    @property
    def values(self): ...
    def __contains__(self, key: Any) -> bool: ...
    @overload
    def __sub__(self, other: Period) -> Index[Any]: ...
    @overload
    def __sub__(self, other: Self) -> Index[Any]: ...
    @overload
    def __sub__(self, other: _PeriodAddSub) -> Self: ...
    @overload
    def __sub__(self, other: NaTType) -> NaTType: ...
    @overload
    def __sub__(self, other: TimedeltaIndex | pd.Timedelta) -> Self: ...
    @overload  # type: ignore[override]
    def __rsub__(self, other: Period) -> Index[Any]: ...
    @overload
    def __rsub__(self, other: Self) -> Index[Any]: ...
    @overload
    def __rsub__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: NaTType
    ) -> NaTType: ...
    def __array__(self, dtype: Any=None) -> np.ndarray[Any, Any]: ...
    def __array_wrap__(self, result: Any, context: Any=None): ...
    def asof_locs(self, where: Any, mask: Any): ...
    def astype(self, dtype: Any, copy: bool = True): ...
    def searchsorted(self, value: Any, side: str = 'left', sorter: Any=None): ...
    @property
    def is_full(self) -> bool: ...
    @property
    def inferred_type(self) -> str: ...
    def get_indexer(self, target: Any, method: Any=None, limit: Any=None, tolerance: Any=None): ...
    def get_indexer_non_unique(self, target: Any): ...
    def insert(self, loc: Any, item: Any): ...
    def join(
        self,
        other: Any,
        *,
        how: str = 'left',
        level=None,
        return_indexers: bool = False,
        sort: bool = False,
    ): ...
    def difference(self, other: Any, sort: Any=None): ...
    def memory_usage(self, deep: bool = False): ...
    @property
    def freqstr(self) -> str: ...
    def shift(self, periods: int = 1, freq: Any=None) -> Self: ...

def period_range(
    start: (
        str | datetime.datetime | datetime.date | pd.Timestamp | pd.Period | None
    ) = None,
    end: (
        str | datetime.datetime | datetime.date | pd.Timestamp | pd.Period | None
    ) = None,
    periods: int | None = None,
    freq: str | BaseOffset | None = None,
    name: Hashable | None = None,
) -> PeriodIndex: ...
