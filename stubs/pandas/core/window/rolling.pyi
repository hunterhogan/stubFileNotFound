from collections.abc import (
    Callable,
    Iterator,
)
import datetime as dt
from typing import (
    Any,
    overload,
)

from pandas import (
    DataFrame,
    Index,
    Series,
)
from pandas.core.base import SelectionMixin
from pandas.core.indexers import BaseIndexer
from typing_extensions import Self

from pandas._libs.tslibs import BaseOffset
from pandas._typing import (
    AggFuncTypeBase,
    AggFuncTypeFrame,
    AggFuncTypeSeriesToFrame,
    AxisInt,
    CalculationMethod,
    IntervalClosedType,
    NDFrameT,
    QuantileInterpolation,
    WindowingEngine,
    WindowingEngineKwargs,
    WindowingRankType,
)

class BaseWindow(SelectionMixin[NDFrameT]):
    on: str | Index[Any] | None
    closed: IntervalClosedType | None
    step: int | None
    window: int | dt.timedelta | str | BaseOffset | BaseIndexer | None
    min_periods: int | None
    center: bool | None
    win_type: str | None
    axis: AxisInt
    method: CalculationMethod
    def __getitem__(self, key: Any) -> Self: ...
    def __getattr__(self, attr: str) -> Self: ...
    def __iter__(self) -> Iterator[NDFrameT]: ...
    @overload
    def aggregate(
        self: BaseWindow[Series], func: AggFuncTypeBase, *args: Any, **kwargs: Any
    ) -> Series: ...
    @overload
    def aggregate(
        self: BaseWindow[Series],
        func: AggFuncTypeSeriesToFrame,
        *args: Any,
        **kwargs: Any,
    ) -> DataFrame: ...
    @overload
    def aggregate(
        self: BaseWindow[DataFrame],
        func: AggFuncTypeFrame,
        *args: Any,
        **kwargs: Any,
    ) -> DataFrame: ...
    agg = aggregate

class BaseWindowGroupby(BaseWindow[NDFrameT]): ...

class Window(BaseWindow[NDFrameT]):
    def sum(self, numeric_only: bool = False, **kwargs: Any) -> NDFrameT: ...
    def mean(self, numeric_only: bool = False, **kwargs: Any) -> NDFrameT: ...
    def var(
        self, ddof: int = 1, numeric_only: bool = False, **kwargs: Any
    ) -> NDFrameT: ...
    def std(
        self, ddof: int = 1, numeric_only: bool = False, **kwargs: Any
    ) -> NDFrameT: ...

class RollingAndExpandingMixin(BaseWindow[NDFrameT]):
    def count(self, numeric_only: bool = False) -> NDFrameT: ...
    def apply(
        self,
        func: Callable[..., Any],
        raw: bool = False,
        engine: WindowingEngine = None,
        engine_kwargs: WindowingEngineKwargs = None,
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> NDFrameT: ...
    def sum(
        self,
        numeric_only: bool = False,
        engine: WindowingEngine = None,
        engine_kwargs: WindowingEngineKwargs = None,
    ) -> NDFrameT: ...
    def max(
        self,
        numeric_only: bool = False,
        engine: WindowingEngine = None,
        engine_kwargs: WindowingEngineKwargs = None,
    ) -> NDFrameT: ...
    def min(
        self,
        numeric_only: bool = False,
        engine: WindowingEngine = None,
        engine_kwargs: WindowingEngineKwargs = None,
    ) -> NDFrameT: ...
    def mean(
        self,
        numeric_only: bool = False,
        engine: WindowingEngine = None,
        engine_kwargs: WindowingEngineKwargs = None,
    ) -> NDFrameT: ...
    def median(
        self,
        numeric_only: bool = False,
        engine: WindowingEngine = None,
        engine_kwargs: WindowingEngineKwargs = None,
    ) -> NDFrameT: ...
    def std(
        self,
        ddof: int = 1,
        numeric_only: bool = False,
        engine: WindowingEngine = None,
        engine_kwargs: WindowingEngineKwargs = None,
    ) -> NDFrameT: ...
    def var(
        self,
        ddof: int = 1,
        numeric_only: bool = False,
        engine: WindowingEngine = None,
        engine_kwargs: WindowingEngineKwargs = None,
    ) -> NDFrameT: ...
    def skew(self, numeric_only: bool = False) -> NDFrameT: ...
    def sem(self, ddof: int = 1, numeric_only: bool = False) -> NDFrameT: ...
    def kurt(self, numeric_only: bool = False) -> NDFrameT: ...
    def quantile(
        self,
        q: float,
        interpolation: QuantileInterpolation = 'linear',
        numeric_only: bool = False,
    ) -> NDFrameT: ...
    def rank(
        self,
        method: WindowingRankType = 'average',
        ascending: bool = True,
        pct: bool = False,
        numeric_only: bool = False,
    ) -> NDFrameT: ...
    def cov(
        self,
        other: DataFrame | Series | None = None,
        pairwise: bool | None = None,
        ddof: int = 1,
        numeric_only: bool = False,
    ) -> NDFrameT: ...
    def corr(
        self,
        other: DataFrame | Series | None = None,
        pairwise: bool | None = None,
        ddof: int = 1,
        numeric_only: bool = False,
    ) -> NDFrameT: ...

class Rolling(RollingAndExpandingMixin[NDFrameT]): ...
class RollingGroupby(BaseWindowGroupby[NDFrameT], Rolling[NDFrameT]): ...
