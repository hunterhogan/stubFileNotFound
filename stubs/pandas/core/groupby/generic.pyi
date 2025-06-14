from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Sequence,
)
from typing import (
    Any,
    Generic,
    Literal,
    NamedTuple,
    TypeVar,
    final,
    overload,
)

from matplotlib.axes import Axes as PlotAxes
import numpy as np
from pandas.core.frame import DataFrame
from pandas.core.groupby.groupby import (
    GroupBy,
    GroupByPlot,
)
from pandas.core.series import Series
from typing_extensions import (
    Self,
    TypeAlias,
)

from pandas._libs.tslibs.timestamps import Timestamp
from pandas._typing import (
    S1,
    AggFuncTypeBase,
    AggFuncTypeFrame,
    ByT,
    CorrelationMethod,
    Dtype,
    IndexLabel,
    Level,
    ListLike,
    NsmallestNlargestKeep,
    Scalar,
    TakeIndexer,
    WindowingEngine,
    WindowingEngineKwargs,
)

AggScalar: TypeAlias = str | Callable[..., Any]

class NamedAgg(NamedTuple):
    column: str
    aggfunc: AggScalar

class SeriesGroupBy(GroupBy[Series[S1]], Generic[S1, ByT]):
    @overload
    def aggregate(
        self,
        func: list[AggFuncTypeBase],
        *args: Any,
        engine: WindowingEngine = ...,
        engine_kwargs: WindowingEngineKwargs = ...,
        **kwargs: Any,
    ) -> DataFrame: ...
    @overload
    def aggregate(
        self,
        func: AggFuncTypeBase | None = ...,
        *args: Any,
        engine: WindowingEngine = ...,
        engine_kwargs: WindowingEngineKwargs = ...,
        **kwargs: Any,
    ) -> Series[Any]: ...
    agg = aggregate
    def transform(
        self,
        func: Callable[..., Any] | str,
        *args: Any,
        engine: WindowingEngine = ...,
        engine_kwargs: WindowingEngineKwargs = ...,
        **kwargs: Any,
    ) -> Series[Any]: ...
    def filter(
        self, func: Callable[..., Any] | str, dropna: bool = ..., *args: Any, **kwargs: Any
    ) -> Series[Any]: ...
    def nunique(self, dropna: bool = ...) -> Series[int]: ...
    # describe delegates to super() method but here it has keyword-only parameters
    def describe(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        *,
        percentiles: Iterable[float] | None = ...,
        include: Literal["all"] | list[Dtype] | None = ...,
        exclude: list[Dtype] | None = ...,
    ) -> DataFrame: ...
    @overload
    def value_counts(
        self,
        normalize: Literal[False] = False,
        sort: bool = ...,
        ascending: bool = ...,
        bins: int | Sequence[int] | None = ...,
        dropna: bool = ...,
    ) -> Series[int]: ...
    @overload
    def value_counts(
        self,
        normalize: Literal[True],
        sort: bool = ...,
        ascending: bool = ...,
        bins: int | Sequence[int] | None = ...,
        dropna: bool = ...,
    ) -> Series[float]: ...
    def take(
        self,
        indices: TakeIndexer,
        **kwargs: Any,
    ) -> Series[S1]: ...
    def skew(
        self,
        skipna: bool = ...,
        numeric_only: bool = ...,
        **kwargs: Any,
    ) -> Series[Any]: ...
    @property
    def plot(self) -> GroupByPlot[Self]: ...
    def nlargest(
        self, n: int = ..., keep: NsmallestNlargestKeep = ...
    ) -> Series[S1]: ...
    def nsmallest(
        self, n: int = ..., keep: NsmallestNlargestKeep = ...
    ) -> Series[S1]: ...
    def idxmin(self, skipna: bool = ...) -> Series[Any]: ...
    def idxmax(self, skipna: bool = ...) -> Series[Any]: ...
    def corr(
        self,
        other: Series[Any],
        method: CorrelationMethod = ...,
        min_periods: int | None = ...,
    ) -> Series[Any]: ...
    def cov(
        self, other: Series[Any], min_periods: int | None = ..., ddof: int | None = ...
    ) -> Series[Any]: ...
    @property
    def is_monotonic_increasing(self) -> Series[bool]: ...
    @property
    def is_monotonic_decreasing(self) -> Series[bool]: ...
    def hist(
        self,
        by: IndexLabel | None = ...,
        ax: PlotAxes | None = ...,
        grid: bool = ...,
        xlabelsize: float | str | None = ...,
        xrot: float | None = ...,
        ylabelsize: float | str | None = ...,
        yrot: float | None = ...,
        figsize: tuple[float, float] | None = ...,
        bins: int | Sequence[int] = ...,
        backend: str | None = ...,
        legend: bool = ...,
        **kwargs: Any,
    ) -> Series[Any]: ...  # Series[Axes] but this is not allowed
    @property
    def dtype(self) -> Series[Any]: ...
    def unique(self) -> Series[Any]: ...
    # Overrides that provide more precise return types over the GroupBy class
    @final  # type: ignore[misc]
    def __iter__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
    ) -> Iterator[tuple[ByT, Series[S1]]]: ...

_TT = TypeVar("_TT", bound=Literal[True, False])

class DataFrameGroupBy(GroupBy[DataFrame], Generic[ByT, _TT]):
    # error: Overload 3 for "apply" will never be used because its parameters overlap overload 1
    @overload  # type: ignore[override]
    def apply(
        self,
        func: Callable[[DataFrame], Scalar | list[Any] | dict[Any, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Series[Any]: ...
    @overload
    def apply(
        self,
        func: Callable[[DataFrame], Series[Any] | DataFrame],
        *args: Any,
        **kwargs: Any,
    ) -> DataFrame: ...
    @overload
    def apply(  # pyright: ignore[reportOverlappingOverload]
        self,
        func: Callable[[Iterable[Any]], float],
        *args: Any,
        **kwargs: Any,
    ) -> DataFrame: ...
    # error: overload 1 overlaps overload 2 because of different return types
    @overload
    def aggregate(self, func: Literal["size"]) -> Series[Any]: ...  # type: ignore[overload-overlap]
    @overload
    def aggregate(
        self,
        func: AggFuncTypeFrame | None = ...,
        *args: Any,
        engine: WindowingEngine = ...,
        engine_kwargs: WindowingEngineKwargs = ...,
        **kwargs: Any,
    ) -> DataFrame: ...
    agg = aggregate
    def transform(
        self,
        func: Callable[..., Any] | str,
        *args: Any,
        engine: WindowingEngine = ...,
        engine_kwargs: WindowingEngineKwargs = ...,
        **kwargs: Any,
    ) -> DataFrame: ...
    def filter(
        self, func: Callable[..., Any], dropna: bool = ..., *args: Any, **kwargs: Any
    ) -> DataFrame: ...
    @overload
    def __getitem__(self, key: Scalar) -> SeriesGroupBy[Any, ByT]: ...  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
    @overload
    def __getitem__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, key: Iterable[Hashable]
    ) -> DataFrameGroupBy[ByT, bool]: ...
    def nunique(self, dropna: bool = ...) -> DataFrame: ...
    def idxmax(
        self,
        skipna: bool = ...,
        numeric_only: bool = ...,
    ) -> DataFrame: ...
    def idxmin(
        self,
        skipna: bool = ...,
        numeric_only: bool = ...,
    ) -> DataFrame: ...
    @overload
    def boxplot(
        self,
        subplots: Literal[True] = True,
        column: IndexLabel | None = ...,
        fontsize: float | str | None = ...,
        rot: float = ...,
        grid: bool = ...,
        ax: PlotAxes | None = ...,
        figsize: tuple[float, float] | None = ...,
        layout: tuple[int, int] | None = ...,
        sharex: bool = ...,
        sharey: bool = ...,
        backend: str | None = ...,
        **kwargs: Any,
    ) -> Series[Any]: ...  # Series[PlotAxes] but this is not allowed
    @overload
    def boxplot(
        self,
        subplots: Literal[False],
        column: IndexLabel | None = ...,
        fontsize: float | str | None = ...,
        rot: float = ...,
        grid: bool = ...,
        ax: PlotAxes | None = ...,
        figsize: tuple[float, float] | None = ...,
        layout: tuple[int, int] | None = ...,
        sharex: bool = ...,
        sharey: bool = ...,
        backend: str | None = ...,
        **kwargs: Any,
    ) -> PlotAxes: ...
    @overload
    def boxplot(
        self,
        subplots: bool,
        column: IndexLabel | None = ...,
        fontsize: float | str | None = ...,
        rot: float = ...,
        grid: bool = ...,
        ax: PlotAxes | None = ...,
        figsize: tuple[float, float] | None = ...,
        layout: tuple[int, int] | None = ...,
        sharex: bool = ...,
        sharey: bool = ...,
        backend: str | None = ...,
        **kwargs: Any,
    ) -> PlotAxes | Series[Any]: ...  # Series[PlotAxes]
    @overload
    def value_counts(
        self: DataFrameGroupBy[ByT, Literal[True]],
        subset: ListLike | None = ...,
        normalize: Literal[False] = False,
        sort: bool = ...,
        ascending: bool = ...,
        dropna: bool = ...,
    ) -> Series[int]: ...
    @overload
    def value_counts(
        self: DataFrameGroupBy[ByT, Literal[True]],
        subset: ListLike | None,
        normalize: Literal[True],
        sort: bool = ...,
        ascending: bool = ...,
        dropna: bool = ...,
    ) -> Series[float]: ...
    @overload
    def value_counts(
        self: DataFrameGroupBy[ByT, Literal[False]],
        subset: ListLike | None = ...,
        normalize: Literal[False] = False,
        sort: bool = ...,
        ascending: bool = ...,
        dropna: bool = ...,
    ) -> DataFrame: ...
    @overload
    def value_counts(
        self: DataFrameGroupBy[ByT, Literal[False]],
        subset: ListLike | None,
        normalize: Literal[True],
        sort: bool = ...,
        ascending: bool = ...,
        dropna: bool = ...,
    ) -> DataFrame: ...
    def take(self, indices: TakeIndexer, **kwargs: Any) -> DataFrame: ...
    @overload
    def skew(
        self,
        skipna: bool = ...,
        numeric_only: bool = ...,
        *,
        level: Level,
        **kwargs: Any,
    ) -> DataFrame: ...
    @overload
    def skew(
        self,
        skipna: bool = ...,
        numeric_only: bool = ...,
        *,
        level: None = None,
        **kwargs: Any,
    ) -> Series[Any]: ...
    @property
    def plot(self) -> GroupByPlot[Self]: ...
    def corr(
        self,
        method: str | Callable[[np.ndarray[Any, Any], np.ndarray[Any, Any]], float] = ...,
        min_periods: int = ...,
        numeric_only: bool = ...,
    ) -> DataFrame: ...
    def cov(
        self,
        min_periods: int | None = ...,
        ddof: int | None = ...,
        numeric_only: bool = ...,
    ) -> DataFrame: ...
    def hist(
        self,
        column: IndexLabel | None = ...,
        by: IndexLabel | None = ...,
        grid: bool = ...,
        xlabelsize: float | str | None = ...,
        xrot: float | None = ...,
        ylabelsize: float | str | None = ...,
        yrot: float | None = ...,
        ax: PlotAxes | None = ...,
        sharex: bool = ...,
        sharey: bool = ...,
        figsize: tuple[float, float] | None = ...,
        layout: tuple[int, int] | None = ...,
        bins: int | Sequence[int] = ...,
        backend: str | None = ...,
        legend: bool = ...,
        **kwargs: Any,
    ) -> Series[Any]: ...  # Series[Axes] but this is not allowed
    @property
    def dtypes(self) -> Series[Any]: ...
    def __getattr__(self, name: str) -> SeriesGroupBy[Any, ByT]: ...
    # Overrides that provide more precise return types over the GroupBy class
    @final  # type: ignore[misc]
    def __iter__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
    ) -> Iterator[tuple[ByT, DataFrame]]: ...
    @overload
    def size(self: DataFrameGroupBy[ByT, Literal[True]]) -> Series[int]: ...
    @overload
    def size(self: DataFrameGroupBy[ByT, Literal[False]]) -> DataFrame: ...
    @overload
    def size(self: DataFrameGroupBy[Timestamp, Literal[True]]) -> Series[int]: ...
    @overload
    def size(self: DataFrameGroupBy[Timestamp, Literal[False]]) -> DataFrame: ...
