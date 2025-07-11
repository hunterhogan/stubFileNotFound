from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Sequence,
)
from typing import (
    Any,
    Concatenate,
    Generic,
    Literal,
    NamedTuple,
    Protocol,
    TypeVar,
    final,
    overload,
)

from matplotlib.axes import Axes as PlotAxes
import numpy as np
from pandas.core.frame import DataFrame
from pandas.core.groupby.base import TransformReductionListType
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
    S2,
    S3,
    AggFuncTypeBase,
    AggFuncTypeFrame,
    ByT,
    CorrelationMethod,
    Dtype,
    IndexLabel,
    Level,
    ListLike,
    NsmallestNlargestKeep,
    P,
    Scalar,
    TakeIndexer,
    WindowingEngine,
    WindowingEngineKwargs,
)

AggScalar: TypeAlias = str | Callable[..., Any]

class NamedAgg(NamedTuple):
    column: str
    aggfunc: AggScalar

class SeriesGroupBy(GroupBy[Series[S2]], Generic[S2, ByT]):
    @overload
    def aggregate(  # pyrefly: ignore
        self,
        func: Callable[Concatenate[Series[S2], P], S3],
        /,
        *args: Any,
        engine: WindowingEngine = None,
        engine_kwargs: WindowingEngineKwargs = None,
        **kwargs: Any,
    ) -> Series[S3]: ...
    @overload
    def aggregate(
        self,
        func: Callable[[Series], S3],
        *args: Any,
        engine: WindowingEngine = None,
        engine_kwargs: WindowingEngineKwargs = None,
        **kwargs: Any,
    ) -> Series[S3]: ...
    @overload
    def aggregate(
        self,
        func: list[AggFuncTypeBase],
        /,
        *args: Any,
        engine: WindowingEngine = None,
        engine_kwargs: WindowingEngineKwargs = None,
        **kwargs: Any,
    ) -> DataFrame: ...
    @overload
    def aggregate(
        self,
        func: AggFuncTypeBase | None = None,
        /,
        *args: Any,
        engine: WindowingEngine = None,
        engine_kwargs: WindowingEngineKwargs = None,
        **kwargs: Any,
    ) -> Series[Any]: ...
    agg = aggregate
    @overload
    def transform(
        self,
        func: Callable[Concatenate[Series[S2], P], Series[S3]],
        /,
        *args: Any,
        engine: WindowingEngine = None,
        engine_kwargs: WindowingEngineKwargs = None,
        **kwargs: Any,
    ) -> Series[S3]: ...
    @overload
    def transform(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Series[Any]: ...
    @overload
    def transform(
        self, func: TransformReductionListType, *args: Any, **kwargs: Any
    ) -> Series[Any]: ...
    def filter(
        self, func: Callable[..., Any] | str, dropna: bool = True, *args: Any, **kwargs: Any
    ) -> Series[Any]: ...
    def nunique(self, dropna: bool = True) -> Series[int]: ...
    # describe delegates to super() method but here it has keyword-only parameters
    def describe(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        *,
        percentiles: Iterable[float] | None = None,
        include: Literal["all"] | list[Dtype] | None = None,
        exclude: list[Dtype] | None = None,
    ) -> DataFrame: ...
    @overload
    def value_counts(
        self,
        normalize: Literal[False] = False,
        sort: bool = True,
        ascending: bool = False,
        bins: int | Sequence[int] | None = None,
        dropna: bool = True,
    ) -> Series[int]: ...
    @overload
    def value_counts(
        self,
        normalize: Literal[True],
        sort: bool = True,
        ascending: bool = False,
        bins: int | Sequence[int] | None = None,
        dropna: bool = True,
    ) -> Series[float]: ...
    def take(
        self,
        indices: TakeIndexer,
        **kwargs: Any,
    ) -> Series[S2]: ...
    def skew(
        self,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs: Any,
    ) -> Series[Any]: ...
    @property
    def plot(self) -> GroupByPlot[Self]: ...
    def nlargest(
        self, n: int = 5, keep: NsmallestNlargestKeep = 'first'
    ) -> Series[S2]: ...
    def nsmallest(
        self, n: int = 5, keep: NsmallestNlargestKeep = 'first'
    ) -> Series[S2]: ...
    def idxmin(self, skipna: bool = True) -> Series[Any]: ...
    def idxmax(self, skipna: bool = True) -> Series[Any]: ...
    def corr(
        self,
        other: Series[Any],
        method: CorrelationMethod = 'pearson',
        min_periods: int | None = None,
    ) -> Series[Any]: ...
    def cov(
        self, other: Series[Any], min_periods: int | None = None, ddof: int | None = 1
    ) -> Series[Any]: ...
    @property
    def is_monotonic_increasing(self) -> Series[bool]: ...
    @property
    def is_monotonic_decreasing(self) -> Series[bool]: ...
    def hist(
        self,
        by: IndexLabel | None = None,
        ax: PlotAxes | None = None,
        grid: bool = True,
        xlabelsize: float | str | None = None,
        xrot: float | None = None,
        ylabelsize: float | str | None = None,
        yrot: float | None = None,
        figsize: tuple[float, float] | None = None,
        bins: int | Sequence[int] = 10,
        backend: str | None = None,
        legend: bool = False,
        **kwargs: Any,
    ) -> Series[Any]: ...  # Series[Axes] but this is not allowed
    @property
    def dtype(self) -> Series[Any]: ...
    def unique(self) -> Series[Any]: ...
    # Overrides that provide more precise return types over the GroupBy class
    @final  # type: ignore[misc]
    def __iter__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
    ) -> Iterator[tuple[ByT, Series[S2]]]: ...

_TT = TypeVar("_TT", bound=Literal[True, False])

# ty ignore needed because of https://github.com/astral-sh/ty/issues/157#issuecomment-3017337945
class DFCallable1(Protocol[P]):  # ty: ignore[invalid-argument-type]
    def __call__(
        self, df: DataFrame, /, *args: P.args, **kwargs: P.kwargs
    ) -> Scalar | list[Any] | dict[Any, Any]: ...

class DFCallable2(Protocol[P]):  # ty: ignore[invalid-argument-type]
    def __call__(
        self, df: DataFrame, /, *args: P.args, **kwargs: P.kwargs
    ) -> DataFrame | Series[Any]: ...

class DFCallable3(Protocol[P]):  # ty: ignore[invalid-argument-type]
    def __call__(self, df: Iterable[Any], /, *args: P.args, **kwargs: P.kwargs) -> float: ...

class DataFrameGroupBy(GroupBy[DataFrame], Generic[ByT, _TT]):
    # error: Overload 3 for "apply" will never be used because its parameters overlap overload 1
    @overload  # type: ignore[override]
    def apply(
        self,
        func: DFCallable1[P],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Series[Any]: ...
    @overload
    def apply(
        self,
        func: DFCallable2[P],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> DataFrame: ...
    @overload
    def apply(
        self,
        func: DFCallable3[P],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> DataFrame: ...
    # error: overload 1 overlaps overload 2 because of different return types
    @overload
    def aggregate(self, func: Literal["size"]) -> Series[Any]: ...  # type: ignore[overload-overlap]
    @overload
    def aggregate(
        self,
        func: AggFuncTypeFrame | None = None,
        *args: Any,
        engine: WindowingEngine = None,
        engine_kwargs: WindowingEngineKwargs = None,
        **kwargs: Any,
    ) -> DataFrame: ...
    agg = aggregate
    @overload
    def transform(
        self,
        func: Callable[Concatenate[DataFrame, P], DataFrame],
        *args: Any,
        engine: WindowingEngine = None,
        engine_kwargs: WindowingEngineKwargs = None,
        **kwargs: Any,
    ) -> DataFrame: ...
    @overload
    def transform(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> DataFrame: ...
    @overload
    def transform(
        self, func: TransformReductionListType, *args: Any, **kwargs: Any
    ) -> DataFrame: ...
    def filter(
        self, func: Callable[..., Any], dropna: bool = True, *args: Any, **kwargs: Any
    ) -> DataFrame: ...
    @overload
    def __getitem__(self, key: Scalar) -> SeriesGroupBy[Any, ByT]: ...  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
    @overload
    def __getitem__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, key: Iterable[Hashable]
    ) -> DataFrameGroupBy[ByT, _TT]: ...
    def nunique(self, dropna: bool = True) -> DataFrame: ...
    def idxmax(
        self,
        skipna: bool = True,
        numeric_only: bool = False,
    ) -> DataFrame: ...
    def idxmin(
        self,
        skipna: bool = True,
        numeric_only: bool = False,
    ) -> DataFrame: ...
    @overload
    def boxplot(
        self,
        subplots: Literal[True] = True,
        column: IndexLabel | None = None,
        fontsize: float | str | None = None,
        rot: float = 0,
        grid: bool = True,
        ax: PlotAxes | None = None,
        figsize: tuple[float, float] | None = None,
        layout: tuple[int, int] | None = None,
        sharex: bool = False,
        sharey: bool = True,
        backend: str | None = None,
        **kwargs: Any,
    ) -> Series[Any]: ...  # Series[PlotAxes] but this is not allowed
    @overload
    def boxplot(
        self,
        subplots: Literal[False],
        column: IndexLabel | None = None,
        fontsize: float | str | None = None,
        rot: float = 0,
        grid: bool = True,
        ax: PlotAxes | None = None,
        figsize: tuple[float, float] | None = None,
        layout: tuple[int, int] | None = None,
        sharex: bool = False,
        sharey: bool = True,
        backend: str | None = None,
        **kwargs: Any,
    ) -> PlotAxes: ...
    @overload
    def boxplot(
        self,
        subplots: bool,
        column: IndexLabel | None = None,
        fontsize: float | str | None = None,
        rot: float = 0,
        grid: bool = True,
        ax: PlotAxes | None = None,
        figsize: tuple[float, float] | None = None,
        layout: tuple[int, int] | None = None,
        sharex: bool = False,
        sharey: bool = True,
        backend: str | None = None,
        **kwargs: Any,
    ) -> PlotAxes | Series[Any]: ...  # Series[PlotAxes]
    @overload
    def value_counts(
        self: DataFrameGroupBy[ByT, Literal[True]],
        subset: ListLike | None = None,
        normalize: Literal[False] = False,
        sort: bool = True,
        ascending: bool = False,
        dropna: bool = True,
    ) -> Series[int]: ...
    @overload
    def value_counts(
        self: DataFrameGroupBy[ByT, Literal[True]],
        subset: ListLike | None,
        normalize: Literal[True],
        sort: bool = True,
        ascending: bool = False,
        dropna: bool = True,
    ) -> Series[float]: ...
    @overload
    def value_counts(
        self: DataFrameGroupBy[ByT, Literal[False]],
        subset: ListLike | None = None,
        normalize: Literal[False] = False,
        sort: bool = True,
        ascending: bool = False,
        dropna: bool = True,
    ) -> DataFrame: ...
    @overload
    def value_counts(
        self: DataFrameGroupBy[ByT, Literal[False]],
        subset: ListLike | None,
        normalize: Literal[True],
        sort: bool = True,
        ascending: bool = False,
        dropna: bool = True,
    ) -> DataFrame: ...
    def take(self, indices: TakeIndexer, **kwargs: Any) -> DataFrame: ...
    @overload
    def skew(
        self,
        skipna: bool = True,
        numeric_only: bool = False,
        *,
        level: Level,
        **kwargs: Any,
    ) -> DataFrame: ...
    @overload
    def skew(
        self,
        skipna: bool = True,
        numeric_only: bool = False,
        *,
        level: None = None,
        **kwargs: Any,
    ) -> Series[Any]: ...
    @property
    def plot(self) -> GroupByPlot[Self]: ...
    def corr(
        self,
        method: str | Callable[[np.ndarray[Any, Any], np.ndarray[Any, Any]], float] = 'pearson',
        min_periods: int = 1,
        numeric_only: bool = False,
    ) -> DataFrame: ...
    def cov(
        self,
        min_periods: int | None = None,
        ddof: int | None = 1,
        numeric_only: bool = False,
    ) -> DataFrame: ...
    def hist(
        self,
        column: IndexLabel | None = None,
        by: IndexLabel | None = None,
        grid: bool = True,
        xlabelsize: float | str | None = None,
        xrot: float | None = None,
        ylabelsize: float | str | None = None,
        yrot: float | None = None,
        ax: PlotAxes | None = None,
        sharex: bool = False,
        sharey: bool = False,
        figsize: tuple[float, float] | None = None,
        layout: tuple[int, int] | None = None,
        bins: int | Sequence[int] = 10,
        backend: str | None = None,
        legend: bool = False,
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
