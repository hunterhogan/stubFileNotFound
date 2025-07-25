from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Sequence,
)
from typing import (
    Any,
    Literal,
    NamedTuple,
    overload,
)

from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from pandas import Series
from pandas.core.frame import DataFrame
from scipy.stats import gaussian_kde
from typing_extensions import TypeAlias

from pandas._typing import (
    ArrayLike,
    HashableT,
    HashableT1,
    HashableT2,
    HashableT3,
    npt,
)

class _BoxPlotT(NamedTuple):
    ax: Axes
    lines: dict[str, list[Line2D]]

_SingleColor: TypeAlias = (
    str | list[float] | tuple[float, float, float] | tuple[float, float, float, float]
)
_PlotAccessorColor: TypeAlias = str | list[_SingleColor] | dict[HashableT, _SingleColor]

@overload
def boxplot(
    data: DataFrame,
    column: Hashable | list[HashableT1] | None = None,
    by: Hashable | list[HashableT2] | None = None,
    ax: Axes | None = None,
    fontsize: float | str | None = None,
    rot: float = 0,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
    layout: tuple[int, int] | None = None,
    return_type: Literal["axes"] | None = None,
    **kwargs,
) -> Axes: ...
@overload
def boxplot(
    data: DataFrame,
    column: Hashable | list[HashableT1] | None = None,
    by: Hashable | list[HashableT2] | None = None,
    ax: Axes | None = None,
    fontsize: float | str | None = None,
    rot: float = 0,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
    layout: tuple[int, int] | None = None,
    *,
    return_type: Literal["dict"],
    **kwargs,
) -> dict[str, list[Line2D]]: ...
@overload
def boxplot(
    data: DataFrame,
    column: Hashable | list[HashableT1] | None = None,
    by: Hashable | list[HashableT2] | None = None,
    ax: Axes | None = None,
    fontsize: float | str | None = None,
    rot: float = 0,
    grid: bool = True,
    figsize: tuple[float, float] | None = None,
    layout: tuple[int, int] | None = None,
    *,
    return_type: Literal["both"],
    **kwargs,
) -> _BoxPlotT: ...

class PlotAccessor:
    def __init__(self, data: Any) -> None: ...
    @overload
    def __call__(
        self,
        *,
        data: Series | DataFrame | None = ...,
        x: Hashable = ...,
        y: Hashable | Sequence[Hashable] = ...,
        kind: Literal[
            "line",
            "bar",
            "barh",
            "hist",
            "box",
            "kde",
            "density",
            "area",
            "pie",
            "scatter",
            "hexbin",
        ] = ...,
        ax: Axes | None = ...,
        subplots: Literal[False] | None = ...,
        sharex: bool = ...,
        sharey: bool = ...,
        layout: tuple[int, int] = ...,
        figsize: tuple[float, float] = ...,
        use_index: bool = ...,
        title: Sequence[str] | None = ...,
        grid: bool | None = ...,
        legend: bool | Literal["reverse"] = ...,
        style: str | list[str] | dict[HashableT1, str] = ...,
        logx: bool | Literal["sym"] = ...,
        logy: bool | Literal["sym"] = ...,
        loglog: bool | Literal["sym"] = ...,
        xticks: Sequence[float] = ...,
        yticks: Sequence[float] = ...,
        xlim: tuple[float, float] | list[float] = ...,
        ylim: tuple[float, float] | list[float] = ...,
        xlabel: str = ...,
        ylabel: str = ...,
        rot: float = ...,
        fontsize: float = ...,
        colormap: str | Colormap | None = ...,
        colorbar: bool = ...,
        position: float = ...,
        table: bool | Series | DataFrame = ...,
        yerr: DataFrame | Series | ArrayLike | dict[Any, Any] | str = ...,
        xerr: DataFrame | Series | ArrayLike | dict[Any, Any] | str = ...,
        stacked: bool = ...,
        secondary_y: bool | list[HashableT2] | tuple[HashableT2, ...] = ...,
        mark_right: bool = ...,
        include_bool: bool = ...,
        backend: str = ...,
        **kwargs: Any,
    ) -> Axes: ...
    @overload
    def __call__(
        self,
        *,
        data: Series | DataFrame | None = ...,
        x: Hashable = ...,
        y: Hashable | Sequence[Hashable] = ...,
        kind: Literal[
            "line",
            "bar",
            "barh",
            "hist",
            "kde",
            "density",
            "area",
            "pie",
            "scatter",
            "hexbin",
        ] = ...,
        ax: Axes | None = ...,
        subplots: Literal[True] | Sequence[Iterable[HashableT1]],
        sharex: bool = ...,
        sharey: bool = ...,
        layout: tuple[int, int] = ...,
        figsize: tuple[float, float] = ...,
        use_index: bool = ...,
        title: Sequence[str] | None = ...,
        grid: bool | None = ...,
        legend: bool | Literal["reverse"] = ...,
        style: str | list[str] | dict[HashableT2, str] = ...,
        logx: bool | Literal["sym"] = ...,
        logy: bool | Literal["sym"] = ...,
        loglog: bool | Literal["sym"] = ...,
        xticks: Sequence[float] = ...,
        yticks: Sequence[float] = ...,
        xlim: tuple[float, float] | list[float] = ...,
        ylim: tuple[float, float] | list[float] = ...,
        xlabel: str = ...,
        ylabel: str = ...,
        rot: float = ...,
        fontsize: float = ...,
        colormap: str | Colormap | None = ...,
        colorbar: bool = ...,
        position: float = ...,
        table: bool | Series | DataFrame = ...,
        yerr: DataFrame | Series | ArrayLike | dict[Any, Any] | str = ...,
        xerr: DataFrame | Series | ArrayLike | dict[Any, Any] | str = ...,
        stacked: bool = ...,
        secondary_y: bool | list[HashableT3] | tuple[HashableT3, ...] = ...,
        mark_right: bool = ...,
        include_bool: bool = ...,
        backend: str = ...,
        **kwargs: Any,
    ) -> npt.NDArray[np.object_]: ...
    @overload
    def __call__(
        self,
        *,
        data: Series | DataFrame | None = ...,
        x: Hashable = ...,
        y: Hashable | Sequence[Hashable] = ...,
        kind: Literal["box"],
        ax: Axes | None = ...,
        subplots: Literal[True] | Sequence[Iterable[HashableT1]],
        sharex: bool = ...,
        sharey: bool = ...,
        layout: tuple[int, int] = ...,
        figsize: tuple[float, float] = ...,
        use_index: bool = ...,
        title: Sequence[str] | None = ...,
        grid: bool | None = ...,
        legend: bool | Literal["reverse"] = ...,
        style: str | list[str] | dict[HashableT2, str] = ...,
        logx: bool | Literal["sym"] = ...,
        logy: bool | Literal["sym"] = ...,
        loglog: bool | Literal["sym"] = ...,
        xticks: Sequence[float] = ...,
        yticks: Sequence[float] = ...,
        xlim: tuple[float, float] | list[float] = ...,
        ylim: tuple[float, float] | list[float] = ...,
        xlabel: str = ...,
        ylabel: str = ...,
        rot: float = ...,
        fontsize: float = ...,
        colormap: str | Colormap | None = ...,
        colorbar: bool = ...,
        position: float = ...,
        table: bool | Series | DataFrame = ...,
        yerr: DataFrame | Series | ArrayLike | dict[Any, Any] | str = ...,
        xerr: DataFrame | Series | ArrayLike | dict[Any, Any] | str = ...,
        stacked: bool = ...,
        secondary_y: bool | list[HashableT3] | tuple[HashableT3, ...] = ...,
        mark_right: bool = ...,
        include_bool: bool = ...,
        backend: str = ...,
        **kwargs: Any,
    ) -> pd.Series: ...
    @overload
    def line(
        self,
        x: Hashable = None,
        y: Hashable = None,
        color: _PlotAccessorColor = ...,
        *,
        subplots: Literal[False] | None = ...,
        **kwargs,
    ) -> Axes: ...
    @overload
    def line(
        self,
        x: Hashable = None,
        y: Hashable = None,
        color: _PlotAccessorColor = ...,
        *,
        subplots: Literal[True],
        **kwargs,
    ) -> npt.NDArray[np.object_]: ...
    @overload
    def bar(
        self,
        x: Hashable = None,
        y: Hashable = None,
        color: _PlotAccessorColor = ...,
        *,
        subplots: Literal[False] | None = ...,
        **kwargs,
    ) -> Axes: ...
    @overload
    def bar(
        self,
        x: Hashable = None,
        y: Hashable = None,
        color: _PlotAccessorColor = ...,
        *,
        subplots: Literal[True],
        **kwargs,
    ) -> npt.NDArray[np.object_]: ...
    @overload
    def barh(
        self,
        x: Hashable = None,
        y: Hashable = None,
        color: _PlotAccessorColor = ...,
        subplots: Literal[False] | None = ...,
        **kwargs,
    ) -> Axes: ...
    @overload
    def barh(
        self,
        x: Hashable = None,
        y: Hashable = None,
        color: _PlotAccessorColor = ...,
        *,
        subplots: Literal[True],
        **kwargs,
    ) -> npt.NDArray[np.object_]: ...
    @overload
    def box(
        self,
        by: Hashable | list[HashableT] = None,
        *,
        subplots: Literal[False] | None = ...,
        **kwargs,
    ) -> Axes: ...
    @overload
    def box(
        self,
        by: Hashable | list[HashableT] = None,
        *,
        subplots: Literal[True],
        **kwargs,
    ) -> Series: ...
    @overload
    def hist(
        self,
        by: Hashable | list[HashableT] | None = None,
        bins: int = 10,
        *,
        subplots: Literal[False] | None = ...,
        **kwargs,
    ) -> Axes: ...
    @overload
    def hist(
        self,
        by: Hashable | list[HashableT] | None = None,
        bins: int = 10,
        *,
        subplots: Literal[True],
        **kwargs,
    ) -> npt.NDArray[np.object_]: ...
    @overload
    def kde(
        self,
        bw_method: (
            Literal["scott", "silverman"]
            | float
            | Callable[[gaussian_kde], float]
            | None
        ) = None,
        ind: npt.NDArray[np.double] | int | None = None,
        *,
        subplots: Literal[False] | None = ...,
        **kwargs,
    ) -> Axes: ...
    @overload
    def kde(
        self,
        bw_method: (
            Literal["scott", "silverman"]
            | float
            | Callable[[gaussian_kde], float]
            | None
        ) = None,
        ind: npt.NDArray[np.double] | int | None = None,
        *,
        subplots: Literal[True],
        **kwargs,
    ) -> npt.NDArray[np.object_]: ...
    @overload
    def area(
        self,
        x: Hashable | None = None,
        y: Hashable | None = None,
        stacked: bool = True,
        *,
        subplots: Literal[False] | None = ...,
        **kwargs,
    ) -> Axes: ...
    @overload
    def area(
        self,
        x: Hashable | None = None,
        y: Hashable | None = None,
        stacked: bool = True,
        *,
        subplots: Literal[True],
        **kwargs,
    ) -> npt.NDArray[np.object_]: ...
    @overload
    def pie(
        self, y: Hashable, *, subplots: Literal[False] | None = ..., **kwargs
    ) -> Axes: ...
    @overload
    def pie(
        self, y: Hashable, *, subplots: Literal[True], **kwargs
    ) -> npt.NDArray[np.object_]: ...
    @overload
    def scatter(
        self,
        x: Hashable,
        y: Hashable,
        s: Hashable | Sequence[float] | None = None,
        c: Hashable | list[str] = None,
        *,
        subplots: Literal[False] | None = ...,
        **kwargs,
    ) -> Axes: ...
    @overload
    def scatter(
        self,
        x: Hashable,
        y: Hashable,
        s: Hashable | Sequence[float] | None = None,
        c: Hashable | list[str] = None,
        *,
        subplots: Literal[True],
        **kwargs,
    ) -> npt.NDArray[np.object_]: ...
    @overload
    def hexbin(
        self,
        x: Hashable,
        y: Hashable,
        C: Hashable | None = None,
        reduce_C_function: Callable[[list], float] | None = None,
        gridsize: int | tuple[int, int] | None = None,
        *,
        subplots: Literal[False] | None = ...,
        **kwargs,
    ) -> Axes: ...
    @overload
    def hexbin(
        self,
        x: Hashable,
        y: Hashable,
        C: Hashable | None = None,
        reduce_C_function: Callable[[list], float] | None = None,
        gridsize: int | tuple[int, int] | None = None,
        *,
        subplots: Literal[True],
        **kwargs,
    ) -> npt.NDArray[np.object_]: ...

    density = kde
