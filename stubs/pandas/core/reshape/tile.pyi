from collections.abc import Sequence
from pandas import Categorical, CategoricalDtype, DatetimeIndex, Index, Interval, IntervalIndex, Series, Timestamp
from pandas._typing import IntervalT, Label, npt
from pandas.core.series import TimestampSeries
from typing import Any, Literal, overload
import numpy as np

@overload
def cut(
    x: Index[Any] | npt.NDArray[Any] | Sequence[int] | Sequence[float],
    bins: int | Series | Index[int] | Index[float] | Sequence[int] | Sequence[float],
    right: bool = True,
    *,
    labels: Literal[False],
    retbins: Literal[True],
    precision: int = 3,
    include_lowest: bool = False,
    duplicates: Literal["raise", "drop"] = 'raise',
    ordered: bool = True,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[Any]]: ...
@overload
def cut(
    x: Index[Any] | npt.NDArray[Any] | Sequence[int] | Sequence[float],
    bins: IntervalIndex[IntervalT],
    right: bool = True,
    *,
    labels: Literal[False],
    retbins: Literal[True],
    precision: int = 3,
    include_lowest: bool = False,
    duplicates: Literal["raise", "drop"] = 'raise',
    ordered: bool = True,
) -> tuple[npt.NDArray[np.intp], IntervalIndex[IntervalT]]: ...
@overload
def cut(  # pyright: ignore[reportOverlappingOverload]
    x: TimestampSeries,
    bins: (
        int
        | TimestampSeries
        | DatetimeIndex
        | Sequence[Timestamp]
        | Sequence[np.datetime64]
    ),
    right: bool = True,
    labels: Literal[False] | Sequence[Label] | None = None,
    *,
    retbins: Literal[True],
    precision: int = 3,
    include_lowest: bool = False,
    duplicates: Literal["raise", "drop"] = 'raise',
    ordered: bool = True,
) -> tuple[Series, DatetimeIndex]: ...
@overload
def cut(
    x: TimestampSeries,
    bins: IntervalIndex[Interval[Timestamp]],
    right: bool = True,
    labels: Sequence[Label] | None = None,
    *,
    retbins: Literal[True],
    precision: int = 3,
    include_lowest: bool = False,
    duplicates: Literal["raise", "drop"] = 'raise',
    ordered: bool = True,
) -> tuple[Series, DatetimeIndex]: ...
@overload
def cut(
    x: Series,
    bins: int | Series | Index[int] | Index[float] | Sequence[int] | Sequence[float],
    right: bool = True,
    labels: Literal[False] | Sequence[Label] | None = None,
    *,
    retbins: Literal[True],
    precision: int = 3,
    include_lowest: bool = False,
    duplicates: Literal["raise", "drop"] = 'raise',
    ordered: bool = True,
) -> tuple[Series, npt.NDArray[Any]]: ...
@overload
def cut(
    x: Series,
    bins: IntervalIndex[Interval[int]] | IntervalIndex[Interval[float]],
    right: bool = True,
    labels: Sequence[Label] | None = None,
    *,
    retbins: Literal[True],
    precision: int = 3,
    include_lowest: bool = False,
    duplicates: Literal["raise", "drop"] = 'raise',
    ordered: bool = True,
) -> tuple[Series, IntervalIndex[Any]]: ...
@overload
def cut(
    x: Index[Any] | npt.NDArray[Any] | Sequence[int] | Sequence[float],
    bins: int | Series | Index[int] | Index[float] | Sequence[int] | Sequence[float],
    right: bool = True,
    labels: Sequence[Label] | None = None,
    *,
    retbins: Literal[True],
    precision: int = 3,
    include_lowest: bool = False,
    duplicates: Literal["raise", "drop"] = 'raise',
    ordered: bool = True,
) -> tuple[Categorical, npt.NDArray[Any]]: ...
@overload
def cut(
    x: Index[Any] | npt.NDArray[Any] | Sequence[int] | Sequence[float],
    bins: IntervalIndex[IntervalT],
    right: bool = True,
    labels: Sequence[Label] | None = None,
    *,
    retbins: Literal[True],
    precision: int = 3,
    include_lowest: bool = False,
    duplicates: Literal["raise", "drop"] = 'raise',
    ordered: bool = True,
) -> tuple[Categorical, IntervalIndex[IntervalT]]: ...
@overload
def cut(
    x: Index[Any] | npt.NDArray[Any] | Sequence[int] | Sequence[float],
    bins: (
        int
        | Series
        | Index[int]
        | Index[float]
        | Sequence[int]
        | Sequence[float]
        | IntervalIndex[Any]
    ),
    right: bool = True,
    *,
    labels: Literal[False],
    retbins: Literal[False] = False,
    precision: int = 3,
    include_lowest: bool = False,
    duplicates: Literal["raise", "drop"] = 'raise',
    ordered: bool = True,
) -> npt.NDArray[np.intp]: ...
@overload
def cut(
    x: TimestampSeries,
    bins: (
        int
        | TimestampSeries
        | DatetimeIndex
        | Sequence[Timestamp]
        | Sequence[np.datetime64]
        | IntervalIndex[Interval[Timestamp]]
    ),
    right: bool = True,
    labels: Literal[False] | Sequence[Label] | None = None,
    retbins: Literal[False] = False,
    precision: int = 3,
    include_lowest: bool = False,
    duplicates: Literal["raise", "drop"] = 'raise',
    ordered: bool = True,
) -> Series[CategoricalDtype]: ...
@overload
def cut(
    x: Series,
    bins: (
        int
        | Series
        | Index[int]
        | Index[float]
        | Sequence[int]
        | Sequence[float]
        | IntervalIndex[Any]
    ),
    right: bool = True,
    labels: Literal[False] | Sequence[Label] | None = None,
    retbins: Literal[False] = False,
    precision: int = 3,
    include_lowest: bool = False,
    duplicates: Literal["raise", "drop"] = 'raise',
    ordered: bool = True,
) -> Series: ...
@overload
def cut(
    x: Index[Any] | npt.NDArray[Any] | Sequence[int] | Sequence[float],
    bins: (
        int
        | Series
        | Index[int]
        | Index[float]
        | Sequence[int]
        | Sequence[float]
        | IntervalIndex[Any]
    ),
    right: bool = True,
    labels: Sequence[Label] | None = None,
    retbins: Literal[False] = False,
    precision: int = 3,
    include_lowest: bool = False,
    duplicates: Literal["raise", "drop"] = 'raise',
    ordered: bool = True,
) -> Categorical: ...
@overload
def qcut(
    x: Index[Any] | npt.NDArray[Any] | Sequence[int] | Sequence[float],
    q: int | Sequence[float] | Series[float] | Index[float] | npt.NDArray[Any],
    *,
    labels: Literal[False],
    retbins: Literal[False] = False,
    precision: int = 3,
    duplicates: Literal["raise", "drop"] = 'raise',
) -> npt.NDArray[np.intp]: ...
@overload
def qcut(
    x: Index[Any] | npt.NDArray[Any] | Sequence[int] | Sequence[float],
    q: int | Sequence[float] | Series[float] | Index[float] | npt.NDArray[Any],
    labels: Sequence[Label] | None = None,
    retbins: Literal[False] = False,
    precision: int = 3,
    duplicates: Literal["raise", "drop"] = 'raise',
) -> Categorical: ...
@overload
def qcut(
    x: Series,
    q: int | Sequence[float] | Series[float] | Index[float] | npt.NDArray[Any],
    labels: Literal[False] | Sequence[Label] | None = None,
    retbins: Literal[False] = False,
    precision: int = 3,
    duplicates: Literal["raise", "drop"] = 'raise',
) -> Series: ...
@overload
def qcut(
    x: Index[Any] | npt.NDArray[Any] | Sequence[int] | Sequence[float],
    q: int | Sequence[float] | Series[float] | Index[float] | npt.NDArray[Any],
    *,
    labels: Literal[False],
    retbins: Literal[True],
    precision: int = 3,
    duplicates: Literal["raise", "drop"] = 'raise',
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.double]]: ...
@overload
def qcut(
    x: Series,
    q: int | Sequence[float] | Series[float] | Index[float] | npt.NDArray[Any],
    labels: Literal[False] | Sequence[Label] | None = None,
    *,
    retbins: Literal[True],
    precision: int = 3,
    duplicates: Literal["raise", "drop"] = 'raise',
) -> tuple[Series, npt.NDArray[np.double]]: ...
@overload
def qcut(
    x: Index[Any] | npt.NDArray[Any] | Sequence[int] | Sequence[float],
    q: int | Sequence[float] | Series[float] | Index[float] | npt.NDArray[Any],
    labels: Sequence[Label] | None = None,
    *,
    retbins: Literal[True],
    precision: int = 3,
    duplicates: Literal["raise", "drop"] = 'raise',
) -> tuple[Categorical, npt.NDArray[np.double]]: ...
