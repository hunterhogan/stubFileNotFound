from builtins import (
    bool as _bool,
    str as _str,
)
from collections import dict_keys  # type: ignore[attr-defined]
from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
)
from datetime import (
    date,
    datetime,
    time,
    timedelta,
)
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Generic,
    Literal,
    NoReturn,
    overload,
)

from matplotlib.axes import (
    Axes as PlotAxes,
    SubplotBase,
)
import numpy as np
from pandas import (
    Index,
    Period,
    PeriodDtype,
    Timedelta,
    Timestamp,
)
from pandas.core.api import (
    Int8Dtype as Int8Dtype,
    Int16Dtype as Int16Dtype,
    Int32Dtype as Int32Dtype,
    Int64Dtype as Int64Dtype,
)
from pandas.core.arrays import TimedeltaArray
from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.categorical import CategoricalAccessor
from pandas.core.arrays.datetimes import DatetimeArray
from pandas.core.arrays.interval import IntervalArray
from pandas.core.base import IndexOpsMixin
from pandas.core.frame import DataFrame
from pandas.core.generic import NDFrame
from pandas.core.groupby.generic import SeriesGroupBy
from pandas.core.groupby.groupby import BaseGroupBy
from pandas.core.indexers import BaseIndexer
from pandas.core.indexes.accessors import (
    CombinedDatetimelikeProperties,
    PeriodProperties,
    TimedeltaProperties,
    TimestampProperties,
)
from pandas.core.indexes.base import UnknownIndex
from pandas.core.indexes.category import CategoricalIndex
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.interval import IntervalIndex
from pandas.core.indexes.multi import MultiIndex
from pandas.core.indexes.period import PeriodIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex
from pandas.core.indexing import (
    _AtIndexer,
    _iAtIndexer,
    _iLocIndexer,
    _IndexSliceTuple,
    _LocIndexer,
)
from pandas.core.strings import StringMethods
from pandas.core.window import (
    Expanding,
    ExponentialMovingWindow,
)
from pandas.core.window.rolling import (
    Rolling,
    Window,
)
from typing_extensions import (
    Never,
    Self,
    TypeAlias,
)
import xarray as xr

from pandas._libs.interval import (
    Interval,
    _OrderableT,
)
from pandas._libs.lib import NoDefault
from pandas._libs.missing import NAType
from pandas._libs.tslibs import BaseOffset
from pandas._libs.tslibs.nattype import NaTType
from pandas._libs.tslibs.offsets import DateOffset
from pandas._typing import (
    S1,
    S2,
    AggFuncTypeBase,
    AggFuncTypeDictFrame,
    AggFuncTypeSeriesToFrame,
    AnyAll,
    AnyArrayLike,
    ArrayLike,
    Axes,
    Axis,
    AxisColumn,
    AxisIndex,
    BooleanDtypeArg,
    BytesDtypeArg,
    CalculationMethod,
    CategoryDtypeArg,
    ComplexDtypeArg,
    CompressionOptions,
    DropKeep,
    Dtype,
    DtypeObj,
    FilePath,
    FillnaOptions,
    FloatDtypeArg,
    FloatFormatType,
    GroupByObjectNonScalar,
    HashableT1,
    IgnoreRaise,
    IndexingInt,
    IndexKeyFunc,
    IndexLabel,
    IntDtypeArg,
    InterpolateOptions,
    IntervalClosedType,
    IntervalT,
    JoinHow,
    JSONSerializable,
    JsonSeriesOrient,
    Label,
    Level,
    ListLike,
    ListLikeU,
    MaskType,
    NaPosition,
    NsmallestNlargestKeep,
    ObjectDtypeArg,
    QuantileInterpolation,
    RandomState,
    ReindexMethod,
    Renamer,
    ReplaceValue,
    Scalar,
    ScalarT,
    SequenceNotStr,
    SeriesByT,
    SortKind,
    StrDtypeArg,
    StrLike,
    Suffixes,
    T as _T,
    TimeAmbiguous,
    TimedeltaDtypeArg,
    TimestampDtypeArg,
    TimeUnit,
    TimeZones,
    ToTimestampHow,
    UIntDtypeArg,
    ValueKeyFunc,
    VoidDtypeArg,
    WriteBuffer,
    np_ndarray_anyint,
    npt,
    num,
)

from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.dtypes import CategoricalDtype

from pandas.plotting import PlotAccessor

class _iLocIndexerSeries(_iLocIndexer, Generic[S1]):
    # get item
    @overload
    def __getitem__(self, idx: IndexingInt) -> S1: ...
    @overload
    def __getitem__(self, idx: Index[Any] | slice | np_ndarray_anyint) -> Series[S1]: ...
    # set item
    @overload
    def __setitem__(self, idx: int, value: S1 | None) -> None: ...
    @overload
    def __setitem__(
        self,
        idx: Index[Any] | slice | np_ndarray_anyint | list[int],
        value: S1 | Series[S1] | None,
    ) -> None: ...

class _LocIndexerSeries(_LocIndexer, Generic[S1]):
    # ignore needed because of mypy.  Overlapping, but we want to distinguish
    # having a tuple of just scalars, versus tuples that include slices or Index
    @overload
    def __getitem__(  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
        self,
        idx: Scalar | tuple[Scalar, ...],
        # tuple case is for getting a specific element when using a MultiIndex
    ) -> S1: ...
    @overload
    def __getitem__(
        self,
        idx: (
            MaskType
            | Index[Any]
            | SequenceNotStr[float | str | Timestamp]
            | slice
            | _IndexSliceTuple[Any]
            | Sequence[_IndexSliceTuple[Any]]
            | Callable[..., Any]
        ),
        # _IndexSliceTuple is when having a tuple that includes a slice.  Could just
        # be s.loc[1, :], or s.loc[pd.IndexSlice[1, :]]
    ) -> Series[S1]: ...
    @overload
    def __setitem__(
        self,
        idx: Index[Any] | MaskType | slice,
        value: S1 | ArrayLike | Series[S1] | None,
    ) -> None: ...
    @overload
    def __setitem__(
        self,
        idx: _str,
        value: S1 | None,
    ) -> None: ...
    @overload
    def __setitem__(
        self,
        idx: MaskType | StrLike | _IndexSliceTuple[Any] | list[ScalarT],
        value: S1 | ArrayLike | Series[S1] | None,
    ) -> None: ...

_ListLike: TypeAlias = (
    ArrayLike | dict[_str, np.ndarray[Any, Any]] | Sequence[S1] | IndexOpsMixin[S1]
)

class Series(IndexOpsMixin[S1], NDFrame):
    __hash__: ClassVar[None]

    @overload
    def __new__(
        cls,
        data: npt.NDArray[np.float64],
        index: Axes | None = ...,
        dtype: Dtype = ...,
        name: Hashable = ...,
        copy: bool = ...,
    ) -> Series[float]: ...
    @overload
    def __new__(  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
        cls,
        data: Sequence[Never],
        index: Axes | None = ...,
        dtype: Dtype = ...,
        name: Hashable = ...,
        copy: bool = ...,
    ) -> UnknownSeries: ...
    @overload
    def __new__(
        cls,
        data: Sequence[list[_str]],
        index: Axes | None = ...,
        dtype: Dtype = ...,
        name: Hashable = ...,
        copy: bool = ...,
    ) -> Series[list[_str]]: ...
    @overload
    def __new__(
        cls,
        data: Sequence[_str],
        index: Axes | None = ...,
        dtype: Dtype = ...,
        name: Hashable = ...,
        copy: bool = ...,
    ) -> Series[_str]: ...
    @overload
    def __new__(
        cls,
        data: (
            DatetimeIndex
            | Sequence[np.datetime64 | datetime | date]
            | dict[HashableT1, np.datetime64 | datetime | date]
            | np.datetime64
            | datetime
            | date
        ),
        index: Axes | None = ...,
        dtype: TimestampDtypeArg = ...,
        name: Hashable = ...,
        copy: bool = ...,
    ) -> TimestampSeries: ...
    @overload
    def __new__(
        cls,
        data: _ListLike[Any],
        index: Axes | None = ...,
        *,
        dtype: TimestampDtypeArg,
        name: Hashable = ...,
        copy: bool = ...,
    ) -> TimestampSeries: ...
    @overload
    def __new__(
        cls,
        data: PeriodIndex | Sequence[Period],
        index: Axes | None = ...,
        dtype: PeriodDtype = ...,
        name: Hashable = ...,
        copy: bool = ...,
    ) -> PeriodSeries: ...
    @overload
    def __new__(
        cls,
        data: (
            TimedeltaIndex
            | Sequence[np.timedelta64 | timedelta]
            | dict[HashableT1, np.timedelta64 | timedelta]
            | np.timedelta64
            | timedelta
        ),
        index: Axes | None = ...,
        dtype: TimedeltaDtypeArg = ...,
        name: Hashable = ...,
        copy: bool = ...,
    ) -> TimedeltaSeries: ...
    @overload
    def __new__(
        cls,
        data: (
            IntervalIndex[Interval[_OrderableT]]
            | Interval[_OrderableT]
            | Sequence[Interval[_OrderableT]]
            | dict[HashableT1, Interval[_OrderableT]]
        ),
        index: Axes | None = ...,
        dtype: Literal["Interval"] = "Interval",
        name: Hashable = ...,
        copy: bool = ...,
    ) -> IntervalSeries[_OrderableT]: ...
    @overload
    def __new__(  # type: ignore[overload-overlap]
        cls,
        data: Scalar | _ListLike[Any] | dict[HashableT1, Any] | None,
        index: Axes | None = ...,
        *,
        dtype: type[S1],
        name: Hashable = ...,
        copy: bool = ...,
    ) -> Self: ...
    @overload
    def __new__(  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
        cls,
        data: Sequence[bool],
        index: Axes | None = ...,
        dtype: Dtype = ...,
        name: Hashable = ...,
        copy: bool = ...,
    ) -> Series[bool]: ...
    @overload
    def __new__(  # type: ignore[overload-overlap]
        cls,
        data: Sequence[int],
        index: Axes | None = ...,
        dtype: Dtype = ...,
        name: Hashable = ...,
        copy: bool = ...,
    ) -> Series[int]: ...
    @overload
    def __new__(
        cls,
        data: Sequence[float],
        index: Axes | None = ...,
        dtype: Dtype = ...,
        name: Hashable = ...,
        copy: bool = ...,
    ) -> Series[float]: ...
    @overload
    def __new__(  # type: ignore[overload-cannot-match] # pyright: ignore[reportOverlappingOverload]
        cls,
        data: Sequence[int | float],
        index: Axes | None = ...,
        dtype: Dtype = ...,
        name: Hashable = ...,
        copy: bool = ...,
    ) -> Series[float]: ...
    @overload
    def __new__(
        cls,
        data: S1 | _ListLike[S1] | dict[HashableT1, S1] | dict_keys[S1, Any],
        index: Axes | None = ...,
        dtype: Dtype = ...,
        name: Hashable = ...,
        copy: bool = ...,
    ) -> Self: ...
    @overload
    def __new__(
        cls,
        data: (
            Scalar
            | _ListLike[Any]
            | Mapping[HashableT1, Any]
            | BaseGroupBy[Any]
            | NaTType
            | NAType
            | None
        ) = ...,
        index: Axes | None = ...,
        dtype: Dtype = ...,
        name: Hashable = ...,
        copy: bool = ...,
    ) -> UnknownSeries: ...
    @property
    def hasnans(self) -> bool: ...
    def div(
        self,
        other: num | _ListLike[Any] | Series[S1],
        level: Level | None = ...,
        fill_value: float | None = ...,
        axis: AxisIndex = ...,
    ) -> Series[float]: ...
    def rdiv(
        self,
        other: Series[S1] | Scalar,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[S1]: ...
    @property
    def dtype(self) -> DtypeObj: ...
    @property
    def dtypes(self) -> DtypeObj: ...
    @property
    def name(self) -> Hashable | None: ...
    @name.setter
    def name(self, value: Hashable | None) -> None: ...
    @property
    def values(self) -> ArrayLike: ...
    @property
    def array(self) -> ExtensionArray: ...
    def ravel(self, order: _str = ...) -> np.ndarray[Any, Any]: ...
    def __len__(self) -> int: ...
    def view(self, dtype=None) -> Series[S1]: ...
    def __array_ufunc__(
        self, ufunc: Callable[..., Any], method: _str, *inputs: Any, **kwargs: Any
    ) -> Any: ...
    def __array__(self, dtype: Any=...) -> np.ndarray[Any, Any]: ...
    @property
    def axes(self) -> list[Any]: ...
    def __getattr__(self, name: _str) -> S1: ...
    @overload
    def __getitem__(
        self,
        idx: (
            list[_str]
            | Index[Any]
            | Series[S1]
            | slice
            | MaskType
            | tuple[Hashable | slice, ...]
        ),
    ) -> Self: ...
    @overload
    def __getitem__(self, idx: Scalar) -> S1: ...
    def __setitem__(self, key: Any, value: Any ) -> None: ...
    @overload
    def get(self, key: Hashable, default: None = None) -> S1 | None: ...
    @overload
    def get(self, key: Hashable, default: S1) -> S1: ...
    @overload
    def get(self, key: Hashable, default: _T) -> S1 | _T: ...
    def repeat(
        self, repeats: int | list[int], axis: AxisIndex | None = None
    ) -> Series[S1]: ...
    @property
    def index(self) -> Index[Any] | MultiIndex: ...
    @index.setter
    def index(self, idx: Index[Any]) -> None: ...
    @overload
    def reset_index(
        self,
        level: Sequence[Level] | Level | None = None,
        *,
        drop: Literal[False] = False,
        name: Level = ...,
        inplace: Literal[False] = False,
        allow_duplicates: bool = False,
    ) -> DataFrame: ...
    @overload
    def reset_index(
        self,
        level: Sequence[Level] | Level | None = None,
        *,
        drop: Literal[True],
        name: Level = ...,
        inplace: Literal[False] = False,
        allow_duplicates: bool = False,
    ) -> Series[S1]: ...
    @overload
    def reset_index(
        self,
        level: Sequence[Level] | Level | None = None,
        *,
        drop: bool = False,
        name: Level = ...,
        inplace: Literal[True],
        allow_duplicates: bool = False,
    ) -> None: ...
    @overload
    def to_string(
        self,
        buf: FilePath | WriteBuffer[_str],
        na_rep: _str = 'NaN',
        float_format: FloatFormatType = None,
        header: _bool = True,
        index: _bool = True,
        length: _bool = False,
        dtype: _bool = False,
        name: _bool = False,
        max_rows: int | None = None,
        min_rows: int | None = None,
    ) -> None: ...
    @overload
    def to_string(
        self,
        buf: None = None,
        na_rep: _str = 'NaN',
        float_format: FloatFormatType = None,
        header: _bool = True,
        index: _bool = True,
        length: _bool = False,
        dtype: _bool = False,
        name: _bool = False,
        max_rows: int | None = None,
        min_rows: int | None = None,
    ) -> _str: ...
    @overload
    def to_json(
        self,
        path_or_buf: FilePath | WriteBuffer[_str],
        *,
        orient: Literal["records"],
        date_format: Literal["epoch", "iso"] | None = None,
        double_precision: int = 10,
        force_ascii: _bool = True,
        date_unit: TimeUnit = 'ms',
        default_handler: Callable[[Any], JSONSerializable] | None = None,
        lines: Literal[True],
        compression: CompressionOptions = 'infer',
        index: _bool = None,
        indent: int | None = None,
        mode: Literal["a"],
    ) -> None: ...
    @overload
    def to_json(
        self,
        path_or_buf: None = None,
        *,
        orient: Literal["records"],
        date_format: Literal["epoch", "iso"] | None = None,
        double_precision: int = 10,
        force_ascii: _bool = True,
        date_unit: TimeUnit = 'ms',
        default_handler: Callable[[Any], JSONSerializable] | None = None,
        lines: Literal[True],
        compression: CompressionOptions = 'infer',
        index: _bool = None,
        indent: int | None = None,
        mode: Literal["a"],
    ) -> _str: ...
    @overload
    def to_json(
        self,
        path_or_buf: FilePath | WriteBuffer[_str] | WriteBuffer[bytes],
        orient: JsonSeriesOrient | None = None,
        date_format: Literal["epoch", "iso"] | None = None,
        double_precision: int = 10,
        force_ascii: _bool = True,
        date_unit: TimeUnit = 'ms',
        default_handler: Callable[[Any], JSONSerializable] | None = None,
        lines: _bool = False,
        compression: CompressionOptions = 'infer',
        index: _bool = None,
        indent: int | None = None,
        mode: Literal["w"] = "w",
    ) -> None: ...
    @overload
    def to_json(
        self,
        path_or_buf: None = None,
        orient: JsonSeriesOrient | None = None,
        date_format: Literal["epoch", "iso"] | None = None,
        double_precision: int = 10,
        force_ascii: _bool = True,
        date_unit: TimeUnit = 'ms',
        default_handler: Callable[[Any], JSONSerializable] | None = None,
        lines: _bool = False,
        compression: CompressionOptions = 'infer',
        index: _bool = None,
        indent: int | None = None,
        mode: Literal["w"] = "w",
    ) -> _str: ...
    def to_xarray(self) -> xr.DataArray: ...
    def items(self) -> Iterator[tuple[Hashable, S1]]: ...
    def keys(self) -> Index[Any]: ...
    @overload
    def to_dict(self, *, into: type[dict[Any, Any]] = ...) -> dict[Any, S1]: ...
    @overload
    def to_dict(
        self, *, into: type[MutableMapping[Any, Any]] | MutableMapping[Any, Any]
    ) -> MutableMapping[Hashable, S1]: ...
    def to_frame(self, name: object | None = ...) -> DataFrame: ...
    @overload
    def groupby(
        self,
        by: Scalar,
        axis: AxisIndex = 0,
        level: IndexLabel | None = None,
        as_index: _bool = True,
        sort: _bool = True,
        group_keys: _bool = True,
        observed: _bool | NoDefault = ...,
        dropna: _bool = True,
    ) -> SeriesGroupBy[S1, Scalar]: ...
    @overload
    def groupby(
        self,
        by: DatetimeIndex,
        axis: AxisIndex = 0,
        level: IndexLabel | None = None,
        as_index: _bool = True,
        sort: _bool = True,
        group_keys: _bool = True,
        observed: _bool | NoDefault = ...,
        dropna: _bool = True,
    ) -> SeriesGroupBy[S1, Timestamp]: ...
    @overload
    def groupby(
        self,
        by: TimedeltaIndex,
        axis: AxisIndex = 0,
        level: IndexLabel | None = None,
        as_index: _bool = True,
        sort: _bool = True,
        group_keys: _bool = True,
        observed: _bool | NoDefault = ...,
        dropna: _bool = True,
    ) -> SeriesGroupBy[S1, Timedelta]: ...
    @overload
    def groupby(
        self,
        by: PeriodIndex,
        axis: AxisIndex = 0,
        level: IndexLabel | None = None,
        as_index: _bool = True,
        sort: _bool = True,
        group_keys: _bool = True,
        observed: _bool | NoDefault = ...,
        dropna: _bool = True,
    ) -> SeriesGroupBy[S1, Period]: ...
    @overload
    def groupby(
        self,
        by: IntervalIndex[IntervalT],
        axis: AxisIndex = 0,
        level: IndexLabel | None = None,
        as_index: _bool = True,
        sort: _bool = True,
        group_keys: _bool = True,
        observed: _bool | NoDefault = ...,
        dropna: _bool = True,
    ) -> SeriesGroupBy[S1, IntervalT]: ...
    @overload
    def groupby(
        self,
        by: MultiIndex | GroupByObjectNonScalar[Any],
        axis: AxisIndex = 0,
        level: IndexLabel | None = None,
        as_index: _bool = True,
        sort: _bool = True,
        group_keys: _bool = True,
        observed: _bool | NoDefault = ...,
        dropna: _bool = ...,
    ) -> SeriesGroupBy[S1, tuple[Any, ...]]: ...
    @overload
    def groupby(
        self,
        by: None,
        axis: AxisIndex,
        level: IndexLabel,  # level is required when by=None (passed as positional)
        as_index: _bool = True,
        sort: _bool = True,
        group_keys: _bool = True,
        observed: _bool | NoDefault = ...,
        dropna: _bool = True,
    ) -> SeriesGroupBy[S1, Scalar]: ...
    @overload
    def groupby(
        self,
        by: None = None,
        axis: AxisIndex = 0,
        *,
        level: IndexLabel,  # level is required when by=None (passed as keyword)
        as_index: _bool = True,
        sort: _bool = True,
        group_keys: _bool = True,
        observed: _bool | NoDefault = ...,
        dropna: _bool = True,
    ) -> SeriesGroupBy[S1, Scalar]: ...
    @overload
    def groupby(
        self,
        by: Series[SeriesByT],
        axis: AxisIndex = 0,
        level: IndexLabel | None = None,
        as_index: _bool = True,
        sort: _bool = True,
        group_keys: _bool = True,
        observed: _bool | NoDefault = ...,
        dropna: _bool = True,
    ) -> SeriesGroupBy[S1, SeriesByT]: ...
    @overload
    def groupby(
        self,
        by: CategoricalIndex[Any] | UnknownIndex | UnknownSeries,
        axis: AxisIndex = ...,
        level: IndexLabel | None = ...,
        as_index: _bool = ...,
        sort: _bool = ...,
        group_keys: _bool = ...,
        observed: _bool | NoDefault = ...,
        dropna: _bool = True,
    ) -> SeriesGroupBy[S1, Any]: ...
    # need the ignore because None is Hashable
    @overload
    def count(self, level: None = None) -> int: ...  # type: ignore[overload-overlap]
    @overload
    def count(self, level: Hashable) -> Series[S1]: ...
    def mode(self, dropna=True) -> Series[S1]: ...
    def unique(self) -> np.ndarray: ...
    @overload
    def drop_duplicates(
        self,
        *,
        keep: DropKeep = 'first',
        inplace: Literal[True],
        ignore_index: _bool = False,
    ) -> None: ...
    @overload
    def drop_duplicates(
        self,
        *,
        keep: DropKeep = 'first',
        inplace: Literal[False] = False,
        ignore_index: _bool = False,
    ) -> Series[S1]: ...
    def duplicated(self, keep: DropKeep = 'first') -> Series[_bool]: ...
    def idxmax(
        self, axis: AxisIndex = 0, skipna: _bool = True, *args: Any, **kwargs: Any
    ) -> int | _str: ...
    def idxmin(
        self, axis: AxisIndex = 0, skipna: _bool = True, *args: Any, **kwargs: Any
    ) -> int | _str: ...
    def round(self, decimals: int = 0, *args: Any, **kwargs: Any) -> Series[S1]: ...
    @overload
    def quantile(
        self,
        q: float = 0.5,
        interpolation: QuantileInterpolation = 'linear',
    ) -> float: ...
    @overload
    def quantile(
        self,
        q: _ListLike,
        interpolation: QuantileInterpolation = 'linear',
    ) -> Series[S1]: ...
    def corr(
        self,
        other: Series[S1],
        method: Literal["pearson", "kendall", "spearman"] = 'pearson',
        min_periods: int = None,
    ) -> float: ...
    def cov(
        self, other: Series[S1], min_periods: int | None = None, ddof: int = 1
    ) -> float: ...
    @overload
    def diff(self: Series[_bool], periods: int = 1) -> Series[type[object]]: ...  # type: ignore[overload-overlap]
    @overload
    def diff(self: Series[complex], periods: int = 1) -> Series[complex]: ...  # type: ignore[overload-overlap]
    @overload
    def diff(self: Series[bytes], periods: int = 1) -> Never: ...
    @overload
    def diff(self: Series[type], periods: int = 1) -> Never: ...
    @overload
    def diff(self: Series[_str], periods: int = 1) -> Never: ...
    @overload
    def diff(self, periods: int = 1) -> Series[float]: ...
    def autocorr(self, lag: int = 1) -> float: ...
    @overload
    def dot(self, other: Series[S1]) -> Scalar: ...
    @overload
    def dot(self, other: DataFrame) -> Series[S1]: ...
    @overload
    def dot(
        self, other: ArrayLike | dict[_str, np.ndarray[Any, Any]] | Sequence[S1] | Index[S1]
    ) -> np.ndarray[Any, Any]: ...
    @overload
    def __matmul__(self, other: UnknownSeries) -> Scalar: ...
    @overload
    def __matmul__(self, other: DataFrame) -> UnknownSeries: ...
    @overload
    def __matmul__(self, other: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]: ...
    @overload
    def __rmatmul__(self, other: UnknownSeries) -> Scalar: ...
    @overload
    def __rmatmul__(self, other: DataFrame) -> UnknownSeries: ...
    @overload
    def __rmatmul__(self, other: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]: ...
    @overload
    def searchsorted(
        self,
        value: _ListLike[Any],
        side: Literal["left", "right"] = ...,
        sorter: _ListLike[Any] | None = ...,
    ) -> list[int]: ...
    @overload
    def searchsorted(
        self,
        value: Scalar,
        side: Literal["left", "right"] = ...,
        sorter: _ListLike[Any] | None = ...,
    ) -> int: ...
    @overload
    def compare(
        self,
        other: UnknownSeries,
        align_axis: AxisIndex,
        keep_shape: bool = False,
        keep_equal: bool = False,
        result_names: Suffixes = ('self', 'other'),
    ) -> UnknownSeries: ...
    @overload
    def compare(
        self,
        other: UnknownSeries,
        align_axis: AxisColumn = 1,
        keep_shape: bool = False,
        keep_equal: bool = False,
        result_names: Suffixes = ('self', 'other'),
    ) -> DataFrame: ...
    def combine(
        self, other: Series[S1], func: Callable[..., Any], fill_value: Scalar | None = ...
    ) -> Series[S1]: ...
    def combine_first(self, other: Series[S1]) -> Series[S1]: ...
    def update(self, other: Series[S1] | Sequence[S1] | Mapping[int, S1]) -> None: ...
    @overload
    def sort_values(
        self,
        *,
        axis: Axis = 0,
        ascending: _bool | Sequence[_bool] = True,
        kind: SortKind = 'quicksort',
        na_position: NaPosition = 'last',
        ignore_index: _bool = False,
        inplace: Literal[True],
        key: ValueKeyFunc = None,
    ) -> None: ...
    @overload
    def sort_values(
        self,
        *,
        axis: Axis = 0,
        ascending: _bool | Sequence[_bool] = True,
        kind: SortKind = 'quicksort',
        na_position: NaPosition = 'last',
        ignore_index: _bool = False,
        inplace: Literal[False] = False,
        key: ValueKeyFunc = None,
    ) -> Series[S1]: ...
    @overload
    def sort_index(
        self,
        *,
        axis: Axis = 0,
        level: Level | None = None,
        ascending: _bool | Sequence[_bool] = True,
        kind: SortKind = 'quicksort',
        na_position: NaPosition = 'last',
        sort_remaining: _bool = True,
        ignore_index: _bool = False,
        inplace: Literal[True],
        key: IndexKeyFunc = None,
    ) -> None: ...
    @overload
    def sort_index(
        self,
        *,
        axis: Axis = 0,
        level: Level | list[int] | list[_str] | None = None,
        ascending: _bool | Sequence[_bool] = True,
        kind: SortKind = 'quicksort',
        na_position: NaPosition = 'last',
        sort_remaining: _bool = True,
        ignore_index: _bool = False,
        inplace: Literal[False] = False,
        key: IndexKeyFunc = None,
    ) -> Series[S1]: ...
    def argsort(
        self,
        axis: AxisIndex = 0,
        kind: SortKind = 'quicksort',
        order: None = None,
    ) -> Series[int]: ...
    def nlargest(
        self, n: int = 5, keep: NsmallestNlargestKeep = 'first'
    ) -> Series[S1]: ...
    def nsmallest(
        self, n: int = 5, keep: NsmallestNlargestKeep = 'first'
    ) -> Series[S1]: ...
    def swaplevel(
        self, i: Level = -2, j: Level = -1, copy: _bool = None
    ) -> Series[S1]: ...
    def reorder_levels(self, order: list[Any]) -> Series[S1]: ...
    def explode(self) -> Series[S1]: ...
    def unstack(
        self,
        level: Level = -1,
        fill_value: int | _str | dict | None = None,
    ) -> DataFrame: ...
    @overload
    def map(
        self,
        arg: Callable[[S1], S2 | NAType] | Mapping[S1, S2] | Series[S2],
        na_action: Literal["ignore"] = "ignore",
    ) -> Series[S2]: ...
    @overload
    def map(
        self,
        arg: Callable[[S1 | NAType], S2 | NAType] | Mapping[S1, S2] | Series[S2],
        na_action: None = None,
    ) -> Series[S2]: ...
    @overload
    def map(
        self,
        arg: Callable[[Any], Any] | Mapping[Any, Any] | UnknownSeries,
        na_action: Literal["ignore"] | None = None,
    ) -> UnknownSeries: ...
    @overload
    def aggregate(
        self: Series[int],
        func: Literal["mean"],
        axis: AxisIndex = 0,
        *args: Any,
        **kwargs: Any,
    ) -> float: ...
    @overload
    def aggregate(
        self,
        func: AggFuncTypeBase,
        axis: AxisIndex = 0,
        *args: Any,
        **kwargs: Any,
    ) -> S1: ...
    @overload
    def aggregate(
        self,
        func: AggFuncTypeSeriesToFrame = None,
        axis: AxisIndex = 0,
        *args: Any,
        **kwargs: Any,
    ) -> UnknownSeries: ...
    agg = aggregate
    @overload
    def transform(
        self,
        func: AggFuncTypeBase,
        axis: AxisIndex = 0,
        *args: Any,
        **kwargs: Any,
    ) -> Series[S1]: ...
    @overload
    def transform(
        self,
        func: list[AggFuncTypeBase] | AggFuncTypeDictFrame[Any],
        axis: AxisIndex = ...,
        *args: Any,
        **kwargs: Any,
    ) -> DataFrame: ...
    @overload
    def apply(
        self,
        func: Callable[
            ..., Scalar | Sequence[Any] | set[Any] | Mapping[Any, Any] | NAType | frozenset[Any] | None
        ],
        convertDType: _bool = ...,
        args: tuple[Any, ...] = ...,
        **kwargs: Any,
    ) -> UnknownSeries: ...
    @overload
    def apply(
        self,
        func: Callable[..., BaseOffset],
        convertDType: _bool = ...,
        args: tuple[Any, ...] = ...,
        **kwargs: Any,
    ) -> OffsetSeries: ...
    @overload
    def apply(
        self,
        func: Callable[..., UnknownSeries],
        convertDType: _bool = ...,
        args: tuple[Any, ...] = ...,
        **kwargs: Any,
    ) -> DataFrame: ...
    def align(
        self,
        other: DataFrame | UnknownSeries,
        join: JoinHow = 'outer',
        axis: Axis | None = None,
        level: Level | None = None,
        copy: _bool = None,
        fill_value: Scalar | NAType | None = None,
    ) -> tuple[UnknownSeries, UnknownSeries]: ...
    @overload
    def rename(
        self,
        index: Callable[[Any], Label],
        *,
        axis: Axis | None = None,
        copy: bool = None,
        inplace: Literal[True],
        level: Level | None = None,
        errors: IgnoreRaise = 'ignore',
    ) -> None: ...
    @overload
    def rename(
        self,
        index: Mapping[Any, Label],
        *,
        axis: Axis | None = None,
        copy: bool = None,
        inplace: Literal[True],
        level: Level | None = None,
        errors: IgnoreRaise = 'ignore',
    ) -> None: ...
    @overload
    def rename(
        self,
        index: Scalar | tuple[Hashable, ...] | None = None,
        *,
        axis: Axis | None = None,
        copy: bool = None,
        inplace: Literal[True],
        level: Level | None = None,
        errors: IgnoreRaise = 'ignore',
    ) -> Self: ...
    @overload
    def rename(
        self,
        index: Renamer | Scalar | tuple[Hashable, ...] | None = None,
        *,
        axis: Axis | None = None,
        copy: bool = None,
        inplace: Literal[False] = False,
        level: Level | None = None,
        errors: IgnoreRaise = 'ignore',
    ) -> Self: ...
    def reindex_like(
        self,
        other: Series[S1],
        method: FillnaOptions | Literal["nearest"] | None = None,
        copy: _bool = None,
        limit: int | None = None,
        tolerance: Scalar | AnyArrayLike | Sequence[Scalar] = None,
    ) -> Self: ...
    @overload
    def fillna(
        self,
        value: Scalar | NAType | dict[Any, Any] | Series[S1] | DataFrame | None = ...,
        *,
        axis: AxisIndex = None,
        limit: int | None = None,
        inplace: Literal[True],
    ) -> None: ...
    @overload
    def fillna(
        self,
        value: Scalar | NAType | dict[Any, Any] | Series[S1] | DataFrame | None = ...,
        *,
        axis: AxisIndex = None,
        limit: int | None = None,
        inplace: Literal[False] = False,
    ) -> Series[S1]: ...
    @overload
    def replace(
        self,
        to_replace: ReplaceValue[Any, Any] = ...,
        value: ReplaceValue[Any, Any] = ...,
        *,
        regex: ReplaceValue[Any, Any] = ...,
        inplace: Literal[True],
    ) -> None: ...
    @overload
    def replace(
        self,
        to_replace: ReplaceValue[Any, Any] = ...,
        value: ReplaceValue[Any, Any] = ...,
        *,
        regex: ReplaceValue[Any, Any] = ...,
        inplace: Literal[False] = False,
    ) -> Series[S1]: ...
    def shift(
        self,
        periods: int | Sequence[int] = 1,
        freq: DateOffset | timedelta | _str | None = None,
        axis: Axis = 0,
        fill_value: Scalar | NAType | None = ...,
    ) -> UnknownSeries: ...
    def info(
        self,
        verbose: bool | None = None,
        buf: WriteBuffer[_str] | None = None,
        memory_usage: bool | Literal["deep"] | None = None,
        show_counts: bool | None = True,
    ) -> None: ...
    def memory_usage(self, index: _bool = ..., deep: _bool = ...) -> int: ...
    def isin(self, values: Iterable[Any] | Series[S1] | dict[Any, Any]) -> Series[_bool]: ...
    def between(
        self,
        left: Scalar | ListLikeU,
        right: Scalar | ListLikeU,
        inclusive: Literal["both", "neither", "left", "right"] = 'both',
    ) -> Series[_bool]: ...
    def isna(self) -> Series[_bool]: ...
    def isnull(self) -> Series[_bool]: ...
    def notna(self) -> Series[_bool]: ...
    def notnull(self) -> Series[_bool]: ...
    @overload
    def dropna(
        self,
        *,
        axis: AxisIndex = 0,
        inplace: Literal[True],
        how: AnyAll | None = None,
        ignore_index: _bool = False,
    ) -> None: ...
    @overload
    def dropna(
        self,
        *,
        axis: AxisIndex = 0,
        inplace: Literal[False] = False,
        how: AnyAll | None = None,
        ignore_index: _bool = False,
    ) -> Series[S1]: ...
    def to_timestamp(
        self,
        freq: Any=...,
        how: ToTimestampHow = ...,
        copy: _bool = ...,
    ) -> Series[S1]: ...
    def to_period(self, freq: _str | None = None, copy: _bool = None) -> DataFrame: ...
    @property
    def str(
        self,
    ) -> StringMethods[
        Self,
        DataFrame,
        Series[bool],
        Series[list[_str]],
        Series[int],
        Series[bytes],
        Series[_str],
        Series[type[object]],
    ]: ...
    @property
    def dt(self) -> CombinedDatetimelikeProperties: ...
    @property
    def plot(self) -> PlotAccessor: ...
    sparse = ...
    def hist(
        self,
        by: object | None = None,
        ax: PlotAxes | None = None,
        grid: _bool = True,
        xlabelsize: float | _str | None = None,
        xrot: float | None = None,
        ylabelsize: float | _str | None = None,
        yrot: float | None = None,
        figsize: tuple[float, float] | None = None,
        bins: int | Sequence[Any] = 10,
        backend: _str | None = None,
        **kwargs: Any,
    ) -> SubplotBase: ...
    def swapaxes(
        self, axis1: AxisIndex, axis2: AxisIndex, copy: _bool = None
    ) -> Series[S1]: ...
    def droplevel(self, level: Level | list[Level], axis: AxisIndex = 0) -> Self: ...
    def pop(self, item: Hashable) -> S1: ...
    def squeeze(self) -> Series[S1] | Scalar: ...
    def __abs__(self) -> Series[S1]: ...
    def add_prefix(self, prefix: _str, axis: AxisIndex | None = None) -> Series[S1]: ...
    def add_suffix(self, suffix: _str, axis: AxisIndex | None = None) -> Series[S1]: ...
    def reindex(
        self,
        index: Axes | None = None,
        method: ReindexMethod | None = None,
        copy: bool = None,
        level: int | _str = None,
        fill_value: Scalar | None = None,
        limit: int | None = None,
        tolerance: float | None = None,
    ) -> Series[S1]: ...
    def filter(
        self,
        items: _ListLike[Any] | None = ...,
        like: _str | None = ...,
        regex: _str | None = ...,
        axis: AxisIndex | None = ...,
    ) -> Series[S1]: ...
    def head(self, n: int = 5) -> Series[S1]: ...
    def tail(self, n: int = 5) -> Series[S1]: ...
    def sample(
        self,
        n: int | None = ...,
        frac: float | None = ...,
        replace: _bool = ...,
        weights: _str | _ListLike[Any] | np.ndarray[Any, Any] | None = ...,
        random_state: RandomState | None = ...,
        axis: AxisIndex | None = ...,
        ignore_index: _bool = ...,
    ) -> Series[S1]: ...
    @overload
    def astype(
        self,
        dtype: BooleanDtypeArg,
        copy: _bool = None,
        errors: IgnoreRaise = 'raise',
    ) -> Series[bool]: ...
    @overload
    def astype(
        self,
        dtype: IntDtypeArg | UIntDtypeArg,
        copy: _bool = None,
        errors: IgnoreRaise = 'raise',
    ) -> Series[int]: ...
    @overload
    def astype(
        self,
        dtype: StrDtypeArg,
        copy: _bool = None,
        errors: IgnoreRaise = 'raise',
    ) -> Series[_str]: ...
    @overload
    def astype(
        self,
        dtype: BytesDtypeArg,
        copy: _bool = None,
        errors: IgnoreRaise = 'raise',
    ) -> Series[bytes]: ...
    @overload
    def astype(
        self,
        dtype: FloatDtypeArg,
        copy: _bool = None,
        errors: IgnoreRaise = 'raise',
    ) -> Series[float]: ...
    @overload
    def astype(
        self,
        dtype: ComplexDtypeArg,
        copy: _bool = None,
        errors: IgnoreRaise = 'raise',
    ) -> Series[complex]: ...
    @overload
    def astype(
        self,
        dtype: TimedeltaDtypeArg,
        copy: _bool = None,
        errors: IgnoreRaise = 'raise',
    ) -> TimedeltaSeries: ...
    @overload
    def astype(
        self,
        dtype: TimestampDtypeArg,
        copy: _bool = None,
        errors: IgnoreRaise = 'raise',
    ) -> TimestampSeries: ...
    @overload
    def astype(
        self,
        dtype: CategoryDtypeArg,
        copy: _bool = None,
        errors: IgnoreRaise = 'raise',
    ) -> Series[CategoricalDtype]: ...
    @overload
    def astype(
        self,
        dtype: ObjectDtypeArg | VoidDtypeArg | ExtensionDtype | DtypeObj,
        copy: _bool = None,
        errors: IgnoreRaise = 'raise',
    ) -> Series[Any]: ...
    def copy(self, deep: _bool = True) -> Series[S1]: ...
    def infer_objects(self) -> Series[S1]: ...
    @overload
    def ffill(
        self,
        *,
        axis: AxisIndex | None = None,
        inplace: Literal[True],
        limit: int | None = None,
        limit_area: Literal["inside", "outside"] | None = None,
    ) -> None: ...
    @overload
    def ffill(
        self,
        *,
        axis: AxisIndex | None = None,
        inplace: Literal[False] = False,
        limit: int | None = None,
        limit_area: Literal["inside", "outside"] | None = None,
    ) -> Series[S1]: ...
    @overload
    def bfill(
        self,
        *,
        axis: AxisIndex | None = None,
        inplace: Literal[True],
        limit: int | None = None,
        limit_area: Literal["inside", "outside"] | None = None,
    ) -> None: ...
    @overload
    def bfill(
        self,
        *,
        axis: AxisIndex | None = None,
        inplace: Literal[False] = False,
        limit: int | None = None,
        limit_area: Literal["inside", "outside"] | None = None,
    ) -> Series[S1]: ...
    @overload
    def interpolate(
        self,
        method: InterpolateOptions = 'linear',
        *,
        axis: AxisIndex | None = 0,
        limit: int | None = None,
        inplace: Literal[True],
        limit_direction: Literal["forward", "backward", "both"] | None = None,
        limit_area: Literal["inside", "outside"] | None = None,
        **kwargs: Any,
    ) -> None: ...
    @overload
    def interpolate(
        self,
        method: InterpolateOptions = 'linear',
        *,
        axis: AxisIndex | None = 0,
        limit: int | None = None,
        inplace: Literal[False] = False,
        limit_direction: Literal["forward", "backward", "both"] | None = None,
        limit_area: Literal["inside", "outside"] | None = None,
        **kwargs: Any,
    ) -> Series[S1]: ...
    def asof(
        self,
        where: Scalar | Sequence[Scalar],
        subset: _str | Sequence[_str] | None = None,
    ) -> Scalar | Series[S1]: ...
    @overload
    def clip(  # pyright: ignore[reportOverlappingOverload]
        self,
        lower: None = None,
        upper: None = None,
        *,
        axis: AxisIndex | None = None,
        inplace: Literal[True],
        **kwargs: Any,
    ) -> Self: ...
    @overload
    def clip(
        self,
        lower: AnyArrayLike | float | None = None,
        upper: AnyArrayLike | float | None = None,
        *,
        axis: AxisIndex | None = None,
        inplace: Literal[True],
        **kwargs: Any,
    ) -> None: ...
    @overload
    def clip(
        self,
        lower: AnyArrayLike | float | None = None,
        upper: AnyArrayLike | float | None = None,
        *,
        axis: AxisIndex | None = None,
        inplace: Literal[False] = False,
        **kwargs: Any,
    ) -> Series[S1]: ...
    def asfreq(
        self,
        freq: Any,
        method: FillnaOptions | None = ...,
        how: Literal["start", "end"] | None = ...,
        normalize: _bool = ...,
        fill_value: Scalar | None = ...,
    ) -> Series[S1]: ...
    def at_time(
        self,
        time: _str | time,
        asof: _bool = False,
        axis: AxisIndex | None = None,
    ) -> Series[S1]: ...
    def between_time(
        self,
        start_time: _str | time,
        end_time: _str | time,
        axis: AxisIndex | None = None,
    ) -> Series[S1]: ...
    def first(self, offset: _str | int | None = ...) -> Series[S1]: ...
    def last(self, offset: _str | int | None = ...) -> Series[S1]: ...
    def rank(
        self,
        axis: AxisIndex = 0,
        method: Literal["average", "min", "max", "first", "dense"] = 'average',
        numeric_only: _bool = False,
        na_option: Literal["keep", "top", "bottom"] = 'keep',
        ascending: _bool = True,
        pct: _bool = False,
    ) -> Series[float]: ...
    @overload
    def where(
        self,
        cond: (
            Series[S1]
            | Series[_bool]
            | np.ndarray[Any, Any]
            | Callable[[Series[S1]], Series[bool]]
            | Callable[[S1], bool]
        ),
        other: Any=...,
        *,
        inplace: Literal[True],
        axis: AxisIndex | None = None,
        level: Level | None = None,
    ) -> None: ...
    @overload
    def where(
        self,
        cond: (
            Series[S1]
            | Series[_bool]
            | np.ndarray[Any, Any]
            | Callable[[Series[S1]], Series[bool]]
            | Callable[[S1], bool]
        ),
        other: Any=...,
        *,
        inplace: Literal[False] = False,
        axis: AxisIndex | None = None,
        level: Level | None = None,
    ) -> Self: ...
    @overload
    def mask(
        self,
        cond: (
            Series[S1]
            | Series[_bool]
            | np.ndarray[Any, Any]
            | Callable[[Series[S1]], Series[bool]]
            | Callable[[S1], bool]
        ),
        other: Scalar | Series[S1] | DataFrame | Callable[..., Any] | NAType | None = ...,
        *,
        inplace: Literal[True],
        axis: AxisIndex | None = None,
        level: Level | None = None,
    ) -> None: ...
    @overload
    def mask(
        self,
        cond: (
            Series[S1]
            | Series[_bool]
            | np.ndarray[Any, Any]
            | Callable[[Series[S1]], Series[bool]]
            | Callable[[S1], bool]
        ),
        other: Scalar | Series[S1] | DataFrame | Callable[..., Any] | NAType | None = ...,
        *,
        inplace: Literal[False] = False,
        axis: AxisIndex | None = None,
        level: Level | None = None,
    ) -> Series[S1]: ...
    def case_when(
        self,
        caselist: list[
            tuple[
                Sequence[bool]
                | Series[bool]
                | Callable[
                    [UnknownSeries], UnknownSeries | np.ndarray[Any, Any] | Sequence[bool]
                ],
                ListLikeU
                | Scalar
                | Callable[[UnknownSeries], UnknownSeries | np.ndarray[Any, Any]],
            ],
        ],
    ) -> UnknownSeries: ...
    def truncate(
        self,
        before: date | _str | int | None = None,
        after: date | _str | int | None = None,
        axis: AxisIndex | None = None,
        copy: _bool = None,
    ) -> Series[S1]: ...
    def tz_convert(
        self,
        tz: TimeZones,
        axis: AxisIndex = 0,
        level: Level | None = None,
        copy: _bool = None,
    ) -> Series[S1]: ...
    def tz_localize(
        self,
        tz: TimeZones,
        axis: AxisIndex = 0,
        level: Level | None = None,
        copy: _bool = None,
        ambiguous: TimeAmbiguous = 'raise',
        nonexistent: _str = 'raise',
    ) -> Series[S1]: ...
    def abs(self) -> Series[S1]: ...
    def describe(
        self,
        percentiles: list[float] | None = None,
        include: Literal["all"] | list[S1] | None = None,
        exclude: S1 | list[S1] | None = None,
    ) -> Series[S1]: ...
    def pct_change(
        self,
        periods: int = 1,
        fill_method: None = None,
        freq: DateOffset | timedelta | _str | None = None,
        *,
        fill_value: Scalar | NAType | None = ...,
    ) -> Series[float]: ...
    def first_valid_index(self) -> Scalar: ...
    def last_valid_index(self) -> Scalar: ...
    @overload
    def value_counts(
        self,
        normalize: Literal[False] = False,
        sort: _bool = True,
        ascending: _bool = False,
        bins: int | None = None,
        dropna: _bool = True,
    ) -> Series[int]: ...
    @overload
    def value_counts(
        self,
        normalize: Literal[True],
        sort: _bool = True,
        ascending: _bool = False,
        bins: int | None = None,
        dropna: _bool = True,
    ) -> Series[float]: ...
    @property
    def T(self) -> Self: ...
    # The rest of these were left over from the old
    # stubs we shipped in preview. They may belong in
    # the base classes in some cases; I expect stubgen
    # just failed to generate these so I couldn't match
    # them up.
    @overload
    def __add__(self, other: S1 | Self) -> Self: ...
    @overload
    def __add__(
        self,
        other: (
            num
            | _str
            | timedelta
            | Timedelta
            | _ListLike[Any]
            | UnknownSeries
            | np.timedelta64
        ),
    ) -> UnknownSeries: ...
    # ignore needed for mypy as we want different results based on the arguments
    @overload  # type: ignore[override]
    def __and__(  # pyright: ignore[reportOverlappingOverload]
        self, other: bool | list[int] | MaskType
    ) -> Series[bool]: ...
    @overload
    def __and__(self, other: int | np_ndarray_anyint | Series[int]) -> Series[int]: ...
    # def __array__(self, dtype: Optional[_bool] = ...) -> _np_ndarray
    def __div__(self, other: num | _ListLike[Any] | Series[S1]) -> Series[S1]: ...
    def __eq__(self, other: object) -> Series[_bool]: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def __floordiv__(self, other: num | _ListLike[Any] | Series[S1]) -> Series[int]: ...
    def __ge__(  # type: ignore[override]
        self, other: S1 | _ListLike[Any] | Series[S1] | datetime | timedelta | date
    ) -> Series[_bool]: ...
    def __gt__(  # type: ignore[override]
        self, other: S1 | _ListLike[Any] | Series[S1] | datetime | timedelta | date
    ) -> Series[_bool]: ...
    def __le__(  # type: ignore[override]
        self, other: S1 | _ListLike[Any] | Series[S1] | datetime | timedelta | date
    ) -> Series[_bool]: ...
    def __lt__(  # type: ignore[override]
        self, other: S1 | _ListLike[Any] | Series[S1] | datetime | timedelta | date
    ) -> Series[_bool]: ...
    @overload
    def __mul__(
        self, other: timedelta | Timedelta | TimedeltaSeries | np.timedelta64
    ) -> TimedeltaSeries: ...
    @overload
    def __mul__(self, other: num | _ListLike[Any] | UnknownSeries) -> UnknownSeries: ...
    def __mod__(self, other: num | _ListLike[Any] | Series[S1]) -> Series[S1]: ...
    def __ne__(self, other: object) -> Series[_bool]: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def __pow__(self, other: num | _ListLike[Any] | Series[S1]) -> Series[S1]: ...
    # ignore needed for mypy as we want different results based on the arguments
    @overload  # type: ignore[override]
    def __or__(  # pyright: ignore[reportOverlappingOverload]
        self, other: bool | list[int] | MaskType
    ) -> Series[bool]: ...
    @overload
    def __or__(self, other: int | np_ndarray_anyint | Series[int]) -> Series[int]: ...
    @overload
    def __radd__(self, other: S1 | Series[S1]) -> Self: ...
    @overload
    def __radd__(
        self, other: num | _str | _ListLike[Any] | UnknownSeries
    ) -> UnknownSeries: ...
    # ignore needed for mypy as we want different results based on the arguments
    @overload  # type: ignore[override]
    def __rand__(  # pyright: ignore[reportOverlappingOverload]
        self, other: bool | MaskType | list[int]
    ) -> Series[bool]: ...
    @overload
    def __rand__(self, other: int | np_ndarray_anyint | Series[int]) -> Series[int]: ...
    def __rdiv__(self, other: num | _ListLike[Any] | Series[S1]) -> Series[S1]: ...
    def __rdivmod__(self, other: num | _ListLike[Any] | Series[S1]) -> Series[S1]: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def __rfloordiv__(self, other: num | _ListLike[Any] | Series[S1]) -> Series[S1]: ...
    def __rmod__(self, other: num | _ListLike[Any] | Series[S1]) -> Series[S1]: ...
    @overload
    def __rmul__(
        self, other: timedelta | Timedelta | TimedeltaSeries | np.timedelta64
    ) -> TimedeltaSeries: ...
    @overload
    def __rmul__(self, other: num | _ListLike[Any] | UnknownSeries) -> UnknownSeries: ...
    def __rnatmul__(self, other: num | _ListLike[Any] | Series[S1]) -> Series[S1]: ...
    def __rpow__(self, other: num | _ListLike[Any] | Series[S1]) -> Series[S1]: ...
    # ignore needed for mypy as we want different results based on the arguments
    @overload  # type: ignore[override]
    def __ror__(  # pyright: ignore[reportOverlappingOverload]
        self, other: bool | MaskType | list[int]
    ) -> Series[bool]: ...
    @overload
    def __ror__(self, other: int | np_ndarray_anyint | Series[int]) -> Series[int]: ...
    def __rsub__(self, other: num | _ListLike[Any] | Series[S1]) -> UnknownSeries: ...
    def __rtruediv__(
        self, other: num | _ListLike[Any] | Series[S1] | Path
    ) -> UnknownSeries: ...
    # ignore needed for mypy as we want different results based on the arguments
    @overload  # type: ignore[override]
    def __rxor__(  # pyright: ignore[reportOverlappingOverload]
        self, other: bool | MaskType | list[int]
    ) -> Series[bool]: ...
    @overload
    def __rxor__(self, other: int | np_ndarray_anyint | Series[int]) -> Series[int]: ...
    @overload
    def __sub__(
        self: Series[Timestamp],
        other: Timedelta | TimedeltaSeries | TimedeltaIndex | np.timedelta64,
    ) -> TimestampSeries: ...
    @overload
    def __sub__(
        self: Series[Timedelta],
        other: Timedelta | TimedeltaSeries | TimedeltaIndex | np.timedelta64,
    ) -> TimedeltaSeries: ...
    @overload
    def __sub__(
        self, other: Timestamp | datetime | TimestampSeries
    ) -> TimedeltaSeries: ...
    @overload
    def __sub__(self, other: num | _ListLike[Any] | UnknownSeries) -> UnknownSeries: ...
    def __truediv__(
        self, other: num | _ListLike[Any] | Series[S1] | Path
    ) -> UnknownSeries: ...
    # ignore needed for mypy as we want different results based on the arguments
    @overload  # type: ignore[override]
    def __xor__(  # pyright: ignore[reportOverlappingOverload]
        self, other: bool | MaskType | list[int]
    ) -> Series[bool]: ...
    @overload
    def __xor__(self, other: int | np_ndarray_anyint | Series[int]) -> Series[int]: ...
    def __invert__(self) -> Series[bool]: ...
    # properties
    # @property
    # def array(self) -> _npndarray
    @property
    def at(self) -> _AtIndexer: ...
    @property
    def cat(self) -> CategoricalAccessor: ...
    @property
    def iat(self) -> _iAtIndexer: ...
    @property
    def iloc(self) -> _iLocIndexerSeries[S1]: ...
    @property
    def loc(self) -> _LocIndexerSeries[S1]: ...
    # Methods
    def add(
        self,
        other: Series[S1] | Scalar,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[S1]: ...
    def all(
        self,
        axis: AxisIndex = 0,
        bool_only: _bool | None = False,
        skipna: _bool = True,
        **kwargs: Any,
    ) -> np.bool: ...
    def any(
        self,
        *,
        axis: AxisIndex = 0,
        bool_only: _bool | None = False,
        skipna: _bool = True,
        **kwargs: Any,
    ) -> np.bool: ...
    def cummax(
        self,
        axis: AxisIndex | None = None,
        skipna: _bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> Series[S1]: ...
    def cummin(
        self,
        axis: AxisIndex | None = None,
        skipna: _bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> Series[S1]: ...
    def cumprod(
        self,
        axis: AxisIndex | None = None,
        skipna: _bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> Series[S1]: ...
    def cumsum(
        self,
        axis: AxisIndex | None = None,
        skipna: _bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> Series[S1]: ...
    def divide(
        self,
        other: num | _ListLike[Any] | Series[S1],
        level: Level | None = ...,
        fill_value: float | None = ...,
        axis: AxisIndex = ...,
    ) -> Series[float]: ...
    def divmod(
        self,
        other: num | _ListLike[Any] | Series[S1],
        level: Level | None = ...,
        fill_value: float | None = ...,
        axis: AxisIndex = ...,
    ) -> Series[S1]: ...
    def eq(
        self,
        other: Scalar | Series[S1],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[_bool]: ...
    def ewm(
        self,
        com: float | None = None,
        span: float | None = None,
        halflife: float | None = None,
        alpha: float | None = None,
        min_periods: int = 0,
        adjust: _bool = True,
        ignore_na: _bool = False,
    ) -> ExponentialMovingWindow[UnknownSeries]: ...
    def expanding(
        self,
        min_periods: int = 1,
        method: CalculationMethod = 'single',
    ) -> Expanding[UnknownSeries]: ...
    def floordiv(
        self,
        other: num | _ListLike[Any] | Series[S1],
        level: Level | None = ...,
        fill_value: float | None = ...,
        axis: AxisIndex | None = ...,
    ) -> Series[int]: ...
    def ge(
        self,
        other: Scalar | Series[S1],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[_bool]: ...
    def gt(
        self,
        other: Scalar | Series[S1],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[_bool]: ...
    def item(self) -> S1: ...
    def kurt(
        self,
        axis: AxisIndex | None = 0,
        skipna: _bool = True,
        level: None = None,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> Scalar: ...
    def kurtosis(
        self,
        axis: AxisIndex | None = 0,
        skipna: _bool = True,
        level: None = None,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> Scalar: ...
    def le(
        self,
        other: Scalar | Series[S1],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[_bool]: ...
    def lt(
        self,
        other: Scalar | Series[S1],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[_bool]: ...
    def max(
        self,
        axis: AxisIndex | None = 0,
        skipna: _bool = True,
        level: None = None,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> S1: ...
    def mean(
        self,
        axis: AxisIndex | None = 0,
        skipna: _bool = True,
        level: None = None,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> float: ...
    def median(
        self,
        axis: AxisIndex | None = 0,
        skipna: _bool = True,
        level: None = None,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> float: ...
    def min(
        self,
        axis: AxisIndex | None = 0,
        skipna: _bool = True,
        level: None = None,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> S1: ...
    def mod(
        self,
        other: num | _ListLike[Any] | Series[S1],
        level: Level | None = ...,
        fill_value: float | None = ...,
        axis: AxisIndex | None = ...,
    ) -> Series[S1]: ...
    @overload
    def mul(
        self,
        other: timedelta | Timedelta | TimedeltaSeries | np.timedelta64,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex | None = 0,
    ) -> TimedeltaSeries: ...
    @overload
    def mul(
        self,
        other: num | _ListLike[Any] | UnknownSeries,
        level: Level | None = ...,
        fill_value: float | None = ...,
        axis: AxisIndex | None = ...,
    ) -> UnknownSeries: ...
    def multiply(
        self,
        other: num | _ListLike[Any] | Series[S1],
        level: Level | None = ...,
        fill_value: float | None = ...,
        axis: AxisIndex | None = ...,
    ) -> Series[S1]: ...
    def ne(
        self,
        other: Scalar | Series[S1],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[_bool]: ...
    def nunique(self, dropna: _bool = True) -> int: ...
    def pow(
        self,
        other: num | _ListLike[Any] | Series[S1],
        level: Level | None = ...,
        fill_value: float | None = ...,
        axis: AxisIndex | None = ...,
    ) -> Series[S1]: ...
    def prod(
        self,
        axis: AxisIndex | None = None,
        skipna: _bool | None = True,
        level: None = None,
        numeric_only: _bool = False,
        min_count: int = 0,
        **kwargs: Any,
    ) -> Scalar: ...
    def product(
        self,
        axis: AxisIndex | None = None,
        skipna: _bool | None = True,
        level: None = None,
        numeric_only: _bool = False,
        min_count: int = 0,
        **kwargs: Any,
    ) -> Scalar: ...
    def radd(
        self,
        other: Series[S1] | Scalar,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[S1]: ...
    def rdivmod(
        self,
        other: Series[S1] | Scalar,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[S1]: ...
    def rfloordiv(
        self,
        other: Any,
        level: Level | None = ...,
        fill_value: float | None = ...,
        axis: AxisIndex = ...,
    ) -> Series[S1]: ...
    def rmod(
        self,
        other: Series[S1] | Scalar,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[S1]: ...
    @overload
    def rmul(
        self,
        other: timedelta | Timedelta | TimedeltaSeries | np.timedelta64,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> TimedeltaSeries: ...
    @overload
    def rmul(
        self,
        other: num | _ListLike[Any] | UnknownSeries,
        level: Level | None = ...,
        fill_value: float | None = ...,
        axis: AxisIndex = ...,
    ) -> UnknownSeries: ...
    @overload
    def rolling(
        self,
        window: int | _str | timedelta | BaseOffset | BaseIndexer,
        min_periods: int | None = None,
        center: _bool = False,
        on: _str | None = None,
        closed: IntervalClosedType | None = None,
        step: int | None = None,
        method: CalculationMethod = 'single',
        *,
        win_type: _str,
    ) -> Window[UnknownSeries]: ...
    @overload
    def rolling(
        self,
        window: int | _str | timedelta | BaseOffset | BaseIndexer,
        min_periods: int | None = None,
        center: _bool = False,
        on: _str | None = None,
        closed: IntervalClosedType | None = None,
        step: int | None = None,
        method: CalculationMethod = 'single',
        *,
        win_type: None = None,
    ) -> Rolling[UnknownSeries]: ...
    def rpow(
        self,
        other: Series[S1] | Scalar,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[S1]: ...
    def rsub(
        self,
        other: Series[S1] | Scalar,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[S1]: ...
    def rtruediv(
        self,
        other: Any,
        level: Level | None = ...,
        fill_value: float | None = ...,
        axis: AxisIndex = ...,
    ) -> Series[S1]: ...
    def sem(
        self,
        axis: AxisIndex | None = None,
        skipna: _bool | None = True,
        level: None = None,
        ddof: int = 1,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> Scalar: ...
    def skew(
        self,
        axis: AxisIndex | None = 0,
        skipna: _bool | None = True,
        level: None = None,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> Scalar: ...
    def std(
        self,
        axis: AxisIndex | None = None,
        skipna: _bool | None = True,
        level: None = None,
        ddof: int = 1,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> float: ...
    def sub(
        self,
        other: num | _ListLike[Any] | Series[S1],
        level: Level | None = ...,
        fill_value: float | None = ...,
        axis: AxisIndex | None = ...,
    ) -> Series[S1]: ...
    def subtract(
        self,
        other: num | _ListLike[Any] | Series[S1],
        level: Level | None = ...,
        fill_value: float | None = ...,
        axis: AxisIndex | None = ...,
    ) -> Series[S1]: ...
    # ignore needed because of mypy, for using `Never` as type-var.
    @overload
    def sum(
        self: Series[Never],
        axis: AxisIndex | None = None,
        skipna: _bool | None = True,
        level: None = None,
        numeric_only: _bool = False,
        min_count: int = 0,
        **kwargs: Any,
    ) -> Any: ...
    # ignore needed because of mypy, for overlapping overloads
    # between `Series[bool]` and `Series[int]`.
    @overload
    def sum(
        self: Series[bool],
        axis: AxisIndex | None = None,
        skipna: _bool | None = True,
        level: None = None,
        numeric_only: _bool = False,
        min_count: int = 0,
        **kwargs: Any,
    ) -> int: ...
    @overload
    def sum(
        self: Series[S1],
        axis: AxisIndex | None = None,
        skipna: _bool | None = True,
        level: None = None,
        numeric_only: _bool = False,
        min_count: int = 0,
        **kwargs: Any,
    ) -> S1: ...
    def to_list(self) -> list[S1]: ...
    def to_numpy(
        self,
        dtype: npt.DTypeLike | None = None,
        copy: bool = False,
        na_value: Scalar = ...,
        **kwargs: Any,
    ) -> np.ndarray[Any, Any]: ...
    def tolist(self) -> list[S1]: ...
    def truediv(
        self,
        other: Any,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[float]: ...
    def var(
        self,
        axis: AxisIndex | None = None,
        skipna: _bool | None = True,
        level: None = None,
        ddof: int = 1,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> Scalar: ...
    # Rename axis with `mapper`, `axis`, and `inplace=True`
    @overload
    def rename_axis(
        self,
        mapper: Scalar | ListLike | None = ...,
        *,
        axis: AxisIndex | None = 0,
        copy: _bool = True,
        inplace: Literal[True],
    ) -> None: ...
    # Rename axis with `mapper`, `axis`, and `inplace=False`
    @overload
    def rename_axis(
        self,
        mapper: Scalar | ListLike | None = ...,
        *,
        axis: AxisIndex | None = 0,
        copy: _bool = True,
        inplace: Literal[False] = False,
    ) -> Self: ...
    # Rename axis with `index` and `inplace=True`
    @overload
    def rename_axis(
        self,
        *,
        index: Scalar | ListLike | Callable[..., Any] | dict[Any, Any] | None = ...,
        copy: _bool = ...,
        inplace: Literal[True],
    ) -> None: ...
    # Rename axis with `index` and `inplace=False`
    @overload
    def rename_axis(
        self,
        *,
        index: Scalar | ListLike | Callable[..., Any] | dict[Any, Any] | None = ...,
        copy: _bool = ...,
        inplace: Literal[False] = False,
    ) -> Self: ...
    def set_axis(self, labels: Any, *, axis: Axis = ..., copy: _bool = ...) -> Self: ...
    def __iter__(self) -> Iterator[S1]: ...
    def xs(
        self,
        key: Hashable,
        axis: AxisIndex = 0,
        level: Level | None = None,
        drop_level: _bool = True,
    ) -> Self: ...
    def __bool__(self) -> NoReturn: ...

class TimestampSeries(Series[Timestamp]):
    @property
    def dt(self) -> TimestampProperties: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def __add__(self, other: TimedeltaSeries | np.timedelta64 | timedelta | BaseOffset) -> TimestampSeries: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def __radd__(self, other: TimedeltaSeries | np.timedelta64 | timedelta) -> TimestampSeries: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    @overload  # type: ignore[override]
    def __sub__(
        self, other: Timestamp | datetime | TimestampSeries
    ) -> TimedeltaSeries: ...
    @overload
    def __sub__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        other: (
            timedelta | TimedeltaSeries | TimedeltaIndex | np.timedelta64 | BaseOffset
        ),
    ) -> TimestampSeries: ...
    def __mul__(self, other: float | Series[int] | Series[float] | Sequence[float]) -> TimestampSeries: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def __truediv__(self, other: float | Series[int] | Series[float] | Sequence[float]) -> TimestampSeries: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def unique(self) -> DatetimeArray: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def mean(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        axis: AxisIndex | None = ...,
        skipna: _bool = ...,
        level: None = None,
        numeric_only: _bool = ...,
        **kwargs: Any,
    ) -> Timestamp: ...
    def median(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        axis: AxisIndex | None = ...,
        skipna: _bool = ...,
        level: None = None,
        numeric_only: _bool = ...,
        **kwargs: Any,
    ) -> Timestamp: ...
    def std(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        axis: AxisIndex | None = ...,
        skipna: _bool | None = ...,
        level: None = None,
        ddof: int = ...,
        numeric_only: _bool = ...,
        **kwargs: Any,
    ) -> Timedelta: ...
    def diff(self, periods: int = ...) -> TimedeltaSeries: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]

class TimedeltaSeries(Series[Timedelta]):
    # ignores needed because of mypy
    @overload  # type: ignore[override]
    def __add__(self, other: Period) -> PeriodSeries: ...
    @overload
    def __add__(
        self, other: datetime | Timestamp | TimestampSeries | DatetimeIndex
    ) -> TimestampSeries: ...
    @overload
    def __add__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: timedelta | Timedelta | np.timedelta64
    ) -> TimedeltaSeries: ...
    def __radd__(self, other: datetime | Timestamp | TimestampSeries) -> TimestampSeries: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def __mul__(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: num | Sequence[num] | Series[int] | Series[float]
    ) -> TimedeltaSeries: ...
    def unique(self) -> TimedeltaArray: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def __sub__(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        other: (
            timedelta | Timedelta | TimedeltaSeries | TimedeltaIndex | np.timedelta64
        ),
    ) -> TimedeltaSeries: ...
    @overload  # type: ignore[override]
    def __truediv__(self, other: float | Sequence[float]) -> Self: ...
    @overload
    def __truediv__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        other: (
            timedelta
            | TimedeltaSeries
            | np.timedelta64
            | TimedeltaIndex
            | Sequence[timedelta]
        ),
    ) -> Series[float]: ...
    def __rtruediv__(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        other: (
            timedelta
            | TimedeltaSeries
            | np.timedelta64
            | TimedeltaIndex
            | Sequence[timedelta]
        ),
    ) -> Series[float]: ...
    @overload  # type: ignore[override]
    def __floordiv__(self, other: float | Sequence[float]) -> Self: ...
    @overload
    def __floordiv__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        other: (
            timedelta
            | TimedeltaSeries
            | np.timedelta64
            | TimedeltaIndex
            | Sequence[timedelta]
        ),
    ) -> Series[int]: ...
    def __rfloordiv__(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        other: (
            timedelta
            | TimedeltaSeries
            | np.timedelta64
            | TimedeltaIndex
            | Sequence[timedelta]
        ),
    ) -> Series[int]: ...
    @property
    def dt(self) -> TimedeltaProperties: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def mean(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        axis: AxisIndex | None = ...,
        skipna: _bool = ...,
        level: None = None,
        numeric_only: _bool = ...,
        **kwargs: Any,
    ) -> Timedelta: ...
    def median(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        axis: AxisIndex | None = ...,
        skipna: _bool = ...,
        level: None = None,
        numeric_only: _bool = ...,
        **kwargs: Any,
    ) -> Timedelta: ...
    def std(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        axis: AxisIndex | None = ...,
        skipna: _bool | None = ...,
        level: None = None,
        ddof: int = ...,
        numeric_only: _bool = ...,
        **kwargs: Any,
    ) -> Timedelta: ...
    def diff(self, periods: int = ...) -> TimedeltaSeries: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def cumsum(
        self,
        axis: AxisIndex | None = ...,
        skipna: _bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> TimedeltaSeries: ...

class PeriodSeries(Series[Period]):
    @property
    def dt(self) -> PeriodProperties: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def __sub__(self, other: PeriodSeries) -> OffsetSeries: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def diff(self, periods: int = ...) -> OffsetSeries: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]

class OffsetSeries(Series[BaseOffset]):
    @overload  # type: ignore[override]
    def __radd__(self, other: Period) -> PeriodSeries: ...
    @overload
    def __radd__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: BaseOffset
    ) -> OffsetSeries: ...

class IntervalSeries(Series[Interval[_OrderableT]], Generic[_OrderableT]):
    @property
    def array(self) -> IntervalArray: ...
    def diff(self, periods: int = ...) -> Never: ...

UnknownSeries: TypeAlias = Series[Any]
