from builtins import bool as _bool, str as _str
from collections import defaultdict, OrderedDict
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, MutableMapping, Sequence
from matplotlib.axes import Axes as PlotAxes
from pandas import Period, Timedelta, Timestamp
from pandas._libs.lib import _NoDefaultDoNotUse
from pandas._libs.missing import NAType
from pandas._libs.tslibs import BaseOffset
from pandas._libs.tslibs.nattype import NaTType
from pandas._libs.tslibs.offsets import DateOffset
from pandas._typing import (
	AggFuncTypeBase, AggFuncTypeDictFrame, AggFuncTypeDictSeries, AggFuncTypeFrame, AlignJoin, AnyAll, AnyArrayLike,
	ArrayLike, AstypeArg, Axes, Axis, AxisColumn, AxisIndex, CalculationMethod, ColspaceArgType, CompressionOptions,
	DropKeep, Dtype, FilePath, FillnaOptions, FloatFormatType, FormattersType, GroupByObjectNonScalar, HashableT,
	HashableT1, HashableT2, HashableT3, IgnoreRaise, IndexingInt, IndexKeyFunc, IndexLabel, IndexType, InterpolateOptions,
	IntervalClosedType, IntervalT, IntoColumn, JoinValidate, JsonFrameOrient, JSONSerializable, Label, Level, ListLike,
	ListLikeExceptSeriesAndStr, ListLikeU, MaskType, MergeHow, MergeValidate, NaPosition, NDFrameT, npt,
	NsmallestNlargestKeep, num, ParquetEngine, QuantileInterpolation, RandomState, ReadBuffer, ReindexMethod, Renamer,
	ReplaceValue, S2, Scalar, ScalarT, SequenceNotStr, SeriesByT, SortKind, StataDateFormat, StorageOptions, StrDtypeArg,
	StrLike, Suffixes, T as _T, TimeAmbiguous, TimeNonexistent, TimeUnit, TimeZones, ToStataByteorder, ToTimestampHow,
	UpdateJoin, ValueKeyFunc, WriteBuffer, XMLParsers)
from pandas.core.arraylike import OpsMixin
from pandas.core.generic import NDFrame
from pandas.core.groupby.generic import DataFrameGroupBy
from pandas.core.indexers import BaseIndexer
from pandas.core.indexes.base import Index, UnknownIndex
from pandas.core.indexes.category import CategoricalIndex
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.interval import IntervalIndex
from pandas.core.indexes.multi import MultiIndex
from pandas.core.indexes.period import PeriodIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex
from pandas.core.indexing import _AtIndexer, _iAtIndexer, _iLocIndexer, _IndexSliceTuple, _LocIndexer
from pandas.core.interchange.dataframe_protocol import DataFrame as DataFrameXchg
from pandas.core.reshape.pivot import _PivotTableColumnsTypes, _PivotTableIndexTypes, _PivotTableValuesTypes
from pandas.core.series import Series
from pandas.core.window import Expanding, ExponentialMovingWindow
from pandas.core.window.rolling import Rolling, Window
from pandas.io.formats.style import Styler
from pandas.plotting import PlotAccessor
from typing import Any, ClassVar, final, Generic, Literal, NoReturn, overload, TypeVar
from typing_extensions import Never, Self, TypeAlias
import datetime as dt
import numpy as np
import sys
import xarray as xr

_T_MUTABLE_MAPPING = TypeVar("_T_MUTABLE_MAPPING", bound=MutableMapping[Any, Any], covariant=True)

class _iLocIndexerFrame(_iLocIndexer, Generic[_T]):
    @overload
    def __getitem__(self, idx: tuple[int, int]) -> Scalar: ...
    @overload
    def __getitem__(self, idx: IndexingInt) -> Series: ...
    @overload
    def __getitem__(self, idx: tuple[IndexType | MaskType, int]) -> Series: ...
    @overload
    def __getitem__(self, idx: tuple[int, IndexType | MaskType]) -> Series: ...
    @overload
    def __getitem__(
        self,
        idx: (
            IndexType
            | MaskType
            | tuple[IndexType | MaskType, IndexType | MaskType]
            | tuple[slice]
        ),
    ) -> _T: ...
    def __setitem__(
        self,
        idx: (
            int
            | IndexType
            | tuple[int, int]
            | tuple[IndexType, int]
            | tuple[IndexType, IndexType]
            | tuple[int, IndexType]
        ),
        value: (
            Scalar
            | Series
            | DataFrame
            | np.ndarray[Any, Any]
            | NAType
            | NaTType
            | Mapping[Hashable, Scalar | NAType | NaTType]
            | None
        ),
    ) -> None: ...

class _LocIndexerFrame(_LocIndexer, Generic[_T]):
    @overload
    def __getitem__(self, idx: Scalar) -> Series | _T: ...
    @overload
    def __getitem__(  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
        self,
        idx: (
            IndexType
            | MaskType
            | Callable[[DataFrame], IndexType | MaskType | Sequence[Hashable]]
            | list[HashableT]
            | tuple[IndexType
                | MaskType
                | list[HashableT]
                | slice
                | _IndexSliceTuple[Any]
                | Callable[..., Any],
                MaskType | list[HashableT] | IndexType | Callable[..., Any],
            ]
        ),
    ) -> _T: ...
    @overload
    def __getitem__(  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
        self,
        idx: tuple[int
            | StrLike
            | Timestamp
            | tuple[Scalar, ...]
            | Callable[[DataFrame], ScalarT],
            int | StrLike | tuple[Scalar, ...],
        ],
    ) -> Scalar: ...
    @overload
    def __getitem__(
        self,
        idx: (
            Callable[[DataFrame], ScalarT]
            | tuple[IndexType
                | MaskType
                | _IndexSliceTuple[Any]
                | SequenceNotStr[float | str | Timestamp]
                | Callable[[DataFrame], ScalarT | list[HashableT] | IndexType | MaskType],
                ScalarT | None,
            ]
            | None
        ),
    ) -> Series: ...
    @overload
    def __getitem__(self, idx: tuple[Scalar, slice]) -> Series | _T: ...
    @overload
    def __setitem__(
        self,
        idx: (
            MaskType | StrLike | _IndexSliceTuple[Any] | list[ScalarT] | IndexingInt | slice
        ),
        value: (
            Scalar
            | NAType
            | NaTType
            | ArrayLike
            | Series
            | DataFrame
            | list[Any]
            | Mapping[Hashable, Scalar | NAType | NaTType]
            | None
        ),
    ) -> None: ...
    @overload
    def __setitem__(
        self,
        idx: tuple[_IndexSliceTuple[Any], Hashable],
        value: Scalar | NAType | NaTType | ArrayLike | Series | list[Any] | dict[Any, Any] | None,
    ) -> None: ...

class _iAtIndexerFrame(_iAtIndexer):
    def __getitem__(self, idx: tuple[int, int]) -> Scalar: ...
    def __setitem__(
        self,
        idx: tuple[int, int],
        value: Scalar | NAType | NaTType | None,
    ) -> None: ...

class _AtIndexerFrame(_AtIndexer):
    def __getitem__(
        self,
        idx: tuple[int
            | StrLike
            | Timestamp
            | tuple[Scalar, ...]
            | Callable[[DataFrame], ScalarT],
            int | StrLike | tuple[Scalar, ...],
        ],
    ) -> Scalar: ...
    def __setitem__(
        self,
        idx: (
            MaskType | StrLike | _IndexSliceTuple[Any] | list[ScalarT] | IndexingInt | slice
        ),
        value: (
            Scalar
            | NAType
            | NaTType
            | ArrayLike
            | Series
            | DataFrame
            | list[Any]
            | Mapping[Hashable, Scalar | NAType | NaTType]
            | None
        ),
    ) -> None: ...

# With mypy 1.14.1 and python 3.12, the second overload needs a type-ignore statement
if sys.version_info >= (3, 12):
    class _GetItemHack:
        @overload
        def __getitem__(self, key: Scalar | tuple[Hashable, ...]) -> Series: ...  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
        @overload
        def __getitem__(  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
            self, key: Iterable[Hashable] | slice
        ) -> Self: ...
        @overload
        def __getitem__(self, key: Hashable) -> Series: ...

else:
    class _GetItemHack:
        @overload
        def __getitem__(self, key: Scalar | tuple[Hashable, ...]) -> Series: ...  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
        @overload
        def __getitem__(  # pyright: ignore[reportOverlappingOverload]
            self, key: Iterable[Hashable] | slice
        ) -> Self: ...
        @overload
        def __getitem__(self, key: Hashable) -> Series: ...

_AstypeArgExt: TypeAlias = (
    AstypeArg
    | Literal["number",
        "datetime64",
        "datetime",
        "integer",
        "timedelta",
        "timedelta64",
        "datetimetz",
        "datetime64[ns]",
    ]
)
_AstypeArgExtList: TypeAlias = _AstypeArgExt | list[_AstypeArgExt]

class DataFrame(NDFrame, OpsMixin, _GetItemHack):

    __hash__: ClassVar[None]  # type: ignore[assignment] # pyright: ignore[reportIncompatibleMethodOverride]

    @overload
    def __new__(
        cls,
        data: (
            ListLikeU
            | DataFrame
            | dict[Any, Any]
            | Iterable[ListLikeU | tuple[Hashable, ListLikeU] | dict[Any, Any]]
            | None
        ) = ...,
        index: Axes | None = ...,
        columns: Axes | None = ...,
        dtype: Any=...,
        copy: _bool = ...,
    ) -> Self: ...
    @overload
    def __new__(
        cls,
        data: Scalar,
        index: Axes,
        columns: Axes,
        dtype: Any=...,
        copy: _bool = ...,
    ) -> Self: ...
    def __dataframe__(
        self, nan_as_null: bool = False, allow_copy: bool = True
    ) -> DataFrameXchg: ...
    def __arrow_c_stream__(self, requested_schema: object | None = None) -> object: ...
    @property
    def axes(self) -> list[Index[Any]]: ...
    @property
    def shape(self) -> tuple[int, int]: ...
    @property
    def style(self) -> Styler: ...
    def items(self) -> Iterator[tuple[Hashable, Series]]: ...
    def iterrows(self) -> Iterator[tuple[Hashable, Series]]: ...
    @overload
    def itertuples(
        self, index: _bool = True, name: _str = 'Pandas'
    ) -> Iterator[_PandasNamedTuple]: ...
    @overload
    def itertuples(
        self, index: _bool = True, name: None = None
    ) -> Iterator[tuple[Any, ...]]: ...
    def __len__(self) -> int: ...
    @overload
    def dot(self, other: DataFrame | ArrayLike) -> Self: ...
    @overload
    def dot(self, other: Series) -> Series: ...
    @overload
    def __matmul__(self, other: DataFrame) -> Self: ...
    @overload
    def __matmul__(self, other: Series) -> Series: ...
    @overload
    def __matmul__(self, other: np.ndarray[Any, Any]) -> Self: ...
    def __rmatmul__(self, other: Any) -> Any: ...
    @overload
    @classmethod
    def from_dict(
        cls,
        data: dict[Any, Any],
        orient: Literal["index"],
        dtype: AstypeArg | None = None,
        columns: Axes | None = None,
    ) -> Self: ...
    @overload
    @classmethod
    def from_dict(
        cls,
        data: dict[Any, Any],
        orient: Literal["columns", "tight"] = 'columns',
        dtype: AstypeArg | None = None,
    ) -> Self: ...
    def to_numpy(
        self,
        dtype: npt.DTypeLike | None = None,
        copy: bool = False,
        na_value: Scalar = ...,
    ) -> np.ndarray[Any, Any]: ...
    @overload
    def to_dict(
        self,
        orient: str = 'dict',
        *,
        into: type[defaultdict[Any, Any]],
        index: Literal[True] = True,
    ) -> Never: ...
    @overload
    def to_dict(
        self,
        orient: Literal["records"],
        *,
        into: _T_MUTABLE_MAPPING | type[_T_MUTABLE_MAPPING],
        index: Literal[True] = True,
    ) -> list[_T_MUTABLE_MAPPING]: ...
    @overload
    def to_dict(
        self,
        orient: Literal["records"],
        *,
        into: type[dict[Any, Any]] = ...,
        index: Literal[True] = True,
    ) -> list[dict[Hashable, Any]]: ...
    @overload
    def to_dict(
        self,
        orient: Literal["index"],
        *,
        into: defaultdict[Any, Any],
        index: Literal[True] = True,
    ) -> defaultdict[Hashable, dict[Hashable, Any]]: ...
    @overload
    def to_dict(
        self,
        orient: Literal["index"],
        *,
        into: OrderedDict[Any, Any] | type[OrderedDict[Any, Any]],
        index: Literal[True] = True,
    ) -> OrderedDict[Hashable, dict[Hashable, Any]]: ...
    @overload
    def to_dict(
        self,
        orient: Literal["index"],
        *,
        into: type[MutableMapping[Any, Any]],
        index: Literal[True] = True,
    ) -> MutableMapping[Hashable, dict[Hashable, Any]]: ...
    @overload
    def to_dict(
        self,
        orient: Literal["index"],
        *,
        into: type[dict[Any, Any]] = ...,
        index: Literal[True] = True,
    ) -> dict[Hashable, dict[Hashable, Any]]: ...
    @overload
    def to_dict(
        self,
        orient: Literal["dict", "list", "series"] = 'dict',
        *,
        into: _T_MUTABLE_MAPPING | type[_T_MUTABLE_MAPPING],
        index: Literal[True] = True,
    ) -> _T_MUTABLE_MAPPING: ...
    @overload
    def to_dict(
        self,
        orient: Literal["dict", "list", "series"] = 'dict',
        *,
        into: type[dict[Any, Any]] = ...,
        index: Literal[True] = True,
    ) -> dict[Hashable, Any]: ...
    @overload
    def to_dict(
        self,
        orient: Literal["split", "tight"],
        *,
        into: MutableMapping[Any, Any] | type[MutableMapping[Any, Any]],
        index: bool = True,
    ) -> MutableMapping[str, list[Any]]: ...
    @overload
    def to_dict(
        self,
        orient: Literal["split", "tight"],
        *,
        into: type[dict[Any, Any]] = ...,
        index: bool = True,
    ) -> dict[str, list[Any]]: ...
    @classmethod
    def from_records(
        cls,
        data: Any,
        index: Any=None,
        exclude: SequenceNotStr[str] | None = None,
        columns: SequenceNotStr[str] | None = None,
        coerce_float: bool = False,
        nrows: int | None = None,
    ) -> Self: ...
    def to_records(
        self,
        index: _bool = True,
        column_dtypes: (
            _str | npt.DTypeLike | Mapping[HashableT1, npt.DTypeLike] | None
        ) = None,
        index_dtypes: (
            _str | npt.DTypeLike | Mapping[HashableT2, npt.DTypeLike] | None
        ) = None,
    ) -> np.recarray[Any, Any]: ...
    @overload
    def to_stata(
        self,
        path: FilePath | WriteBuffer[bytes],
        *,
        convert_dates: dict[HashableT1, StataDateFormat] | None = None,
        write_index: _bool = True,
        byteorder: ToStataByteorder | None = None,
        time_stamp: dt.datetime | None = None,
        data_label: _str | None = None,
        variable_labels: dict[HashableT2, str] | None = None,
        version: Literal[117, 118, 119],
        convert_strl: SequenceNotStr[Hashable] | None = None,
        compression: CompressionOptions = 'infer',
        storage_options: StorageOptions = None,
        value_labels: dict[Hashable, dict[float, str]] | None = None,
    ) -> None: ...
    @overload
    def to_stata(
        self,
        path: FilePath | WriteBuffer[bytes],
        *,
        convert_dates: dict[HashableT1, StataDateFormat] | None = None,
        write_index: _bool = True,
        byteorder: Literal["<", ">", "little", "big"] | None = None,
        time_stamp: dt.datetime | None = None,
        data_label: _str | None = None,
        variable_labels: dict[HashableT2, str] | None = None,
        version: Literal[114, 117, 118, 119] | None = 114,
        convert_strl: None = None,
        compression: CompressionOptions = 'infer',
        storage_options: StorageOptions = None,
        value_labels: dict[Hashable, dict[float, str]] | None = None,
    ) -> None: ...
    def to_feather(
        self, path: FilePath | WriteBuffer[bytes], **kwargs: Any
    ) -> None: ...
    @overload
    def to_parquet(
        self,
        path: FilePath | WriteBuffer[bytes],
        *,
        engine: ParquetEngine = 'auto',
        compression: Literal["snappy", "gzip", "brotli", "lz4", "zstd"] | None = 'snappy',
        index: bool | None = None,
        partition_cols: Sequence[Hashable] | None = None,
        storage_options: StorageOptions = None,
        **kwargs: Any,
    ) -> None: ...
    @overload
    def to_parquet(
        self,
        path: None = None,
        *,
        engine: ParquetEngine = 'auto',
        compression: Literal["snappy", "gzip", "brotli", "lz4", "zstd"] | None = 'snappy',
        index: bool | None = None,
        partition_cols: Sequence[Hashable] | None = None,
        storage_options: StorageOptions = None,
        **kwargs: Any,
    ) -> bytes: ...
    @overload
    def to_orc(
        self,
        path: FilePath | WriteBuffer[bytes],
        *,
        engine: Literal["pyarrow"] = "pyarrow",
        index: bool | None = None,
        engine_kwargs: dict[str, Any] | None = None,
    ) -> None: ...
    @overload
    def to_orc(
        self,
        path: None = None,
        *,
        engine: Literal["pyarrow"] = "pyarrow",
        index: bool | None = None,
        engine_kwargs: dict[str, Any] | None = None,
    ) -> bytes: ...
    @overload
    def to_html(
        self,
        buf: FilePath | WriteBuffer[str],
        *,
        columns: SequenceNotStr[Hashable] | Index[Any] | Series | None = None,
        col_space: ColspaceArgType | None = None,
        header: _bool = True,
        index: _bool = True,
        na_rep: _str = 'NaN',
        formatters: (
            list[Callable[[object], str]]
            | tuple[Callable[[object], str], ...]
            | Mapping[Hashable, Callable[[object], str]]
            | None
        ) = None,
        float_format: Callable[[float], str] | None = None,
        sparsify: _bool | None = None,
        index_names: _bool = True,
        justify: (
            Literal["left",
                "right",
                "center",
                "justify",
                "justify-all",
                "start",
                "end",
                "inherit",
                "match-parent",
                "initial",
                "unset",
            ]
            | None
        ) = None,
        max_rows: int | None = None,
        max_cols: int | None = None,
        show_dimensions: _bool = False,
        decimal: _str = '.',
        bold_rows: _bool = True,
        classes: Sequence[str] | None = None,
        escape: _bool = True,
        notebook: _bool = False,
        border: int | None = None,
        table_id: _str | None = None,
        render_links: _bool = False,
        encoding: _str | None = None,
    ) -> None: ...
    @overload
    def to_html(
        self,
        buf: None = None,
        *,
        columns: Sequence[Hashable] | None = None,
        col_space: ColspaceArgType | None = None,
        header: _bool = True,
        index: _bool = True,
        na_rep: _str = 'NaN',
        formatters: (
            list[Callable[[object], str]]
            | tuple[Callable[[object], str], ...]
            | Mapping[Hashable, Callable[[object], str]]
            | None
        ) = None,
        float_format: Callable[[float], str] | None = None,
        sparsify: _bool | None = None,
        index_names: _bool = True,
        justify: (
            Literal["left",
                "right",
                "center",
                "justify",
                "justify-all",
                "start",
                "end",
                "inherit",
                "match-parent",
                "initial",
                "unset",
            ]
            | None
        ) = None,
        max_rows: int | None = None,
        max_cols: int | None = None,
        show_dimensions: _bool = False,
        decimal: _str = '.',
        bold_rows: _bool = True,
        classes: Sequence[str] | None = None,
        escape: _bool = True,
        notebook: _bool = False,
        border: int | None = None,
        table_id: _str | None = None,
        render_links: _bool = False,
        encoding: _str | None = None,
    ) -> _str: ...
    @overload
    def to_xml(
        self,
        path_or_buffer: FilePath | WriteBuffer[bytes] | WriteBuffer[str],
        index: bool = True,
        root_name: str = 'data',
        row_name: str = 'row',
        na_rep: str | None = None,
        attr_cols: SequenceNotStr[Hashable] | None = None,
        elem_cols: SequenceNotStr[Hashable] | None = None,
        namespaces: dict[str | None, str] | None = None,
        prefix: str | None = None,
        encoding: str = 'utf-8',
        xml_declaration: bool = True,
        pretty_print: bool = True,
        parser: XMLParsers = 'lxml',
        stylesheet: FilePath | ReadBuffer[str] | ReadBuffer[bytes] | None = None,
        compression: CompressionOptions = 'infer',
        storage_options: StorageOptions = None,
    ) -> None: ...
    @overload
    def to_xml(
        self,
        path_or_buffer: Literal[None] = None,
        index: bool = True,
        root_name: str | None = 'data',
        row_name: str | None = 'row',
        na_rep: str | None = None,
        attr_cols: list[Hashable] | None = None,
        elem_cols: list[Hashable] | None = None,
        namespaces: dict[str | None, str] | None = None,
        prefix: str | None = None,
        encoding: str = 'utf-8',
        xml_declaration: bool | None = True,
        pretty_print: bool | None = True,
        parser: str | None = 'lxml',
        stylesheet: FilePath | ReadBuffer[str] | ReadBuffer[bytes] | None = None,
        compression: CompressionOptions = 'infer',
        storage_options: StorageOptions = None,
    ) -> str: ...
    def info(
        self,
        verbose: bool | None = None,
        buf: WriteBuffer[str] | None = None,
        max_cols: int | None = None,
        memory_usage: bool | Literal["deep"] | None = None,
        show_counts: bool | None = None,
    ) -> None: ...
    def memory_usage(self, index: _bool = True, deep: _bool = False) -> Series: ...
    def transpose(self, *args: Any, copy: _bool = False) -> Self: ...
    @property
    def T(self) -> Self: ...
    @final
    def __getattr__(self, name: str) -> Series: ...
    def isetitem(
        self, loc: int | Sequence[int], value: Scalar | ArrayLike | list[Any]
    ) -> None: ...
    def __setitem__(self, key: Any, value: Any) -> None: ...
    @overload
    def query(
        self,
        expr: _str,
        *,
        parser: Literal["pandas", "python"] = ...,
        engine: Literal["python", "numexpr"] | None = ...,
        local_dict: dict[_str, Any] | None = ...,
        global_dict: dict[_str, Any] | None = ...,
        resolvers: list[Mapping[Any, Any]] | None = ...,
        level: int = ...,
        target: object | None = ...,
        inplace: Literal[True],
    ) -> None: ...
    @overload
    def query(
        self,
        expr: _str,
        *,
        inplace: Literal[False] = False,
        parser: Literal["pandas", "python"] = ...,
        engine: Literal["python", "numexpr"] | None = ...,
        local_dict: dict[_str, Any] | None = ...,
        global_dict: dict[_str, Any] | None = ...,
        resolvers: list[Mapping[Any, Any]] | None = ...,
        level: int = ...,
        target: object | None = ...,
    ) -> Self: ...
    @overload
    def eval(self, expr: _str, *, inplace: Literal[True], **kwargs: Any) -> None: ...
    @overload
    def eval(
        self, expr: _str, *, inplace: Literal[False] = False, **kwargs: Any
    ) -> Scalar | np.ndarray[Any, Any] | Self | Series: ...
    @overload
    def select_dtypes(
        self, include: StrDtypeArg, exclude: _AstypeArgExtList | None = None
    ) -> Never: ...
    @overload
    def select_dtypes(
        self, include: _AstypeArgExtList | None, exclude: StrDtypeArg
    ) -> Never: ...
    @overload
    def select_dtypes(self, exclude: StrDtypeArg) -> Never: ...
    @overload
    def select_dtypes(self, include: list[Never], exclude: list[Never]) -> Never: ...
    @overload
    def select_dtypes(
        self,
        include: _AstypeArgExtList,
        exclude: _AstypeArgExtList | None = None,
    ) -> Self: ...
    @overload
    def select_dtypes(
        self,
        include: _AstypeArgExtList | None,
        exclude: _AstypeArgExtList,
    ) -> Self: ...
    @overload
    def select_dtypes(
        self,
        exclude: _AstypeArgExtList,
    ) -> Self: ...
    def insert(
        self,
        loc: int,
        column: Hashable,
        value: Scalar | ListLikeU | None,
        allow_duplicates: _bool = ...,
    ) -> None: ...
    def assign(self, **kwargs: IntoColumn) -> Self: ...
    @final
    def align(
        self,
        other: NDFrameT,
        join: AlignJoin = "outer",
        axis: Axis | None = None,
        level: Level | None = None,
        copy: _bool = True,
        fill_value: Scalar | NAType | None = None,
    ) -> tuple[Self, NDFrameT]: ...
    def reindex(
        self,
        labels: Axes | None = None,
        *,
        index: Axes | None = None,
        columns: Axes | None = None,
        axis: Axis | None = None,
        method: ReindexMethod | None = None,
        copy: bool = True,
        level: int | _str | None = None,
        fill_value: Scalar | None = ...,
        limit: int | None = None,
        tolerance: float | Timedelta | None = None,
    ) -> Self: ...
    @overload
    def rename(
        self,
        mapper: Renamer | None = None,
        *,
        index: Renamer | None = None,
        columns: Renamer | None = None,
        axis: Axis | None = None,
        copy: bool | None = None,
        inplace: Literal[True],
        level: Level | None = None,
        errors: IgnoreRaise = 'ignore',
    ) -> None: ...
    @overload
    def rename(
        self,
        mapper: Renamer | None = None,
        *,
        index: Renamer | None = None,
        columns: Renamer | None = None,
        axis: Axis | None = None,
        copy: bool | None = None,
        inplace: Literal[False] = False,
        level: Level | None = None,
        errors: IgnoreRaise = 'ignore',
    ) -> Self: ...
    @overload
    def fillna(
        self,
        value: Scalar | NAType | dict[Any, Any] | Series | DataFrame | None = None,
        *,
        axis: Axis | None = None,
        limit: int| None = None,
        inplace: Literal[True],
    ) -> None: ...
    @overload
    def fillna(
        self,
        value: Scalar | NAType | dict[Any, Any] | Series | DataFrame | None = None,
        *,
        axis: Axis | None = None,
        limit: int| None = None,
        inplace: Literal[False] = False,
    ) -> Self: ...
    @overload
    def replace(
        self,
        to_replace: ReplaceValue[Any, Any] | Mapping[HashableT2, ReplaceValue[Any, Any]] = None,
        value: ReplaceValue[Any, Any] | Mapping[HashableT3, ReplaceValue[Any, Any]] = ...,
        *,
        inplace: Literal[True],
        regex: ReplaceValue[Any, Any] | Mapping[HashableT3, ReplaceValue[Any, Any]] = False,
    ) -> None: ...
    @overload
    def replace(
        self,
        to_replace: ReplaceValue[Any, Any] | Mapping[HashableT2, ReplaceValue[Any, Any]] = None,
        value: ReplaceValue[Any, Any] | Mapping[HashableT3, ReplaceValue[Any, Any]] = ...,
        *,
        inplace: Literal[False] = False,
        regex: ReplaceValue[Any, Any] | Mapping[HashableT3, ReplaceValue[Any, Any]] = False,
    ) -> Self: ...
    def shift(
        self,
        periods: int | Sequence[int] = 1,
        freq: BaseOffset | dt.timedelta | _str | None = None,
        axis: Axis | None = None,
        fill_value: Scalar | NAType | None = ...,
    ) -> Self: ...
    @overload
    def set_index(
        self,
        keys: (
            Label
            | Series
            | Index[Any]
            | np.ndarray[Any, Any]
            | Iterator[Hashable]
            | Sequence[Hashable]
        ),
        *,
        drop: _bool = True,
        append: _bool = False,
        verify_integrity: _bool = False,
        inplace: Literal[True],
    ) -> None: ...
    @overload
    def set_index(
        self,
        keys: (
            Label
            | Series
            | Index[Any]
            | np.ndarray[Any, Any]
            | Iterator[Hashable]
            | Sequence[Hashable]
        ),
        *,
        drop: _bool = True,
        append: _bool = False,
        verify_integrity: _bool = False,
        inplace: Literal[False] = False,
    ) -> Self: ...
    @overload
    def reset_index(
        self,
        level: Level | Sequence[Level] = None,
        *,
        drop: _bool = False,
        col_level: int | _str = 0,
        col_fill: Hashable = '',
        inplace: Literal[True],
        allow_duplicates: _bool = ...,
        names: Hashable | Sequence[Hashable] = None,
    ) -> None: ...
    @overload
    def reset_index(
        self,
        level: Level | Sequence[Level] = None,
        *,
        col_level: int | _str = 0,
        col_fill: Hashable = '',
        drop: _bool = False,
        inplace: Literal[False] = False,
        allow_duplicates: _bool = ...,
        names: Hashable | Sequence[Hashable] = None,
    ) -> Self: ...
    def isna(self) -> Self: ...
    def isnull(self) -> Self: ...
    def notna(self) -> Self: ...
    def notnull(self) -> Self: ...
    @overload
    def dropna(
        self,
        *,
        axis: Axis = 0,
        how: AnyAll = ...,
        thresh: int | None = ...,
        subset: ListLikeU | Scalar | None = None,
        inplace: Literal[True],
        ignore_index: _bool = False,
    ) -> None: ...
    @overload
    def dropna(
        self,
        *,
        axis: Axis = 0,
        how: AnyAll = ...,
        thresh: int | None = ...,
        subset: ListLikeU | Scalar | None = None,
        inplace: Literal[False] = False,
        ignore_index: _bool = False,
    ) -> Self: ...
    @overload
    def drop_duplicates(
        self,
        subset: Hashable | Iterable[Hashable] | None = None,
        *,
        keep: DropKeep = 'first',
        inplace: Literal[True],
        ignore_index: _bool = False,
    ) -> None: ...
    @overload
    def drop_duplicates(
        self,
        subset: Hashable | Iterable[Hashable] | None = None,
        *,
        keep: DropKeep = 'first',
        inplace: Literal[False] = False,
        ignore_index: _bool = False,
    ) -> Self: ...
    def duplicated(
        self,
        subset: Hashable | Iterable[Hashable] | None = None,
        keep: DropKeep = "first",
    ) -> Series: ...
    @overload
    def sort_values(
        self,
        by: _str | Sequence[_str],
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
        by: _str | Sequence[_str],
        *,
        axis: Axis = 0,
        ascending: _bool | Sequence[_bool] = True,
        kind: SortKind = 'quicksort',
        na_position: NaPosition = 'last',
        ignore_index: _bool = False,
        inplace: Literal[False] = False,
        key: ValueKeyFunc = None,
    ) -> Self: ...
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
    ) -> Self: ...
    @overload
    def value_counts(
        self,
        subset: Sequence[Hashable] | None = None,
        normalize: Literal[False] = False,
        sort: _bool = True,
        ascending: _bool = False,
        dropna: _bool = True,
    ) -> Series[int]: ...
    @overload
    def value_counts(
        self,
        normalize: Literal[True],
        subset: Sequence[Hashable] | None = None,
        sort: _bool = True,
        ascending: _bool = False,
        dropna: _bool = True,
    ) -> Series[float]: ...
    def nlargest(
        self,
        n: int,
        columns: _str | list[_str],
        keep: NsmallestNlargestKeep = "first",
    ) -> Self: ...
    def nsmallest(
        self,
        n: int,
        columns: _str | list[_str],
        keep: NsmallestNlargestKeep = "first",
    ) -> Self: ...
    def swaplevel(self, i: Level = -2, j: Level = -1, axis: Axis = 0) -> Self: ...
    def reorder_levels(self, order: list[Any], axis: Axis = 0) -> Self: ...
    def compare(
        self,
        other: DataFrame,
        align_axis: Axis = 1,
        keep_shape: bool = False,
        keep_equal: bool = False,
        result_names: Suffixes = ('self', 'other'),
    ) -> Self: ...
    def combine(
        self,
        other: DataFrame,
        func: Callable[..., Any],
        fill_value: Scalar | None = None,
        overwrite: _bool = True,
    ) -> Self: ...
    def combine_first(self, other: DataFrame) -> Self: ...
    def update(
        self,
        other: DataFrame | Series,
        join: UpdateJoin = "left",
        overwrite: _bool = True,
        filter_func: Callable[..., Any] | None = None,
        errors: IgnoreRaise = "ignore",
    ) -> None: ...
    @overload
    def groupby(  # pyright: ignore reportOverlappingOverload
        self,
        by: Scalar,
        level: IndexLabel | None = None,
        as_index: Literal[True] = True,
        sort: _bool = True,
        group_keys: _bool = True,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = True,
    ) -> DataFrameGroupBy[Scalar, Literal[True]]: ...
    @overload
    def groupby(
        self,
        by: Scalar,
        level: IndexLabel | None = None,
        as_index: Literal[False] = False,
        sort: _bool = True,
        group_keys: _bool = True,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = True,
    ) -> DataFrameGroupBy[Scalar, Literal[False]]: ...
    @overload
    def groupby(  # pyright: ignore reportOverlappingOverload
        self,
        by: DatetimeIndex,
        level: IndexLabel | None = None,
        as_index: Literal[True] = True,
        sort: _bool = True,
        group_keys: _bool = True,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = True,
    ) -> DataFrameGroupBy[Timestamp, Literal[True]]: ...
    @overload
    def groupby(  # pyright: ignore reportOverlappingOverload
        self,
        by: DatetimeIndex,
        level: IndexLabel | None = None,
        as_index: Literal[False] = False,
        sort: _bool = True,
        group_keys: _bool = True,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = True,
    ) -> DataFrameGroupBy[Timestamp, Literal[False]]: ...
    @overload
    def groupby(  # pyright: ignore reportOverlappingOverload
        self,
        by: TimedeltaIndex,
        level: IndexLabel | None = None,
        as_index: Literal[True] = True,
        sort: _bool = True,
        group_keys: _bool = True,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = True,
    ) -> DataFrameGroupBy[Timedelta, Literal[True]]: ...
    @overload
    def groupby(
        self,
        by: TimedeltaIndex,
        level: IndexLabel | None = None,
        as_index: Literal[False] = False,
        sort: _bool = True,
        group_keys: _bool = True,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = True,
    ) -> DataFrameGroupBy[Timedelta, Literal[False]]: ...
    @overload
    def groupby(  # pyright: ignore reportOverlappingOverload
        self,
        by: PeriodIndex,
        level: IndexLabel | None = None,
        as_index: Literal[True] = True,
        sort: _bool = True,
        group_keys: _bool = True,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = True,
    ) -> DataFrameGroupBy[Period, Literal[True]]: ...
    @overload
    def groupby(
        self,
        by: PeriodIndex,
        level: IndexLabel | None = None,
        as_index: Literal[False] = False,
        sort: _bool = True,
        group_keys: _bool = True,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = True,
    ) -> DataFrameGroupBy[Period, Literal[False]]: ...
    @overload
    def groupby(  # pyright: ignore reportOverlappingOverload
        self,
        by: IntervalIndex[IntervalT],
        level: IndexLabel | None = None,
        as_index: Literal[True] = True,
        sort: _bool = True,
        group_keys: _bool = True,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = True,
    ) -> DataFrameGroupBy[IntervalT, Literal[True]]: ...
    @overload
    def groupby(
        self,
        by: IntervalIndex[IntervalT],
        level: IndexLabel | None = None,
        as_index: Literal[False] = False,
        sort: _bool = True,
        group_keys: _bool = True,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = True,
    ) -> DataFrameGroupBy[IntervalT, Literal[False]]: ...
    @overload
    def groupby(  # type: ignore[overload-overlap] # pyright: ignore reportOverlappingOverload
        self,
        by: MultiIndex | GroupByObjectNonScalar[Any] | None = None,
        level: IndexLabel | None = None,
        as_index: Literal[True] = True,
        sort: _bool = True,
        group_keys: _bool = True,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = True,
    ) -> DataFrameGroupBy[tuple[Any, ...], Literal[True]]: ...
    @overload
    def groupby(  # type: ignore[overload-overlap]
        self,
        by: MultiIndex | GroupByObjectNonScalar[Any] | None = None,
        level: IndexLabel | None = None,
        as_index: Literal[False] = False,
        sort: _bool = True,
        group_keys: _bool = True,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = True,
    ) -> DataFrameGroupBy[tuple[Any, ...], Literal[False]]: ...
    @overload
    def groupby(  # pyright: ignore reportOverlappingOverload
        self,
        by: Series[SeriesByT],
        level: IndexLabel | None = None,
        as_index: Literal[True] = True,
        sort: _bool = True,
        group_keys: _bool = True,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = True,
    ) -> DataFrameGroupBy[SeriesByT, Literal[True]]: ...
    @overload
    def groupby(
        self,
        by: Series[SeriesByT],
        level: IndexLabel | None = None,
        as_index: Literal[False] = False,
        sort: _bool = True,
        group_keys: _bool = True,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = True,
    ) -> DataFrameGroupBy[SeriesByT, Literal[False]]: ...
    @overload
    def groupby(
        self,
        by: CategoricalIndex[Any] | Index[Any] | Series,
        level: IndexLabel | None = None,
        as_index: Literal[True] = True,
        sort: _bool = True,
        group_keys: _bool = True,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = True,
    ) -> DataFrameGroupBy[Any, Literal[True]]: ...
    @overload
    def groupby(
        self,
        by: CategoricalIndex[Any] | Index[Any] | Series,
        level: IndexLabel | None = None,
        as_index: Literal[False] = False,
        sort: _bool = True,
        group_keys: _bool = True,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = True,
    ) -> DataFrameGroupBy[Any, Literal[False]]: ...
    def pivot(
        self,
        *,
        columns: IndexLabel,
        index: IndexLabel = ...,
        values: IndexLabel = ...,
    ) -> Self: ...
    def pivot_table(
        self,
        values: _PivotTableValuesTypes[Any] = None,
        index: _PivotTableIndexTypes[Any] = None,
        columns: _PivotTableColumnsTypes[Any] = None,
        aggfunc: Any="mean",
        fill_value: Scalar | None = None,
        margins: _bool = False,
        dropna: _bool = True,
        margins_name: _str = "All",
        observed: _bool = True,
        sort: _bool = True,
    ) -> Self: ...
    @overload
    def stack(
        self,
        level: IndexLabel = -1,
        *,
        future_stack: Literal[True],
    ) -> Self | Series: ...
    @overload
    def stack(
        self,
        level: IndexLabel = -1,
        dropna: _bool = ...,
        sort: _bool = ...,
        future_stack: Literal[False] = False,
    ) -> Self | Series: ...
    def explode(
        self, column: Sequence[Hashable], ignore_index: _bool = False
    ) -> Self: ...
    def unstack(
        self,
        level: IndexLabel = -1,
        fill_value: Scalar | None = None,
        sort: _bool = True,
    ) -> Self | Series: ...
    def melt(
        self,
        id_vars: tuple[Any, ...] | Sequence[Any] | np.ndarray[Any, Any] | None = None,
        value_vars: tuple[Any, ...] | Sequence[Any] | np.ndarray[Any, Any] | None = None,
        var_name: Scalar | None = None,
        value_name: Scalar = "value",
        col_level: int | _str | None = None,
        ignore_index: _bool = True,
    ) -> Self: ...
    def diff(self, periods: int = 1, axis: Axis = 0) -> Self: ...
    @overload
    def agg(  # pyright: ignore[reportOverlappingOverload]
        self,
        func: AggFuncTypeBase | AggFuncTypeDictSeries[Any],
        axis: Axis = 0,
        **kwargs: Any,
    ) -> Series: ...
    @overload
    def agg(
        self,
        func: list[AggFuncTypeBase] | AggFuncTypeDictFrame[Any]| None = None,
        axis: Axis = 0,
        **kwargs: Any,
    ) -> Self: ...
    @overload
    def aggregate(  # pyright: ignore[reportOverlappingOverload]
        self,
        func: AggFuncTypeBase | AggFuncTypeDictSeries[Any],
        axis: Axis = 0,
        **kwargs: Any,
    ) -> Series: ...
    @overload
    def aggregate(
        self,
        func: list[AggFuncTypeBase] | AggFuncTypeDictFrame[Any],
        axis: Axis = 0,
        **kwargs: Any,
    ) -> Self: ...
    def transform(
        self,
        func: AggFuncTypeFrame,
        axis: Axis = 0,
        *args: Any,
        **kwargs: Any,
    ) -> Self: ...

    # apply() overloads with default result_type of None, and is indifferent to axis
    @overload
    def apply(
        self,
        f: Callable[..., ListLikeExceptSeriesAndStr | Series],
        axis: AxisIndex = 0,
        raw: _bool = False,
        result_type: None = None,
        args: Any = (),
        **kwargs: Any,
    ) -> Self: ...
    @overload
    def apply(
        self,
        # Use S2 (TypeVar without `default=Any`) instead of S1 due to https://github.com/python/mypy/issues/19182.
        f: Callable[..., S2 | NAType],
        axis: AxisIndex = 0,
        raw: _bool = False,
        result_type: None = None,
        args: Any = (),
        **kwargs: Any,
    ) -> Series[S2]: ...
    # Since non-scalar type T is not supported in Series[T],
    # we separate this overload from the above one
    @overload
    def apply(
        self,
        f: Callable[..., Mapping[Any, Any]],
        axis: AxisIndex = 0,
        raw: _bool = False,
        result_type: None = None,
        args: Any = (),
        **kwargs: Any,
    ) -> Series: ...

    # apply() overloads with keyword result_type, and axis does not matter
    @overload
    def apply(
        self,
        # Use S2 (TypeVar without `default=Any`) instead of S1 due to https://github.com/python/mypy/issues/19182.
        f: Callable[..., S2 | NAType],
        axis: Axis = 0,
        raw: _bool = False,
        args: Any = (),
        *,
        result_type: Literal["expand", "reduce"],
        **kwargs: Any,
    ) -> Series[S2]: ...
    @overload
    def apply(
        self,
        f: Callable[..., ListLikeExceptSeriesAndStr | Series | Mapping[Any, Any]],
        axis: Axis = 0,
        raw: _bool = False,
        args: Any = (),
        *,
        result_type: Literal["expand"],
        **kwargs: Any,
    ) -> Self: ...
    @overload
    def apply(
        self,
        f: Callable[..., ListLikeExceptSeriesAndStr | Mapping[Any, Any]],
        axis: Axis = 0,
        raw: _bool = False,
        args: Any = (),
        *,
        result_type: Literal["reduce"],
        **kwargs: Any,
    ) -> Series: ...
    @overload
    def apply(
        self,
        f: Callable[..., ListLikeExceptSeriesAndStr | Series | Scalar | Mapping[Any, Any]],
        axis: Axis = 0,
        raw: _bool = False,
        args: Any = (),
        *,
        result_type: Literal["broadcast"],
        **kwargs: Any,
    ) -> Self: ...

    # apply() overloads with keyword result_type, and axis does matter
    @overload
    def apply(
        self,
        f: Callable[..., Series],
        axis: AxisIndex = 0,
        raw: _bool = False,
        args: Any = (),
        *,
        result_type: Literal["reduce"],
        **kwargs: Any,
    ) -> Series: ...

    # apply() overloads with default result_type of None, and keyword axis=1 matters
    @overload
    def apply(
        self,
        # Use S2 (TypeVar without `default=Any`) instead of S1 due to https://github.com/python/mypy/issues/19182.
        f: Callable[..., S2 | NAType],
        raw: _bool = False,
        result_type: None = None,
        args: Any = (),
        *,
        axis: AxisColumn,
        **kwargs: Any,
    ) -> Series[S2]: ...
    @overload
    def apply(
        self,
        f: Callable[..., ListLikeExceptSeriesAndStr | Mapping[Any, Any]],
        raw: _bool = False,
        result_type: None = None,
        args: Any = (),
        *,
        axis: AxisColumn,
        **kwargs: Any,
    ) -> Series: ...
    @overload
    def apply(
        self,
        f: Callable[..., Series],
        raw: _bool = False,
        result_type: None = None,
        args: Any = (),
        *,
        axis: AxisColumn,
        **kwargs: Any,
    ) -> Self: ...

    # apply() overloads with keyword axis=1 and keyword result_type
    @overload
    def apply(
        self,
        f: Callable[..., Series],
        raw: _bool = False,
        args: Any = (),
        *,
        axis: AxisColumn,
        result_type: Literal["reduce"],
        **kwargs: Any,
    ) -> Self: ...

    # Add spacing between apply() overloads and remaining annotations
    def map(
        self, func: Callable[..., Any], na_action: Literal["ignore"] | None = None, **kwargs: Any
    ) -> Self: ...
    def join(
        self,
        other: DataFrame | Series | list[DataFrame | Series],
        on: _str | list[_str] | None = None,
        how: MergeHow = "left",
        lsuffix: _str = "",
        rsuffix: _str = "",
        sort: _bool = False,
        validate: JoinValidate | None = None,
    ) -> Self: ...
    def merge(
        self,
        right: DataFrame | Series,
        how: MergeHow = "inner",
        on: IndexLabel | AnyArrayLike | None = None,
        left_on: IndexLabel | AnyArrayLike | None = None,
        right_on: IndexLabel | AnyArrayLike | None = None,
        left_index: _bool = False,
        right_index: _bool = False,
        sort: _bool = False,
        suffixes: Suffixes = ('_x', '_y'),
        copy: _bool = True,
        indicator: _bool | _str = False,
        validate: MergeValidate | None = None,
    ) -> Self: ...
    def round(
        self, decimals: int | dict[Any, Any] | Series = 0, *args: Any, **kwargs: Any
    ) -> Self: ...
    def corr(
        self,
        method: Literal["pearson", "kendall", "spearman"] = "pearson",
        min_periods: int = 1,
        numeric_only: _bool = False,
    ) -> Self: ...
    def cov(
        self,
        min_periods: int | None = None,
        ddof: int = 1,
        numeric_only: _bool = False,
    ) -> Self: ...
    def corrwith(
        self,
        other: DataFrame | Series,
        axis: Axis | None = 0,
        drop: _bool = False,
        method: Literal["pearson", "kendall", "spearman"] = "pearson",
        numeric_only: _bool = False,
    ) -> Series: ...
    def count(self, axis: Axis = 0, numeric_only: _bool = False) -> Series[int]: ...
    def nunique(self, axis: Axis = 0, dropna: bool = True) -> Series[int]: ...
    def idxmax(
        self,
        axis: Axis = 0,
        skipna: _bool = True,
        numeric_only: _bool = False,
    ) -> Series[int]: ...
    def idxmin(
        self,
        axis: Axis = 0,
        skipna: _bool = True,
        numeric_only: _bool = False,
    ) -> Series[int]: ...
    def mode(
        self,
        axis: Axis = 0,
        numeric_only: _bool = False,
        dropna: _bool = True,
    ) -> Series: ...
    @overload
    def quantile(
        self,
        q: float = 0.5,
        axis: Axis = 0,
        numeric_only: _bool = False,
        interpolation: QuantileInterpolation = 'linear',
        method: CalculationMethod = 'single',
    ) -> Series: ...
    @overload
    def quantile(
        self,
        q: list[float] | np.ndarray[Any, Any],
        axis: Axis = 0,
        numeric_only: _bool = False,
        interpolation: QuantileInterpolation = 'linear',
        method: CalculationMethod = 'single',
    ) -> Self: ...
    def to_timestamp(
        self,
        freq: Any=None,
        how: ToTimestampHow = 'start',
        axis: Axis = 0,
        copy: _bool = True,
    ) -> Self: ...
    def to_period(
        self,
        freq: _str | None = None,
        axis: Axis = 0,
        copy: _bool = True,
    ) -> Self: ...
    def isin(self, values: Iterable[Any] | Series | DataFrame | dict[Any, Any]) -> Self: ...
    @property
    def plot(self) -> PlotAccessor: ...
    def hist(
        self,
        column: _str | list[_str] | None = None,
        by: _str | ListLike | None = None,
        grid: _bool = True,
        xlabelsize: float | str | None = None,
        xrot: float | None = None,
        ylabelsize: float | str | None = None,
        yrot: float | None = None,
        ax: PlotAxes | None = None,
        sharex: _bool = False,
        sharey: _bool = False,
        figsize: tuple[float, float] | None = None,
        layout: tuple[int, int] | None = None,
        bins: int | list[Any] = 10,
        backend: _str | None = None,
        **kwargs: Any,
    ) -> Any: ...
    def boxplot(
        self,
        column: _str | list[_str] | None = None,
        by: _str | ListLike | None = None,
        ax: PlotAxes | None = None,
        fontsize: float | _str | None = None,
        rot: float = 0,
        grid: _bool = True,
        figsize: tuple[float, float] | None = None,
        layout: tuple[int, int] | None = None,
        return_type: Literal["axes", "dict", "both"] | None = None,
        backend: _str | None = None,
        **kwargs: Any,
    ) -> Any: ...
    sparse = ...

    # The rest of these are remnants from the
    # stubs shipped at preview. They may belong in
    # base classes, or stubgen just failed to generate
    # these.

    Name: _str
    #
    # dunder methods
    def __iter__(self) -> Iterator[Hashable]: ...
    # properties
    @property
    def at(self) -> _AtIndexerFrame: ...
    @property
    def columns(self) -> Index[str]: ...
    @columns.setter  # setter needs to be right next to getter; otherwise mypy complains
    def columns(
        self, cols: AnyArrayLike | SequenceNotStr[Hashable] | tuple[Hashable, ...]
    ) -> None: ...
    @property
    def dtypes(self) -> Series: ...
    @property
    def empty(self) -> _bool: ...
    @property
    def iat(self) -> _iAtIndexerFrame: ...
    @property
    def iloc(self) -> _iLocIndexerFrame[Self]: ...
    @property
    # mypy complains if we use Index[Any] instead of UnknownIndex here, even though
    # the latter is aliased to the former ¯\_(ツ)_/¯.
    def index(self) -> UnknownIndex: ...
    @index.setter
    def index(self, idx: Index[Any]) -> None: ...
    @property
    def loc(self) -> _LocIndexerFrame[Self]: ...
    @property
    def ndim(self) -> int: ...
    @property
    def size(self) -> int: ...
    @property
    def values(self) -> np.ndarray[Any, Any]: ...
    # methods
    @final
    def abs(self) -> Self: ...
    def add(
        self,
        other: num | ListLike | DataFrame,
        axis: Axis | None = "columns",
        level: Level | None = None,
        fill_value: float | None = None,
    ) -> Self: ...
    @final
    def add_prefix(self, prefix: _str, axis: Axis | None = None) -> Self: ...
    @final
    def add_suffix(self, suffix: _str, axis: Axis | None = None) -> Self: ...
    @overload
    def all(
        self,
        axis: None,
        bool_only: _bool | None = False,
        skipna: _bool = True,
        **kwargs: Any,
    ) -> np.bool: ...
    @overload
    def all(
        self,
        axis: Axis = 0,
        bool_only: _bool | None = False,
        skipna: _bool = True,
        **kwargs: Any,
    ) -> Series[_bool]: ...
    @overload
    def any(
        self,
        *,
        axis: None,
        bool_only: _bool | None = False,
        skipna: _bool = True,
        **kwargs: Any,
    ) -> np.bool: ...
    @overload
    def any(
        self,
        *,
        axis: Axis = 0,
        bool_only: _bool | None = False,
        skipna: _bool = True,
        **kwargs: Any,
    ) -> Series[_bool]: ...
    @final
    def asof(self, where: Any, subset: _str | list[_str] | None = None) -> Self: ...
    @final
    def asfreq(
        self,
        freq: Any,
        method: FillnaOptions | None = None,
        how: Literal["start", "end"] | None = None,
        normalize: _bool = False,
        fill_value: Scalar | None = None,
    ) -> Self: ...
    @final
    def astype(
        self,
        dtype: AstypeArg | Mapping[Any, Dtype] | Series,
        copy: _bool = True,
        errors: IgnoreRaise = "raise",
    ) -> Self: ...
    @final
    def at_time(
        self,
        time: _str | dt.time,
        asof: _bool = False,
        axis: Axis | None = 0,
    ) -> Self: ...
    @final
    def between_time(
        self,
        start_time: _str | dt.time,
        end_time: _str | dt.time,
        inclusive: IntervalClosedType = "both",
        axis: Axis | None = 0,
    ) -> Self: ...
    @overload
    def bfill(
        self,
        *,
        axis: Axis | None = None,
        inplace: Literal[True],
        limit: int | None = None,
        limit_area: Literal["inside", "outside"] | None = None,
    ) -> None: ...
    @overload
    def bfill(
        self,
        *,
        axis: Axis | None = None,
        inplace: Literal[False] = False,
        limit: int | None = None,
        limit_area: Literal["inside", "outside"] | None = None,
    ) -> Self: ...
    @overload
    def clip(
        self,
        lower: float | None = None,
        upper: float | None = None,
        *,
        axis: Axis | None = None,
        inplace: Literal[False] = False,
        **kwargs: Any,
    ) -> Self: ...
    @overload
    def clip(
        self,
        lower: AnyArrayLike| None = None,
        upper: AnyArrayLike | None = None,
        *,
        axis: Axis| None = None,
        inplace: Literal[False] = False,
        **kwargs: Any,
    ) -> Self: ...
    @overload
    def clip( # pyright: ignore[reportOverlappingOverload]
        self,
        lower: AnyArrayLike | None = None,
        upper: AnyArrayLike| None = None,
        *,
        axis: Axis| None = None,
        inplace: Literal[False] = False,
        **kwargs: Any,
    ) -> Self: ...
    @overload
    def clip(  # pyright: ignore[reportOverlappingOverload]
        self,
        lower: None = None,
        upper: None = None,
        *,
        axis: Axis | None = None,
        inplace: Literal[True],
        **kwargs: Any,
    ) -> Self: ...
    @overload
    def clip(
        self,
        lower: float | None = None,
        upper: float | None = None,
        *,
        axis: Axis | None = None,
        inplace: Literal[True],
        **kwargs: Any,
    ) -> None: ...
    @overload
    def clip(
        self,
        lower: AnyArrayLike| None = None,
        upper: AnyArrayLike | None = None,
        *,
        axis: Axis| None = None,
        inplace: Literal[True],
        **kwargs: Any,
    ) -> None: ...
    @overload
    def clip( # pyright: ignore[reportOverlappingOverload]
        self,
        lower: AnyArrayLike | None = None,
        upper: AnyArrayLike| None = None,
        *,
        axis: Axis| None = None,
        inplace: Literal[True],
        **kwargs: Any,
    ) -> None: ...
    @final
    def copy(self, deep: _bool = True) -> Self: ...
    def cummax(
        self,
        axis: Axis | None = None,
        skipna: _bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> Self: ...
    def cummin(
        self,
        axis: Axis | None = None,
        skipna: _bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> Self: ...
    def cumprod(
        self,
        axis: Axis | None = None,
        skipna: _bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> Self: ...
    def cumsum(
        self,
        axis: Axis | None = None,
        skipna: _bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> Self: ...
    @final
    def describe(
        self,
        percentiles: list[float] | None = None,
        include: Literal["all"] | list[Dtype] | None = None,
        exclude: list[Dtype] | None = None,
    ) -> Self: ...
    def div(
        self,
        other: num | ListLike | DataFrame,
        axis: Axis | None = 'columns',
        level: Level | None = None,
        fill_value: float | None = None,
    ) -> Self: ...
    def divide(
        self,
        other: num | ListLike | DataFrame,
        axis: Axis | None = 'columns',
        level: Level | None = None,
        fill_value: float | None = None,
    ) -> Self: ...
    @final
    def droplevel(self, level: Level | list[Level], axis: Axis = 0) -> Self: ...
    def eq(self, other: Any, axis: Axis = "columns", level: Level | None = None) -> Self: ...
    @final
    def equals(self, other: Series | DataFrame) -> _bool: ...
    @final
    def ewm(
        self,
        com: float | None = None,
        span: float | None = None,
        halflife: float | None = None,
        alpha: float | None = None,
        min_periods: int = 0,
        adjust: _bool = True,
        ignore_na: _bool = False,
        axis: Axis = 0,
    ) -> ExponentialMovingWindow[Self]: ...
    @final
    def expanding(
        self,
        min_periods: int = 1,
        axis: AxisIndex = 0,
        method: CalculationMethod = "single",
    ) -> Expanding[Self]: ...
    @overload
    def ffill(
        self,
        *,
        axis: Axis | None = None,
        inplace: Literal[True],
        limit: int | None = None,
        limit_area: Literal["inside", "outside"] | None = None,
    ) -> None: ...
    @overload
    def ffill(
        self,
        *,
        axis: Axis | None = None,
        inplace: Literal[False] = False,
        limit: int | None = None,
        limit_area: Literal["inside", "outside"] | None = None,
    ) -> Self: ...
    def filter(
        self,
        items: ListLike | None = None,
        like: _str | None = None,
        regex: _str | None = None,
        axis: Axis | None = None,
    ) -> Self: ...
    @final
    def first(self, offset: Any) -> Self: ...
    @final
    def first_valid_index(self) -> Scalar: ...
    def floordiv(
        self,
        other: num | ListLike | DataFrame,
        axis: Axis | None = 'columns',
        level: Level | None = None,
        fill_value: float | None = None,
    ) -> Self: ...
    def ge(self, other: Any, axis: Axis = "columns", level: Level | None = None) -> Self: ...
    @overload
    def get(self, key: Hashable, default: None = None) -> Series | None: ...
    @overload
    def get(self, key: Hashable, default: _T) -> Series | _T: ...
    @overload
    def get(self, key: list[Hashable], default: None = None) -> Self | None: ...
    @overload
    def get(self, key: list[Hashable], default: _T) -> Self | _T: ...
    def gt(self, other: Any, axis: Axis = "columns", level: Level | None = None) -> Self: ...
    @final
    def head(self, n: int = 5) -> Self: ...
    @final
    def infer_objects(self, copy: _bool | None = None) -> Self: ...
    @overload
    def interpolate(
        self,
        method: InterpolateOptions = 'linear',
        *,
        axis: Axis = 0,
        limit: int | None = None,
        limit_direction: Literal["forward", "backward", "both"]| None = None,
        limit_area: Literal["inside", "outside"] | None = None,
        inplace: Literal[True],
        **kwargs: Any,
    ) -> None: ...
    @overload
    def interpolate(
        self,
        method: InterpolateOptions = 'linear',
        *,
        axis: Axis = 0,
        limit: int | None = None,
        limit_direction: Literal["forward", "backward", "both"]| None = None,
        limit_area: Literal["inside", "outside"] | None = None,
        inplace: Literal[False] = False,
        **kwargs: Any,
    ) -> Self: ...
    def keys(self) -> Index[Any]: ...
    def kurt(
        self,
        axis: Axis | None = 0,
        skipna: _bool | None = True,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> Series: ...
    def kurtosis(
        self,
        axis: Axis | None = 0,
        skipna: _bool | None = True,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> Series: ...
    @final
    def last(self, offset: Any) -> Self: ...
    @final
    def last_valid_index(self) -> Scalar: ...
    def le(self, other: Any, axis: Axis = "columns", level: Level | None = None) -> Self: ...
    def lt(self, other: Any, axis: Axis = "columns", level: Level | None = None) -> Self: ...
    @overload
    def mask(
        self,
        cond: (
            Series
            | DataFrame
            | np.ndarray[Any, Any]
            | Callable[[DataFrame], DataFrame]
            | Callable[[Any], _bool]
        ),
        other: Scalar | Series | DataFrame | Callable[..., Any] | NAType | None = ...,
        *,
        inplace: Literal[True],
        axis: Axis | None = None,
        level: Level | None = None,
    ) -> None: ...
    @overload
    def mask(
        self,
        cond: (
            Series
            | DataFrame
            | np.ndarray[Any, Any]
            | Callable[[DataFrame], DataFrame]
            | Callable[[Any], _bool]
        ),
        other: Scalar | Series | DataFrame | Callable[..., Any] | NAType | None = ...,
        *,
        inplace: Literal[False] = False,
        axis: Axis | None = None,
        level: Level | None = None,
    ) -> Self: ...
    def max(
        self,
        axis: Axis | None = 0,
        skipna: _bool | None = True,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> Series: ...
    def mean(
        self,
        axis: Axis | None = 0,
        skipna: _bool | None = True,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> Series: ...
    def median(
        self,
        axis: Axis | None = 0,
        skipna: _bool | None = True,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> Series: ...
    def min(
        self,
        axis: Axis | None = 0,
        skipna: _bool | None = True,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> Series: ...
    def mod(
        self,
        other: num | ListLike | DataFrame,
        axis: Axis | None = 'columns',
        level: Level | None = None,
        fill_value: float | None = None,
    ) -> Self: ...
    def mul(
        self,
        other: num | ListLike | DataFrame,
        axis: Axis | None = 'columns',
        level: Level | None = None,
        fill_value: float | None = None,
    ) -> Self: ...
    def multiply(
        self,
        other: num | ListLike | DataFrame,
        axis: Axis | None = 'columns',
        level: Level | None = None,
        fill_value: float | None = None,
    ) -> Self: ...
    def ne(self, other: Any, axis: Axis = "columns", level: Level | None = None) -> Self: ...
    @final
    def pct_change(
        self,
        periods: int = 1,
        fill_method: None = None,
        freq: DateOffset | dt.timedelta | _str | None = None,
        fill_value: Scalar | NAType | None = ...,
    ) -> Self: ...
    def pop(self, item: _str) -> Series: ...
    def pow(
        self,
        other: num | ListLike | DataFrame,
        axis: Axis | None = 'columns',
        level: Level | None = None,
        fill_value: float | None = None,
    ) -> Self: ...
    def prod(
        self,
        axis: Axis | None = 0,
        skipna: _bool | None = True,
        numeric_only: _bool = False,
        min_count: int = 0,
        **kwargs: Any,
    ) -> Series: ...
    def product(
        self,
        axis: Axis | None = 0,
        skipna: _bool = True,
        numeric_only: _bool = False,
        min_count: int = 0,
        **kwargs: Any,
    ) -> Series: ...
    def radd(
        self,
        other: Any,
        axis: Axis = "columns",
        level: Level | None = None,
        fill_value: float | None = None,
    ) -> Self: ...
    @final
    def rank(
        self,
        axis: Axis = 0,
        method: Literal["average", "min", "max", "first", "dense"] = "average",
        numeric_only: _bool = False,
        na_option: Literal["keep", "top", "bottom"] = "keep",
        ascending: _bool = True,
        pct: _bool = False,
    ) -> Self: ...
    def rdiv(
        self,
        other: Any,
        axis: Axis = 'columns',
        level: Level | None = None,
        fill_value: float | None = None,
    ) -> Self: ...
    @final
    def reindex_like(
        self,
        other: DataFrame,
        method: FillnaOptions | Literal["nearest"] | None = None,
        copy: _bool = True,
        limit: int | None = None,
        tolerance: Scalar | AnyArrayLike | Sequence[Scalar]| None = None,
    ) -> Self: ...
    # Rename axis with `mapper`, `axis`, and `inplace=True`
    @overload
    def rename_axis(
        self,
        mapper: Scalar | ListLike | None = ...,
        *,
        axis: Axis | None = 0,
        copy: _bool| None = None,
        inplace: Literal[True],
    ) -> None: ...
    # Rename axis with `mapper`, `axis`, and `inplace=False`
    @overload
    def rename_axis(
        self,
        mapper: Scalar | ListLike | None = ...,
        *,
        axis: Axis | None = 0,
        copy: _bool| None = None,
        inplace: Literal[False] = False,
    ) -> Self: ...
    # Rename axis with `index` and/or `columns` and `inplace=True`
    @overload
    def rename_axis(
        self,
        *,
        index: _str | Sequence[_str] | dict[_str | int, _str] | Callable[..., Any] | None = ...,
        columns: _str | Sequence[_str] | dict[_str | int, _str] | Callable[..., Any] | None = ...,
        copy: _bool| None = None,
        inplace: Literal[True],
    ) -> None: ...
    # Rename axis with `index` and/or `columns` and `inplace=False`
    @overload
    def rename_axis(
        self,
        *,
        index: _str | Sequence[_str] | dict[_str | int, _str] | Callable[..., Any] | None = ...,
        columns: _str | Sequence[_str] | dict[_str | int, _str] | Callable[..., Any] | None = ...,
        copy: _bool| None = None,
        inplace: Literal[False] = False,
    ) -> Self: ...
    def rfloordiv(
        self,
        other: Any,
        axis: Axis = 'columns',
        level: Level | None = None,
        fill_value: float | None = None,
    ) -> Self: ...
    def rmod(
        self,
        other: Any,
        axis: Axis = 'columns',
        level: Level | None = None,
        fill_value: float | None = None,
    ) -> Self: ...
    def rmul(
        self,
        other: Any,
        axis: Axis = 'columns',
        level: Level | None = None,
        fill_value: float | None = None,
    ) -> Self: ...
    @overload
    def rolling(
        self,
        window: int | str | dt.timedelta | BaseOffset | BaseIndexer,
        min_periods: int | None = None,
        center: _bool = False,
        on: Hashable | None = None,
        axis: AxisIndex = ...,
        closed: IntervalClosedType | None = None,
        step: int | None = None,
        method: CalculationMethod = 'single',
        *,
        win_type: _str,
    ) -> Window[Self]: ...
    @overload
    def rolling(
        self,
        window: int | str | dt.timedelta | BaseOffset | BaseIndexer,
        min_periods: int | None = None,
        center: _bool = False,
        on: Hashable | None = None,
        axis: AxisIndex = ...,
        closed: IntervalClosedType | None = None,
        step: int | None = None,
        method: CalculationMethod = 'single',
        *,
        win_type: None = None,
    ) -> Rolling[Self]: ...
    def rpow(
        self,
        other: Any,
        axis: Axis = 'columns',
        level: Level | None = None,
        fill_value: float | None = None,
    ) -> Self: ...
    def rsub(
        self,
        other: Any,
        axis: Axis = 'columns',
        level: Level | None = None,
        fill_value: float | None = None,
    ) -> Self: ...
    def rtruediv(
        self,
        other: Any,
        axis: Axis = "columns",
        level: Level | None = None,
        fill_value: float | None = None,
    ) -> Self: ...
    @final
    def sample(
        self,
        n: int | None = None,
        frac: float | None = None,
        replace: _bool = False,
        weights: _str | ListLike | None = None,
        random_state: RandomState | None = None,
        axis: Axis | None = None,
        ignore_index: _bool = False,
    ) -> Self: ...
    def sem(
        self,
        axis: Axis | None = 0,
        skipna: _bool | None = True,
        ddof: int = 1,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> Series: ...
    # Not actually positional, but used to handle removal of deprecated
    def set_axis(self, labels: Any, *, axis: Axis = 0, copy: _bool| None = None) -> Self: ...
    def skew(
        self,
        axis: Axis | None = 0,
        skipna: _bool | None = True,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> Series: ...
    @final
    def squeeze(self, axis: Axis | None = None) -> DataFrame | Series | Scalar: ...
    def std(
        self,
        axis: Axis = 0,
        skipna: _bool = True,
        ddof: int = 1,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> Series: ...
    def sub(
        self,
        other: num | ListLike | DataFrame,
        axis: Axis | None = 'columns',
        level: Level | None = None,
        fill_value: float | None = None,
    ) -> Self: ...
    def subtract(
        self,
        other: num | ListLike | DataFrame,
        axis: Axis | None = 'columns',
        level: Level | None = None,
        fill_value: float | None = None,
    ) -> Self: ...
    def sum(
        self,
        axis: Axis = 0,
        skipna: _bool | None = True,
        numeric_only: _bool = False,
        min_count: int = 0,
        **kwargs: Any,
    ) -> Series: ...
    @final
    def swapaxes(self, axis1: Axis, axis2: Axis, copy: _bool| None = None) -> Self: ...
    @final
    def tail(self, n: int = 5) -> Self: ...
    @overload
    def to_json(
        self,
        path_or_buf: FilePath | WriteBuffer[str],
        *,
        orient: Literal["records"],
        date_format: Literal["epoch", "iso"] | None = None,
        double_precision: int = 10,
        force_ascii: _bool = True,
        date_unit: TimeUnit = 'ms',
        default_handler: Callable[[Any], JSONSerializable] | None = None,
        lines: Literal[True],
        compression: CompressionOptions = 'infer',
        index: _bool | None = None,
        indent: int | None = None,
        storage_options: dict[Any, Any] | None = None,
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
        index: _bool | None = None,
        indent: int | None = None,
        storage_options: dict[Any, Any] | None = None,
        mode: Literal["a"],
    ) -> _str: ...
    @overload
    def to_json(
        self,
        path_or_buf: None = None,
        *,
        orient: JsonFrameOrient | None = None,
        date_format: Literal["epoch", "iso"] | None = None,
        double_precision: int = 10,
        force_ascii: _bool = True,
        date_unit: TimeUnit = 'ms',
        default_handler: Callable[[Any], JSONSerializable] | None = None,
        lines: _bool = False,
        compression: CompressionOptions = 'infer',
        index: _bool | None = None,
        indent: int | None = None,
        storage_options: dict[Any, Any] | None = None,
        mode: Literal["w"] = "w",
    ) -> _str: ...
    @overload
    def to_json(
        self,
        path_or_buf: FilePath | WriteBuffer[str] | WriteBuffer[bytes],
        *,
        orient: JsonFrameOrient | None = None,
        date_format: Literal["epoch", "iso"] | None = None,
        double_precision: int = 10,
        force_ascii: _bool = True,
        date_unit: TimeUnit = 'ms',
        default_handler: Callable[[Any], JSONSerializable] | None = None,
        lines: _bool = False,
        compression: CompressionOptions = 'infer',
        index: _bool | None = None,
        indent: int | None = None,
        storage_options: dict[Any, Any] | None = None,
        mode: Literal["w"] = "w",
    ) -> None: ...
    @overload
    def to_string(
        self,
        buf: FilePath | WriteBuffer[str],
        *,
        columns: SequenceNotStr[Hashable] | Index[Any] | Series | None = None,
        col_space: int | list[int] | dict[HashableT, int] | None = None,
        header: _bool | list[_str] | tuple[str, ...] = True,
        index: _bool = True,
        na_rep: _str = 'NaN',
        formatters: FormattersType | None = None,
        float_format: FloatFormatType | None = None,
        sparsify: _bool | None = None,
        index_names: _bool = True,
        justify: _str | None = None,
        max_rows: int | None = None,
        max_cols: int | None = None,
        show_dimensions: _bool = False,
        decimal: _str = '.',
        line_width: int | None = None,
        min_rows: int | None = None,
        max_colwidth: int | None = None,
        encoding: _str | None = None,
    ) -> None: ...
    @overload
    def to_string(
        self,
        buf: None = None,
        *,
        columns: Sequence[Hashable] | Index[Any] | Series | None = None,
        col_space: int | list[int] | dict[Hashable, int] | None = None,
        header: _bool | Sequence[_str] = True,
        index: _bool = True,
        na_rep: _str = 'NaN',
        formatters: FormattersType | None = None,
        float_format: FloatFormatType | None = None,
        sparsify: _bool | None = None,
        index_names: _bool = True,
        justify: _str | None = None,
        max_rows: int | None = None,
        max_cols: int | None = None,
        show_dimensions: _bool = False,
        decimal: _str = '.',
        line_width: int | None = None,
        min_rows: int | None = None,
        max_colwidth: int | None = None,
        encoding: _str | None = None,
    ) -> _str: ...
    @final
    def to_xarray(self) -> xr.Dataset: ...
    def truediv(
        self,
        other: num | ListLike | DataFrame,
        axis: Axis | None = "columns",
        level: Level | None = None,
        fill_value: float | None = None,
    ) -> Self: ...
    @final
    def truncate(
        self,
        before: dt.date | _str | int | None = None,
        after: dt.date | _str | int | None = None,
        axis: Axis | None = None,
        copy: _bool| None = None,
    ) -> Self: ...
    @final
    def tz_convert(
        self,
        tz: TimeZones,
        axis: Axis = 0,
        level: Level | None = None,
        copy: _bool = True,
    ) -> Self: ...
    @final
    def tz_localize(
        self,
        tz: TimeZones,
        axis: Axis = 0,
        level: Level | None = None,
        copy: _bool = True,
        ambiguous: TimeAmbiguous = "raise",
        nonexistent: TimeNonexistent = "raise",
    ) -> Self: ...
    def var(
        self,
        axis: Axis = 0,
        skipna: _bool | None = True,
        ddof: int = 1,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> Series: ...
    @overload
    def where(
        self,
        cond: (
            Series
            | DataFrame
            | np.ndarray[Any, Any]
            | Callable[[DataFrame], DataFrame]
            | Callable[[Any], _bool]
        ),
        other: Any=...,
        *,
        inplace: Literal[True],
        axis: Axis | None = None,
        level: Level | None = None,
    ) -> None: ...
    @overload
    def where(
        self,
        cond: (
            Series
            | DataFrame
            | np.ndarray[Any, Any]
            | Callable[[DataFrame], DataFrame]
            | Callable[[Any], _bool]
        ),
        other: Any=...,
        *,
        inplace: Literal[False] = False,
        axis: Axis | None = None,
        level: Level | None = None,
    ) -> Self: ...
    # Move from generic because Series is Generic and it returns Series[bool] there
    @final
    def __invert__(self) -> Self: ...
    @final
    def xs(
        self,
        key: Hashable,
        axis: Axis = 0,
        level: Level | None = None,
        drop_level: _bool = True,
    ) -> Self | Series: ...
    # floordiv overload
    def __floordiv__(
        self, other: float | DataFrame | Series[int] | Series[float] | Sequence[float]
    ) -> Self: ...
    def __rfloordiv__(
        self, other: float | DataFrame | Series[int] | Series[float] | Sequence[float]
    ) -> Self: ...
    def __truediv__(self, other: float | DataFrame | Series | Sequence[Any]) -> Self: ...
    def __rtruediv__(self, other: float | DataFrame | Series | Sequence[Any]) -> Self: ...
    @final
    def __bool__(self) -> NoReturn: ...

class _PandasNamedTuple(tuple[Any, ...]):
    def __getattr__(self, field: str) -> Scalar: ...
