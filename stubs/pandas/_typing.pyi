import numpy as np
from _typeshed import Incomplete
from collections.abc import Hashable, Iterator, Mapping, MutableMapping, Sequence
from datetime import date, tzinfo
from pandas import Interval as Interval
from pandas._libs import NaTType as NaTType, Period as Period, Timedelta as Timedelta, Timestamp as Timestamp
from pandas._libs.tslibs import BaseOffset as BaseOffset
from pandas.arrays import DatetimeArray as DatetimeArray, TimedeltaArray as TimedeltaArray
from pandas.core.arrays.base import ExtensionArray as ExtensionArray
from pandas.core.dtypes.dtypes import ExtensionDtype as ExtensionDtype
from pandas.core.frame import DataFrame as DataFrame
from pandas.core.generic import NDFrame as NDFrame
from pandas.core.groupby.generic import DataFrameGroupBy as DataFrameGroupBy, GroupBy as GroupBy, SeriesGroupBy as SeriesGroupBy
from pandas.core.indexes.base import Index as Index
from pandas.core.internals import ArrayManager as ArrayManager, BlockManager as BlockManager, SingleArrayManager as SingleArrayManager, SingleBlockManager as SingleBlockManager
from pandas.core.resample import Resampler as Resampler
from pandas.core.series import Series as Series
from pandas.core.window.rolling import BaseWindow as BaseWindow
from pandas.io.formats.format import EngFormatter as EngFormatter
from pandas.tseries.holiday import AbstractHolidayCalendar as AbstractHolidayCalendar
from typing import Any, Callable, Protocol, SupportsIndex, TypeVar, overload

ScalarLike_co = int | float | complex | str | bytes | np.generic
NumpyValueArrayLike: Incomplete
NumpySorter: Incomplete
HashableT = TypeVar('HashableT', bound=Hashable)
MutableMappingT = TypeVar('MutableMappingT', bound=MutableMapping)
ArrayLike: Incomplete
AnyArrayLike: Incomplete
TimeArrayLike: Incomplete
_T_co = TypeVar('_T_co', covariant=True)

class SequenceNotStr(Protocol[_T_co]):
    @overload
    def __getitem__(self, index: SupportsIndex) -> _T_co: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[_T_co]: ...
    def __contains__(self, value: object) -> bool: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[_T_co]: ...
    def index(self, value: Any, /, start: int = 0, stop: int = ...) -> int: ...
    def count(self, value: Any, /) -> int: ...
    def __reversed__(self) -> Iterator[_T_co]: ...
ListLike = AnyArrayLike | SequenceNotStr | range
PythonScalar = str | float | bool
DatetimeLikeScalar: Incomplete
PandasScalar: Incomplete
Scalar = PythonScalar | PandasScalar | np.datetime64 | np.timedelta64 | date
IntStrT = TypeVar('IntStrT', bound=int | str)
TimestampConvertibleTypes: Incomplete
TimestampNonexistent: Incomplete
TimedeltaConvertibleTypes: Incomplete
Timezone = str | tzinfo
ToTimestampHow: Incomplete
NDFrameT = TypeVar('NDFrameT', bound='NDFrame')
NumpyIndexT = TypeVar('NumpyIndexT', np.ndarray, 'Index')
AxisInt = int
Axis: Incomplete
IndexLabel = Hashable | Sequence[Hashable]
Level = Hashable
Shape = tuple[int, ...]
Suffixes = tuple[str | None, str | None]
Ordered = bool | None
JSONSerializable = PythonScalar | list | dict | None
Frequency: Incomplete
Axes = ListLike
RandomState = int | np.ndarray | np.random.Generator | np.random.BitGenerator | np.random.RandomState
NpDtype = str | np.dtype | type[str | complex | bool | object]
Dtype: Incomplete
AstypeArg: Incomplete
DtypeArg = Dtype | dict[Hashable, Dtype]
DtypeObj: Incomplete
ConvertersArg = dict[Hashable, Callable[[Dtype], Dtype]]
ParseDatesArg = bool | list[Hashable] | list[list[Hashable]] | dict[Hashable, list[Hashable]]
Renamer = Mapping[Any, Hashable] | Callable[[Any], Hashable]
T = TypeVar('T')
FuncType = Callable[..., Any]
F = TypeVar('F', bound=FuncType)
ValueKeyFunc: Incomplete
IndexKeyFunc: Incomplete
AggFuncTypeBase = Callable | str
AggFuncTypeDict = MutableMapping[Hashable, AggFuncTypeBase | list[AggFuncTypeBase]]
AggFuncType = AggFuncTypeBase | list[AggFuncTypeBase] | AggFuncTypeDict
AggObjType: Incomplete
PythonFuncType = Callable[[Any], Any]
AnyStr_co = TypeVar('AnyStr_co', str, bytes, covariant=True)
AnyStr_contra = TypeVar('AnyStr_contra', str, bytes, contravariant=True)

class BaseBuffer(Protocol):
    @property
    def mode(self) -> str: ...
    def seek(self, __offset: int, /, __whence: int = ...) -> int: ...
    def seekable(self) -> bool: ...
    def tell(self) -> int: ...

class ReadBuffer(BaseBuffer, Protocol[AnyStr_co]):
    def read(self, /, __n: int = ...) -> AnyStr_co: ...

class WriteBuffer(BaseBuffer, Protocol[AnyStr_contra]):
    def write(self, /, __b: AnyStr_contra) -> Any: ...
    def flush(self) -> Any: ...

class ReadPickleBuffer(ReadBuffer[bytes], Protocol):
    def readline(self) -> bytes: ...

class WriteExcelBuffer(WriteBuffer[bytes], Protocol):
    def truncate(self, size: int | None = ...) -> int: ...

class ReadCsvBuffer(ReadBuffer[AnyStr_co], Protocol):
    def __iter__(self) -> Iterator[AnyStr_co]: ...
    def fileno(self) -> int: ...
    def readline(self) -> AnyStr_co: ...
    @property
    def closed(self) -> bool: ...

FilePath: Incomplete
StorageOptions = dict[str, Any] | None
CompressionDict = dict[str, Any]
CompressionOptions: Incomplete
FormattersType = list[Callable] | tuple[Callable, ...] | Mapping[str | int, Callable]
ColspaceType = Mapping[Hashable, str | int]
FloatFormatType: Incomplete
ColspaceArgType = str | int | Sequence[str | int] | Mapping[Hashable, str | int]
FillnaOptions: Incomplete
InterpolateOptions: Incomplete
Manager: Incomplete
SingleManager: Incomplete
Manager2D: Incomplete
ScalarIndexer = int | np.integer
SequenceIndexer = slice | list[int] | np.ndarray
PositionalIndexer = ScalarIndexer | SequenceIndexer
PositionalIndexerTuple = tuple[PositionalIndexer, PositionalIndexer]
PositionalIndexer2D = PositionalIndexer | PositionalIndexerTuple
TakeIndexer: Incomplete
IgnoreRaise: Incomplete
WindowingRankType: Incomplete
CSVEngine: Incomplete
JSONEngine: Incomplete
XMLParsers: Incomplete
HTMLFlavors: Incomplete
IntervalLeftRight: Incomplete
IntervalClosedType: Incomplete
DatetimeNaTType: Incomplete
DateTimeErrorChoices: Incomplete
SortKind: Incomplete
NaPosition: Incomplete
NsmallestNlargestKeep: Incomplete
QuantileInterpolation: Incomplete
PlottingOrientation: Incomplete
AnyAll: Incomplete
MergeHow: Incomplete
MergeValidate: Incomplete
JoinHow: Incomplete
JoinValidate: Incomplete
ReindexMethod: Incomplete
MatplotlibColor = str | Sequence[float]
TimeGrouperOrigin: Incomplete
TimeAmbiguous: Incomplete
TimeNonexistent: Incomplete
DropKeep: Incomplete
CorrelationMethod: Incomplete
AlignJoin: Incomplete
DtypeBackend: Incomplete
TimeUnit: Incomplete
OpenFileErrors: Incomplete
UpdateJoin: Incomplete
NaAction: Incomplete
FromDictOrient: Incomplete
ToGbqIfexist: Incomplete
ToStataByteorder: Incomplete
ExcelWriterIfSheetExists: Incomplete
OffsetCalendar: Incomplete
UsecolsArgType = SequenceNotStr[Hashable] | range | AnyArrayLike | Callable[[HashableT], bool] | None
