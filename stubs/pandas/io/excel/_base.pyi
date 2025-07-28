from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Mapping,
    Sequence,
)
from types import TracebackType
from typing import (
    Any,
    Literal,
    overload,
)

from odf.opendocument import OpenDocument
from openpyxl.workbook.workbook import Workbook
from pandas.core.frame import DataFrame
import pyxlsb.workbook
from typing_extensions import Self
from xlrd.book import Book

from pandas._libs.lib import _NoDefaultDoNotUse
from pandas._typing import (
    Dtype,
    DtypeBackend,
    ExcelReadEngine,
    ExcelWriteEngine,
    ExcelWriterIfSheetExists,
    FilePath,
    IntStrT,
    ListLikeHashable,
    ReadBuffer,
    StorageOptions,
    UsecolsArgType,
    WriteExcelBuffer,
)

@overload
def read_excel(
    io: (
        FilePath
        | ReadBuffer[bytes]
        | ExcelFile
        | Workbook
        | Book
        | OpenDocument
        | pyxlsb.workbook.Workbook
    ),
    sheet_name: list[IntStrT],
    *,
    header: int | Sequence[int] | None = 0,
    names: ListLikeHashable[Any] | None = None,
    index_col: int | Sequence[int] | str | None = None,
    usecols: str | UsecolsArgType[Any] = None,
    dtype: str | Dtype | Mapping[str, str | Dtype] | None = None,
    engine: ExcelReadEngine | None = None,
    converters: Mapping[int | str, Callable[[object], object]] | None = None,
    true_values: Iterable[Hashable] | None = None,
    false_values: Iterable[Hashable] | None = None,
    skiprows: int | Sequence[int] | Callable[[object], bool] | None = None,
    nrows: int | None = None,
    na_values: Sequence[str] | dict[str | int, Sequence[str]] | None = None,
    keep_default_na: bool = True,
    na_filter: bool = True,
    verbose: bool = False,
    parse_dates: (
        bool
        | Sequence[int]
        | Sequence[Sequence[str] | Sequence[int]]
        | dict[str, Sequence[int] | list[str]]
    ) = False,
    date_format: dict[Hashable, str] | str | None = None,
    thousands: str | None = None,
    decimal: str = '.',
    comment: str | None = None,
    skipfooter: int = 0,
    storage_options: StorageOptions = None,
    dtype_backend: DtypeBackend | _NoDefaultDoNotUse = ...,
    engine_kwargs: dict[str, Any] | None = None,
) -> dict[IntStrT, DataFrame]: ...
@overload
def read_excel(
    io: (
        FilePath
        | ReadBuffer[bytes]
        | ExcelFile
        | Workbook
        | Book
        | OpenDocument
        | pyxlsb.workbook.Workbook
    ),
    sheet_name: None,
    *,
    header: int | Sequence[int] | None = 0,
    names: ListLikeHashable[Any] | None = None,
    index_col: int | Sequence[int] | str | None = None,
    usecols: str | UsecolsArgType[Any] = None,
    dtype: str | Dtype | Mapping[str, str | Dtype] | None = None,
    engine: ExcelReadEngine | None = None,
    converters: Mapping[int | str, Callable[[object], object]] | None = None,
    true_values: Iterable[Hashable] | None = None,
    false_values: Iterable[Hashable] | None = None,
    skiprows: int | Sequence[int] | Callable[[object], bool] | None = None,
    nrows: int | None = None,
    na_values: Sequence[str] | dict[str | int, Sequence[str]] | None = None,
    keep_default_na: bool = True,
    na_filter: bool = True,
    verbose: bool = False,
    parse_dates: (
        bool
        | Sequence[int]
        | Sequence[Sequence[str] | Sequence[int]]
        | dict[str, Sequence[int] | list[str]]
    ) = False,
    date_format: dict[Hashable, str] | str | None = None,
    thousands: str | None = None,
    decimal: str = '.',
    comment: str | None = None,
    skipfooter: int = 0,
    storage_options: StorageOptions = None,
    dtype_backend: DtypeBackend | _NoDefaultDoNotUse = ...,
    engine_kwargs: dict[str, Any] | None = None,
) -> dict[str, DataFrame]: ...
@overload
# mypy says this won't be matched
def read_excel(  # type: ignore[overload-cannot-match]
    io: (
        FilePath
        | ReadBuffer[bytes]
        | ExcelFile
        | Workbook
        | Book
        | OpenDocument
        | pyxlsb.workbook.Workbook
    ),
    sheet_name: list[int | str],
    *,
    header: int | Sequence[int] | None = 0,
    names: ListLikeHashable[Any] | None = None,
    index_col: int | Sequence[int] | str | None = None,
    usecols: str | UsecolsArgType[Any] = None,
    dtype: str | Dtype | Mapping[str, str | Dtype] | None = None,
    engine: ExcelReadEngine | None = None,
    converters: Mapping[int | str, Callable[[object], object]] | None = None,
    true_values: Iterable[Hashable] | None = None,
    false_values: Iterable[Hashable] | None = None,
    skiprows: int | Sequence[int] | Callable[[object], bool] | None = None,
    nrows: int | None = None,
    na_values: Sequence[str] | dict[str | int, Sequence[str]] | None = None,
    keep_default_na: bool = True,
    na_filter: bool = True,
    verbose: bool = False,
    parse_dates: (
        bool
        | Sequence[int]
        | Sequence[Sequence[str] | Sequence[int]]
        | dict[str, Sequence[int] | list[str]]
    ) = False,
    date_format: dict[Hashable, str] | str | None = None,
    thousands: str | None = None,
    decimal: str = '.',
    comment: str | None = None,
    skipfooter: int = 0,
    storage_options: StorageOptions = None,
    dtype_backend: DtypeBackend | _NoDefaultDoNotUse = ...,
    engine_kwargs: dict[str, Any] | None = None,
) -> dict[int | str, DataFrame]: ...
@overload
def read_excel(
    io: (
        FilePath
        | ReadBuffer[bytes]
        | ExcelFile
        | Workbook
        | Book
        | OpenDocument
        | pyxlsb.workbook.Workbook
    ),
    sheet_name: int | str = 0,
    *,
    header: int | Sequence[int] | None = 0,
    names: ListLikeHashable[Any] | None = None,
    index_col: int | Sequence[int] | str | None = None,
    usecols: str | UsecolsArgType[Any] = None,
    dtype: str | Dtype | Mapping[str, str | Dtype] | None = None,
    engine: ExcelReadEngine | None = None,
    converters: Mapping[int | str, Callable[[object], object]] | None = None,
    true_values: Iterable[Hashable] | None = None,
    false_values: Iterable[Hashable] | None = None,
    skiprows: int | Sequence[int] | Callable[[object], bool] | None = None,
    nrows: int | None = None,
    na_values: Sequence[str] | dict[str | int, Sequence[str]] | None = None,
    keep_default_na: bool = True,
    na_filter: bool = True,
    verbose: bool = False,
    parse_dates: (
        bool
        | Sequence[int]
        | Sequence[Sequence[str] | Sequence[int]]
        | dict[str, Sequence[int] | list[str]]
    ) = False,
    date_format: dict[Hashable, str] | str | None = None,
    thousands: str | None = None,
    decimal: str = '.',
    comment: str | None = None,
    skipfooter: int = 0,
    storage_options: StorageOptions = None,
    dtype_backend: DtypeBackend | _NoDefaultDoNotUse = ...,
    engine_kwargs: dict[str, Any] | None = None,
) -> DataFrame: ...

class ExcelWriter:
    def __init__(
        self,
        path: FilePath | WriteExcelBuffer | ExcelWriter,
        engine: ExcelWriteEngine | Literal["auto"] | None = None,
        date_format: str | None = None,
        datetime_format: str | None = None,
        mode: Literal["w", "a"] = 'w',
        storage_options: StorageOptions = None,
        if_sheet_exists: ExcelWriterIfSheetExists | None = None,
        engine_kwargs: dict[str, Any] | None = None,
    ) -> None: ...
    @property
    def supported_extensions(self) -> tuple[str, ...]: ...
    @property
    def engine(self) -> ExcelWriteEngine: ...
    @property
    def sheets(self) -> dict[str, Any]: ...
    @property
    def book(self) -> Workbook | OpenDocument: ...
    @property
    def date_format(self) -> str: ...
    @property
    def datetime_format(self) -> str: ...
    @property
    def if_sheet_exists(self) -> Literal["error", "new", "replace", "overlay"]: ...
    def __fspath__(self) -> str: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...
    def close(self) -> None: ...

class ExcelFile:
    engine = ...
    io: FilePath | ReadBuffer[bytes] | bytes = ...
    def __init__(
        self,
        path_or_buffer: FilePath | ReadBuffer[bytes] | bytes,
        engine: ExcelReadEngine | None = None,
        storage_options: StorageOptions = None,
        engine_kwargs: dict[str, Any] | None = None,
    ) -> None: ...
    def __fspath__(self) -> Any: ...
    @overload
    def parse(
        self,
        sheet_name: list[int | str] | None,
        header: int | Sequence[int] | None = 0,
        names: ListLikeHashable[Any] | None = None,
        index_col: int | Sequence[int] | None = None,
        usecols: str | UsecolsArgType[Any] = None,
        converters: dict[int | str, Callable[[object], object]] | None = None,
        true_values: Iterable[Hashable] | None = None,
        false_values: Iterable[Hashable] | None = None,
        skiprows: int | Sequence[int] | Callable[[object], bool] | None = None,
        nrows: int | None = None,
        na_values: Sequence[str] | dict[str | int, Sequence[str]] = None,
        parse_dates: (
            bool
            | Sequence[int]
            | Sequence[Sequence[str] | Sequence[int]]
            | dict[str, Sequence[int] | list[str]]
        ) = False,
        date_parser: Callable[..., Any] | None = ...,
        thousands: str | None = None,
        comment: str | None = None,
        skipfooter: int = 0,
        keep_default_na: bool = ...,
        na_filter: bool = ...,
        **kwds: Any,
    ) -> dict[int | str, DataFrame]: ...
    @overload
    def parse(
        self,
        sheet_name: int | str,
        header: int | Sequence[int] | None = 0,
        names: ListLikeHashable[Any] | None = None,
        index_col: int | Sequence[int] | None = None,
        usecols: str | UsecolsArgType[Any] = None,
        converters: dict[int | str, Callable[[object], object]] | None = None,
        true_values: Iterable[Hashable] | None = None,
        false_values: Iterable[Hashable] | None = None,
        skiprows: int | Sequence[int] | Callable[[object], bool] | None = None,
        nrows: int | None = None,
        na_values: Sequence[str] | dict[str | int, Sequence[str]] = None,
        parse_dates: (
            bool
            | Sequence[int]
            | Sequence[Sequence[str] | Sequence[int]]
            | dict[str, Sequence[int] | list[str]]
        ) = False,
        date_parser: Callable[..., Any] | None = ...,
        thousands: str | None = None,
        comment: str | None = None,
        skipfooter: int = 0,
        keep_default_na: bool = ...,
        na_filter: bool = ...,
        **kwds: Any,
    ) -> DataFrame: ...
    @property
    def book(self) -> Workbook | Book | OpenDocument | pyxlsb.workbook.Workbook: ...
    @property
    def sheet_names(self) -> list[int | str]: ...
    def close(self) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...
    def __del__(self) -> None: ...
