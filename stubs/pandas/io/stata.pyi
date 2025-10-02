from collections import abc
from collections.abc import Sequence
import datetime
from io import BytesIO
from types import TracebackType
from typing import (
    Literal,
    overload,
)

from pandas.core.frame import DataFrame
from typing_extensions import Self

from pandas._typing import (
    CompressionOptions,
    FilePath,
    HashableT,
    HashableT1,
    HashableT2,
    HashableT3,
    ReadBuffer,
    StataDateFormat,
    StorageOptions,
    WriteBuffer,
)
from typing import Any

@overload
def read_stata(
    filepath_or_buffer: FilePath | ReadBuffer[bytes],
    *,
    convert_dates: bool = True,
    convert_categoricals: bool = True,
    index_col: str | None = None,
    convert_missing: bool = False,
    preserve_dtypes: bool = True,
    columns: list[HashableT] | None = None,
    order_categoricals: bool = True,
    chunksize: int | None = None,
    iterator: Literal[True],
    compression: CompressionOptions = 'infer',
    storage_options: StorageOptions = None,
) -> StataReader: ...
@overload
def read_stata(
    filepath_or_buffer: FilePath | ReadBuffer[bytes],
    *,
    convert_dates: bool = True,
    convert_categoricals: bool = True,
    index_col: str | None = None,
    convert_missing: bool = False,
    preserve_dtypes: bool = True,
    columns: list[HashableT] | None = None,
    order_categoricals: bool = True,
    chunksize: int,
    iterator: bool = False,
    compression: CompressionOptions = 'infer',
    storage_options: StorageOptions = None,
) -> StataReader: ...
@overload
def read_stata(
    filepath_or_buffer: FilePath | ReadBuffer[bytes],
    *,
    convert_dates: bool = True,
    convert_categoricals: bool = True,
    index_col: str | None = None,
    convert_missing: bool = False,
    preserve_dtypes: bool = True,
    columns: list[HashableT] | None = None,
    order_categoricals: bool = True,
    chunksize: None = None,
    iterator: Literal[False] = False,
    compression: CompressionOptions = 'infer',
    storage_options: StorageOptions = None,
) -> DataFrame: ...

class StataParser:
    def __init__(self) -> None: ...

class StataReader(StataParser, abc.Iterator[Any]):
    col_sizes: list[int] = ...
    path_or_buf: BytesIO = ...
    def __init__(
        self,
        path_or_buf: FilePath | ReadBuffer[bytes],
        convert_dates: bool = True,
        convert_categoricals: bool = True,
        index_col: str | None = None,
        convert_missing: bool = False,
        preserve_dtypes: bool = True,
        columns: Sequence[str] | None = None,
        order_categoricals: bool = True,
        chunksize: int | None = None,
        compression: CompressionOptions = 'infer',
        storage_options: StorageOptions = None,
    ) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...
    def __next__(self) -> DataFrame: ...
    @property
    def data_label(self) -> str: ...
    def variable_labels(self) -> dict[str, str]: ...
    def value_labels(self) -> dict[str, dict[float, str]]: ...

class StataWriter(StataParser):
    def __init__(
        self,
        fname: FilePath | WriteBuffer[bytes],
        data: DataFrame,
        convert_dates: dict[HashableT1, StataDateFormat] | None = None,
        write_index: bool = True,
        byteorder: str | None = None,
        time_stamp: datetime.datetime | None = None,
        data_label: str | None = None,
        variable_labels: dict[HashableT2, str] | None = None,
        compression: CompressionOptions = 'infer',
        storage_options: StorageOptions = None,
        *,
        value_labels: dict[HashableT3, dict[float, str]] | None = None,
    ) -> None: ...
    def write_file(self) -> None: ...
