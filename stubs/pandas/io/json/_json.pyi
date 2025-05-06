import abc
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Hashable, Mapping
from pandas import ArrowDtype as ArrowDtype, DataFrame as DataFrame, Index as Index, MultiIndex as MultiIndex, Series as Series, isna as isna, notna as notna, to_datetime as to_datetime
from pandas._libs import lib as lib
from pandas._libs.json import ujson_dumps as ujson_dumps, ujson_loads as ujson_loads
from pandas._typing import CompressionOptions as CompressionOptions, DtypeArg as DtypeArg, DtypeBackend as DtypeBackend, FilePath as FilePath, IndexLabel as IndexLabel, JSONEngine as JSONEngine, JSONSerializable as JSONSerializable, ReadBuffer as ReadBuffer, Self as Self, StorageOptions as StorageOptions, WriteBuffer as WriteBuffer
from pandas.core.dtypes.common import ensure_str as ensure_str, is_string_dtype as is_string_dtype
from pandas.core.generic import NDFrame as NDFrame
from pandas.io.common import IOHandles as IOHandles, dedup_names as dedup_names, extension_to_compression as extension_to_compression, file_exists as file_exists, get_handle as get_handle, is_fsspec_url as is_fsspec_url, is_potential_multi_index as is_potential_multi_index, is_url as is_url, stringify_path as stringify_path
from pandas.io.json._table_schema import build_table_schema as build_table_schema, parse_table_schema as parse_table_schema
from types import TracebackType
from typing import Any, Generic, Literal, TypeVar, overload

from collections.abc import Callable

FrameSeriesStrT = TypeVar('FrameSeriesStrT', bound=Literal['frame', 'series'])

@overload
def to_json(path_or_buf: FilePath | WriteBuffer[str] | WriteBuffer[bytes], obj: NDFrame, orient: str | None = ..., date_format: str = ..., double_precision: int = ..., force_ascii: bool = ..., date_unit: str = ..., default_handler: Callable[[Any], JSONSerializable] | None = ..., lines: bool = ..., compression: CompressionOptions = ..., index: bool | None = ..., indent: int = ..., storage_options: StorageOptions = ..., mode: Literal['a', 'w'] = ...) -> None: ...
@overload
def to_json(path_or_buf: None, obj: NDFrame, orient: str | None = ..., date_format: str = ..., double_precision: int = ..., force_ascii: bool = ..., date_unit: str = ..., default_handler: Callable[[Any], JSONSerializable] | None = ..., lines: bool = ..., compression: CompressionOptions = ..., index: bool | None = ..., indent: int = ..., storage_options: StorageOptions = ..., mode: Literal['a', 'w'] = ...) -> str: ...

class Writer(ABC, metaclass=abc.ABCMeta):
    _default_orient: str
    obj: Incomplete
    orient: Incomplete
    date_format: Incomplete
    double_precision: Incomplete
    ensure_ascii: Incomplete
    date_unit: Incomplete
    default_handler: Incomplete
    index: Incomplete
    indent: Incomplete
    is_copy: Incomplete
    def __init__(self, obj: NDFrame, orient: str | None, date_format: str, double_precision: int, ensure_ascii: bool, date_unit: str, index: bool, default_handler: Callable[[Any], JSONSerializable] | None = None, indent: int = 0) -> None: ...
    def _format_axes(self) -> None: ...
    def write(self) -> str: ...
    @property
    @abstractmethod
    def obj_to_write(self) -> NDFrame | Mapping[IndexLabel, Any]:
        """Object to write in JSON format."""

class SeriesWriter(Writer):
    _default_orient: str
    @property
    def obj_to_write(self) -> NDFrame | Mapping[IndexLabel, Any]: ...
    def _format_axes(self) -> None: ...

class FrameWriter(Writer):
    _default_orient: str
    @property
    def obj_to_write(self) -> NDFrame | Mapping[IndexLabel, Any]: ...
    def _format_axes(self) -> None:
        """
        Try to format axes if they are datelike.
        """

class JSONTableWriter(FrameWriter):
    _default_orient: str
    schema: Incomplete
    obj: Incomplete
    date_format: str
    orient: str
    index: Incomplete
    def __init__(self, obj, orient: str | None, date_format: str, double_precision: int, ensure_ascii: bool, date_unit: str, index: bool, default_handler: Callable[[Any], JSONSerializable] | None = None, indent: int = 0) -> None:
        """
        Adds a `schema` attribute with the Table Schema, resets
        the index (can't do in caller, because the schema inference needs
        to know what the index is, forces orient to records, and forces
        date_format to 'iso'.
        """
    @property
    def obj_to_write(self) -> NDFrame | Mapping[IndexLabel, Any]: ...

@overload
def read_json(path_or_buf: FilePath | ReadBuffer[str] | ReadBuffer[bytes], *, orient: str | None = ..., typ: Literal['frame'] = ..., dtype: DtypeArg | None = ..., convert_axes: bool | None = ..., convert_dates: bool | list[str] = ..., keep_default_dates: bool = ..., precise_float: bool = ..., date_unit: str | None = ..., encoding: str | None = ..., encoding_errors: str | None = ..., lines: bool = ..., chunksize: int, compression: CompressionOptions = ..., nrows: int | None = ..., storage_options: StorageOptions = ..., dtype_backend: DtypeBackend | lib.NoDefault = ..., engine: JSONEngine = ...) -> JsonReader[Literal['frame']]: ...
@overload
def read_json(path_or_buf: FilePath | ReadBuffer[str] | ReadBuffer[bytes], *, orient: str | None = ..., typ: Literal['series'], dtype: DtypeArg | None = ..., convert_axes: bool | None = ..., convert_dates: bool | list[str] = ..., keep_default_dates: bool = ..., precise_float: bool = ..., date_unit: str | None = ..., encoding: str | None = ..., encoding_errors: str | None = ..., lines: bool = ..., chunksize: int, compression: CompressionOptions = ..., nrows: int | None = ..., storage_options: StorageOptions = ..., dtype_backend: DtypeBackend | lib.NoDefault = ..., engine: JSONEngine = ...) -> JsonReader[Literal['series']]: ...
@overload
def read_json(path_or_buf: FilePath | ReadBuffer[str] | ReadBuffer[bytes], *, orient: str | None = ..., typ: Literal['series'], dtype: DtypeArg | None = ..., convert_axes: bool | None = ..., convert_dates: bool | list[str] = ..., keep_default_dates: bool = ..., precise_float: bool = ..., date_unit: str | None = ..., encoding: str | None = ..., encoding_errors: str | None = ..., lines: bool = ..., chunksize: None = ..., compression: CompressionOptions = ..., nrows: int | None = ..., storage_options: StorageOptions = ..., dtype_backend: DtypeBackend | lib.NoDefault = ..., engine: JSONEngine = ...) -> Series: ...
@overload
def read_json(path_or_buf: FilePath | ReadBuffer[str] | ReadBuffer[bytes], *, orient: str | None = ..., typ: Literal['frame'] = ..., dtype: DtypeArg | None = ..., convert_axes: bool | None = ..., convert_dates: bool | list[str] = ..., keep_default_dates: bool = ..., precise_float: bool = ..., date_unit: str | None = ..., encoding: str | None = ..., encoding_errors: str | None = ..., lines: bool = ..., chunksize: None = ..., compression: CompressionOptions = ..., nrows: int | None = ..., storage_options: StorageOptions = ..., dtype_backend: DtypeBackend | lib.NoDefault = ..., engine: JSONEngine = ...) -> DataFrame: ...

class JsonReader(abc.Iterator, Generic[FrameSeriesStrT]):
    """
    JsonReader provides an interface for reading in a JSON file.

    If initialized with ``lines=True`` and ``chunksize``, can be iterated over
    ``chunksize`` lines at a time. Otherwise, calling ``read`` reads in the
    whole document.
    """
    orient: Incomplete
    typ: Incomplete
    dtype: Incomplete
    convert_axes: Incomplete
    convert_dates: Incomplete
    keep_default_dates: Incomplete
    precise_float: Incomplete
    date_unit: Incomplete
    encoding: Incomplete
    engine: Incomplete
    compression: Incomplete
    storage_options: Incomplete
    lines: Incomplete
    chunksize: Incomplete
    nrows_seen: int
    nrows: Incomplete
    encoding_errors: Incomplete
    handles: IOHandles[str] | None
    dtype_backend: Incomplete
    data: Incomplete
    def __init__(self, filepath_or_buffer, orient, typ: FrameSeriesStrT, dtype, convert_axes: bool | None, convert_dates, keep_default_dates: bool, precise_float: bool, date_unit, encoding, lines: bool, chunksize: int | None, compression: CompressionOptions, nrows: int | None, storage_options: StorageOptions | None = None, encoding_errors: str | None = 'strict', dtype_backend: DtypeBackend | lib.NoDefault = ..., engine: JSONEngine = 'ujson') -> None: ...
    def _preprocess_data(self, data):
        """
        At this point, the data either has a `read` attribute (e.g. a file
        object or a StringIO) or is a string that is a JSON document.

        If self.chunksize, we prepare the data for the `__next__` method.
        Otherwise, we read it into memory for the `read` method.
        """
    def _get_data_from_filepath(self, filepath_or_buffer):
        """
        The function read_json accepts three input types:
            1. filepath (string-like)
            2. file-like object (e.g. open file object, StringIO)
            3. JSON string

        This method turns (1) into (2) to simplify the rest of the processing.
        It returns input types (2) and (3) unchanged.

        It raises FileNotFoundError if the input is a string ending in
        one of .json, .json.gz, .json.bz2, etc. but no such file exists.
        """
    def _combine_lines(self, lines) -> str:
        """
        Combines a list of JSON objects into one JSON object.
        """
    @overload
    def read(self) -> DataFrame: ...
    @overload
    def read(self) -> Series: ...
    @overload
    def read(self) -> DataFrame | Series: ...
    def _get_object_parser(self, json) -> DataFrame | Series:
        """
        Parses a json document into a pandas object.
        """
    def close(self) -> None:
        """
        If we opened a stream earlier, in _get_data_from_filepath, we should
        close it.

        If an open stream or file was passed, we leave it open.
        """
    def __iter__(self) -> Self: ...
    @overload
    def __next__(self) -> DataFrame: ...
    @overload
    def __next__(self) -> Series: ...
    @overload
    def __next__(self) -> DataFrame | Series: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None: ...

class Parser:
    _split_keys: tuple[str, ...]
    _default_orient: str
    _STAMP_UNITS: Incomplete
    _MIN_STAMPS: Incomplete
    json: str
    orient: Incomplete
    dtype: Incomplete
    min_stamp: Incomplete
    precise_float: Incomplete
    convert_axes: Incomplete
    convert_dates: Incomplete
    date_unit: Incomplete
    keep_default_dates: Incomplete
    obj: DataFrame | Series | None
    dtype_backend: Incomplete
    def __init__(self, json: str, orient, dtype: DtypeArg | None = None, convert_axes: bool = True, convert_dates: bool | list[str] = True, keep_default_dates: bool = False, precise_float: bool = False, date_unit: Incomplete | None = None, dtype_backend: DtypeBackend | lib.NoDefault = ...) -> None: ...
    def check_keys_split(self, decoded: dict) -> None:
        """
        Checks that dict has only the appropriate keys for orient='split'.
        """
    def parse(self): ...
    def _parse(self) -> None: ...
    def _convert_axes(self) -> None:
        """
        Try to convert axes.
        """
    def _try_convert_types(self) -> None: ...
    def _try_convert_data(self, name: Hashable, data: Series, use_dtypes: bool = True, convert_dates: bool | list[str] = True, is_axis: bool = False) -> tuple[Series, bool]:
        """
        Try to parse a Series into a column by inferring dtype.
        """
    def _try_convert_to_date(self, data: Series) -> tuple[Series, bool]:
        """
        Try to parse a ndarray like into a date column.

        Try to coerce object in epoch/iso formats and integer/float in epoch
        formats. Return a boolean if parsing was successful.
        """

class SeriesParser(Parser):
    _default_orient: str
    _split_keys: Incomplete
    obj: Series | None
    def _parse(self) -> None: ...
    def _try_convert_types(self) -> None: ...

class FrameParser(Parser):
    _default_orient: str
    _split_keys: Incomplete
    obj: DataFrame | None
    def _parse(self) -> None: ...
    def _process_converter(self, f: Callable[[Hashable, Series], tuple[Series, bool]], filt: Callable[[Hashable], bool] | None = None) -> None:
        """
        Take a conversion function and possibly recreate the frame.
        """
    def _try_convert_types(self) -> None: ...
    def _try_convert_dates(self) -> None: ...
