import csv
from _typeshed import Incomplete
from collections import abc
from collections.abc import Hashable, Iterable, Mapping, Sequence
from pandas import Series as Series
from pandas._config import using_copy_on_write as using_copy_on_write
from pandas._libs import lib as lib
from pandas._libs.parsers import STR_NA_VALUES as STR_NA_VALUES
from pandas._typing import CSVEngine as CSVEngine, CompressionOptions as CompressionOptions, DtypeArg as DtypeArg, DtypeBackend as DtypeBackend, FilePath as FilePath, IndexLabel as IndexLabel, ReadCsvBuffer as ReadCsvBuffer, Self as Self, StorageOptions as StorageOptions, UsecolsArgType as UsecolsArgType
from pandas.core.dtypes.common import is_file_like as is_file_like, is_float as is_float, is_hashable as is_hashable, is_integer as is_integer, is_list_like as is_list_like, pandas_dtype as pandas_dtype
from pandas.core.frame import DataFrame as DataFrame
from pandas.core.indexes.api import RangeIndex as RangeIndex
from pandas.core.shared_docs import _shared_docs as _shared_docs
from pandas.errors import AbstractMethodError as AbstractMethodError, ParserWarning as ParserWarning
from pandas.io.common import IOHandles as IOHandles, get_handle as get_handle, stringify_path as stringify_path, validate_header_arg as validate_header_arg
from pandas.io.parsers.arrow_parser_wrapper import ArrowParserWrapper as ArrowParserWrapper
from pandas.io.parsers.base_parser import ParserBase as ParserBase, is_index_col as is_index_col, parser_defaults as parser_defaults
from pandas.io.parsers.c_parser_wrapper import CParserWrapper as CParserWrapper
from pandas.io.parsers.python_parser import FixedWidthFieldParser as FixedWidthFieldParser, PythonParser as PythonParser
from pandas.util._decorators import Appender as Appender
from pandas.util._exceptions import find_stack_level as find_stack_level
from pandas.util._validators import check_dtype_backend as check_dtype_backend
from types import TracebackType
from typing import Any, Callable, IO, Literal, NamedTuple, TypedDict, overload

_doc_read_csv_and_table: Incomplete

class _C_Parser_Defaults(TypedDict):
    delim_whitespace: Literal[False]
    na_filter: Literal[True]
    low_memory: Literal[True]
    memory_map: Literal[False]
    float_precision: None

_c_parser_defaults: _C_Parser_Defaults

class _Fwf_Defaults(TypedDict):
    colspecs: Literal['infer']
    infer_nrows: Literal[100]
    widths: None

_fwf_defaults: _Fwf_Defaults
_c_unsupported: Incomplete
_python_unsupported: Incomplete
_pyarrow_unsupported: Incomplete

class _DeprecationConfig(NamedTuple):
    default_value: Any
    msg: str | None

@overload
def validate_integer(name: str, val: None, min_val: int = ...) -> None: ...
@overload
def validate_integer(name: str, val: float, min_val: int = ...) -> int: ...
@overload
def validate_integer(name: str, val: int | None, min_val: int = ...) -> int | None: ...
def _validate_names(names: Sequence[Hashable] | None) -> None:
    """
    Raise ValueError if the `names` parameter contains duplicates or has an
    invalid data type.

    Parameters
    ----------
    names : array-like or None
        An array containing a list of the names used for the output DataFrame.

    Raises
    ------
    ValueError
        If names are not unique or are not ordered (e.g. set).
    """
def _read(filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str], kwds) -> DataFrame | TextFileReader:
    """Generic reader of line files."""
@overload
def read_csv(filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str], *, sep: str | None | lib.NoDefault = ..., delimiter: str | None | lib.NoDefault = ..., header: int | Sequence[int] | None | Literal['infer'] = ..., names: Sequence[Hashable] | None | lib.NoDefault = ..., index_col: IndexLabel | Literal[False] | None = ..., usecols: UsecolsArgType = ..., dtype: DtypeArg | None = ..., engine: CSVEngine | None = ..., converters: Mapping[Hashable, Callable] | None = ..., true_values: list | None = ..., false_values: list | None = ..., skipinitialspace: bool = ..., skiprows: list[int] | int | Callable[[Hashable], bool] | None = ..., skipfooter: int = ..., nrows: int | None = ..., na_values: Hashable | Iterable[Hashable] | Mapping[Hashable, Iterable[Hashable]] | None = ..., na_filter: bool = ..., verbose: bool | lib.NoDefault = ..., skip_blank_lines: bool = ..., parse_dates: bool | Sequence[Hashable] | None = ..., infer_datetime_format: bool | lib.NoDefault = ..., keep_date_col: bool | lib.NoDefault = ..., date_parser: Callable | lib.NoDefault = ..., date_format: str | dict[Hashable, str] | None = ..., dayfirst: bool = ..., cache_dates: bool = ..., iterator: Literal[True], chunksize: int | None = ..., compression: CompressionOptions = ..., thousands: str | None = ..., decimal: str = ..., lineterminator: str | None = ..., quotechar: str = ..., quoting: int = ..., doublequote: bool = ..., escapechar: str | None = ..., comment: str | None = ..., encoding: str | None = ..., encoding_errors: str | None = ..., dialect: str | csv.Dialect | None = ..., on_bad_lines=..., delim_whitespace: bool | lib.NoDefault = ..., low_memory: bool = ..., memory_map: bool = ..., float_precision: Literal['high', 'legacy'] | None = ..., storage_options: StorageOptions = ..., dtype_backend: DtypeBackend | lib.NoDefault = ...) -> TextFileReader: ...
@overload
def read_csv(filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str], *, sep: str | None | lib.NoDefault = ..., delimiter: str | None | lib.NoDefault = ..., header: int | Sequence[int] | None | Literal['infer'] = ..., names: Sequence[Hashable] | None | lib.NoDefault = ..., index_col: IndexLabel | Literal[False] | None = ..., usecols: UsecolsArgType = ..., dtype: DtypeArg | None = ..., engine: CSVEngine | None = ..., converters: Mapping[Hashable, Callable] | None = ..., true_values: list | None = ..., false_values: list | None = ..., skipinitialspace: bool = ..., skiprows: list[int] | int | Callable[[Hashable], bool] | None = ..., skipfooter: int = ..., nrows: int | None = ..., na_values: Hashable | Iterable[Hashable] | Mapping[Hashable, Iterable[Hashable]] | None = ..., keep_default_na: bool = ..., na_filter: bool = ..., verbose: bool | lib.NoDefault = ..., skip_blank_lines: bool = ..., parse_dates: bool | Sequence[Hashable] | None = ..., infer_datetime_format: bool | lib.NoDefault = ..., keep_date_col: bool | lib.NoDefault = ..., date_parser: Callable | lib.NoDefault = ..., date_format: str | dict[Hashable, str] | None = ..., dayfirst: bool = ..., cache_dates: bool = ..., iterator: bool = ..., chunksize: int, compression: CompressionOptions = ..., thousands: str | None = ..., decimal: str = ..., lineterminator: str | None = ..., quotechar: str = ..., quoting: int = ..., doublequote: bool = ..., escapechar: str | None = ..., comment: str | None = ..., encoding: str | None = ..., encoding_errors: str | None = ..., dialect: str | csv.Dialect | None = ..., on_bad_lines=..., delim_whitespace: bool | lib.NoDefault = ..., low_memory: bool = ..., memory_map: bool = ..., float_precision: Literal['high', 'legacy'] | None = ..., storage_options: StorageOptions = ..., dtype_backend: DtypeBackend | lib.NoDefault = ...) -> TextFileReader: ...
@overload
def read_csv(filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str], *, sep: str | None | lib.NoDefault = ..., delimiter: str | None | lib.NoDefault = ..., header: int | Sequence[int] | None | Literal['infer'] = ..., names: Sequence[Hashable] | None | lib.NoDefault = ..., index_col: IndexLabel | Literal[False] | None = ..., usecols: UsecolsArgType = ..., dtype: DtypeArg | None = ..., engine: CSVEngine | None = ..., converters: Mapping[Hashable, Callable] | None = ..., true_values: list | None = ..., false_values: list | None = ..., skipinitialspace: bool = ..., skiprows: list[int] | int | Callable[[Hashable], bool] | None = ..., skipfooter: int = ..., nrows: int | None = ..., na_values: Hashable | Iterable[Hashable] | Mapping[Hashable, Iterable[Hashable]] | None = ..., keep_default_na: bool = ..., na_filter: bool = ..., verbose: bool | lib.NoDefault = ..., skip_blank_lines: bool = ..., parse_dates: bool | Sequence[Hashable] | None = ..., infer_datetime_format: bool | lib.NoDefault = ..., keep_date_col: bool | lib.NoDefault = ..., date_parser: Callable | lib.NoDefault = ..., date_format: str | dict[Hashable, str] | None = ..., dayfirst: bool = ..., cache_dates: bool = ..., iterator: Literal[False] = ..., chunksize: None = ..., compression: CompressionOptions = ..., thousands: str | None = ..., decimal: str = ..., lineterminator: str | None = ..., quotechar: str = ..., quoting: int = ..., doublequote: bool = ..., escapechar: str | None = ..., comment: str | None = ..., encoding: str | None = ..., encoding_errors: str | None = ..., dialect: str | csv.Dialect | None = ..., on_bad_lines=..., delim_whitespace: bool | lib.NoDefault = ..., low_memory: bool = ..., memory_map: bool = ..., float_precision: Literal['high', 'legacy'] | None = ..., storage_options: StorageOptions = ..., dtype_backend: DtypeBackend | lib.NoDefault = ...) -> DataFrame: ...
@overload
def read_csv(filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str], *, sep: str | None | lib.NoDefault = ..., delimiter: str | None | lib.NoDefault = ..., header: int | Sequence[int] | None | Literal['infer'] = ..., names: Sequence[Hashable] | None | lib.NoDefault = ..., index_col: IndexLabel | Literal[False] | None = ..., usecols: UsecolsArgType = ..., dtype: DtypeArg | None = ..., engine: CSVEngine | None = ..., converters: Mapping[Hashable, Callable] | None = ..., true_values: list | None = ..., false_values: list | None = ..., skipinitialspace: bool = ..., skiprows: list[int] | int | Callable[[Hashable], bool] | None = ..., skipfooter: int = ..., nrows: int | None = ..., na_values: Hashable | Iterable[Hashable] | Mapping[Hashable, Iterable[Hashable]] | None = ..., keep_default_na: bool = ..., na_filter: bool = ..., verbose: bool | lib.NoDefault = ..., skip_blank_lines: bool = ..., parse_dates: bool | Sequence[Hashable] | None = ..., infer_datetime_format: bool | lib.NoDefault = ..., keep_date_col: bool | lib.NoDefault = ..., date_parser: Callable | lib.NoDefault = ..., date_format: str | dict[Hashable, str] | None = ..., dayfirst: bool = ..., cache_dates: bool = ..., iterator: bool = ..., chunksize: int | None = ..., compression: CompressionOptions = ..., thousands: str | None = ..., decimal: str = ..., lineterminator: str | None = ..., quotechar: str = ..., quoting: int = ..., doublequote: bool = ..., escapechar: str | None = ..., comment: str | None = ..., encoding: str | None = ..., encoding_errors: str | None = ..., dialect: str | csv.Dialect | None = ..., on_bad_lines=..., delim_whitespace: bool | lib.NoDefault = ..., low_memory: bool = ..., memory_map: bool = ..., float_precision: Literal['high', 'legacy'] | None = ..., storage_options: StorageOptions = ..., dtype_backend: DtypeBackend | lib.NoDefault = ...) -> DataFrame | TextFileReader: ...
@overload
def read_table(filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str], *, sep: str | None | lib.NoDefault = ..., delimiter: str | None | lib.NoDefault = ..., header: int | Sequence[int] | None | Literal['infer'] = ..., names: Sequence[Hashable] | None | lib.NoDefault = ..., index_col: IndexLabel | Literal[False] | None = ..., usecols: UsecolsArgType = ..., dtype: DtypeArg | None = ..., engine: CSVEngine | None = ..., converters: Mapping[Hashable, Callable] | None = ..., true_values: list | None = ..., false_values: list | None = ..., skipinitialspace: bool = ..., skiprows: list[int] | int | Callable[[Hashable], bool] | None = ..., skipfooter: int = ..., nrows: int | None = ..., na_values: Sequence[str] | Mapping[str, Sequence[str]] | None = ..., keep_default_na: bool = ..., na_filter: bool = ..., verbose: bool | lib.NoDefault = ..., skip_blank_lines: bool = ..., parse_dates: bool | Sequence[Hashable] = ..., infer_datetime_format: bool | lib.NoDefault = ..., keep_date_col: bool | lib.NoDefault = ..., date_parser: Callable | lib.NoDefault = ..., date_format: str | dict[Hashable, str] | None = ..., dayfirst: bool = ..., cache_dates: bool = ..., iterator: Literal[True], chunksize: int | None = ..., compression: CompressionOptions = ..., thousands: str | None = ..., decimal: str = ..., lineterminator: str | None = ..., quotechar: str = ..., quoting: int = ..., doublequote: bool = ..., escapechar: str | None = ..., comment: str | None = ..., encoding: str | None = ..., encoding_errors: str | None = ..., dialect: str | csv.Dialect | None = ..., on_bad_lines=..., delim_whitespace: bool = ..., low_memory: bool = ..., memory_map: bool = ..., float_precision: str | None = ..., storage_options: StorageOptions = ..., dtype_backend: DtypeBackend | lib.NoDefault = ...) -> TextFileReader: ...
@overload
def read_table(filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str], *, sep: str | None | lib.NoDefault = ..., delimiter: str | None | lib.NoDefault = ..., header: int | Sequence[int] | None | Literal['infer'] = ..., names: Sequence[Hashable] | None | lib.NoDefault = ..., index_col: IndexLabel | Literal[False] | None = ..., usecols: UsecolsArgType = ..., dtype: DtypeArg | None = ..., engine: CSVEngine | None = ..., converters: Mapping[Hashable, Callable] | None = ..., true_values: list | None = ..., false_values: list | None = ..., skipinitialspace: bool = ..., skiprows: list[int] | int | Callable[[Hashable], bool] | None = ..., skipfooter: int = ..., nrows: int | None = ..., na_values: Sequence[str] | Mapping[str, Sequence[str]] | None = ..., keep_default_na: bool = ..., na_filter: bool = ..., verbose: bool | lib.NoDefault = ..., skip_blank_lines: bool = ..., parse_dates: bool | Sequence[Hashable] = ..., infer_datetime_format: bool | lib.NoDefault = ..., keep_date_col: bool | lib.NoDefault = ..., date_parser: Callable | lib.NoDefault = ..., date_format: str | dict[Hashable, str] | None = ..., dayfirst: bool = ..., cache_dates: bool = ..., iterator: bool = ..., chunksize: int, compression: CompressionOptions = ..., thousands: str | None = ..., decimal: str = ..., lineterminator: str | None = ..., quotechar: str = ..., quoting: int = ..., doublequote: bool = ..., escapechar: str | None = ..., comment: str | None = ..., encoding: str | None = ..., encoding_errors: str | None = ..., dialect: str | csv.Dialect | None = ..., on_bad_lines=..., delim_whitespace: bool = ..., low_memory: bool = ..., memory_map: bool = ..., float_precision: str | None = ..., storage_options: StorageOptions = ..., dtype_backend: DtypeBackend | lib.NoDefault = ...) -> TextFileReader: ...
@overload
def read_table(filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str], *, sep: str | None | lib.NoDefault = ..., delimiter: str | None | lib.NoDefault = ..., header: int | Sequence[int] | None | Literal['infer'] = ..., names: Sequence[Hashable] | None | lib.NoDefault = ..., index_col: IndexLabel | Literal[False] | None = ..., usecols: UsecolsArgType = ..., dtype: DtypeArg | None = ..., engine: CSVEngine | None = ..., converters: Mapping[Hashable, Callable] | None = ..., true_values: list | None = ..., false_values: list | None = ..., skipinitialspace: bool = ..., skiprows: list[int] | int | Callable[[Hashable], bool] | None = ..., skipfooter: int = ..., nrows: int | None = ..., na_values: Sequence[str] | Mapping[str, Sequence[str]] | None = ..., keep_default_na: bool = ..., na_filter: bool = ..., verbose: bool | lib.NoDefault = ..., skip_blank_lines: bool = ..., parse_dates: bool | Sequence[Hashable] = ..., infer_datetime_format: bool | lib.NoDefault = ..., keep_date_col: bool | lib.NoDefault = ..., date_parser: Callable | lib.NoDefault = ..., date_format: str | dict[Hashable, str] | None = ..., dayfirst: bool = ..., cache_dates: bool = ..., iterator: Literal[False] = ..., chunksize: None = ..., compression: CompressionOptions = ..., thousands: str | None = ..., decimal: str = ..., lineterminator: str | None = ..., quotechar: str = ..., quoting: int = ..., doublequote: bool = ..., escapechar: str | None = ..., comment: str | None = ..., encoding: str | None = ..., encoding_errors: str | None = ..., dialect: str | csv.Dialect | None = ..., on_bad_lines=..., delim_whitespace: bool = ..., low_memory: bool = ..., memory_map: bool = ..., float_precision: str | None = ..., storage_options: StorageOptions = ..., dtype_backend: DtypeBackend | lib.NoDefault = ...) -> DataFrame: ...
@overload
def read_table(filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str], *, sep: str | None | lib.NoDefault = ..., delimiter: str | None | lib.NoDefault = ..., header: int | Sequence[int] | None | Literal['infer'] = ..., names: Sequence[Hashable] | None | lib.NoDefault = ..., index_col: IndexLabel | Literal[False] | None = ..., usecols: UsecolsArgType = ..., dtype: DtypeArg | None = ..., engine: CSVEngine | None = ..., converters: Mapping[Hashable, Callable] | None = ..., true_values: list | None = ..., false_values: list | None = ..., skipinitialspace: bool = ..., skiprows: list[int] | int | Callable[[Hashable], bool] | None = ..., skipfooter: int = ..., nrows: int | None = ..., na_values: Sequence[str] | Mapping[str, Sequence[str]] | None = ..., keep_default_na: bool = ..., na_filter: bool = ..., verbose: bool | lib.NoDefault = ..., skip_blank_lines: bool = ..., parse_dates: bool | Sequence[Hashable] = ..., infer_datetime_format: bool | lib.NoDefault = ..., keep_date_col: bool | lib.NoDefault = ..., date_parser: Callable | lib.NoDefault = ..., date_format: str | dict[Hashable, str] | None = ..., dayfirst: bool = ..., cache_dates: bool = ..., iterator: bool = ..., chunksize: int | None = ..., compression: CompressionOptions = ..., thousands: str | None = ..., decimal: str = ..., lineterminator: str | None = ..., quotechar: str = ..., quoting: int = ..., doublequote: bool = ..., escapechar: str | None = ..., comment: str | None = ..., encoding: str | None = ..., encoding_errors: str | None = ..., dialect: str | csv.Dialect | None = ..., on_bad_lines=..., delim_whitespace: bool = ..., low_memory: bool = ..., memory_map: bool = ..., float_precision: str | None = ..., storage_options: StorageOptions = ..., dtype_backend: DtypeBackend | lib.NoDefault = ...) -> DataFrame | TextFileReader: ...
@overload
def read_fwf(filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str], *, colspecs: Sequence[tuple[int, int]] | str | None = ..., widths: Sequence[int] | None = ..., infer_nrows: int = ..., dtype_backend: DtypeBackend | lib.NoDefault = ..., iterator: Literal[True], chunksize: int | None = ..., **kwds) -> TextFileReader: ...
@overload
def read_fwf(filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str], *, colspecs: Sequence[tuple[int, int]] | str | None = ..., widths: Sequence[int] | None = ..., infer_nrows: int = ..., dtype_backend: DtypeBackend | lib.NoDefault = ..., iterator: bool = ..., chunksize: int, **kwds) -> TextFileReader: ...
@overload
def read_fwf(filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str], *, colspecs: Sequence[tuple[int, int]] | str | None = ..., widths: Sequence[int] | None = ..., infer_nrows: int = ..., dtype_backend: DtypeBackend | lib.NoDefault = ..., iterator: Literal[False] = ..., chunksize: None = ..., **kwds) -> DataFrame: ...

class TextFileReader(abc.Iterator):
    """

    Passed dialect overrides any of the related parser options

    """
    engine: Incomplete
    _engine_specified: Incomplete
    orig_options: Incomplete
    _currow: int
    chunksize: Incomplete
    nrows: Incomplete
    handles: IOHandles | None
    _engine: Incomplete
    def __init__(self, f: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str] | list, engine: CSVEngine | None = None, **kwds) -> None: ...
    def close(self) -> None: ...
    def _get_options_with_defaults(self, engine: CSVEngine) -> dict[str, Any]: ...
    def _check_file_or_buffer(self, f, engine: CSVEngine) -> None: ...
    def _clean_options(self, options: dict[str, Any], engine: CSVEngine) -> tuple[dict[str, Any], CSVEngine]: ...
    def __next__(self) -> DataFrame: ...
    def _make_engine(self, f: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str] | list | IO, engine: CSVEngine = 'c') -> ParserBase: ...
    def _failover_to_python(self) -> None: ...
    def read(self, nrows: int | None = None) -> DataFrame: ...
    def get_chunk(self, size: int | None = None) -> DataFrame: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None: ...

def TextParser(*args, **kwds) -> TextFileReader:
    """
    Converts lists of lists/tuples into DataFrames with proper type inference
    and optional (e.g. string to datetime) conversion. Also enables iterating
    lazily over chunks of large files

    Parameters
    ----------
    data : file-like object or list
    delimiter : separator character to use
    dialect : str or csv.Dialect instance, optional
        Ignored if delimiter is longer than 1 character
    names : sequence, default
    header : int, default 0
        Row to use to parse column labels. Defaults to the first row. Prior
        rows will be discarded
    index_col : int or list, optional
        Column or columns to use as the (possibly hierarchical) index
    has_index_names: bool, default False
        True if the cols defined in index_col have an index name and are
        not in the header.
    na_values : scalar, str, list-like, or dict, optional
        Additional strings to recognize as NA/NaN.
    keep_default_na : bool, default True
    thousands : str, optional
        Thousands separator
    comment : str, optional
        Comment out remainder of line
    parse_dates : bool, default False
    keep_date_col : bool, default False
    date_parser : function, optional

        .. deprecated:: 2.0.0
    date_format : str or dict of column -> format, default ``None``

        .. versionadded:: 2.0.0
    skiprows : list of integers
        Row numbers to skip
    skipfooter : int
        Number of line at bottom of file to skip
    converters : dict, optional
        Dict of functions for converting values in certain columns. Keys can
        either be integers or column labels, values are functions that take one
        input argument, the cell (not column) content, and return the
        transformed content.
    encoding : str, optional
        Encoding to use for UTF when reading/writing (ex. 'utf-8')
    float_precision : str, optional
        Specifies which converter the C engine should use for floating-point
        values. The options are `None` or `high` for the ordinary converter,
        `legacy` for the original lower precision pandas converter, and
        `round_trip` for the round-trip converter.
    """
def _clean_na_values(na_values, keep_default_na: bool = True, floatify: bool = True): ...
def _floatify_na_values(na_values): ...
def _stringify_na_values(na_values, floatify: bool):
    """return a stringified and numeric for these values"""
def _refine_defaults_read(dialect: str | csv.Dialect | None, delimiter: str | None | lib.NoDefault, delim_whitespace: bool, engine: CSVEngine | None, sep: str | None | lib.NoDefault, on_bad_lines: str | Callable, names: Sequence[Hashable] | None | lib.NoDefault, defaults: dict[str, Any], dtype_backend: DtypeBackend | lib.NoDefault):
    '''Validate/refine default values of input parameters of read_csv, read_table.

    Parameters
    ----------
    dialect : str or csv.Dialect
        If provided, this parameter will override values (default or not) for the
        following parameters: `delimiter`, `doublequote`, `escapechar`,
        `skipinitialspace`, `quotechar`, and `quoting`. If it is necessary to
        override values, a ParserWarning will be issued. See csv.Dialect
        documentation for more details.
    delimiter : str or object
        Alias for sep.
    delim_whitespace : bool
        Specifies whether or not whitespace (e.g. ``\' \'`` or ``\'\t\'``) will be
        used as the sep. Equivalent to setting ``sep=\'\\s+\'``. If this option
        is set to True, nothing should be passed in for the ``delimiter``
        parameter.

        .. deprecated:: 2.2.0
            Use ``sep="\\s+"`` instead.
    engine : {{\'c\', \'python\'}}
        Parser engine to use. The C engine is faster while the python engine is
        currently more feature-complete.
    sep : str or object
        A delimiter provided by the user (str) or a sentinel value, i.e.
        pandas._libs.lib.no_default.
    on_bad_lines : str, callable
        An option for handling bad lines or a sentinel value(None).
    names : array-like, optional
        List of column names to use. If the file contains a header row,
        then you should explicitly pass ``header=0`` to override the column names.
        Duplicates in this list are not allowed.
    defaults: dict
        Default values of input parameters.

    Returns
    -------
    kwds : dict
        Input parameters with correct values.

    Raises
    ------
    ValueError :
        If a delimiter was specified with ``sep`` (or ``delimiter``) and
        ``delim_whitespace=True``.
    '''
def _extract_dialect(kwds: dict[str, Any]) -> csv.Dialect | None:
    """
    Extract concrete csv dialect instance.

    Returns
    -------
    csv.Dialect or None
    """

MANDATORY_DIALECT_ATTRS: Incomplete

def _validate_dialect(dialect: csv.Dialect) -> None:
    """
    Validate csv dialect instance.

    Raises
    ------
    ValueError
        If incorrect dialect is provided.
    """
def _merge_with_dialect_properties(dialect: csv.Dialect, defaults: dict[str, Any]) -> dict[str, Any]:
    """
    Merge default kwargs in TextFileReader with dialect parameters.

    Parameters
    ----------
    dialect : csv.Dialect
        Concrete csv dialect. See csv.Dialect documentation for more details.
    defaults : dict
        Keyword arguments passed to TextFileReader.

    Returns
    -------
    kwds : dict
        Updated keyword arguments, merged with dialect parameters.
    """
def _validate_skipfooter(kwds: dict[str, Any]) -> None:
    """
    Check whether skipfooter is compatible with other kwargs in TextFileReader.

    Parameters
    ----------
    kwds : dict
        Keyword arguments passed to TextFileReader.

    Raises
    ------
    ValueError
        If skipfooter is not compatible with other parameters.
    """
