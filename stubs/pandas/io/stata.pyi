import numpy as np
from _typeshed import Incomplete
from collections import abc
from collections.abc import Hashable, Sequence
from datetime import datetime
from pandas import Categorical as Categorical, DatetimeIndex as DatetimeIndex, NaT as NaT, Timestamp as Timestamp, isna as isna, to_datetime as to_datetime, to_timedelta as to_timedelta
from pandas._libs import lib as lib
from pandas._libs.lib import infer_dtype as infer_dtype
from pandas._libs.writers import max_len_string_array as max_len_string_array
from pandas._typing import CompressionOptions as CompressionOptions, FilePath as FilePath, ReadBuffer as ReadBuffer, Self as Self, StorageOptions as StorageOptions, WriteBuffer as WriteBuffer
from pandas.core.dtypes.base import ExtensionDtype as ExtensionDtype
from pandas.core.dtypes.common import ensure_object as ensure_object, is_numeric_dtype as is_numeric_dtype, is_string_dtype as is_string_dtype
from pandas.core.dtypes.dtypes import CategoricalDtype as CategoricalDtype
from pandas.core.frame import DataFrame as DataFrame
from pandas.core.indexes.base import Index as Index
from pandas.core.indexes.range import RangeIndex as RangeIndex
from pandas.core.series import Series as Series
from pandas.core.shared_docs import _shared_docs as _shared_docs
from pandas.errors import CategoricalConversionWarning as CategoricalConversionWarning, InvalidColumnName as InvalidColumnName, PossiblePrecisionLoss as PossiblePrecisionLoss, ValueLabelTypeMismatch as ValueLabelTypeMismatch
from pandas.io.common import get_handle as get_handle
from pandas.util._decorators import Appender as Appender, doc as doc
from pandas.util._exceptions import find_stack_level as find_stack_level
from types import TracebackType
from typing import AnyStr, Final, IO, Literal

from collections.abc import Callable

_version_error: str
_statafile_processing_params1: str
_statafile_processing_params2: str
_chunksize_params: str
_iterator_params: str
_reader_notes: str
_read_stata_doc: Incomplete
_read_method_doc: Incomplete
_stata_reader_doc: Incomplete
_date_formats: Incomplete
stata_epoch: Final[Incomplete]

def _stata_elapsed_date_to_datetime_vec(dates: Series, fmt: str) -> Series:
    '''
    Convert from SIF to datetime. https://www.stata.com/help.cgi?datetime

    Parameters
    ----------
    dates : Series
        The Stata Internal Format date to convert to datetime according to fmt
    fmt : str
        The format to convert to. Can be, tc, td, tw, tm, tq, th, ty
        Returns

    Returns
    -------
    converted : Series
        The converted dates

    Examples
    --------
    >>> dates = pd.Series([52])
    >>> _stata_elapsed_date_to_datetime_vec(dates , "%tw")
    0   1961-01-01
    dtype: datetime64[ns]

    Notes
    -----
    datetime/c - tc
        milliseconds since 01jan1960 00:00:00.000, assuming 86,400 s/day
    datetime/C - tC - NOT IMPLEMENTED
        milliseconds since 01jan1960 00:00:00.000, adjusted for leap seconds
    date - td
        days since 01jan1960 (01jan1960 = 0)
    weekly date - tw
        weeks since 1960w1
        This assumes 52 weeks in a year, then adds 7 * remainder of the weeks.
        The datetime value is the start of the week in terms of days in the
        year, not ISO calendar weeks.
    monthly date - tm
        months since 1960m1
    quarterly date - tq
        quarters since 1960q1
    half-yearly date - th
        half-years since 1960h1 yearly
    date - ty
        years since 0000
    '''
def _datetime_to_stata_elapsed_vec(dates: Series, fmt: str) -> Series:
    """
    Convert from datetime to SIF. https://www.stata.com/help.cgi?datetime

    Parameters
    ----------
    dates : Series
        Series or array containing datetime or datetime64[ns] to
        convert to the Stata Internal Format given by fmt
    fmt : str
        The format to convert to. Can be, tc, td, tw, tm, tq, th, ty
    """

excessive_string_length_error: Final[str]
precision_loss_doc: Final[str]
value_label_mismatch_doc: Final[str]
invalid_name_doc: Final[str]
categorical_conversion_warning: Final[str]

def _cast_to_stata_types(data: DataFrame) -> DataFrame:
    """
    Checks the dtypes of the columns of a pandas DataFrame for
    compatibility with the data types and ranges supported by Stata, and
    converts if necessary.

    Parameters
    ----------
    data : DataFrame
        The DataFrame to check and convert

    Notes
    -----
    Numeric columns in Stata must be one of int8, int16, int32, float32 or
    float64, with some additional value restrictions.  int8 and int16 columns
    are checked for violations of the value restrictions and upcast if needed.
    int64 data is not usable in Stata, and so it is downcast to int32 whenever
    the value are in the int32 range, and sidecast to float64 when larger than
    this range.  If the int64 values are outside of the range of those
    perfectly representable as float64 values, a warning is raised.

    bool columns are cast to int8.  uint columns are converted to int of the
    same size if there is no loss in precision, otherwise are upcast to a
    larger type.  uint64 is currently not supported since it is concerted to
    object in a DataFrame.
    """

class StataValueLabel:
    '''
    Parse a categorical column and prepare formatted output

    Parameters
    ----------
    catarray : Series
        Categorical Series to encode
    encoding : {"latin-1", "utf-8"}
        Encoding to use for value labels.
    '''
    labname: Incomplete
    _encoding: Incomplete
    value_labels: Incomplete
    def __init__(self, catarray: Series, encoding: Literal['latin-1', 'utf-8'] = 'latin-1') -> None: ...
    text_len: int
    txt: list[bytes]
    n: int
    off: Incomplete
    val: Incomplete
    len: int
    def _prepare_value_labels(self) -> None:
        """Encode value labels."""
    def generate_value_label(self, byteorder: str) -> bytes:
        """
        Generate the binary representation of the value labels.

        Parameters
        ----------
        byteorder : str
            Byte order of the output

        Returns
        -------
        value_label : bytes
            Bytes containing the formatted value label
        """

class StataNonCatValueLabel(StataValueLabel):
    '''
    Prepare formatted version of value labels

    Parameters
    ----------
    labname : str
        Value label name
    value_labels: Dictionary
        Mapping of values to labels
    encoding : {"latin-1", "utf-8"}
        Encoding to use for value labels.
    '''
    labname: Incomplete
    _encoding: Incomplete
    value_labels: Incomplete
    def __init__(self, labname: str, value_labels: dict[float, str], encoding: Literal['latin-1', 'utf-8'] = 'latin-1') -> None: ...

class StataMissingValue:
    """
    An observation's missing value.

    Parameters
    ----------
    value : {int, float}
        The Stata missing value code

    Notes
    -----
    More information: <https://www.stata.com/help.cgi?missing>

    Integer missing values make the code '.', '.a', ..., '.z' to the ranges
    101 ... 127 (for int8), 32741 ... 32767  (for int16) and 2147483621 ...
    2147483647 (for int32).  Missing values for floating point data types are
    more complex but the pattern is simple to discern from the following table.

    np.float32 missing values (float in Stata)
    0000007f    .
    0008007f    .a
    0010007f    .b
    ...
    00c0007f    .x
    00c8007f    .y
    00d0007f    .z

    np.float64 missing values (double in Stata)
    000000000000e07f    .
    000000000001e07f    .a
    000000000002e07f    .b
    ...
    000000000018e07f    .x
    000000000019e07f    .y
    00000000001ae07f    .z
    """
    MISSING_VALUES: dict[float, str]
    bases: Final[Incomplete]
    float32_base: bytes
    increment_32: int
    key: Incomplete
    int_value: Incomplete
    float64_base: bytes
    increment_64: Incomplete
    BASE_MISSING_VALUES: Final[Incomplete]
    _value: Incomplete
    _str: Incomplete
    def __init__(self, value: float) -> None: ...
    @property
    def string(self) -> str:
        """
        The Stata representation of the missing value: '.', '.a'..'.z'

        Returns
        -------
        str
            The representation of the missing value.
        """
    @property
    def value(self) -> float:
        """
        The binary representation of the missing value.

        Returns
        -------
        {int, float}
            The binary representation of the missing value.
        """
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    @classmethod
    def get_base_missing_value(cls, dtype: np.dtype) -> float: ...

class StataParser:
    DTYPE_MAP: Incomplete
    DTYPE_MAP_XML: dict[int, np.dtype]
    TYPE_MAP: Incomplete
    TYPE_MAP_XML: Incomplete
    VALID_RANGE: Incomplete
    OLD_TYPE_MAPPING: Incomplete
    MISSING_VALUES: Incomplete
    NUMPY_TYPE_MAP: Incomplete
    RESERVED_WORDS: Incomplete
    def __init__(self) -> None: ...

class StataReader(StataParser, abc.Iterator):
    __doc__ = _stata_reader_doc
    _path_or_buf: IO[bytes]
    _convert_dates: Incomplete
    _convert_categoricals: Incomplete
    _index_col: Incomplete
    _convert_missing: Incomplete
    _preserve_dtypes: Incomplete
    _columns: Incomplete
    _order_categoricals: Incomplete
    _original_path_or_buf: Incomplete
    _compression: Incomplete
    _storage_options: Incomplete
    _encoding: str
    _chunksize: Incomplete
    _using_iterator: bool
    _entered: bool
    _close_file: Callable[[], None] | None
    _missing_values: bool
    _can_read_value_labels: bool
    _column_selector_set: bool
    _value_labels_read: bool
    _data_read: bool
    _dtype: np.dtype | None
    _lines_read: int
    _native_byteorder: Incomplete
    def __init__(self, path_or_buf: FilePath | ReadBuffer[bytes], convert_dates: bool = True, convert_categoricals: bool = True, index_col: str | None = None, convert_missing: bool = False, preserve_dtypes: bool = True, columns: Sequence[str] | None = None, order_categoricals: bool = True, chunksize: int | None = None, compression: CompressionOptions = 'infer', storage_options: StorageOptions | None = None) -> None: ...
    def _ensure_open(self) -> None:
        """
        Ensure the file has been opened and its header data read.
        """
    def _open_file(self) -> None:
        """
        Open the file (with compression options, etc.), and read header information.
        """
    def __enter__(self) -> Self:
        """enter context manager"""
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None: ...
    def close(self) -> None:
        """Close the handle if its open.

        .. deprecated: 2.0.0

           The close method is not part of the public API.
           The only supported way to use StataReader is to use it as a context manager.
        """
    def _set_encoding(self) -> None:
        """
        Set string encoding which depends on file version
        """
    def _read_int8(self) -> int: ...
    def _read_uint8(self) -> int: ...
    def _read_uint16(self) -> int: ...
    def _read_uint32(self) -> int: ...
    def _read_uint64(self) -> int: ...
    def _read_int16(self) -> int: ...
    def _read_int32(self) -> int: ...
    def _read_int64(self) -> int: ...
    def _read_char8(self) -> bytes: ...
    def _read_int16_count(self, count: int) -> tuple[int, ...]: ...
    def _read_header(self) -> None: ...
    _format_version: Incomplete
    _byteorder: Incomplete
    _nvar: Incomplete
    _nobs: Incomplete
    _data_label: Incomplete
    _time_stamp: Incomplete
    _seek_vartypes: Incomplete
    _seek_varnames: Incomplete
    _seek_sortlist: Incomplete
    _seek_formats: Incomplete
    _seek_value_label_names: Incomplete
    _seek_variable_labels: Incomplete
    _data_location: Incomplete
    _seek_strls: Incomplete
    _seek_value_labels: Incomplete
    _varlist: Incomplete
    _srtlist: Incomplete
    _fmtlist: Incomplete
    _lbllist: Incomplete
    _variable_labels: Incomplete
    def _read_new_header(self) -> None: ...
    def _get_dtypes(self, seek_vartypes: int) -> tuple[list[int | str], list[str | np.dtype]]: ...
    def _get_varlist(self) -> list[str]: ...
    def _get_fmtlist(self) -> list[str]: ...
    def _get_lbllist(self) -> list[str]: ...
    def _get_variable_labels(self) -> list[str]: ...
    def _get_nobs(self) -> int: ...
    def _get_data_label(self) -> str: ...
    def _get_time_stamp(self) -> str: ...
    def _get_seek_variable_labels(self) -> int: ...
    _filetype: Incomplete
    _typlist: Incomplete
    _dtyplist: Incomplete
    def _read_old_header(self, first_char: bytes) -> None: ...
    def _setup_dtype(self) -> np.dtype:
        """Map between numpy and state dtypes"""
    def _decode(self, s: bytes) -> str: ...
    _value_label_dict: dict[str, dict[float, str]]
    def _read_value_labels(self) -> None: ...
    GSO: Incomplete
    def _read_strls(self) -> None: ...
    def __next__(self) -> DataFrame: ...
    def get_chunk(self, size: int | None = None) -> DataFrame:
        """
        Reads lines from Stata file and returns as dataframe

        Parameters
        ----------
        size : int, defaults to None
            Number of lines to read.  If None, reads whole file.

        Returns
        -------
        DataFrame
        """
    def read(self, nrows: int | None = None, convert_dates: bool | None = None, convert_categoricals: bool | None = None, index_col: str | None = None, convert_missing: bool | None = None, preserve_dtypes: bool | None = None, columns: Sequence[str] | None = None, order_categoricals: bool | None = None) -> DataFrame: ...
    def _do_convert_missing(self, data: DataFrame, convert_missing: bool) -> DataFrame: ...
    def _insert_strls(self, data: DataFrame) -> DataFrame: ...
    def _do_select_columns(self, data: DataFrame, columns: Sequence[str]) -> DataFrame: ...
    def _do_convert_categoricals(self, data: DataFrame, value_label_dict: dict[str, dict[float, str]], lbllist: Sequence[str], order_categoricals: bool) -> DataFrame:
        """
        Converts categorical columns to Categorical type.
        """
    @property
    def data_label(self) -> str:
        '''
        Return data label of Stata file.

        Examples
        --------
        >>> df = pd.DataFrame([(1,)], columns=["variable"])
        >>> time_stamp = pd.Timestamp(2000, 2, 29, 14, 21)
        >>> data_label = "This is a data file."
        >>> path = "/My_path/filename.dta"
        >>> df.to_stata(path, time_stamp=time_stamp,    # doctest: +SKIP
        ...             data_label=data_label,  # doctest: +SKIP
        ...             version=None)  # doctest: +SKIP
        >>> with pd.io.stata.StataReader(path) as reader:  # doctest: +SKIP
        ...     print(reader.data_label)  # doctest: +SKIP
        This is a data file.
        '''
    @property
    def time_stamp(self) -> str:
        """
        Return time stamp of Stata file.
        """
    def variable_labels(self) -> dict[str, str]:
        '''
        Return a dict associating each variable name with corresponding label.

        Returns
        -------
        dict

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=["col_1", "col_2"])
        >>> time_stamp = pd.Timestamp(2000, 2, 29, 14, 21)
        >>> path = "/My_path/filename.dta"
        >>> variable_labels = {"col_1": "This is an example"}
        >>> df.to_stata(path, time_stamp=time_stamp,  # doctest: +SKIP
        ...             variable_labels=variable_labels, version=None)  # doctest: +SKIP
        >>> with pd.io.stata.StataReader(path) as reader:  # doctest: +SKIP
        ...     print(reader.variable_labels())  # doctest: +SKIP
        {\'index\': \'\', \'col_1\': \'This is an example\', \'col_2\': \'\'}
        >>> pd.read_stata(path)  # doctest: +SKIP
            index col_1 col_2
        0       0    1    2
        1       1    3    4
        '''
    def value_labels(self) -> dict[str, dict[float, str]]:
        '''
        Return a nested dict associating each variable name to its value and label.

        Returns
        -------
        dict

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=["col_1", "col_2"])
        >>> time_stamp = pd.Timestamp(2000, 2, 29, 14, 21)
        >>> path = "/My_path/filename.dta"
        >>> value_labels = {"col_1": {3: "x"}}
        >>> df.to_stata(path, time_stamp=time_stamp,  # doctest: +SKIP
        ...             value_labels=value_labels, version=None)  # doctest: +SKIP
        >>> with pd.io.stata.StataReader(path) as reader:  # doctest: +SKIP
        ...     print(reader.value_labels())  # doctest: +SKIP
        {\'col_1\': {3: \'x\'}}
        >>> pd.read_stata(path)  # doctest: +SKIP
            index col_1 col_2
        0       0    1    2
        1       1    x    4
        '''

def read_stata(filepath_or_buffer: FilePath | ReadBuffer[bytes], *, convert_dates: bool = True, convert_categoricals: bool = True, index_col: str | None = None, convert_missing: bool = False, preserve_dtypes: bool = True, columns: Sequence[str] | None = None, order_categoricals: bool = True, chunksize: int | None = None, iterator: bool = False, compression: CompressionOptions = 'infer', storage_options: StorageOptions | None = None) -> DataFrame | StataReader: ...
def _set_endianness(endianness: str) -> str: ...
def _pad_bytes(name: AnyStr, length: int) -> AnyStr:
    """
    Take a char string and pads it with null bytes until it's length chars.
    """
def _convert_datetime_to_stata_type(fmt: str) -> np.dtype:
    """
    Convert from one of the stata date formats to a type in TYPE_MAP.
    """
def _maybe_convert_to_int_keys(convert_dates: dict, varlist: list[Hashable]) -> dict: ...
def _dtype_to_stata_type(dtype: np.dtype, column: Series) -> int:
    """
    Convert dtype types to stata types. Returns the byte of the given ordinal.
    See TYPE_MAP and comments for an explanation. This is also explained in
    the dta spec.
    1 - 244 are strings of this length
                         Pandas    Stata
    251 - for int8      byte
    252 - for int16     int
    253 - for int32     long
    254 - for float32   float
    255 - for double    double

    If there are dates to convert, then dtype will already have the correct
    type inserted.
    """
def _dtype_to_default_stata_fmt(dtype, column: Series, dta_version: int = 114, force_strl: bool = False) -> str:
    '''
    Map numpy dtype to stata\'s default format for this type. Not terribly
    important since users can change this in Stata. Semantics are

    object  -> "%DDs" where DD is the length of the string.  If not a string,
                raise ValueError
    float64 -> "%10.0g"
    float32 -> "%9.0g"
    int64   -> "%9.0g"
    int32   -> "%12.0g"
    int16   -> "%8.0g"
    int8    -> "%8.0g"
    strl    -> "%9s"
    '''

class StataWriter(StataParser):
    '''
    A class for writing Stata binary dta files

    Parameters
    ----------
    fname : path (string), buffer or path object
        string, path object (pathlib.Path or py._path.local.LocalPath) or
        object implementing a binary write() functions. If using a buffer
        then the buffer will not be automatically closed after the file
        is written.
    data : DataFrame
        Input to save
    convert_dates : dict
        Dictionary mapping columns containing datetime types to stata internal
        format to use when writing the dates. Options are \'tc\', \'td\', \'tm\',
        \'tw\', \'th\', \'tq\', \'ty\'. Column can be either an integer or a name.
        Datetime columns that do not have a conversion type specified will be
        converted to \'tc\'. Raises NotImplementedError if a datetime column has
        timezone information
    write_index : bool
        Write the index to Stata dataset.
    byteorder : str
        Can be ">", "<", "little", or "big". default is `sys.byteorder`
    time_stamp : datetime
        A datetime to use as file creation date.  Default is the current time
    data_label : str
        A label for the data set.  Must be 80 characters or smaller.
    variable_labels : dict
        Dictionary containing columns as keys and variable labels as values.
        Each label must be 80 characters or smaller.
    {compression_options}

        .. versionchanged:: 1.4.0 Zstandard support.

    {storage_options}

    value_labels : dict of dicts
        Dictionary containing columns as keys and dictionaries of column value
        to labels as values. The combined length of all labels for a single
        variable must be 32,000 characters or smaller.

        .. versionadded:: 1.4.0

    Returns
    -------
    writer : StataWriter instance
        The StataWriter instance has a write_file method, which will
        write the file to the given `fname`.

    Raises
    ------
    NotImplementedError
        * If datetimes contain timezone information
    ValueError
        * Columns listed in convert_dates are neither datetime64[ns]
          or datetime
        * Column dtype is not representable in Stata
        * Column listed in convert_dates is not in DataFrame
        * Categorical label contains more than 32,000 characters

    Examples
    --------
    >>> data = pd.DataFrame([[1.0, 1]], columns=[\'a\', \'b\'])
    >>> writer = StataWriter(\'./data_file.dta\', data)
    >>> writer.write_file()

    Directly write a zip file
    >>> compression = {{"method": "zip", "archive_name": "data_file.dta"}}
    >>> writer = StataWriter(\'./data_file.zip\', data, compression=compression)
    >>> writer.write_file()

    Save a DataFrame with dates
    >>> from datetime import datetime
    >>> data = pd.DataFrame([[datetime(2000,1,1)]], columns=[\'date\'])
    >>> writer = StataWriter(\'./date_data_file.dta\', data, {{\'date\' : \'tw\'}})
    >>> writer.write_file()
    '''
    _max_string_length: int
    _encoding: Literal['latin-1', 'utf-8']
    data: Incomplete
    _convert_dates: Incomplete
    _write_index: Incomplete
    _time_stamp: Incomplete
    _data_label: Incomplete
    _variable_labels: Incomplete
    _non_cat_value_labels: Incomplete
    _value_labels: list[StataValueLabel]
    _has_value_labels: Incomplete
    _compression: Incomplete
    _output_file: IO[bytes] | None
    _converted_names: dict[Hashable, str]
    storage_options: Incomplete
    _byteorder: Incomplete
    _fname: Incomplete
    type_converters: Incomplete
    def __init__(self, fname: FilePath | WriteBuffer[bytes], data: DataFrame, convert_dates: dict[Hashable, str] | None = None, write_index: bool = True, byteorder: str | None = None, time_stamp: datetime | None = None, data_label: str | None = None, variable_labels: dict[Hashable, str] | None = None, compression: CompressionOptions = 'infer', storage_options: StorageOptions | None = None, *, value_labels: dict[Hashable, dict[float, str]] | None = None) -> None: ...
    def _write(self, to_write: str) -> None:
        """
        Helper to call encode before writing to file for Python 3 compat.
        """
    def _write_bytes(self, value: bytes) -> None:
        """
        Helper to assert file is open before writing.
        """
    def _prepare_non_cat_value_labels(self, data: DataFrame) -> list[StataNonCatValueLabel]:
        """
        Check for value labels provided for non-categorical columns. Value
        labels
        """
    def _prepare_categoricals(self, data: DataFrame) -> DataFrame:
        """
        Check for categorical columns, retain categorical information for
        Stata file and convert categorical data to int
        """
    def _replace_nans(self, data: DataFrame) -> DataFrame:
        """
        Checks floating point data columns for nans, and replaces these with
        the generic Stata for missing value (.)
        """
    def _update_strl_names(self) -> None:
        """No-op, forward compatibility"""
    def _validate_variable_name(self, name: str) -> str:
        """
        Validate variable names for Stata export.

        Parameters
        ----------
        name : str
            Variable name

        Returns
        -------
        str
            The validated name with invalid characters replaced with
            underscores.

        Notes
        -----
        Stata 114 and 117 support ascii characters in a-z, A-Z, 0-9
        and _.
        """
    def _check_column_names(self, data: DataFrame) -> DataFrame:
        """
        Checks column names to ensure that they are valid Stata column names.
        This includes checks for:
            * Non-string names
            * Stata keywords
            * Variables that start with numbers
            * Variables with names that are too long

        When an illegal variable name is detected, it is converted, and if
        dates are exported, the variable name is propagated to the date
        conversion dictionary
        """
    fmtlist: list[str]
    typlist: list[int]
    def _set_formats_and_types(self, dtypes: Series) -> None: ...
    varlist: Incomplete
    def _prepare_pandas(self, data: DataFrame) -> None: ...
    def _encode_strings(self) -> None:
        """
        Encode strings in dta-specific encoding

        Do not encode columns marked for date conversion or for strL
        conversion. The strL converter independently handles conversion and
        also accepts empty string arrays.
        """
    def write_file(self) -> None:
        '''
        Export DataFrame object to Stata dta format.

        Examples
        --------
        >>> df = pd.DataFrame({"fully_labelled": [1, 2, 3, 3, 1],
        ...                    "partially_labelled": [1.0, 2.0, np.nan, 9.0, np.nan],
        ...                    "Y": [7, 7, 9, 8, 10],
        ...                    "Z": pd.Categorical(["j", "k", "l", "k", "j"]),
        ...                    })
        >>> path = "/My_path/filename.dta"
        >>> labels = {"fully_labelled": {1: "one", 2: "two", 3: "three"},
        ...           "partially_labelled": {1.0: "one", 2.0: "two"},
        ...           }
        >>> writer = pd.io.stata.StataWriter(path,
        ...                                  df,
        ...                                  value_labels=labels)  # doctest: +SKIP
        >>> writer.write_file()  # doctest: +SKIP
        >>> df = pd.read_stata(path)  # doctest: +SKIP
        >>> df  # doctest: +SKIP
            index fully_labelled  partially_labeled  Y  Z
        0       0            one                one  7  j
        1       1            two                two  7  k
        2       2          three                NaN  9  l
        3       3          three                9.0  8  k
        4       4            one                NaN 10  j
        '''
    def _close(self) -> None:
        """
        Close the file if it was created by the writer.

        If a buffer or file-like object was passed in, for example a GzipFile,
        then leave this file open for the caller to close.
        """
    def _write_map(self) -> None:
        """No-op, future compatibility"""
    def _write_file_close_tag(self) -> None:
        """No-op, future compatibility"""
    def _write_characteristics(self) -> None:
        """No-op, future compatibility"""
    def _write_strls(self) -> None:
        """No-op, future compatibility"""
    def _write_expansion_fields(self) -> None:
        """Write 5 zeros for expansion fields"""
    def _write_value_labels(self) -> None: ...
    def _write_header(self, data_label: str | None = None, time_stamp: datetime | None = None) -> None: ...
    def _write_variable_types(self) -> None: ...
    def _write_varnames(self) -> None: ...
    def _write_sortlist(self) -> None: ...
    def _write_formats(self) -> None: ...
    def _write_value_label_names(self) -> None: ...
    def _write_variable_labels(self) -> None: ...
    def _convert_strls(self, data: DataFrame) -> DataFrame:
        """No-op, future compatibility"""
    def _prepare_data(self) -> np.rec.recarray: ...
    def _write_data(self, records: np.rec.recarray) -> None: ...
    @staticmethod
    def _null_terminate_str(s: str) -> str: ...
    def _null_terminate_bytes(self, s: str) -> bytes: ...

def _dtype_to_stata_type_117(dtype: np.dtype, column: Series, force_strl: bool) -> int:
    """
    Converts dtype types to stata types. Returns the byte of the given ordinal.
    See TYPE_MAP and comments for an explanation. This is also explained in
    the dta spec.
    1 - 2045 are strings of this length
                Pandas    Stata
    32768 - for object    strL
    65526 - for int8      byte
    65527 - for int16     int
    65528 - for int32     long
    65529 - for float32   float
    65530 - for double    double

    If there are dates to convert, then dtype will already have the correct
    type inserted.
    """
def _pad_bytes_new(name: str | bytes, length: int) -> bytes:
    """
    Takes a bytes instance and pads it with null bytes until it's length chars.
    """

class StataStrLWriter:
    '''
    Converter for Stata StrLs

    Stata StrLs map 8 byte values to strings which are stored using a
    dictionary-like format where strings are keyed to two values.

    Parameters
    ----------
    df : DataFrame
        DataFrame to convert
    columns : Sequence[str]
        List of columns names to convert to StrL
    version : int, optional
        dta version.  Currently supports 117, 118 and 119
    byteorder : str, optional
        Can be ">", "<", "little", or "big". default is `sys.byteorder`

    Notes
    -----
    Supports creation of the StrL block of a dta file for dta versions
    117, 118 and 119.  These differ in how the GSO is stored.  118 and
    119 store the GSO lookup value as a uint32 and a uint64, while 117
    uses two uint32s. 118 and 119 also encode all strings as unicode
    which is required by the format.  117 uses \'latin-1\' a fixed width
    encoding that extends the 7-bit ascii table with an additional 128
    characters.
    '''
    _dta_ver: Incomplete
    df: Incomplete
    columns: Incomplete
    _gso_table: Incomplete
    _byteorder: Incomplete
    _encoding: str
    _o_offet: Incomplete
    _gso_o_type: Incomplete
    _gso_v_type: Incomplete
    def __init__(self, df: DataFrame, columns: Sequence[str], version: int = 117, byteorder: str | None = None) -> None: ...
    def _convert_key(self, key: tuple[int, int]) -> int: ...
    def generate_table(self) -> tuple[dict[str, tuple[int, int]], DataFrame]:
        """
        Generates the GSO lookup table for the DataFrame

        Returns
        -------
        gso_table : dict
            Ordered dictionary using the string found as keys
            and their lookup position (v,o) as values
        gso_df : DataFrame
            DataFrame where strl columns have been converted to
            (v,o) values

        Notes
        -----
        Modifies the DataFrame in-place.

        The DataFrame returned encodes the (v,o) values as uint64s. The
        encoding depends on the dta version, and can be expressed as

        enc = v + o * 2 ** (o_size * 8)

        so that v is stored in the lower bits and o is in the upper
        bits. o_size is

          * 117: 4
          * 118: 6
          * 119: 5
        """
    def generate_blob(self, gso_table: dict[str, tuple[int, int]]) -> bytes:
        """
        Generates the binary blob of GSOs that is written to the dta file.

        Parameters
        ----------
        gso_table : dict
            Ordered dictionary (str, vo)

        Returns
        -------
        gso : bytes
            Binary content of dta file to be placed between strl tags

        Notes
        -----
        Output format depends on dta version.  117 uses two uint32s to
        express v and o while 118+ uses a uint32 for v and a uint64 for o.
        """

class StataWriter117(StataWriter):
    '''
    A class for writing Stata binary dta files in Stata 13 format (117)

    Parameters
    ----------
    fname : path (string), buffer or path object
        string, path object (pathlib.Path or py._path.local.LocalPath) or
        object implementing a binary write() functions. If using a buffer
        then the buffer will not be automatically closed after the file
        is written.
    data : DataFrame
        Input to save
    convert_dates : dict
        Dictionary mapping columns containing datetime types to stata internal
        format to use when writing the dates. Options are \'tc\', \'td\', \'tm\',
        \'tw\', \'th\', \'tq\', \'ty\'. Column can be either an integer or a name.
        Datetime columns that do not have a conversion type specified will be
        converted to \'tc\'. Raises NotImplementedError if a datetime column has
        timezone information
    write_index : bool
        Write the index to Stata dataset.
    byteorder : str
        Can be ">", "<", "little", or "big". default is `sys.byteorder`
    time_stamp : datetime
        A datetime to use as file creation date.  Default is the current time
    data_label : str
        A label for the data set.  Must be 80 characters or smaller.
    variable_labels : dict
        Dictionary containing columns as keys and variable labels as values.
        Each label must be 80 characters or smaller.
    convert_strl : list
        List of columns names to convert to Stata StrL format.  Columns with
        more than 2045 characters are automatically written as StrL.
        Smaller columns can be converted by including the column name.  Using
        StrLs can reduce output file size when strings are longer than 8
        characters, and either frequently repeated or sparse.
    {compression_options}

        .. versionchanged:: 1.4.0 Zstandard support.

    value_labels : dict of dicts
        Dictionary containing columns as keys and dictionaries of column value
        to labels as values. The combined length of all labels for a single
        variable must be 32,000 characters or smaller.

        .. versionadded:: 1.4.0

    Returns
    -------
    writer : StataWriter117 instance
        The StataWriter117 instance has a write_file method, which will
        write the file to the given `fname`.

    Raises
    ------
    NotImplementedError
        * If datetimes contain timezone information
    ValueError
        * Columns listed in convert_dates are neither datetime64[ns]
          or datetime
        * Column dtype is not representable in Stata
        * Column listed in convert_dates is not in DataFrame
        * Categorical label contains more than 32,000 characters

    Examples
    --------
    >>> data = pd.DataFrame([[1.0, 1, \'a\']], columns=[\'a\', \'b\', \'c\'])
    >>> writer = pd.io.stata.StataWriter117(\'./data_file.dta\', data)
    >>> writer.write_file()

    Directly write a zip file
    >>> compression = {"method": "zip", "archive_name": "data_file.dta"}
    >>> writer = pd.io.stata.StataWriter117(
    ...     \'./data_file.zip\', data, compression=compression
    ...     )
    >>> writer.write_file()

    Or with long strings stored in strl format
    >>> data = pd.DataFrame([[\'A relatively long string\'], [\'\'], [\'\']],
    ...                     columns=[\'strls\'])
    >>> writer = pd.io.stata.StataWriter117(
    ...     \'./data_file_with_long_strings.dta\', data, convert_strl=[\'strls\'])
    >>> writer.write_file()
    '''
    _max_string_length: int
    _dta_version: int
    _convert_strl: list[Hashable]
    _map: dict[str, int]
    _strl_blob: bytes
    def __init__(self, fname: FilePath | WriteBuffer[bytes], data: DataFrame, convert_dates: dict[Hashable, str] | None = None, write_index: bool = True, byteorder: str | None = None, time_stamp: datetime | None = None, data_label: str | None = None, variable_labels: dict[Hashable, str] | None = None, convert_strl: Sequence[Hashable] | None = None, compression: CompressionOptions = 'infer', storage_options: StorageOptions | None = None, *, value_labels: dict[Hashable, dict[float, str]] | None = None) -> None: ...
    @staticmethod
    def _tag(val: str | bytes, tag: str) -> bytes:
        """Surround val with <tag></tag>"""
    def _update_map(self, tag: str) -> None:
        """Update map location for tag with file position"""
    def _write_header(self, data_label: str | None = None, time_stamp: datetime | None = None) -> None:
        """Write the file header"""
    def _write_map(self) -> None:
        """
        Called twice during file write. The first populates the values in
        the map with 0s.  The second call writes the final map locations when
        all blocks have been written.
        """
    def _write_variable_types(self) -> None: ...
    def _write_varnames(self) -> None: ...
    def _write_sortlist(self) -> None: ...
    def _write_formats(self) -> None: ...
    def _write_value_label_names(self) -> None: ...
    def _write_variable_labels(self) -> None: ...
    def _write_characteristics(self) -> None: ...
    def _write_data(self, records) -> None: ...
    def _write_strls(self) -> None: ...
    def _write_expansion_fields(self) -> None:
        """No-op in dta 117+"""
    def _write_value_labels(self) -> None: ...
    def _write_file_close_tag(self) -> None: ...
    def _update_strl_names(self) -> None:
        """
        Update column names for conversion to strl if they might have been
        changed to comply with Stata naming rules
        """
    def _convert_strls(self, data: DataFrame) -> DataFrame:
        """
        Convert columns to StrLs if either very large or in the
        convert_strl variable
        """
    typlist: Incomplete
    fmtlist: Incomplete
    def _set_formats_and_types(self, dtypes: Series) -> None: ...

class StataWriterUTF8(StataWriter117):
    '''
    Stata binary dta file writing in Stata 15 (118) and 16 (119) formats

    DTA 118 and 119 format files support unicode string data (both fixed
    and strL) format. Unicode is also supported in value labels, variable
    labels and the dataset label. Format 119 is automatically used if the
    file contains more than 32,767 variables.

    Parameters
    ----------
    fname : path (string), buffer or path object
        string, path object (pathlib.Path or py._path.local.LocalPath) or
        object implementing a binary write() functions. If using a buffer
        then the buffer will not be automatically closed after the file
        is written.
    data : DataFrame
        Input to save
    convert_dates : dict, default None
        Dictionary mapping columns containing datetime types to stata internal
        format to use when writing the dates. Options are \'tc\', \'td\', \'tm\',
        \'tw\', \'th\', \'tq\', \'ty\'. Column can be either an integer or a name.
        Datetime columns that do not have a conversion type specified will be
        converted to \'tc\'. Raises NotImplementedError if a datetime column has
        timezone information
    write_index : bool, default True
        Write the index to Stata dataset.
    byteorder : str, default None
        Can be ">", "<", "little", or "big". default is `sys.byteorder`
    time_stamp : datetime, default None
        A datetime to use as file creation date.  Default is the current time
    data_label : str, default None
        A label for the data set.  Must be 80 characters or smaller.
    variable_labels : dict, default None
        Dictionary containing columns as keys and variable labels as values.
        Each label must be 80 characters or smaller.
    convert_strl : list, default None
        List of columns names to convert to Stata StrL format.  Columns with
        more than 2045 characters are automatically written as StrL.
        Smaller columns can be converted by including the column name.  Using
        StrLs can reduce output file size when strings are longer than 8
        characters, and either frequently repeated or sparse.
    version : int, default None
        The dta version to use. By default, uses the size of data to determine
        the version. 118 is used if data.shape[1] <= 32767, and 119 is used
        for storing larger DataFrames.
    {compression_options}

        .. versionchanged:: 1.4.0 Zstandard support.

    value_labels : dict of dicts
        Dictionary containing columns as keys and dictionaries of column value
        to labels as values. The combined length of all labels for a single
        variable must be 32,000 characters or smaller.

        .. versionadded:: 1.4.0

    Returns
    -------
    StataWriterUTF8
        The instance has a write_file method, which will write the file to the
        given `fname`.

    Raises
    ------
    NotImplementedError
        * If datetimes contain timezone information
    ValueError
        * Columns listed in convert_dates are neither datetime64[ns]
          or datetime
        * Column dtype is not representable in Stata
        * Column listed in convert_dates is not in DataFrame
        * Categorical label contains more than 32,000 characters

    Examples
    --------
    Using Unicode data and column names

    >>> from pandas.io.stata import StataWriterUTF8
    >>> data = pd.DataFrame([[1.0, 1, \'ᴬ\']], columns=[\'a\', \'β\', \'ĉ\'])
    >>> writer = StataWriterUTF8(\'./data_file.dta\', data)
    >>> writer.write_file()

    Directly write a zip file
    >>> compression = {"method": "zip", "archive_name": "data_file.dta"}
    >>> writer = StataWriterUTF8(\'./data_file.zip\', data, compression=compression)
    >>> writer.write_file()

    Or with long strings stored in strl format

    >>> data = pd.DataFrame([[\'ᴀ relatively long ŝtring\'], [\'\'], [\'\']],
    ...                     columns=[\'strls\'])
    >>> writer = StataWriterUTF8(\'./data_file_with_long_strings.dta\', data,
    ...                          convert_strl=[\'strls\'])
    >>> writer.write_file()
    '''
    _encoding: Literal['utf-8']
    _dta_version: Incomplete
    def __init__(self, fname: FilePath | WriteBuffer[bytes], data: DataFrame, convert_dates: dict[Hashable, str] | None = None, write_index: bool = True, byteorder: str | None = None, time_stamp: datetime | None = None, data_label: str | None = None, variable_labels: dict[Hashable, str] | None = None, convert_strl: Sequence[Hashable] | None = None, version: int | None = None, compression: CompressionOptions = 'infer', storage_options: StorageOptions | None = None, *, value_labels: dict[Hashable, dict[float, str]] | None = None) -> None: ...
    def _validate_variable_name(self, name: str) -> str:
        """
        Validate variable names for Stata export.

        Parameters
        ----------
        name : str
            Variable name

        Returns
        -------
        str
            The validated name with invalid characters replaced with
            underscores.

        Notes
        -----
        Stata 118+ support most unicode characters. The only limitation is in
        the ascii range where the characters supported are a-z, A-Z, 0-9 and _.
        """
