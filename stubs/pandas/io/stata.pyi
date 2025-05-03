import _abc
import collections.abc
import datetime
import np
import np.rec
import pandas._libs.lib as lib
from pandas._libs.algos import ensure_object as ensure_object
from pandas._libs.lib import infer_dtype as infer_dtype
from pandas._libs.tslibs.nattype import NaT as NaT
from pandas._libs.tslibs.timestamps import Timestamp as Timestamp
from pandas._libs.writers import max_len_string_array as max_len_string_array
from pandas.core.arrays.categorical import Categorical as Categorical
from pandas.core.dtypes.base import ExtensionDtype as ExtensionDtype
from pandas.core.dtypes.common import is_numeric_dtype as is_numeric_dtype, is_string_dtype as is_string_dtype
from pandas.core.dtypes.dtypes import CategoricalDtype as CategoricalDtype
from pandas.core.dtypes.missing import isna as isna
from pandas.core.frame import DataFrame as DataFrame
from pandas.core.indexes.base import Index as Index
from pandas.core.indexes.datetimes import DatetimeIndex as DatetimeIndex
from pandas.core.indexes.range import RangeIndex as RangeIndex
from pandas.core.series import Series as Series
from pandas.core.tools.datetimes import to_datetime as to_datetime
from pandas.core.tools.timedeltas import to_timedelta as to_timedelta
from pandas.errors import CategoricalConversionWarning as CategoricalConversionWarning, InvalidColumnName as InvalidColumnName, PossiblePrecisionLoss as PossiblePrecisionLoss, ValueLabelTypeMismatch as ValueLabelTypeMismatch
from pandas.io.common import get_handle as get_handle
from pandas.util._decorators import Appender as Appender, doc as doc
from pandas.util._exceptions import find_stack_level as find_stack_level
from typing import AnyStr, ClassVar

TYPE_CHECKING: bool
_shared_docs: dict
_version_error: str
_statafile_processing_params1: str
_statafile_processing_params2: str
_chunksize_params: str
_iterator_params: str
_reader_notes: str
_read_stata_doc: str
_read_method_doc: str
_stata_reader_doc: str
_date_formats: list
stata_epoch: datetime.datetime
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

excessive_string_length_error: str
precision_loss_doc: str
value_label_mismatch_doc: str
invalid_name_doc: str
categorical_conversion_warning: str
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
    def __init__(self, catarray: Series, encoding: Literal['latin-1', 'utf-8'] = ...) -> None: ...
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
    def __init__(self, labname: str, value_labels: dict[float, str], encoding: Literal['latin-1', 'utf-8'] = ...) -> None: ...

class StataMissingValue:
    MISSING_VALUES: ClassVar[dict] = ...
    bases: ClassVar[tuple] = ...
    b: ClassVar[int] = ...
    i: ClassVar[int] = ...
    float32_base: ClassVar[bytes] = ...
    increment_32: ClassVar[int] = ...
    key: ClassVar[float] = ...
    int_value: ClassVar[int] = ...
    float64_base: ClassVar[bytes] = ...
    increment_64: ClassVar[int] = ...
    BASE_MISSING_VALUES: ClassVar[dict] = ...
    def __init__(self, value: float) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    @classmethod
    def get_base_missing_value(cls, dtype: np.dtype) -> float: ...
    @property
    def string(self): ...
    @property
    def value(self): ...

class StataParser:
    def __init__(self) -> None: ...

class StataReader(StataParser, collections.abc.Iterator):
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    def __init__(self, path_or_buf: FilePath | ReadBuffer[bytes], convert_dates: bool = ..., convert_categoricals: bool = ..., index_col: str | None, convert_missing: bool = ..., preserve_dtypes: bool = ..., columns: Sequence[str] | None, order_categoricals: bool = ..., chunksize: int | None, compression: CompressionOptions = ..., storage_options: StorageOptions | None) -> None: ...
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
    def _read_old_header(self, first_char: bytes) -> None: ...
    def _setup_dtype(self) -> np.dtype:
        """Map between numpy and state dtypes"""
    def _decode(self, s: bytes) -> str: ...
    def _read_value_labels(self) -> None: ...
    def _read_strls(self) -> None: ...
    def __next__(self) -> DataFrame: ...
    def get_chunk(self, size: int | None) -> DataFrame:
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
    def read(self, nrows: int | None, convert_dates: bool | None, convert_categoricals: bool | None, index_col: str | None, convert_missing: bool | None, preserve_dtypes: bool | None, columns: Sequence[str] | None, order_categoricals: bool | None) -> DataFrame:
        """Reads observations from Stata file, converting them into a dataframe

        Parameters
        ----------
        nrows : int
            Number of lines to read from data file, if None read whole file.
        convert_dates : bool, default True
            Convert date variables to DataFrame time values.
        convert_categoricals : bool, default True
            Read value labels and convert columns to Categorical/Factor variables.
        index_col : str, optional
            Column to set as index.
        convert_missing : bool, default False
            Flag indicating whether to convert missing values to their Stata
            representations.  If False, missing values are replaced with nan.
            If True, columns containing missing values are returned with
            object data types and missing values are represented by
            StataMissingValue objects.
        preserve_dtypes : bool, default True
            Preserve Stata datatypes. If False, numeric data are upcast to pandas
            default types for foreign data (float64 or int64).
        columns : list or None
            Columns to retain.  Columns will be returned in the given order.  None
            returns all columns.
        order_categoricals : bool, default True
            Flag indicating whether converted categorical data are ordered.

        Returns
        -------
        DataFrame
        """
    def _do_convert_missing(self, data: DataFrame, convert_missing: bool) -> DataFrame: ...
    def _insert_strls(self, data: DataFrame) -> DataFrame: ...
    def _do_select_columns(self, data: DataFrame, columns: Sequence[str]) -> DataFrame: ...
    def _do_convert_categoricals(self, data: DataFrame, value_label_dict: dict[str, dict[float, str]], lbllist: Sequence[str], order_categoricals: bool) -> DataFrame:
        """
        Converts categorical columns to Categorical type.
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
    @property
    def data_label(self): ...
    @property
    def time_stamp(self): ...
def read_stata(filepath_or_buffer: FilePath | ReadBuffer[bytes], *, convert_dates: bool = ..., convert_categoricals: bool = ..., index_col: str | None, convert_missing: bool = ..., preserve_dtypes: bool = ..., columns: Sequence[str] | None, order_categoricals: bool = ..., chunksize: int | None, iterator: bool = ..., compression: CompressionOptions = ..., storage_options: StorageOptions | None) -> DataFrame | StataReader:
    '''
    Read Stata file into DataFrame.

    Parameters
    ----------
    filepath_or_buffer : str, path object or file-like object
        Any valid string path is acceptable. The string could be a URL. Valid
        URL schemes include http, ftp, s3, and file. For file URLs, a host is
        expected. A local file could be: ``file://localhost/path/to/table.dta``.

        If you want to pass in a path object, pandas accepts any ``os.PathLike``.

        By file-like object, we refer to objects with a ``read()`` method,
        such as a file handle (e.g. via builtin ``open`` function)
        or ``StringIO``.
    convert_dates : bool, default True
        Convert date variables to DataFrame time values.
    convert_categoricals : bool, default True
        Read value labels and convert columns to Categorical/Factor variables.
    index_col : str, optional
        Column to set as index.
    convert_missing : bool, default False
        Flag indicating whether to convert missing values to their Stata
        representations.  If False, missing values are replaced with nan.
        If True, columns containing missing values are returned with
        object data types and missing values are represented by
        StataMissingValue objects.
    preserve_dtypes : bool, default True
        Preserve Stata datatypes. If False, numeric data are upcast to pandas
        default types for foreign data (float64 or int64).
    columns : list or None
        Columns to retain.  Columns will be returned in the given order.  None
        returns all columns.
    order_categoricals : bool, default True
        Flag indicating whether converted categorical data are ordered.
    chunksize : int, default None
        Return StataReader object for iterations, returns chunks with
        given number of lines.
    iterator : bool, default False
        Return StataReader object.
    compression : str or dict, default \'infer\'
        For on-the-fly decompression of on-disk data. If \'infer\' and \'filepath_or_buffer\' is
        path-like, then detect compression from the following extensions: \'.gz\',
        \'.bz2\', \'.zip\', \'.xz\', \'.zst\', \'.tar\', \'.tar.gz\', \'.tar.xz\' or \'.tar.bz2\'
        (otherwise no compression).
        If using \'zip\' or \'tar\', the ZIP file must contain only one data file to be read in.
        Set to ``None`` for no decompression.
        Can also be a dict with key ``\'method\'`` set
        to one of {``\'zip\'``, ``\'gzip\'``, ``\'bz2\'``, ``\'zstd\'``, ``\'xz\'``, ``\'tar\'``} and
        other key-value pairs are forwarded to
        ``zipfile.ZipFile``, ``gzip.GzipFile``,
        ``bz2.BZ2File``, ``zstandard.ZstdDecompressor``, ``lzma.LZMAFile`` or
        ``tarfile.TarFile``, respectively.
        As an example, the following could be passed for Zstandard decompression using a
        custom compression dictionary:
        ``compression={\'method\': \'zstd\', \'dict_data\': my_compression_dict}``.

        .. versionadded:: 1.5.0
            Added support for `.tar` files.
    storage_options : dict, optional
        Extra options that make sense for a particular storage connection, e.g.
        host, port, username, password, etc. For HTTP(S) URLs the key-value pairs
        are forwarded to ``urllib.request.Request`` as header options. For other
        URLs (e.g. starting with "s3://", and "gcs://") the key-value pairs are
        forwarded to ``fsspec.open``. Please see ``fsspec`` and ``urllib`` for more
        details, and for more examples on storage options refer `here
        <https://pandas.pydata.org/docs/user_guide/io.html?
        highlight=storage_options#reading-writing-remote-files>`_.

    Returns
    -------
    DataFrame or pandas.api.typing.StataReader

    See Also
    --------
    io.stata.StataReader : Low-level reader for Stata data files.
    DataFrame.to_stata: Export Stata data files.

    Notes
    -----
    Categorical variables read through an iterator may not have the same
    categories and dtype. This occurs when  a variable stored in a DTA
    file is associated to an incomplete set of value labels that only
    label a strict subset of the values.

    Examples
    --------

    Creating a dummy stata for this example

    >>> df = pd.DataFrame({\'animal\': [\'falcon\', \'parrot\', \'falcon\', \'parrot\'],
    ...                     \'speed\': [350, 18, 361, 15]})  # doctest: +SKIP
    >>> df.to_stata(\'animals.dta\')  # doctest: +SKIP

    Read a Stata dta file:

    >>> df = pd.read_stata(\'animals.dta\')  # doctest: +SKIP

    Read a Stata dta file in 10,000 line chunks:

    >>> values = np.random.randint(0, 10, size=(20_000, 1), dtype="uint8")  # doctest: +SKIP
    >>> df = pd.DataFrame(values, columns=["i"])  # doctest: +SKIP
    >>> df.to_stata(\'filename.dta\')  # doctest: +SKIP

    >>> with pd.read_stata(\'filename.dta\', chunksize=10000) as itr: # doctest: +SKIP
    >>>     for chunk in itr:
    ...         # Operate on a single chunk, e.g., chunk.mean()
    ...         pass  # doctest: +SKIP
    '''
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
def _dtype_to_default_stata_fmt(dtype, column: Series, dta_version: int = ..., force_strl: bool = ...) -> str:
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
    _max_string_length: ClassVar[int] = ...
    _encoding: ClassVar[str] = ...
    _docstring_components: ClassVar[list] = ...
    def __init__(self, fname: FilePath | WriteBuffer[bytes], data: DataFrame, convert_dates: dict[Hashable, str] | None, write_index: bool = ..., byteorder: str | None, time_stamp: datetime | None, data_label: str | None, variable_labels: dict[Hashable, str] | None, compression: CompressionOptions = ..., storage_options: StorageOptions | None, *, value_labels: dict[Hashable, dict[float, str]] | None) -> None: ...
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
    def _set_formats_and_types(self, dtypes: Series) -> None: ...
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
    def _write_header(self, data_label: str | None, time_stamp: datetime | None) -> None: ...
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
    def __init__(self, df: DataFrame, columns: Sequence[str], version: int = ..., byteorder: str | None) -> None: ...
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
    _max_string_length: ClassVar[int] = ...
    _dta_version: ClassVar[int] = ...
    def __init__(self, fname: FilePath | WriteBuffer[bytes], data: DataFrame, convert_dates: dict[Hashable, str] | None, write_index: bool = ..., byteorder: str | None, time_stamp: datetime | None, data_label: str | None, variable_labels: dict[Hashable, str] | None, convert_strl: Sequence[Hashable] | None, compression: CompressionOptions = ..., storage_options: StorageOptions | None, *, value_labels: dict[Hashable, dict[float, str]] | None) -> None: ...
    @staticmethod
    def _tag(val: str | bytes, tag: str) -> bytes:
        """Surround val with <tag></tag>"""
    def _update_map(self, tag: str) -> None:
        """Update map location for tag with file position"""
    def _write_header(self, data_label: str | None, time_stamp: datetime | None) -> None:
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
    def _set_formats_and_types(self, dtypes: Series) -> None: ...

class StataWriterUTF8(StataWriter117):
    _encoding: ClassVar[str] = ...
    def __init__(self, fname: FilePath | WriteBuffer[bytes], data: DataFrame, convert_dates: dict[Hashable, str] | None, write_index: bool = ..., byteorder: str | None, time_stamp: datetime | None, data_label: str | None, variable_labels: dict[Hashable, str] | None, convert_strl: Sequence[Hashable] | None, version: int | None, compression: CompressionOptions = ..., storage_options: StorageOptions | None, *, value_labels: dict[Hashable, dict[float, str]] | None) -> None: ...
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
