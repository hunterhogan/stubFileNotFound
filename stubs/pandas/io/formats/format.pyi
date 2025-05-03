import numpy as np
from _typeshed import Incomplete
from collections.abc import Generator, Hashable, Sequence
from io import StringIO
from pandas import DataFrame as DataFrame, Series as Series
from pandas._config.config import get_option as get_option, set_option as set_option
from pandas._libs import lib as lib
from pandas._libs.missing import NA as NA
from pandas._libs.tslibs import NaT as NaT, Timedelta as Timedelta, Timestamp as Timestamp
from pandas._libs.tslibs.nattype import NaTType as NaTType
from pandas._typing import ArrayLike as ArrayLike, Axes as Axes, ColspaceArgType as ColspaceArgType, ColspaceType as ColspaceType, CompressionOptions as CompressionOptions, FilePath as FilePath, FloatFormatType as FloatFormatType, FormattersType as FormattersType, IndexLabel as IndexLabel, SequenceNotStr as SequenceNotStr, StorageOptions as StorageOptions, WriteBuffer as WriteBuffer
from pandas.core.arrays import Categorical as Categorical, DatetimeArray as DatetimeArray, ExtensionArray as ExtensionArray, TimedeltaArray as TimedeltaArray
from pandas.core.arrays.string_ import StringDtype as StringDtype
from pandas.core.base import PandasObject as PandasObject
from pandas.core.dtypes.common import is_complex_dtype as is_complex_dtype, is_float as is_float, is_integer as is_integer, is_list_like as is_list_like, is_numeric_dtype as is_numeric_dtype, is_scalar as is_scalar
from pandas.core.dtypes.dtypes import CategoricalDtype as CategoricalDtype, DatetimeTZDtype as DatetimeTZDtype, ExtensionDtype as ExtensionDtype
from pandas.core.dtypes.missing import isna as isna, notna as notna
from pandas.core.indexes.api import Index as Index, MultiIndex as MultiIndex, PeriodIndex as PeriodIndex, ensure_index as ensure_index
from pandas.core.indexes.datetimes import DatetimeIndex as DatetimeIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex as TimedeltaIndex
from pandas.core.reshape.concat import concat as concat
from pandas.io.common import check_parent_directory as check_parent_directory, stringify_path as stringify_path
from pandas.io.formats import printing as printing
from typing import Any, Callable, Final

common_docstring: Final[str]
VALID_JUSTIFY_PARAMETERS: Incomplete
return_docstring: Final[str]

class SeriesFormatter:
    """
    Implement the main logic of Series.to_string, which underlies
    Series.__repr__.
    """
    series: Incomplete
    buf: Incomplete
    name: Incomplete
    na_rep: Incomplete
    header: Incomplete
    length: Incomplete
    index: Incomplete
    max_rows: Incomplete
    min_rows: Incomplete
    float_format: Incomplete
    dtype: Incomplete
    adj: Incomplete
    def __init__(self, series: Series, *, length: bool | str = True, header: bool = True, index: bool = True, na_rep: str = 'NaN', name: bool = False, float_format: str | None = None, dtype: bool = True, max_rows: int | None = None, min_rows: int | None = None) -> None: ...
    tr_row_num: int | None
    tr_series: Incomplete
    is_truncated_vertically: Incomplete
    def _chk_truncate(self) -> None: ...
    def _get_footer(self) -> str: ...
    def _get_formatted_values(self) -> list[str]: ...
    def to_string(self) -> str: ...

def get_dataframe_repr_params() -> dict[str, Any]:
    """Get the parameters used to repr(dataFrame) calls using DataFrame.to_string.

    Supplying these parameters to DataFrame.to_string is equivalent to calling
    ``repr(DataFrame)``. This is useful if you want to adjust the repr output.

    .. versionadded:: 1.4.0

    Example
    -------
    >>> import pandas as pd
    >>>
    >>> df = pd.DataFrame([[1, 2], [3, 4]])
    >>> repr_params = pd.io.formats.format.get_dataframe_repr_params()
    >>> repr(df) == df.to_string(**repr_params)
    True
    """
def get_series_repr_params() -> dict[str, Any]:
    """Get the parameters used to repr(Series) calls using Series.to_string.

    Supplying these parameters to Series.to_string is equivalent to calling
    ``repr(series)``. This is useful if you want to adjust the series repr output.

    .. versionadded:: 1.4.0

    Example
    -------
    >>> import pandas as pd
    >>>
    >>> ser = pd.Series([1, 2, 3, 4])
    >>> repr_params = pd.io.formats.format.get_series_repr_params()
    >>> repr(ser) == ser.to_string(**repr_params)
    True
    """

class DataFrameFormatter:
    """
    Class for processing dataframe formatting options and data.

    Used by DataFrame.to_string, which backs DataFrame.__repr__.
    """
    __doc__: Incomplete
    frame: Incomplete
    columns: Incomplete
    col_space: Incomplete
    header: Incomplete
    index: Incomplete
    na_rep: Incomplete
    formatters: Incomplete
    justify: Incomplete
    float_format: Incomplete
    sparsify: Incomplete
    show_index_names: Incomplete
    decimal: Incomplete
    bold_rows: Incomplete
    escape: Incomplete
    max_rows: Incomplete
    min_rows: Incomplete
    max_cols: Incomplete
    show_dimensions: Incomplete
    max_cols_fitted: Incomplete
    max_rows_fitted: Incomplete
    tr_frame: Incomplete
    adj: Incomplete
    def __init__(self, frame: DataFrame, columns: Axes | None = None, col_space: ColspaceArgType | None = None, header: bool | SequenceNotStr[str] = True, index: bool = True, na_rep: str = 'NaN', formatters: FormattersType | None = None, justify: str | None = None, float_format: FloatFormatType | None = None, sparsify: bool | None = None, index_names: bool = True, max_rows: int | None = None, min_rows: int | None = None, max_cols: int | None = None, show_dimensions: bool | str = False, decimal: str = '.', bold_rows: bool = False, escape: bool = True) -> None: ...
    def get_strcols(self) -> list[list[str]]:
        """
        Render a DataFrame to a list of columns (as lists of strings).
        """
    @property
    def should_show_dimensions(self) -> bool: ...
    @property
    def is_truncated(self) -> bool: ...
    @property
    def is_truncated_horizontally(self) -> bool: ...
    @property
    def is_truncated_vertically(self) -> bool: ...
    @property
    def dimensions_info(self) -> str: ...
    @property
    def has_index_names(self) -> bool: ...
    @property
    def has_column_names(self) -> bool: ...
    @property
    def show_row_idx_names(self) -> bool: ...
    @property
    def show_col_idx_names(self) -> bool: ...
    @property
    def max_rows_displayed(self) -> int: ...
    def _initialize_sparsify(self, sparsify: bool | None) -> bool: ...
    def _initialize_formatters(self, formatters: FormattersType | None) -> FormattersType: ...
    def _initialize_justify(self, justify: str | None) -> str: ...
    def _initialize_columns(self, columns: Axes | None) -> Index: ...
    def _initialize_colspace(self, col_space: ColspaceArgType | None) -> ColspaceType: ...
    def _calc_max_cols_fitted(self) -> int | None:
        """Number of columns fitting the screen."""
    def _calc_max_rows_fitted(self) -> int | None:
        """Number of rows with data fitting the screen."""
    def _adjust_max_rows(self, max_rows: int | None) -> int | None:
        """Adjust max_rows using display logic.

        See description here:
        https://pandas.pydata.org/docs/dev/user_guide/options.html#frequently-used-options

        GH #37359
        """
    def _is_in_terminal(self) -> bool:
        """Check if the output is to be shown in terminal."""
    def _is_screen_narrow(self, max_width) -> bool: ...
    def _is_screen_short(self, max_height) -> bool: ...
    def _get_number_of_auxiliary_rows(self) -> int:
        """Get number of rows occupied by prompt, dots and dimension info."""
    def truncate(self) -> None:
        """
        Check whether the frame should be truncated. If so, slice the frame up.
        """
    tr_col_num: Incomplete
    def _truncate_horizontally(self) -> None:
        """Remove columns, which are not to be displayed and adjust formatters.

        Attributes affected:
            - tr_frame
            - formatters
            - tr_col_num
        """
    tr_row_num: Incomplete
    def _truncate_vertically(self) -> None:
        """Remove rows, which are not to be displayed.

        Attributes affected:
            - tr_frame
            - tr_row_num
        """
    def _get_strcols_without_index(self) -> list[list[str]]: ...
    def format_col(self, i: int) -> list[str]: ...
    def _get_formatter(self, i: str | int) -> Callable | None: ...
    def _get_formatted_column_labels(self, frame: DataFrame) -> list[list[str]]: ...
    def _get_formatted_index(self, frame: DataFrame) -> list[str]: ...
    def _get_column_name_list(self) -> list[Hashable]: ...

class DataFrameRenderer:
    """Class for creating dataframe output in multiple formats.

    Called in pandas.core.generic.NDFrame:
        - to_csv
        - to_latex

    Called in pandas.core.frame.DataFrame:
        - to_html
        - to_string

    Parameters
    ----------
    fmt : DataFrameFormatter
        Formatter with the formatting options.
    """
    fmt: Incomplete
    def __init__(self, fmt: DataFrameFormatter) -> None: ...
    def to_html(self, buf: FilePath | WriteBuffer[str] | None = None, encoding: str | None = None, classes: str | list | tuple | None = None, notebook: bool = False, border: int | bool | None = None, table_id: str | None = None, render_links: bool = False) -> str | None:
        '''
        Render a DataFrame to a html table.

        Parameters
        ----------
        buf : str, path object, file-like object, or None, default None
            String, path object (implementing ``os.PathLike[str]``), or file-like
            object implementing a string ``write()`` function. If None, the result is
            returned as a string.
        encoding : str, default “utf-8”
            Set character encoding.
        classes : str or list-like
            classes to include in the `class` attribute of the opening
            ``<table>`` tag, in addition to the default "dataframe".
        notebook : {True, False}, optional, default False
            Whether the generated HTML is for IPython Notebook.
        border : int
            A ``border=border`` attribute is included in the opening
            ``<table>`` tag. Default ``pd.options.display.html.border``.
        table_id : str, optional
            A css id is included in the opening `<table>` tag if specified.
        render_links : bool, default False
            Convert URLs to HTML links.
        '''
    def to_string(self, buf: FilePath | WriteBuffer[str] | None = None, encoding: str | None = None, line_width: int | None = None) -> str | None:
        """
        Render a DataFrame to a console-friendly tabular output.

        Parameters
        ----------
        buf : str, path object, file-like object, or None, default None
            String, path object (implementing ``os.PathLike[str]``), or file-like
            object implementing a string ``write()`` function. If None, the result is
            returned as a string.
        encoding: str, default “utf-8”
            Set character encoding.
        line_width : int, optional
            Width to wrap a line in characters.
        """
    def to_csv(self, path_or_buf: FilePath | WriteBuffer[bytes] | WriteBuffer[str] | None = None, encoding: str | None = None, sep: str = ',', columns: Sequence[Hashable] | None = None, index_label: IndexLabel | None = None, mode: str = 'w', compression: CompressionOptions = 'infer', quoting: int | None = None, quotechar: str = '"', lineterminator: str | None = None, chunksize: int | None = None, date_format: str | None = None, doublequote: bool = True, escapechar: str | None = None, errors: str = 'strict', storage_options: StorageOptions | None = None) -> str | None:
        """
        Render dataframe as comma-separated file.
        """

def save_to_buffer(string: str, buf: FilePath | WriteBuffer[str] | None = None, encoding: str | None = None) -> str | None:
    """
    Perform serialization. Write to buf or return as string if buf is None.
    """
def _get_buffer(buf: FilePath | WriteBuffer[str] | None, encoding: str | None = None) -> Generator[WriteBuffer[str], None, None] | Generator[StringIO, None, None]:
    """
    Context manager to open, yield and close buffer for filenames or Path-like
    objects, otherwise yield buf unchanged.
    """
def format_array(values: ArrayLike, formatter: Callable | None, float_format: FloatFormatType | None = None, na_rep: str = 'NaN', digits: int | None = None, space: str | int | None = None, justify: str = 'right', decimal: str = '.', leading_space: bool | None = True, quoting: int | None = None, fallback_formatter: Callable | None = None) -> list[str]:
    """
    Format an array for printing.

    Parameters
    ----------
    values : np.ndarray or ExtensionArray
    formatter
    float_format
    na_rep
    digits
    space
    justify
    decimal
    leading_space : bool, optional, default True
        Whether the array should be formatted with a leading space.
        When an array as a column of a Series or DataFrame, we do want
        the leading space to pad between columns.

        When formatting an Index subclass
        (e.g. IntervalIndex._get_values_for_csv), we don't want the
        leading space since it should be left-aligned.
    fallback_formatter

    Returns
    -------
    List[str]
    """

class _GenericArrayFormatter:
    values: Incomplete
    digits: Incomplete
    na_rep: Incomplete
    space: Incomplete
    formatter: Incomplete
    float_format: Incomplete
    justify: Incomplete
    decimal: Incomplete
    quoting: Incomplete
    fixed_width: Incomplete
    leading_space: Incomplete
    fallback_formatter: Incomplete
    def __init__(self, values: ArrayLike, digits: int = 7, formatter: Callable | None = None, na_rep: str = 'NaN', space: str | int = 12, float_format: FloatFormatType | None = None, justify: str = 'right', decimal: str = '.', quoting: int | None = None, fixed_width: bool = True, leading_space: bool | None = True, fallback_formatter: Callable | None = None) -> None: ...
    def get_result(self) -> list[str]: ...
    def _format_strings(self) -> list[str]: ...

class FloatArrayFormatter(_GenericArrayFormatter):
    fixed_width: bool
    formatter: Incomplete
    float_format: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def _value_formatter(self, float_format: FloatFormatType | None = None, threshold: float | None = None) -> Callable:
        """Returns a function to be applied on each value to format it"""
    def get_result_as_array(self) -> np.ndarray:
        """
        Returns the float values converted into strings using
        the parameters given at initialisation, as a numpy array
        """
    def _format_strings(self) -> list[str]: ...

class _IntArrayFormatter(_GenericArrayFormatter):
    def _format_strings(self) -> list[str]: ...

class _Datetime64Formatter(_GenericArrayFormatter):
    values: DatetimeArray
    nat_rep: Incomplete
    date_format: Incomplete
    def __init__(self, values: DatetimeArray, nat_rep: str = 'NaT', date_format: None = None, **kwargs) -> None: ...
    def _format_strings(self) -> list[str]:
        """we by definition have DO NOT have a TZ"""

class _ExtensionArrayFormatter(_GenericArrayFormatter):
    values: ExtensionArray
    def _format_strings(self) -> list[str]: ...

def format_percentiles(percentiles: np.ndarray | Sequence[float]) -> list[str]:
    """
    Outputs rounded and formatted percentiles.

    Parameters
    ----------
    percentiles : list-like, containing floats from interval [0,1]

    Returns
    -------
    formatted : list of strings

    Notes
    -----
    Rounding precision is chosen so that: (1) if any two elements of
    ``percentiles`` differ, they remain different after rounding
    (2) no entry is *rounded* to 0% or 100%.
    Any non-integer is always rounded to at least 1 decimal place.

    Examples
    --------
    Keeps all entries different after rounding:

    >>> format_percentiles([0.01999, 0.02001, 0.5, 0.666666, 0.9999])
    ['1.999%', '2.001%', '50%', '66.667%', '99.99%']

    No element is rounded to 0% or 100% (unless already equal to it).
    Duplicates are allowed:

    >>> format_percentiles([0, 0.5, 0.02001, 0.5, 0.666666, 0.9999])
    ['0%', '50%', '2.0%', '50%', '66.67%', '99.99%']
    """
def get_precision(array: np.ndarray | Sequence[float]) -> int: ...
def _format_datetime64(x: NaTType | Timestamp, nat_rep: str = 'NaT') -> str: ...
def _format_datetime64_dateonly(x: NaTType | Timestamp, nat_rep: str = 'NaT', date_format: str | None = None) -> str: ...
def get_format_datetime64(is_dates_only: bool, nat_rep: str = 'NaT', date_format: str | None = None) -> Callable:
    """Return a formatter callable taking a datetime64 as input and providing
    a string as output"""

class _Datetime64TZFormatter(_Datetime64Formatter):
    values: DatetimeArray
    def _format_strings(self) -> list[str]:
        """we by definition have a TZ"""

class _Timedelta64Formatter(_GenericArrayFormatter):
    values: TimedeltaArray
    nat_rep: Incomplete
    def __init__(self, values: TimedeltaArray, nat_rep: str = 'NaT', **kwargs) -> None: ...
    def _format_strings(self) -> list[str]: ...

def get_format_timedelta64(values: TimedeltaArray, nat_rep: str | float = 'NaT', box: bool = False) -> Callable:
    """
    Return a formatter function for a range of timedeltas.
    These will all have the same format argument

    If box, then show the return in quotes
    """
def _make_fixed_width(strings: list[str], justify: str = 'right', minimum: int | None = None, adj: printing._TextAdjustment | None = None) -> list[str]: ...
def _trim_zeros_complex(str_complexes: ArrayLike, decimal: str = '.') -> list[str]:
    """
    Separates the real and imaginary parts from the complex number, and
    executes the _trim_zeros_float method on each of those.
    """
def _trim_zeros_single_float(str_float: str) -> str:
    """
    Trims trailing zeros after a decimal point,
    leaving just one if necessary.
    """
def _trim_zeros_float(str_floats: ArrayLike | list[str], decimal: str = '.') -> list[str]:
    """
    Trims the maximum number of trailing zeros equally from
    all numbers containing decimals, leaving just one if
    necessary.
    """
def _has_names(index: Index) -> bool: ...

class EngFormatter:
    """
    Formats float values according to engineering format.

    Based on matplotlib.ticker.EngFormatter
    """
    ENG_PREFIXES: Incomplete
    accuracy: Incomplete
    use_eng_prefix: Incomplete
    def __init__(self, accuracy: int | None = None, use_eng_prefix: bool = False) -> None: ...
    def __call__(self, num: float) -> str:
        '''
        Formats a number in engineering notation, appending a letter
        representing the power of 1000 of the original number. Some examples:
        >>> format_eng = EngFormatter(accuracy=0, use_eng_prefix=True)
        >>> format_eng(0)
        \' 0\'
        >>> format_eng = EngFormatter(accuracy=1, use_eng_prefix=True)
        >>> format_eng(1_000_000)
        \' 1.0M\'
        >>> format_eng = EngFormatter(accuracy=2, use_eng_prefix=False)
        >>> format_eng("-1e-6")
        \'-1.00E-06\'

        @param num: the value to represent
        @type num: either a numeric value or a string that can be converted to
                   a numeric value (as per decimal.Decimal constructor)

        @return: engineering formatted string
        '''

def set_eng_float_format(accuracy: int = 3, use_eng_prefix: bool = False) -> None:
    '''
    Format float representation in DataFrame with SI notation.

    Parameters
    ----------
    accuracy : int, default 3
        Number of decimal digits after the floating point.
    use_eng_prefix : bool, default False
        Whether to represent a value with SI prefixes.

    Returns
    -------
    None

    Examples
    --------
    >>> df = pd.DataFrame([1e-9, 1e-3, 1, 1e3, 1e6])
    >>> df
                  0
    0  1.000000e-09
    1  1.000000e-03
    2  1.000000e+00
    3  1.000000e+03
    4  1.000000e+06

    >>> pd.set_eng_float_format(accuracy=1)
    >>> df
             0
    0  1.0E-09
    1  1.0E-03
    2  1.0E+00
    3  1.0E+03
    4  1.0E+06

    >>> pd.set_eng_float_format(use_eng_prefix=True)
    >>> df
            0
    0  1.000n
    1  1.000m
    2   1.000
    3  1.000k
    4  1.000M

    >>> pd.set_eng_float_format(accuracy=1, use_eng_prefix=True)
    >>> df
          0
    0  1.0n
    1  1.0m
    2   1.0
    3  1.0k
    4  1.0M

    >>> pd.set_option("display.float_format", None)  # unset option
    '''
def get_level_lengths(levels: Any, sentinel: bool | object | str = '') -> list[dict[int, int]]:
    """
    For each index in each level the function returns lengths of indexes.

    Parameters
    ----------
    levels : list of lists
        List of values on for level.
    sentinel : string, optional
        Value which states that no new index starts on there.

    Returns
    -------
    Returns list of maps. For each level returns map of indexes (key is index
    in row and value is length of index).
    """
def buffer_put_lines(buf: WriteBuffer[str], lines: list[str]) -> None:
    """
    Appends lines to a buffer.

    Parameters
    ----------
    buf
        The buffer to write to
    lines
        The lines to append.
    """
