import abc
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pandas import DataFrame as DataFrame, Index as Index, Series as Series
from pandas._config import get_option as get_option
from pandas._typing import Dtype as Dtype, WriteBuffer as WriteBuffer
from pandas.io.formats.printing import pprint_thing as pprint_thing

frame_max_cols_sub: Incomplete
show_counts_sub: Incomplete
frame_examples_sub: Incomplete
frame_see_also_sub: Incomplete
frame_sub_kwargs: Incomplete
series_examples_sub: Incomplete
series_see_also_sub: Incomplete
series_sub_kwargs: Incomplete
INFO_DOCSTRING: Incomplete

def _put_str(s: str | Dtype, space: int) -> str:
    '''
    Make string of specified length, padding to the right if necessary.

    Parameters
    ----------
    s : Union[str, Dtype]
        String to be formatted.
    space : int
        Length to force string to be of.

    Returns
    -------
    str
        String coerced to given length.

    Examples
    --------
    >>> pd.io.formats.info._put_str("panda", 6)
    \'panda \'
    >>> pd.io.formats.info._put_str("panda", 4)
    \'pand\'
    '''
def _sizeof_fmt(num: float, size_qualifier: str) -> str:
    """
    Return size in human readable format.

    Parameters
    ----------
    num : int
        Size in bytes.
    size_qualifier : str
        Either empty, or '+' (if lower bound).

    Returns
    -------
    str
        Size in human readable format.

    Examples
    --------
    >>> _sizeof_fmt(23028, '')
    '22.5 KB'

    >>> _sizeof_fmt(23028, '+')
    '22.5+ KB'
    """
def _initialize_memory_usage(memory_usage: bool | str | None = None) -> bool | str:
    """Get memory usage based on inputs and display options."""

class _BaseInfo(ABC, metaclass=abc.ABCMeta):
    '''
    Base class for DataFrameInfo and SeriesInfo.

    Parameters
    ----------
    data : DataFrame or Series
        Either dataframe or series.
    memory_usage : bool or str, optional
        If "deep", introspect the data deeply by interrogating object dtypes
        for system-level memory consumption, and include it in the returned
        values.
    '''
    data: DataFrame | Series
    memory_usage: bool | str
    @property
    @abstractmethod
    def dtypes(self) -> Iterable[Dtype]:
        """
        Dtypes.

        Returns
        -------
        dtypes : sequence
            Dtype of each of the DataFrame's columns (or one series column).
        """
    @property
    @abstractmethod
    def dtype_counts(self) -> Mapping[str, int]:
        """Mapping dtype - number of counts."""
    @property
    @abstractmethod
    def non_null_counts(self) -> Sequence[int]:
        """Sequence of non-null counts for all columns or column (if series)."""
    @property
    @abstractmethod
    def memory_usage_bytes(self) -> int:
        """
        Memory usage in bytes.

        Returns
        -------
        memory_usage_bytes : int
            Object's total memory usage in bytes.
        """
    @property
    def memory_usage_string(self) -> str:
        """Memory usage in a form of human readable string."""
    @property
    def size_qualifier(self) -> str: ...
    @abstractmethod
    def render(self, *, buf: WriteBuffer[str] | None, max_cols: int | None, verbose: bool | None, show_counts: bool | None) -> None: ...

class DataFrameInfo(_BaseInfo):
    """
    Class storing dataframe-specific info.
    """
    data: DataFrame
    memory_usage: Incomplete
    def __init__(self, data: DataFrame, memory_usage: bool | str | None = None) -> None: ...
    @property
    def dtype_counts(self) -> Mapping[str, int]: ...
    @property
    def dtypes(self) -> Iterable[Dtype]:
        """
        Dtypes.

        Returns
        -------
        dtypes
            Dtype of each of the DataFrame's columns.
        """
    @property
    def ids(self) -> Index:
        """
        Column names.

        Returns
        -------
        ids : Index
            DataFrame's column names.
        """
    @property
    def col_count(self) -> int:
        """Number of columns to be summarized."""
    @property
    def non_null_counts(self) -> Sequence[int]:
        """Sequence of non-null counts for all columns or column (if series)."""
    @property
    def memory_usage_bytes(self) -> int: ...
    def render(self, *, buf: WriteBuffer[str] | None, max_cols: int | None, verbose: bool | None, show_counts: bool | None) -> None: ...

class SeriesInfo(_BaseInfo):
    """
    Class storing series-specific info.
    """
    data: Series
    memory_usage: Incomplete
    def __init__(self, data: Series, memory_usage: bool | str | None = None) -> None: ...
    def render(self, *, buf: WriteBuffer[str] | None = None, max_cols: int | None = None, verbose: bool | None = None, show_counts: bool | None = None) -> None: ...
    @property
    def non_null_counts(self) -> Sequence[int]: ...
    @property
    def dtypes(self) -> Iterable[Dtype]: ...
    @property
    def dtype_counts(self) -> Mapping[str, int]: ...
    @property
    def memory_usage_bytes(self) -> int:
        """Memory usage in bytes.

        Returns
        -------
        memory_usage_bytes : int
            Object's total memory usage in bytes.
        """

class _InfoPrinterAbstract(metaclass=abc.ABCMeta):
    """
    Class for printing dataframe or series info.
    """
    def to_buffer(self, buf: WriteBuffer[str] | None = None) -> None:
        """Save dataframe info into buffer."""
    @abstractmethod
    def _create_table_builder(self) -> _TableBuilderAbstract:
        """Create instance of table builder."""

class _DataFrameInfoPrinter(_InfoPrinterAbstract):
    """
    Class for printing dataframe info.

    Parameters
    ----------
    info : DataFrameInfo
        Instance of DataFrameInfo.
    max_cols : int, optional
        When to switch from the verbose to the truncated output.
    verbose : bool, optional
        Whether to print the full summary.
    show_counts : bool, optional
        Whether to show the non-null counts.
    """
    info: Incomplete
    data: Incomplete
    verbose: Incomplete
    max_cols: Incomplete
    show_counts: Incomplete
    def __init__(self, info: DataFrameInfo, max_cols: int | None = None, verbose: bool | None = None, show_counts: bool | None = None) -> None: ...
    @property
    def max_rows(self) -> int:
        """Maximum info rows to be displayed."""
    @property
    def exceeds_info_cols(self) -> bool:
        """Check if number of columns to be summarized does not exceed maximum."""
    @property
    def exceeds_info_rows(self) -> bool:
        """Check if number of rows to be summarized does not exceed maximum."""
    @property
    def col_count(self) -> int:
        """Number of columns to be summarized."""
    def _initialize_max_cols(self, max_cols: int | None) -> int: ...
    def _initialize_show_counts(self, show_counts: bool | None) -> bool: ...
    def _create_table_builder(self) -> _DataFrameTableBuilder:
        """
        Create instance of table builder based on verbosity and display settings.
        """

class _SeriesInfoPrinter(_InfoPrinterAbstract):
    """Class for printing series info.

    Parameters
    ----------
    info : SeriesInfo
        Instance of SeriesInfo.
    verbose : bool, optional
        Whether to print the full summary.
    show_counts : bool, optional
        Whether to show the non-null counts.
    """
    info: Incomplete
    data: Incomplete
    verbose: Incomplete
    show_counts: Incomplete
    def __init__(self, info: SeriesInfo, verbose: bool | None = None, show_counts: bool | None = None) -> None: ...
    def _create_table_builder(self) -> _SeriesTableBuilder:
        """
        Create instance of table builder based on verbosity.
        """
    def _initialize_show_counts(self, show_counts: bool | None) -> bool: ...

class _TableBuilderAbstract(ABC, metaclass=abc.ABCMeta):
    """
    Abstract builder for info table.
    """
    _lines: list[str]
    info: _BaseInfo
    @abstractmethod
    def get_lines(self) -> list[str]:
        """Product in a form of list of lines (strings)."""
    @property
    def data(self) -> DataFrame | Series: ...
    @property
    def dtypes(self) -> Iterable[Dtype]:
        """Dtypes of each of the DataFrame's columns."""
    @property
    def dtype_counts(self) -> Mapping[str, int]:
        """Mapping dtype - number of counts."""
    @property
    def display_memory_usage(self) -> bool:
        """Whether to display memory usage."""
    @property
    def memory_usage_string(self) -> str:
        """Memory usage string with proper size qualifier."""
    @property
    def non_null_counts(self) -> Sequence[int]: ...
    def add_object_type_line(self) -> None:
        """Add line with string representation of dataframe to the table."""
    def add_index_range_line(self) -> None:
        """Add line with range of indices to the table."""
    def add_dtypes_line(self) -> None:
        """Add summary line with dtypes present in dataframe."""

class _DataFrameTableBuilder(_TableBuilderAbstract, metaclass=abc.ABCMeta):
    """
    Abstract builder for dataframe info table.

    Parameters
    ----------
    info : DataFrameInfo.
        Instance of DataFrameInfo.
    """
    info: DataFrameInfo
    def __init__(self, *, info: DataFrameInfo) -> None: ...
    _lines: Incomplete
    def get_lines(self) -> list[str]: ...
    def _fill_empty_info(self) -> None:
        """Add lines to the info table, pertaining to empty dataframe."""
    @abstractmethod
    def _fill_non_empty_info(self) -> None:
        """Add lines to the info table, pertaining to non-empty dataframe."""
    @property
    def data(self) -> DataFrame:
        """DataFrame."""
    @property
    def ids(self) -> Index:
        """Dataframe columns."""
    @property
    def col_count(self) -> int:
        """Number of dataframe columns to be summarized."""
    def add_memory_usage_line(self) -> None:
        """Add line containing memory usage."""

class _DataFrameTableBuilderNonVerbose(_DataFrameTableBuilder):
    """
    Dataframe info table builder for non-verbose output.
    """
    def _fill_non_empty_info(self) -> None:
        """Add lines to the info table, pertaining to non-empty dataframe."""
    def add_columns_summary_line(self) -> None: ...

class _TableBuilderVerboseMixin(_TableBuilderAbstract, metaclass=abc.ABCMeta):
    """
    Mixin for verbose info output.
    """
    SPACING: str
    strrows: Sequence[Sequence[str]]
    gross_column_widths: Sequence[int]
    with_counts: bool
    @property
    @abstractmethod
    def headers(self) -> Sequence[str]:
        """Headers names of the columns in verbose table."""
    @property
    def header_column_widths(self) -> Sequence[int]:
        """Widths of header columns (only titles)."""
    def _get_gross_column_widths(self) -> Sequence[int]:
        """Get widths of columns containing both headers and actual content."""
    def _get_body_column_widths(self) -> Sequence[int]:
        """Get widths of table content columns."""
    def _gen_rows(self) -> Iterator[Sequence[str]]:
        """
        Generator function yielding rows content.

        Each element represents a row comprising a sequence of strings.
        """
    @abstractmethod
    def _gen_rows_with_counts(self) -> Iterator[Sequence[str]]:
        """Iterator with string representation of body data with counts."""
    @abstractmethod
    def _gen_rows_without_counts(self) -> Iterator[Sequence[str]]:
        """Iterator with string representation of body data without counts."""
    def add_header_line(self) -> None: ...
    def add_separator_line(self) -> None: ...
    def add_body_lines(self) -> None: ...
    def _gen_non_null_counts(self) -> Iterator[str]:
        """Iterator with string representation of non-null counts."""
    def _gen_dtypes(self) -> Iterator[str]:
        """Iterator with string representation of column dtypes."""

class _DataFrameTableBuilderVerbose(_DataFrameTableBuilder, _TableBuilderVerboseMixin):
    """
    Dataframe info table builder for verbose output.
    """
    info: Incomplete
    with_counts: Incomplete
    strrows: Sequence[Sequence[str]]
    gross_column_widths: Sequence[int]
    def __init__(self, *, info: DataFrameInfo, with_counts: bool) -> None: ...
    def _fill_non_empty_info(self) -> None:
        """Add lines to the info table, pertaining to non-empty dataframe."""
    @property
    def headers(self) -> Sequence[str]:
        """Headers names of the columns in verbose table."""
    def add_columns_summary_line(self) -> None: ...
    def _gen_rows_without_counts(self) -> Iterator[Sequence[str]]:
        """Iterator with string representation of body data without counts."""
    def _gen_rows_with_counts(self) -> Iterator[Sequence[str]]:
        """Iterator with string representation of body data with counts."""
    def _gen_line_numbers(self) -> Iterator[str]:
        """Iterator with string representation of column numbers."""
    def _gen_columns(self) -> Iterator[str]:
        """Iterator with string representation of column names."""

class _SeriesTableBuilder(_TableBuilderAbstract, metaclass=abc.ABCMeta):
    """
    Abstract builder for series info table.

    Parameters
    ----------
    info : SeriesInfo.
        Instance of SeriesInfo.
    """
    info: SeriesInfo
    def __init__(self, *, info: SeriesInfo) -> None: ...
    _lines: Incomplete
    def get_lines(self) -> list[str]: ...
    @property
    def data(self) -> Series:
        """Series."""
    def add_memory_usage_line(self) -> None:
        """Add line containing memory usage."""
    @abstractmethod
    def _fill_non_empty_info(self) -> None:
        """Add lines to the info table, pertaining to non-empty series."""

class _SeriesTableBuilderNonVerbose(_SeriesTableBuilder):
    """
    Series info table builder for non-verbose output.
    """
    def _fill_non_empty_info(self) -> None:
        """Add lines to the info table, pertaining to non-empty series."""

class _SeriesTableBuilderVerbose(_SeriesTableBuilder, _TableBuilderVerboseMixin):
    """
    Series info table builder for verbose output.
    """
    info: Incomplete
    with_counts: Incomplete
    strrows: Sequence[Sequence[str]]
    gross_column_widths: Sequence[int]
    def __init__(self, *, info: SeriesInfo, with_counts: bool) -> None: ...
    def _fill_non_empty_info(self) -> None:
        """Add lines to the info table, pertaining to non-empty series."""
    def add_series_name_line(self) -> None: ...
    @property
    def headers(self) -> Sequence[str]:
        """Headers names of the columns in verbose table."""
    def _gen_rows_without_counts(self) -> Iterator[Sequence[str]]:
        """Iterator with string representation of body data without counts."""
    def _gen_rows_with_counts(self) -> Iterator[Sequence[str]]:
        """Iterator with string representation of body data with counts."""

def _get_dataframe_dtype_counts(df: DataFrame) -> Mapping[str, int]:
    """
    Create mapping between datatypes and their number of occurrences.
    """
