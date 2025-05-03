import _abc
import abc
import pandas.io.formats.format as fmt
from pandas._config.config import get_option as get_option
from pandas.io.formats.printing import pprint_thing as pprint_thing
from typing import ClassVar

TYPE_CHECKING: bool
frame_max_cols_sub: str
show_counts_sub: str
frame_examples_sub: str
frame_see_also_sub: str
frame_sub_kwargs: dict
series_examples_sub: str
series_see_also_sub: str
series_sub_kwargs: dict
INFO_DOCSTRING: str
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
def _initialize_memory_usage(memory_usage: bool | str | None) -> bool | str:
    """Get memory usage based on inputs and display options."""

class _BaseInfo(abc.ABC):
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    def render(self, *, buf: WriteBuffer[str] | None, max_cols: int | None, verbose: bool | None, show_counts: bool | None) -> None: ...
    @property
    def dtypes(self): ...
    @property
    def dtype_counts(self): ...
    @property
    def non_null_counts(self): ...
    @property
    def memory_usage_bytes(self): ...
    @property
    def memory_usage_string(self): ...
    @property
    def size_qualifier(self): ...

class DataFrameInfo(_BaseInfo):
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    def __init__(self, data: DataFrame, memory_usage: bool | str | None) -> None: ...
    def render(self, *, buf: WriteBuffer[str] | None, max_cols: int | None, verbose: bool | None, show_counts: bool | None) -> None: ...
    @property
    def dtype_counts(self): ...
    @property
    def dtypes(self): ...
    @property
    def ids(self): ...
    @property
    def col_count(self): ...
    @property
    def non_null_counts(self): ...
    @property
    def memory_usage_bytes(self): ...

class SeriesInfo(_BaseInfo):
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    def __init__(self, data: Series, memory_usage: bool | str | None) -> None: ...
    def render(self, *, buf: WriteBuffer[str] | None, max_cols: int | None, verbose: bool | None, show_counts: bool | None) -> None: ...
    @property
    def non_null_counts(self): ...
    @property
    def dtypes(self): ...
    @property
    def dtype_counts(self): ...
    @property
    def memory_usage_bytes(self): ...

class _InfoPrinterAbstract:
    def to_buffer(self, buf: WriteBuffer[str] | None) -> None:
        """Save dataframe info into buffer."""
    def _create_table_builder(self) -> _TableBuilderAbstract:
        """Create instance of table builder."""

class _DataFrameInfoPrinter(_InfoPrinterAbstract):
    def __init__(self, info: DataFrameInfo, max_cols: int | None, verbose: bool | None, show_counts: bool | None) -> None: ...
    def _initialize_max_cols(self, max_cols: int | None) -> int: ...
    def _initialize_show_counts(self, show_counts: bool | None) -> bool: ...
    def _create_table_builder(self) -> _DataFrameTableBuilder:
        """
        Create instance of table builder based on verbosity and display settings.
        """
    @property
    def max_rows(self): ...
    @property
    def exceeds_info_cols(self): ...
    @property
    def exceeds_info_rows(self): ...
    @property
    def col_count(self): ...

class _SeriesInfoPrinter(_InfoPrinterAbstract):
    def __init__(self, info: SeriesInfo, verbose: bool | None, show_counts: bool | None) -> None: ...
    def _create_table_builder(self) -> _SeriesTableBuilder:
        """
        Create instance of table builder based on verbosity.
        """
    def _initialize_show_counts(self, show_counts: bool | None) -> bool: ...

class _TableBuilderAbstract(abc.ABC):
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    def get_lines(self) -> list[str]:
        """Product in a form of list of lines (strings)."""
    def add_object_type_line(self) -> None:
        """Add line with string representation of dataframe to the table."""
    def add_index_range_line(self) -> None:
        """Add line with range of indices to the table."""
    def add_dtypes_line(self) -> None:
        """Add summary line with dtypes present in dataframe."""
    @property
    def data(self): ...
    @property
    def dtypes(self): ...
    @property
    def dtype_counts(self): ...
    @property
    def display_memory_usage(self): ...
    @property
    def memory_usage_string(self): ...
    @property
    def non_null_counts(self): ...

class _DataFrameTableBuilder(_TableBuilderAbstract):
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    def __init__(self, *, info: DataFrameInfo) -> None: ...
    def get_lines(self) -> list[str]: ...
    def _fill_empty_info(self) -> None:
        """Add lines to the info table, pertaining to empty dataframe."""
    def _fill_non_empty_info(self) -> None:
        """Add lines to the info table, pertaining to non-empty dataframe."""
    def add_memory_usage_line(self) -> None:
        """Add line containing memory usage."""
    @property
    def data(self): ...
    @property
    def ids(self): ...
    @property
    def col_count(self): ...

class _DataFrameTableBuilderNonVerbose(_DataFrameTableBuilder):
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    def _fill_non_empty_info(self) -> None:
        """Add lines to the info table, pertaining to non-empty dataframe."""
    def add_columns_summary_line(self) -> None: ...

class _TableBuilderVerboseMixin(_TableBuilderAbstract):
    SPACING: ClassVar[str] = ...
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    def _get_gross_column_widths(self) -> Sequence[int]:
        """Get widths of columns containing both headers and actual content."""
    def _get_body_column_widths(self) -> Sequence[int]:
        """Get widths of table content columns."""
    def _gen_rows(self) -> Iterator[Sequence[str]]:
        """
        Generator function yielding rows content.

        Each element represents a row comprising a sequence of strings.
        """
    def _gen_rows_with_counts(self) -> Iterator[Sequence[str]]:
        """Iterator with string representation of body data with counts."""
    def _gen_rows_without_counts(self) -> Iterator[Sequence[str]]:
        """Iterator with string representation of body data without counts."""
    def add_header_line(self) -> None: ...
    def add_separator_line(self) -> None: ...
    def add_body_lines(self) -> None: ...
    def _gen_non_null_counts(self) -> Iterator[str]:
        """Iterator with string representation of non-null counts."""
    def _gen_dtypes(self) -> Iterator[str]:
        """Iterator with string representation of column dtypes."""
    @property
    def headers(self): ...
    @property
    def header_column_widths(self): ...

class _DataFrameTableBuilderVerbose(_DataFrameTableBuilder, _TableBuilderVerboseMixin):
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    def __init__(self, *, info: DataFrameInfo, with_counts: bool) -> None: ...
    def _fill_non_empty_info(self) -> None:
        """Add lines to the info table, pertaining to non-empty dataframe."""
    def add_columns_summary_line(self) -> None: ...
    def _gen_rows_without_counts(self) -> Iterator[Sequence[str]]:
        """Iterator with string representation of body data without counts."""
    def _gen_rows_with_counts(self) -> Iterator[Sequence[str]]:
        """Iterator with string representation of body data with counts."""
    def _gen_line_numbers(self) -> Iterator[str]:
        """Iterator with string representation of column numbers."""
    def _gen_columns(self) -> Iterator[str]:
        """Iterator with string representation of column names."""
    @property
    def headers(self): ...

class _SeriesTableBuilder(_TableBuilderAbstract):
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    def __init__(self, *, info: SeriesInfo) -> None: ...
    def get_lines(self) -> list[str]: ...
    def add_memory_usage_line(self) -> None:
        """Add line containing memory usage."""
    def _fill_non_empty_info(self) -> None:
        """Add lines to the info table, pertaining to non-empty series."""
    @property
    def data(self): ...

class _SeriesTableBuilderNonVerbose(_SeriesTableBuilder):
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    def _fill_non_empty_info(self) -> None:
        """Add lines to the info table, pertaining to non-empty series."""

class _SeriesTableBuilderVerbose(_SeriesTableBuilder, _TableBuilderVerboseMixin):
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    def __init__(self, *, info: SeriesInfo, with_counts: bool) -> None: ...
    def _fill_non_empty_info(self) -> None:
        """Add lines to the info table, pertaining to non-empty series."""
    def add_series_name_line(self) -> None: ...
    def _gen_rows_without_counts(self) -> Iterator[Sequence[str]]:
        """Iterator with string representation of body data without counts."""
    def _gen_rows_with_counts(self) -> Iterator[Sequence[str]]:
        """Iterator with string representation of body data with counts."""
    @property
    def headers(self): ...
def _get_dataframe_dtype_counts(df: DataFrame) -> Mapping[str, int]:
    """
    Create mapping between datatypes and their number of occurrences.
    """
