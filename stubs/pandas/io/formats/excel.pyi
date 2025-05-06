from _typeshed import Incomplete
from collections.abc import Hashable, Iterable, Mapping, Sequence
from pandas import DataFrame as DataFrame, ExcelWriter as ExcelWriter, Index as Index, MultiIndex as MultiIndex, PeriodIndex as PeriodIndex
from pandas._libs.lib import is_list_like as is_list_like
from pandas._typing import FilePath as FilePath, IndexLabel as IndexLabel, StorageOptions as StorageOptions, WriteExcelBuffer as WriteExcelBuffer
from pandas.core.dtypes import missing as missing
from pandas.core.dtypes.common import is_float as is_float, is_scalar as is_scalar
from pandas.core.shared_docs import _shared_docs as _shared_docs
from pandas.io.formats._color_data import CSS4_COLORS as CSS4_COLORS
from pandas.io.formats.css import CSSResolver as CSSResolver, CSSWarning as CSSWarning
from pandas.io.formats.format import get_level_lengths as get_level_lengths
from pandas.io.formats.printing import pprint_thing as pprint_thing
from pandas.util._decorators import doc as doc
from pandas.util._exceptions import find_stack_level as find_stack_level
from typing import Any

from collections.abc import Callable

class ExcelCell:
    __fields__: Incomplete
    __slots__ = __fields__
    row: Incomplete
    col: Incomplete
    val: Incomplete
    style: Incomplete
    mergestart: Incomplete
    mergeend: Incomplete
    def __init__(self, row: int, col: int, val, style: Incomplete | None = None, mergestart: int | None = None, mergeend: int | None = None) -> None: ...

class CssExcelCell(ExcelCell):
    def __init__(self, row: int, col: int, val, style: dict | None, css_styles: dict[tuple[int, int], list[tuple[str, Any]]] | None, css_row: int, css_col: int, css_converter: Callable | None, **kwargs) -> None: ...

class CSSToExcelConverter:
    """
    A callable for converting CSS declarations to ExcelWriter styles

    Supports parts of CSS 2.2, with minimal CSS 3.0 support (e.g. text-shadow),
    focusing on font styling, backgrounds, borders and alignment.

    Operates by first computing CSS styles in a fairly generic
    way (see :meth:`compute_css`) then determining Excel style
    properties from CSS properties (see :meth:`build_xlstyle`).

    Parameters
    ----------
    inherited : str, optional
        CSS declarations understood to be the containing scope for the
        CSS processed by :meth:`__call__`.
    """
    NAMED_COLORS = CSS4_COLORS
    VERTICAL_MAP: Incomplete
    BOLD_MAP: Incomplete
    ITALIC_MAP: Incomplete
    FAMILY_MAP: Incomplete
    BORDER_STYLE_MAP: Incomplete
    inherited: dict[str, str] | None
    _call_cached: Incomplete
    def __init__(self, inherited: str | None = None) -> None: ...
    compute_css: Incomplete
    def __call__(self, declarations: str | frozenset[tuple[str, str]]) -> dict[str, dict[str, str]]:
        '''
        Convert CSS declarations to ExcelWriter style.

        Parameters
        ----------
        declarations : str | frozenset[tuple[str, str]]
            CSS string or set of CSS declaration tuples.
            e.g. "font-weight: bold; background: blue" or
            {("font-weight", "bold"), ("background", "blue")}

        Returns
        -------
        xlstyle : dict
            A style as interpreted by ExcelWriter when found in
            ExcelCell.style.
        '''
    def _call_uncached(self, declarations: str | frozenset[tuple[str, str]]) -> dict[str, dict[str, str]]: ...
    def build_xlstyle(self, props: Mapping[str, str]) -> dict[str, dict[str, str]]: ...
    def build_alignment(self, props: Mapping[str, str]) -> dict[str, bool | str | None]: ...
    def _get_vertical_alignment(self, props: Mapping[str, str]) -> str | None: ...
    def _get_is_wrap_text(self, props: Mapping[str, str]) -> bool | None: ...
    def build_border(self, props: Mapping[str, str]) -> dict[str, dict[str, str | None]]: ...
    def _border_style(self, style: str | None, width: str | None, color: str | None): ...
    def _get_width_name(self, width_input: str | None) -> str | None: ...
    def _width_to_float(self, width: str | None) -> float: ...
    def _pt_to_float(self, pt_string: str) -> float: ...
    def build_fill(self, props: Mapping[str, str]): ...
    def build_number_format(self, props: Mapping[str, str]) -> dict[str, str | None]: ...
    def build_font(self, props: Mapping[str, str]) -> dict[str, bool | float | str | None]: ...
    def _get_is_bold(self, props: Mapping[str, str]) -> bool | None: ...
    def _get_is_italic(self, props: Mapping[str, str]) -> bool | None: ...
    def _get_decoration(self, props: Mapping[str, str]) -> Sequence[str]: ...
    def _get_underline(self, decoration: Sequence[str]) -> str | None: ...
    def _get_shadow(self, props: Mapping[str, str]) -> bool | None: ...
    def _get_font_names(self, props: Mapping[str, str]) -> Sequence[str]: ...
    def _get_font_size(self, props: Mapping[str, str]) -> float | None: ...
    def _select_font_family(self, font_names: Sequence[str]) -> int | None: ...
    def color_to_excel(self, val: str | None) -> str | None: ...
    def _is_hex_color(self, color_string: str) -> bool: ...
    def _convert_hex_to_excel(self, color_string: str) -> str: ...
    def _is_shorthand_color(self, color_string: str) -> bool:
        """Check if color code is shorthand.

        #FFF is a shorthand as opposed to full #FFFFFF.
        """

class ExcelFormatter:
    """
    Class for formatting a DataFrame to a list of ExcelCells,

    Parameters
    ----------
    df : DataFrame or Styler
    na_rep: na representation
    float_format : str, default None
        Format string for floating point numbers
    cols : sequence, optional
        Columns to write
    header : bool or sequence of str, default True
        Write out column names. If a list of string is given it is
        assumed to be aliases for the column names
    index : bool, default True
        output row names (index)
    index_label : str or sequence, default None
        Column label for index column(s) if desired. If None is given, and
        `header` and `index` are True, then the index names are used. A
        sequence should be given if the DataFrame uses MultiIndex.
    merge_cells : bool, default False
        Format MultiIndex and Hierarchical Rows as merged cells.
    inf_rep : str, default `'inf'`
        representation for np.inf values (which aren't representable in Excel)
        A `'-'` sign will be added in front of -inf.
    style_converter : callable, optional
        This translates Styler styles (CSS) into ExcelWriter styles.
        Defaults to ``CSSToExcelConverter()``.
        It should have signature css_declarations string -> excel style.
        This is only called for body cells.
    """
    max_rows: Incomplete
    max_cols: Incomplete
    rowcounter: int
    na_rep: Incomplete
    styler: Incomplete
    style_converter: Callable | None
    df: Incomplete
    columns: Incomplete
    float_format: Incomplete
    index: Incomplete
    index_label: Incomplete
    header: Incomplete
    merge_cells: Incomplete
    inf_rep: Incomplete
    def __init__(self, df, na_rep: str = '', float_format: str | None = None, cols: Sequence[Hashable] | None = None, header: Sequence[Hashable] | bool = True, index: bool = True, index_label: IndexLabel | None = None, merge_cells: bool = False, inf_rep: str = 'inf', style_converter: Callable | None = None) -> None: ...
    @property
    def header_style(self) -> dict[str, dict[str, str | bool]]: ...
    def _format_value(self, val): ...
    def _format_header_mi(self) -> Iterable[ExcelCell]: ...
    def _format_header_regular(self) -> Iterable[ExcelCell]: ...
    def _format_header(self) -> Iterable[ExcelCell]: ...
    def _format_body(self) -> Iterable[ExcelCell]: ...
    def _format_regular_rows(self) -> Iterable[ExcelCell]: ...
    def _format_hierarchical_rows(self) -> Iterable[ExcelCell]: ...
    @property
    def _has_aliases(self) -> bool:
        """Whether the aliases for column names are present."""
    def _generate_body(self, coloffset: int) -> Iterable[ExcelCell]: ...
    def get_formatted_cells(self) -> Iterable[ExcelCell]: ...
    def write(self, writer: FilePath | WriteExcelBuffer | ExcelWriter, sheet_name: str = 'Sheet1', startrow: int = 0, startcol: int = 0, freeze_panes: tuple[int, int] | None = None, engine: str | None = None, storage_options: StorageOptions | None = None, engine_kwargs: dict | None = None) -> None:
        """
        writer : path-like, file-like, or ExcelWriter object
            File path or existing ExcelWriter
        sheet_name : str, default 'Sheet1'
            Name of sheet which will contain DataFrame
        startrow :
            upper left cell row to dump data frame
        startcol :
            upper left cell column to dump data frame
        freeze_panes : tuple of integer (length 2), default None
            Specifies the one-based bottommost row and rightmost column that
            is to be frozen
        engine : string, default None
            write engine to use if writer is a path - you can also set this
            via the options ``io.excel.xlsx.writer``,
            or ``io.excel.xlsm.writer``.

        {storage_options}

        engine_kwargs: dict, optional
            Arbitrary keyword arguments passed to excel engine.
        """
