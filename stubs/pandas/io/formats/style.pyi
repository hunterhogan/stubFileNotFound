from collections.abc import (
    Callable,
    Sequence,
)
from typing import (
    Any,
    Literal,
    Protocol,
    overload,
)

from matplotlib.colors import Colormap
import numpy as np
from pandas.core.frame import DataFrame
from pandas.core.series import Series

from pandas._typing import (
    Axis,
    ExcelWriterMergeCells,
    FilePath,
    HashableT,
    HashableT1,
    HashableT2,
    IndexLabel,
    IntervalClosedType,
    Level,
    QuantileInterpolation,
    Scalar,
    StorageOptions,
    T,
    WriteBuffer,
    WriteExcelBuffer,
    npt,
)

from pandas.io.excel import ExcelWriter
from pandas.io.formats.style_render import (
    CSSProperties,
    CSSStyles,
    ExtFormatter,
    StyleExportDict,
    StylerRenderer,
    Subset,
)

class _SeriesFunc(Protocol):
    def __call__(
        self, series: Series, /, *args: Any, **kwargs: Any
    ) -> list[Any] | Series: ...

class _DataFrameFunc(Protocol):
    def __call__(
        self, series: DataFrame, /, *args: Any, **kwargs: Any
    ) -> npt.NDArray[Any] | DataFrame: ...

class _MapCallable(Protocol):
    def __call__(
        self, first_arg: Scalar, /, *args: Any, **kwargs: Any
    ) -> str | None: ...

class Styler(StylerRenderer):
    def __init__(
        self,
        data: DataFrame | Series,
        precision: int | None = None,
        table_styles: CSSStyles | None = None,
        uuid: str | None = None,
        caption: str | tuple[str, str] | None = None,
        table_attributes: str | None = None,
        cell_ids: bool = True,
        na_rep: str | None = None,
        uuid_len: int = 5,
        decimal: str | None = None,
        thousands: str | None = None,
        escape: str | None = None,
        formatter: ExtFormatter | None = None,
    ) -> None: ...
    def concat(self, other: Styler) -> Styler: ...
    @overload
    def map(
        self,
        func: Callable[[Scalar], str | None],
        subset: Subset | None = None,
    ) -> Styler: ...
    @overload
    def map(
        self,
        func: _MapCallable,
        subset: Subset | None = None,
        **kwargs: Any,
    ) -> Styler: ...
    def set_tooltips(
        self,
        ttips: DataFrame,
        props: CSSProperties | None = None,
        css_class: str | None = None,
        as_title_attribute: bool = ...,
    ) -> Styler: ...
    def to_excel(
        self,
        excel_writer: FilePath | WriteExcelBuffer | ExcelWriter,
        sheet_name: str = 'Sheet1',
        na_rep: str = '',
        float_format: str | None = None,
        columns: list[HashableT1] | None = None,
        header: list[HashableT2] | bool = True,
        index: bool = True,
        index_label: IndexLabel | None = None,
        startrow: int = 0,
        startcol: int = 0,
        engine: Literal["openpyxl", "xlsxwriter"] | None = None,
        merge_cells: ExcelWriterMergeCells = True,
        encoding: str | None = None,
        inf_rep: str = 'inf',
        verbose: bool = True,
        freeze_panes: tuple[int, int] | None = None,
        storage_options: StorageOptions | None = None,
    ) -> None: ...
    @overload
    def to_latex(
        self,
        buf: FilePath | WriteBuffer[str],
        *,
        column_format: str | None = None,
        position: str | None = None,
        position_float: Literal["centering", "raggedleft", "raggedright"] | None = None,
        hrules: bool | None = None,
        clines: (
            Literal["all;data", "all;index", "skip-last;data", "skip-last;index"] | None
        ) = None,
        label: str | None = None,
        caption: str | tuple[str, str] | None = None,
        sparse_index: bool | None = None,
        sparse_columns: bool | None = None,
        multirow_align: Literal["c", "t", "b", "naive"] | None = None,
        multicol_align: Literal["r", "c", "l", "naive-l", "naive-r"] | None = None,
        siunitx: bool = False,
        environment: str | None = None,
        encoding: str | None = None,
        convert_css: bool = False,
    ) -> None: ...
    @overload
    def to_latex(
        self,
        buf: None = None,
        *,
        column_format: str | None = None,
        position: str | None = None,
        position_float: Literal["centering", "raggedleft", "raggedright"] | None = None,
        hrules: bool | None = None,
        clines: (
            Literal["all;data", "all;index", "skip-last;data", "skip-last;index"] | None
        ) = None,
        label: str | None = None,
        caption: str | tuple[str, str] | None = None,
        sparse_index: bool | None = None,
        sparse_columns: bool | None = None,
        multirow_align: Literal["c", "t", "b", "naive"] | None = None,
        multicol_align: Literal["r", "c", "l", "naive-l", "naive-r"] | None = None,
        siunitx: bool = False,
        environment: str | None = None,
        encoding: str | None = None,
        convert_css: bool = False,
    ) -> str: ...
    @overload
    def to_html(
        self,
        buf: FilePath | WriteBuffer[str],
        *,
        table_uuid: str | None = None,
        table_attributes: str | None = None,
        sparse_index: bool | None = None,
        sparse_columns: bool | None = None,
        bold_headers: bool = False,
        caption: str | None = None,
        max_rows: int | None = None,
        max_columns: int | None = None,
        encoding: str | None = None,
        doctype_html: bool = False,
        exclude_styles: bool = False,
        **kwargs: Any,
    ) -> None: ...
    @overload
    def to_html(
        self,
        buf: None = None,
        *,
        table_uuid: str | None = None,
        table_attributes: str | None = None,
        sparse_index: bool | None = None,
        sparse_columns: bool | None = None,
        bold_headers: bool = False,
        caption: str | None = None,
        max_rows: int | None = None,
        max_columns: int | None = None,
        encoding: str | None = None,
        doctype_html: bool = False,
        exclude_styles: bool = False,
        **kwargs: Any,
    ) -> str: ...
    @overload
    def to_string(
        self,
        buf: FilePath | WriteBuffer[str],
        *,
        encoding: str | None = None,
        sparse_index: bool | None = None,
        sparse_columns: bool | None = None,
        max_rows: int | None = None,
        max_columns: int | None = None,
        delimiter: str = ' ',
    ) -> None: ...
    @overload
    def to_string(
        self,
        buf: None = None,
        *,
        encoding: str | None = None,
        sparse_index: bool | None = None,
        sparse_columns: bool | None = None,
        max_rows: int | None = None,
        max_columns: int | None = None,
        delimiter: str = ' ',
    ) -> str: ...
    def set_td_classes(self, classes: DataFrame) -> Styler: ...
    def __copy__(self) -> Styler: ...
    def __deepcopy__(self, memo: Any) -> Styler: ...
    def clear(self) -> None: ...
    @overload
    def apply(
        self,
        func: _SeriesFunc | Callable[[Series], list | Series],
        axis: Axis = 0,
        subset: Subset | None = None,
        **kwargs: Any,
    ) -> Styler: ...
    @overload
    def apply(
        self,
        func: _DataFrameFunc | Callable[[DataFrame], npt.NDArray | DataFrame],
        axis: None,
        subset: Subset | None = None,
        **kwargs: Any,
    ) -> Styler: ...
    def apply_index(
        self,
        func: Callable[[Series], npt.NDArray[np.str_] | list[str] | Series[str]],
        axis: Axis = 0,
        level: Level | list[Level] | None = None,
        **kwargs: Any,
    ) -> Styler: ...
    def map_index(
        self,
        func: Callable[[Scalar], str | None],
        axis: Axis = 0,
        level: Level | list[Level] | None = None,
        **kwargs,
    ) -> Styler: ...
    def set_table_attributes(self, attributes: str) -> Styler: ...
    def export(self) -> StyleExportDict: ...
    def use(self, styles: StyleExportDict) -> Styler: ...
    def set_uuid(self, uuid: str) -> Styler: ...
    def set_caption(self, caption: str | tuple[str, str]) -> Styler: ...
    def set_sticky(
        self,
        axis: Axis = 0,
        pixel_size: int | None = None,
        levels: Level | list[Level] | None = None,
    ) -> Styler: ...
    def set_table_styles(
        self,
        table_styles: dict[HashableT, CSSStyles] | CSSStyles | None = None,
        axis: Axis = 0,
        overwrite: bool = True,
        css_class_names: dict[str, str] | None = None,
    ) -> Styler: ...
    def hide(
        self,
        subset: Subset | None = None,
        axis: Axis = 0,
        level: Level | list[Level] | None = None,
        names: bool = False,
    ) -> Styler: ...
    def background_gradient(
        self,
        cmap: str | Colormap = 'PuBu',
        low: float = 0,
        high: float = 0,
        axis: Axis | None = 0,
        subset: Subset | None = None,
        text_color_threshold: float = 0.408,
        vmin: float | None = None,
        vmax: float | None = None,
        gmap: (
            Sequence[float]
            | Sequence[Sequence[float]]
            | npt.NDArray[Any]
            | DataFrame
            | Series
            | None
        ) = None,
    ) -> Styler: ...
    def text_gradient(
        self,
        cmap: str | Colormap = 'PuBu',
        low: float = 0,
        high: float = 0,
        axis: Axis | None = 0,
        subset: Subset | None = None,
        # In docs but not in function declaration
        # text_color_threshold: float
        vmin: float | None = None,
        vmax: float | None = None,
        gmap: (
            Sequence[float]
            | Sequence[Sequence[float]]
            | npt.NDArray[Any]
            | DataFrame
            | Series
            | None
        ) = None,
    ) -> Styler: ...
    def set_properties(
        self, subset: Subset | None = None, **kwargs: str | int
    ) -> Styler: ...
    def bar(
        self,
        subset: Subset | None = None,
        axis: Axis | None = 0,
        *,
        color: str | list[str] | tuple[str, str] | None = None,
        cmap: str | Colormap | None = None,
        width: float = 100,
        height: float = 100,
        align: (
            Literal["left", "right", "zero", "mid", "mean"]
            | float
            | Callable[[Series | npt.NDArray | DataFrame], float]
        ) = 'mid',
        vmin: float | None = None,
        vmax: float | None = None,
        props: str = 'width: 10em;',
    ) -> Styler: ...
    def highlight_null(
        self,
        color: str | None = 'red',
        subset: Subset | None = None,
        props: str | None = None,
    ) -> Styler: ...
    def highlight_max(
        self,
        subset: Subset | None = None,
        color: str = 'yellow',
        axis: Axis | None = 0,
        props: str | None = None,
    ) -> Styler: ...
    def highlight_min(
        self,
        subset: Subset | None = None,
        color: str = 'yellow',
        axis: Axis | None = 0,
        props: str | None = None,
    ) -> Styler: ...
    def highlight_between(
        self,
        subset: Subset | None = None,
        color: str = 'yellow',
        axis: Axis | None = 0,
        left: Scalar | list[Scalar] | None = None,
        right: Scalar | list[Scalar] | None = None,
        inclusive: IntervalClosedType = 'both',
        props: str | None = None,
    ) -> Styler: ...
    def highlight_quantile(
        self,
        subset: Subset | None = None,
        color: str = 'yellow',
        axis: Axis | None = 0,
        q_left: float = 0.0,
        q_right: float = 1.0,
        interpolation: QuantileInterpolation = 'linear',
        inclusive: IntervalClosedType = 'both',
        props: str | None = None,
    ) -> Styler: ...
    @classmethod
    def from_custom_template(
        cls,
        searchpath: str | list[str],
        html_table: str | None = None,
        html_style: str | None = None,
    ) -> type[Styler]: ...
    def pipe(
        self,
        func: Callable[..., T] | tuple[Callable[..., T], str],
        *args: Any,
        **kwargs: Any,
    ) -> T: ...
