import abc
import numpy as np
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterator, Sequence
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.figure import Figure
from pandas import Series as Series
from pandas._libs import lib as lib
from pandas._typing import IndexLabel as IndexLabel, NDFrameT as NDFrameT, PlottingOrientation as PlottingOrientation, npt as npt
from pandas.core.dtypes.common import is_any_real_numeric_dtype as is_any_real_numeric_dtype, is_bool as is_bool, is_float as is_float, is_float_dtype as is_float_dtype, is_hashable as is_hashable, is_integer as is_integer, is_integer_dtype as is_integer_dtype, is_iterator as is_iterator, is_list_like as is_list_like, is_number as is_number, is_numeric_dtype as is_numeric_dtype
from pandas.core.dtypes.dtypes import CategoricalDtype as CategoricalDtype, ExtensionDtype as ExtensionDtype
from pandas.core.dtypes.generic import ABCDataFrame as ABCDataFrame, ABCDatetimeIndex as ABCDatetimeIndex, ABCIndex as ABCIndex, ABCMultiIndex as ABCMultiIndex, ABCPeriodIndex as ABCPeriodIndex, ABCSeries as ABCSeries
from pandas.core.dtypes.missing import isna as isna
from pandas.core.frame import DataFrame as DataFrame
from pandas.errors import AbstractMethodError as AbstractMethodError
from pandas.io.formats.printing import pprint_thing as pprint_thing
from pandas.plotting._matplotlib import tools as tools
from pandas.plotting._matplotlib.converter import register_pandas_matplotlib_converters as register_pandas_matplotlib_converters
from pandas.plotting._matplotlib.groupby import reconstruct_data_with_by as reconstruct_data_with_by
from pandas.plotting._matplotlib.misc import unpack_single_str_list as unpack_single_str_list
from pandas.plotting._matplotlib.style import get_standard_colors as get_standard_colors
from pandas.plotting._matplotlib.timeseries import decorate_axes as decorate_axes, format_dateaxis as format_dateaxis, maybe_convert_index as maybe_convert_index, maybe_resample as maybe_resample, use_dynamic_x as use_dynamic_x
from pandas.plotting._matplotlib.tools import create_subplots as create_subplots, flatten_axes as flatten_axes, format_date_labels as format_date_labels, get_all_lines as get_all_lines, get_xlim as get_xlim, handle_shared_axes as handle_shared_axes
from pandas.util._decorators import cache_readonly as cache_readonly
from pandas.util._exceptions import find_stack_level as find_stack_level
from pandas.util.version import Version as Version
from typing import Any, Literal

def _color_in_style(style: str) -> bool:
    """
    Check if there is a color letter in the style string.
    """

class MPLPlot(ABC, metaclass=abc.ABCMeta):
    """
    Base class for assembling a pandas plot using matplotlib

    Parameters
    ----------
    data :

    """
    @property
    @abstractmethod
    def _kind(self) -> str:
        """Specify kind str. Must be overridden in child class"""
    _layout_type: str
    _default_rot: int
    @property
    def orientation(self) -> str | None: ...
    data: DataFrame
    by: Incomplete
    columns: Incomplete
    _grouped: Incomplete
    kind: Incomplete
    subplots: Incomplete
    sharex: Incomplete
    sharey: Incomplete
    figsize: Incomplete
    layout: Incomplete
    xticks: Incomplete
    yticks: Incomplete
    xlim: Incomplete
    ylim: Incomplete
    title: Incomplete
    use_index: Incomplete
    xlabel: Incomplete
    ylabel: Incomplete
    fontsize: Incomplete
    rot: Incomplete
    _rot_set: bool
    grid: Incomplete
    legend: Incomplete
    legend_handles: list[Artist]
    legend_labels: list[Hashable]
    logx: Incomplete
    logy: Incomplete
    loglog: Incomplete
    label: Incomplete
    style: Incomplete
    mark_right: Incomplete
    stacked: Incomplete
    ax: Incomplete
    errors: Incomplete
    secondary_y: Incomplete
    colormap: Incomplete
    table: Incomplete
    include_bool: Incomplete
    kwds: Incomplete
    color: Incomplete
    def __init__(self, data, kind: Incomplete | None = None, by: IndexLabel | None = None, subplots: bool | Sequence[Sequence[str]] = False, sharex: bool | None = None, sharey: bool = False, use_index: bool = True, figsize: tuple[float, float] | None = None, grid: Incomplete | None = None, legend: bool | str = True, rot: Incomplete | None = None, ax: Incomplete | None = None, fig: Incomplete | None = None, title: Incomplete | None = None, xlim: Incomplete | None = None, ylim: Incomplete | None = None, xticks: Incomplete | None = None, yticks: Incomplete | None = None, xlabel: Hashable | None = None, ylabel: Hashable | None = None, fontsize: int | None = None, secondary_y: bool | tuple | list | np.ndarray = False, colormap: Incomplete | None = None, table: bool = False, layout: Incomplete | None = None, include_bool: bool = False, column: IndexLabel | None = None, *, logx: bool | None | Literal['sym'] = False, logy: bool | None | Literal['sym'] = False, loglog: bool | None | Literal['sym'] = False, mark_right: bool = True, stacked: bool = False, label: Hashable | None = None, style: Incomplete | None = None, **kwds) -> None: ...
    @staticmethod
    def _validate_sharex(sharex: bool | None, ax, by) -> bool: ...
    @classmethod
    def _validate_log_kwd(cls, kwd: str, value: bool | None | Literal['sym']) -> bool | None | Literal['sym']: ...
    @staticmethod
    def _validate_subplots_kwarg(subplots: bool | Sequence[Sequence[str]], data: Series | DataFrame, kind: str) -> bool | list[tuple[int, ...]]:
        """
        Validate the subplots parameter

        - check type and content
        - check for duplicate columns
        - check for invalid column names
        - convert column names into indices
        - add missing columns in a group of their own
        See comments in code below for more details.

        Parameters
        ----------
        subplots : subplots parameters as passed to PlotAccessor

        Returns
        -------
        validated subplots : a bool or a list of tuples of column indices. Columns
        in the same tuple will be grouped together in the resulting plot.
        """
    def _validate_color_args(self, color, colormap): ...
    @staticmethod
    def _iter_data(data: DataFrame | dict[Hashable, Series | DataFrame]) -> Iterator[tuple[Hashable, np.ndarray]]: ...
    def _get_nseries(self, data: Series | DataFrame) -> int: ...
    @property
    def nseries(self) -> int: ...
    def draw(self) -> None: ...
    def generate(self) -> None: ...
    @staticmethod
    def _has_plotted_object(ax: Axes) -> bool:
        """check whether ax has data"""
    def _maybe_right_yaxis(self, ax: Axes, axes_num: int) -> Axes: ...
    def fig(self) -> Figure: ...
    def axes(self) -> Sequence[Axes]: ...
    def _axes_and_fig(self) -> tuple[Sequence[Axes], Figure]: ...
    @property
    def result(self):
        """
        Return result axes
        """
    @staticmethod
    def _convert_to_ndarray(data): ...
    def _ensure_frame(self, data) -> DataFrame: ...
    def _compute_plot_data(self) -> None: ...
    def _make_plot(self, fig: Figure) -> None: ...
    def _add_table(self) -> None: ...
    def _post_plot_logic_common(self, ax: Axes) -> None:
        """Common post process for each axes"""
    @abstractmethod
    def _post_plot_logic(self, ax: Axes, data) -> None:
        """Post process for each axes. Overridden in child classes"""
    def _adorn_subplots(self, fig: Figure) -> None:
        """Common post process unrelated to data"""
    @staticmethod
    def _apply_axis_properties(axis: Axis, rot: Incomplete | None = None, fontsize: int | None = None) -> None:
        """
        Tick creation within matplotlib is reasonably expensive and is
        internally deferred until accessed as Ticks are created/destroyed
        multiple times per draw. It's therefore beneficial for us to avoid
        accessing unless we will act on the Tick.
        """
    @property
    def legend_title(self) -> str | None: ...
    def _mark_right_label(self, label: str, index: int) -> str:
        """
        Append ``(right)`` to the label of a line if it's plotted on the right axis.

        Note that ``(right)`` is only appended when ``subplots=False``.
        """
    def _append_legend_handles_labels(self, handle: Artist, label: str) -> None:
        """
        Append current handle and label to ``legend_handles`` and ``legend_labels``.

        These will be used to make the legend.
        """
    def _make_legend(self) -> None: ...
    @staticmethod
    def _get_ax_legend(ax: Axes):
        """
        Take in axes and return ax and legend under different scenarios
        """
    def plt(self): ...
    _need_to_set_index: bool
    def _get_xticks(self): ...
    @classmethod
    def _plot(cls, ax: Axes, x, y: np.ndarray, style: Incomplete | None = None, is_errorbar: bool = False, **kwds): ...
    def _get_custom_index_name(self):
        """Specify whether xlabel/ylabel should be used to override index name"""
    def _get_index_name(self) -> str | None: ...
    @classmethod
    def _get_ax_layer(cls, ax, primary: bool = True):
        """get left (primary) or right (secondary) axes"""
    def _col_idx_to_axis_idx(self, col_idx: int) -> int:
        """Return the index of the axis where the column at col_idx should be plotted"""
    def _get_ax(self, i: int): ...
    def on_right(self, i: int): ...
    def _apply_style_colors(self, colors, kwds: dict[str, Any], col_num: int, label: str):
        '''
        Manage style and color based on column number and its label.
        Returns tuple of appropriate style and kwds which "color" may be added.
        '''
    def _get_colors(self, num_colors: int | None = None, color_kwds: str = 'color'): ...
    @staticmethod
    def _parse_errorbars(label: str, err, data: NDFrameT, nseries: int) -> tuple[Any, NDFrameT]:
        """
        Look for error keyword arguments and return the actual errorbar data
        or return the error DataFrame/dict

        Error bars can be specified in several ways:
            Series: the user provides a pandas.Series object of the same
                    length as the data
            ndarray: provides a np.ndarray of the same length as the data
            DataFrame/dict: error values are paired with keys matching the
                    key in the plotted DataFrame
            str: the name of the column within the plotted DataFrame

        Asymmetrical error bars are also supported, however raw error values
        must be provided in this case. For a ``N`` length :class:`Series`, a
        ``2xN`` array should be provided indicating lower and upper (or left
        and right) errors. For a ``MxN`` :class:`DataFrame`, asymmetrical errors
        should be in a ``Mx2xN`` array.
        """
    def _get_errorbars(self, label: Incomplete | None = None, index: Incomplete | None = None, xerr: bool = True, yerr: bool = True) -> dict[str, Any]: ...
    def _get_subplots(self, fig: Figure): ...
    def _get_axes_layout(self, fig: Figure) -> tuple[int, int]: ...

class PlanePlot(MPLPlot, ABC, metaclass=abc.ABCMeta):
    """
    Abstract class for plotting on plane, currently scatter and hexbin.
    """
    _layout_type: str
    x: Incomplete
    y: Incomplete
    def __init__(self, data, x, y, **kwargs) -> None: ...
    def _get_nseries(self, data: Series | DataFrame) -> int: ...
    def _post_plot_logic(self, ax: Axes, data) -> None: ...
    def _plot_colorbar(self, ax: Axes, *, fig: Figure, **kwds): ...

class ScatterPlot(PlanePlot):
    @property
    def _kind(self) -> Literal['scatter']: ...
    s: Incomplete
    colorbar: Incomplete
    norm: Incomplete
    c: Incomplete
    def __init__(self, data, x, y, s: Incomplete | None = None, c: Incomplete | None = None, *, colorbar: bool | lib.NoDefault = ..., norm: Incomplete | None = None, **kwargs) -> None: ...
    def _make_plot(self, fig: Figure) -> None: ...
    def _get_c_values(self, color, color_by_categorical: bool, c_is_column: bool): ...
    def _get_norm_and_cmap(self, c_values, color_by_categorical: bool): ...
    def _get_colorbar(self, c_values, c_is_column: bool) -> bool: ...

class HexBinPlot(PlanePlot):
    @property
    def _kind(self) -> Literal['hexbin']: ...
    C: Incomplete
    colorbar: Incomplete
    def __init__(self, data, x, y, C: Incomplete | None = None, *, colorbar: bool = True, **kwargs) -> None: ...
    def _make_plot(self, fig: Figure) -> None: ...
    def _make_legend(self) -> None: ...

class LinePlot(MPLPlot):
    _default_rot: int
    @property
    def orientation(self) -> PlottingOrientation: ...
    @property
    def _kind(self) -> Literal['line', 'area', 'hist', 'kde', 'box']: ...
    data: Incomplete
    x_compat: Incomplete
    def __init__(self, data, **kwargs) -> None: ...
    def _is_ts_plot(self) -> bool: ...
    def _use_dynamic_x(self) -> bool: ...
    def _make_plot(self, fig: Figure) -> None: ...
    @classmethod
    def _plot(cls, ax: Axes, x, y: np.ndarray, style: Incomplete | None = None, column_num: Incomplete | None = None, stacking_id: Incomplete | None = None, **kwds): ...
    def _ts_plot(self, ax: Axes, x, data: Series, style: Incomplete | None = None, **kwds): ...
    def _get_stacking_id(self) -> int | None: ...
    @classmethod
    def _initialize_stacker(cls, ax: Axes, stacking_id, n: int) -> None: ...
    @classmethod
    def _get_stacked_values(cls, ax: Axes, stacking_id: int | None, values: np.ndarray, label) -> np.ndarray: ...
    @classmethod
    def _update_stacker(cls, ax: Axes, stacking_id: int | None, values) -> None: ...
    rot: int
    def _post_plot_logic(self, ax: Axes, data) -> None: ...

class AreaPlot(LinePlot):
    @property
    def _kind(self) -> Literal['area']: ...
    def __init__(self, data, **kwargs) -> None: ...
    @classmethod
    def _plot(cls, ax: Axes, x, y: np.ndarray, style: Incomplete | None = None, column_num: Incomplete | None = None, stacking_id: Incomplete | None = None, is_errorbar: bool = False, **kwds): ...
    def _post_plot_logic(self, ax: Axes, data) -> None: ...

class BarPlot(MPLPlot):
    @property
    def _kind(self) -> Literal['bar', 'barh']: ...
    _default_rot: int
    @property
    def orientation(self) -> PlottingOrientation: ...
    _is_series: Incomplete
    bar_width: Incomplete
    _align: Incomplete
    _position: Incomplete
    tick_pos: Incomplete
    bottom: Incomplete
    left: Incomplete
    log: Incomplete
    def __init__(self, data, *, align: str = 'center', bottom: int = 0, left: int = 0, width: float = 0.5, position: float = 0.5, log: bool = False, **kwargs) -> None: ...
    def ax_pos(self) -> np.ndarray: ...
    def tickoffset(self): ...
    def lim_offset(self): ...
    @classmethod
    def _plot(cls, ax: Axes, x, y: np.ndarray, w, start: int | npt.NDArray[np.intp] = 0, log: bool = False, **kwds): ...
    @property
    def _start_base(self): ...
    def _make_plot(self, fig: Figure) -> None: ...
    def _post_plot_logic(self, ax: Axes, data) -> None: ...
    def _decorate_ticks(self, ax: Axes, name: str | None, ticklabels: list[str], start_edge: float, end_edge: float) -> None: ...

class BarhPlot(BarPlot):
    @property
    def _kind(self) -> Literal['barh']: ...
    _default_rot: int
    @property
    def orientation(self) -> Literal['horizontal']: ...
    @property
    def _start_base(self): ...
    @classmethod
    def _plot(cls, ax: Axes, x, y: np.ndarray, w, start: int | npt.NDArray[np.intp] = 0, log: bool = False, **kwds): ...
    def _get_custom_index_name(self): ...
    def _decorate_ticks(self, ax: Axes, name: str | None, ticklabels: list[str], start_edge: float, end_edge: float) -> None: ...

class PiePlot(MPLPlot):
    @property
    def _kind(self) -> Literal['pie']: ...
    _layout_type: str
    def __init__(self, data, kind: Incomplete | None = None, **kwargs) -> None: ...
    @classmethod
    def _validate_log_kwd(cls, kwd: str, value: bool | None | Literal['sym']) -> bool | None | Literal['sym']: ...
    def _validate_color_args(self, color, colormap) -> None: ...
    def _make_plot(self, fig: Figure) -> None: ...
    def _post_plot_logic(self, ax: Axes, data) -> None: ...
