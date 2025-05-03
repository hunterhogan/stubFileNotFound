import _abc
import np
import pandas.plotting._matplotlib.core
from pandas._libs.lib import is_integer as is_integer, is_list_like as is_list_like
from pandas.core.dtypes.generic import ABCDataFrame as ABCDataFrame, ABCIndex as ABCIndex
from pandas.core.dtypes.missing import isna as isna, remove_na_arraylike as remove_na_arraylike
from pandas.io.formats.printing import pprint_thing as pprint_thing
from pandas.plotting._matplotlib.core import LinePlot as LinePlot, MPLPlot as MPLPlot
from pandas.plotting._matplotlib.groupby import create_iter_data_given_by as create_iter_data_given_by, reformat_hist_y_given_by as reformat_hist_y_given_by
from pandas.plotting._matplotlib.misc import unpack_single_str_list as unpack_single_str_list
from pandas.plotting._matplotlib.tools import create_subplots as create_subplots, flatten_axes as flatten_axes, maybe_adjust_figure as maybe_adjust_figure, set_ticks_props as set_ticks_props
from typing import Any, ClassVar

TYPE_CHECKING: bool

class HistPlot(pandas.plotting._matplotlib.core.LinePlot):
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    def __init__(self, data, bins: int | np.ndarray | list[np.ndarray] = ..., bottom: int | np.ndarray = ..., *, range, weights, **kwargs) -> None: ...
    def _adjust_bins(self, bins: int | np.ndarray | list[np.ndarray]): ...
    def _calculate_bins(self, data: Series | DataFrame, bins) -> np.ndarray:
        """Calculate bins given data"""
    @classmethod
    def _plot(cls, ax: Axes, y: np.ndarray, style, bottom: int | np.ndarray = ..., column_num: int = ..., stacking_id, *, bins, **kwds): ...
    def _make_plot(self, fig: Figure) -> None: ...
    def _make_plot_keywords(self, kwds: dict[str, Any], y: np.ndarray) -> None:
        """merge BoxPlot/KdePlot properties to passed kwds"""
    @staticmethod
    def _get_column_weights(weights, i: int, y): ...
    def _post_plot_logic(self, ax: Axes, data) -> None: ...
    @property
    def _kind(self): ...
    @property
    def orientation(self): ...

class KdePlot(HistPlot):
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    def __init__(self, data, bw_method, ind, *, weights, **kwargs) -> None: ...
    @staticmethod
    def _get_ind(y: np.ndarray, ind): ...
    @classmethod
    def _plot(cls, ax: Axes, y: np.ndarray, style, bw_method, ind, column_num, stacking_id: int | None, **kwds): ...
    def _make_plot_keywords(self, kwds: dict[str, Any], y: np.ndarray) -> None: ...
    def _post_plot_logic(self, ax: Axes, data) -> None: ...
    @property
    def _kind(self): ...
    @property
    def orientation(self): ...
def _grouped_plot(plotf, data: Series | DataFrame, column, by, numeric_only: bool = ..., figsize: tuple[float, float] | None, sharex: bool = ..., sharey: bool = ..., layout, rot: float = ..., ax, **kwargs): ...
def _grouped_hist(data: Series | DataFrame, column, by, ax, bins: int = ..., figsize: tuple[float, float] | None, layout, sharex: bool = ..., sharey: bool = ..., rot: float = ..., grid: bool = ..., xlabelsize: int | None, xrot, ylabelsize: int | None, yrot, legend: bool = ..., **kwargs):
    """
    Grouped histogram

    Parameters
    ----------
    data : Series/DataFrame
    column : object, optional
    by : object, optional
    ax : axes, optional
    bins : int, default 50
    figsize : tuple, optional
    layout : optional
    sharex : bool, default False
    sharey : bool, default False
    rot : float, default 90
    grid : bool, default True
    legend: : bool, default False
    kwargs : dict, keyword arguments passed to matplotlib.Axes.hist

    Returns
    -------
    collection of Matplotlib Axes
    """
def hist_series(self: Series, by, ax, grid: bool = ..., xlabelsize: int | None, xrot, ylabelsize: int | None, yrot, figsize: tuple[float, float] | None, bins: int = ..., legend: bool = ..., **kwds): ...
def hist_frame(data: DataFrame, column, by, grid: bool = ..., xlabelsize: int | None, xrot, ylabelsize: int | None, yrot, ax, sharex: bool = ..., sharey: bool = ..., figsize: tuple[float, float] | None, layout, bins: int = ..., legend: bool = ..., **kwds): ...
