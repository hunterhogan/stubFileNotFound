import numpy as np
from _typeshed import Incomplete
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame as DataFrame, Series as Series
from pandas._typing import PlottingOrientation as PlottingOrientation
from pandas.core.dtypes.common import is_integer as is_integer, is_list_like as is_list_like
from pandas.core.dtypes.generic import ABCDataFrame as ABCDataFrame, ABCIndex as ABCIndex
from pandas.core.dtypes.missing import isna as isna, remove_na_arraylike as remove_na_arraylike
from pandas.io.formats.printing import pprint_thing as pprint_thing
from pandas.plotting._matplotlib.core import LinePlot as LinePlot, MPLPlot as MPLPlot
from pandas.plotting._matplotlib.groupby import create_iter_data_given_by as create_iter_data_given_by, reformat_hist_y_given_by as reformat_hist_y_given_by
from pandas.plotting._matplotlib.misc import unpack_single_str_list as unpack_single_str_list
from pandas.plotting._matplotlib.tools import create_subplots as create_subplots, flatten_axes as flatten_axes, maybe_adjust_figure as maybe_adjust_figure, set_ticks_props as set_ticks_props
from typing import Any, Literal

class HistPlot(LinePlot):
    @property
    def _kind(self) -> Literal['hist', 'kde']: ...
    bottom: Incomplete
    _bin_range: Incomplete
    weights: Incomplete
    xlabel: Incomplete
    ylabel: Incomplete
    bins: Incomplete
    def __init__(self, data, bins: int | np.ndarray | list[np.ndarray] = 10, bottom: int | np.ndarray = 0, *, range: Incomplete | None = None, weights: Incomplete | None = None, **kwargs) -> None: ...
    def _adjust_bins(self, bins: int | np.ndarray | list[np.ndarray]): ...
    def _calculate_bins(self, data: Series | DataFrame, bins) -> np.ndarray:
        """Calculate bins given data"""
    @classmethod
    def _plot(cls, ax: Axes, y: np.ndarray, style: Incomplete | None = None, bottom: int | np.ndarray = 0, column_num: int = 0, stacking_id: Incomplete | None = None, *, bins, **kwds): ...
    def _make_plot(self, fig: Figure) -> None: ...
    def _make_plot_keywords(self, kwds: dict[str, Any], y: np.ndarray) -> None:
        """merge BoxPlot/KdePlot properties to passed kwds"""
    @staticmethod
    def _get_column_weights(weights, i: int, y): ...
    def _post_plot_logic(self, ax: Axes, data) -> None: ...
    @property
    def orientation(self) -> PlottingOrientation: ...

class KdePlot(HistPlot):
    @property
    def _kind(self) -> Literal['kde']: ...
    @property
    def orientation(self) -> Literal['vertical']: ...
    bw_method: Incomplete
    ind: Incomplete
    weights: Incomplete
    def __init__(self, data, bw_method: Incomplete | None = None, ind: Incomplete | None = None, *, weights: Incomplete | None = None, **kwargs) -> None: ...
    @staticmethod
    def _get_ind(y: np.ndarray, ind): ...
    @classmethod
    def _plot(cls, ax: Axes, y: np.ndarray, style: Incomplete | None = None, bw_method: Incomplete | None = None, ind: Incomplete | None = None, column_num: Incomplete | None = None, stacking_id: int | None = None, **kwds): ...
    def _make_plot_keywords(self, kwds: dict[str, Any], y: np.ndarray) -> None: ...
    def _post_plot_logic(self, ax: Axes, data) -> None: ...

def _grouped_plot(plotf, data: Series | DataFrame, column: Incomplete | None = None, by: Incomplete | None = None, numeric_only: bool = True, figsize: tuple[float, float] | None = None, sharex: bool = True, sharey: bool = True, layout: Incomplete | None = None, rot: float = 0, ax: Incomplete | None = None, **kwargs): ...
def _grouped_hist(data: Series | DataFrame, column: Incomplete | None = None, by: Incomplete | None = None, ax: Incomplete | None = None, bins: int = 50, figsize: tuple[float, float] | None = None, layout: Incomplete | None = None, sharex: bool = False, sharey: bool = False, rot: float = 90, grid: bool = True, xlabelsize: int | None = None, xrot: Incomplete | None = None, ylabelsize: int | None = None, yrot: Incomplete | None = None, legend: bool = False, **kwargs):
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
def hist_series(self, by: Incomplete | None = None, ax: Incomplete | None = None, grid: bool = True, xlabelsize: int | None = None, xrot: Incomplete | None = None, ylabelsize: int | None = None, yrot: Incomplete | None = None, figsize: tuple[float, float] | None = None, bins: int = 10, legend: bool = False, **kwds): ...
def hist_frame(data: DataFrame, column: Incomplete | None = None, by: Incomplete | None = None, grid: bool = True, xlabelsize: int | None = None, xrot: Incomplete | None = None, ylabelsize: int | None = None, yrot: Incomplete | None = None, ax: Incomplete | None = None, sharex: bool = False, sharey: bool = False, figsize: tuple[float, float] | None = None, layout: Incomplete | None = None, bins: int = 10, legend: bool = False, **kwds): ...
