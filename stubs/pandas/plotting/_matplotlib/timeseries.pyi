from datetime import timedelta
from matplotlib.axes import Axes
from pandas import DataFrame as DataFrame, DatetimeIndex as DatetimeIndex, Index as Index, PeriodIndex as PeriodIndex, Series as Series
from pandas._libs.tslibs import BaseOffset as BaseOffset, Period as Period, to_offset as to_offset
from pandas._libs.tslibs.dtypes import FreqGroup as FreqGroup, OFFSET_TO_PERIOD_FREQSTR as OFFSET_TO_PERIOD_FREQSTR
from pandas._typing import NDFrameT as NDFrameT
from pandas.core.dtypes.generic import ABCDatetimeIndex as ABCDatetimeIndex, ABCPeriodIndex as ABCPeriodIndex, ABCTimedeltaIndex as ABCTimedeltaIndex
from pandas.plotting._matplotlib.converter import TimeSeries_DateFormatter as TimeSeries_DateFormatter, TimeSeries_DateLocator as TimeSeries_DateLocator, TimeSeries_TimedeltaFormatter as TimeSeries_TimedeltaFormatter
from pandas.tseries.frequencies import get_period_alias as get_period_alias, is_subperiod as is_subperiod, is_superperiod as is_superperiod
from typing import Any

def maybe_resample(series: Series, ax: Axes, kwargs: dict[str, Any]): ...
def _is_sub(f1: str, f2: str) -> bool: ...
def _is_sup(f1: str, f2: str) -> bool: ...
def _upsample_others(ax: Axes, freq: BaseOffset, kwargs: dict[str, Any]) -> None: ...
def _replot_ax(ax: Axes, freq: BaseOffset): ...
def decorate_axes(ax: Axes, freq: BaseOffset) -> None:
    """Initialize axes for time-series plotting"""
def _get_ax_freq(ax: Axes):
    """
    Get the freq attribute of the ax object if set.
    Also checks shared axes (eg when using secondary yaxis, sharex=True
    or twinx)
    """
def _get_period_alias(freq: timedelta | BaseOffset | str) -> str | None: ...
def _get_freq(ax: Axes, series: Series): ...
def use_dynamic_x(ax: Axes, data: DataFrame | Series) -> bool: ...
def _get_index_freq(index: Index) -> BaseOffset | None: ...
def maybe_convert_index(ax: Axes, data: NDFrameT) -> NDFrameT: ...
def _format_coord(freq, t, y) -> str: ...
def format_dateaxis(subplot, freq: BaseOffset, index: DatetimeIndex | PeriodIndex) -> None:
    """
    Pretty-formats the date axis (x-axis).

    Major and minor ticks are automatically set for the frequency of the
    current underlying series.  As the dynamic mode is activated by
    default, changing the limits of the x axis will intelligently change
    the positions of the ticks.
    """
