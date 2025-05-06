import datetime as pydt
import matplotlib.dates as mdates
import matplotlib.units as munits
import numpy as np
from _typeshed import Incomplete
from collections.abc import Generator
from datetime import datetime, tzinfo
from matplotlib.axis import Axis
from matplotlib.ticker import Formatter, Locator
from pandas import Index as Index, Series as Series, get_option as get_option
from pandas._libs.tslibs import Timestamp as Timestamp, to_offset as to_offset
from pandas._libs.tslibs.dtypes import FreqGroup as FreqGroup, periods_per_day as periods_per_day
from pandas._libs.tslibs.offsets import BaseOffset as BaseOffset
from pandas._typing import F as F, npt as npt
from pandas.core.dtypes.common import is_float as is_float, is_float_dtype as is_float_dtype, is_integer as is_integer, is_integer_dtype as is_integer_dtype, is_nested_list_like as is_nested_list_like
from pandas.core.indexes.period import Period as Period, PeriodIndex as PeriodIndex, period_range as period_range
from typing import Any

_mpl_units: Incomplete

def get_pairs(): ...
def register_pandas_matplotlib_converters(func: F) -> F:
    """
    Decorator applying pandas_converters.
    """
def pandas_converters() -> Generator[None, None, None]:
    """
    Context manager registering pandas' converters for a plot.

    See Also
    --------
    register_pandas_matplotlib_converters : Decorator that applies this.
    """
def register() -> None: ...
def deregister() -> None: ...
def _to_ordinalf(tm: pydt.time) -> float: ...
def time2num(d): ...

class TimeConverter(munits.ConversionInterface):
    @staticmethod
    def convert(value, unit, axis): ...
    @staticmethod
    def axisinfo(unit, axis) -> munits.AxisInfo | None: ...
    @staticmethod
    def default_units(x, axis) -> str: ...

class TimeFormatter(Formatter):
    locs: Incomplete
    def __init__(self, locs) -> None: ...
    def __call__(self, x, pos: int | None = 0) -> str:
        """
        Return the time of day as a formatted string.

        Parameters
        ----------
        x : float
            The time of day specified as seconds since 00:00 (midnight),
            with up to microsecond precision.
        pos
            Unused

        Returns
        -------
        str
            A string in HH:MM:SS.mmmuuu format. Microseconds,
            milliseconds and seconds are only displayed if non-zero.
        """

class PeriodConverter(mdates.DateConverter):
    @staticmethod
    def convert(values, units, axis): ...
    @staticmethod
    def _convert_1d(values, units, axis): ...

def get_datevalue(date, freq): ...

class DatetimeConverter(mdates.DateConverter):
    @staticmethod
    def convert(values, unit, axis): ...
    @staticmethod
    def _convert_1d(values, unit, axis): ...
    @staticmethod
    def axisinfo(unit: tzinfo | None, axis) -> munits.AxisInfo:
        """
        Return the :class:`~matplotlib.units.AxisInfo` for *unit*.

        *unit* is a tzinfo instance or None.
        The *axis* argument is required but not used.
        """

class PandasAutoDateFormatter(mdates.AutoDateFormatter):
    def __init__(self, locator, tz: Incomplete | None = None, defaultfmt: str = '%Y-%m-%d') -> None: ...

class PandasAutoDateLocator(mdates.AutoDateLocator):
    _freq: int
    def get_locator(self, dmin, dmax):
        """Pick the best locator based on a distance."""
    def _get_unit(self): ...

class MilliSecondLocator(mdates.DateLocator):
    UNIT: Incomplete
    _interval: float
    def __init__(self, tz) -> None: ...
    def _get_unit(self): ...
    @staticmethod
    def get_unit_generic(freq): ...
    def __call__(self): ...
    def _get_interval(self): ...
    def autoscale(self):
        """
        Set the view limits to include the data range.
        """

def _from_ordinal(x, tz: tzinfo | None = None) -> datetime: ...
def _get_default_annual_spacing(nyears) -> tuple[int, int]:
    """
    Returns a default spacing between consecutive ticks for annual data.
    """
def _period_break(dates: PeriodIndex, period: str) -> npt.NDArray[np.intp]:
    """
    Returns the indices where the given period changes.

    Parameters
    ----------
    dates : PeriodIndex
        Array of intervals to monitor.
    period : str
        Name of the period to monitor.
    """
def _period_break_mask(dates: PeriodIndex, period: str) -> npt.NDArray[np.bool_]: ...
def has_level_label(label_flags: npt.NDArray[np.intp], vmin: float) -> bool:
    """
    Returns true if the ``label_flags`` indicate there is at least one label
    for this level.

    if the minimum view limit is not an exact integer, then the first tick
    label won't be shown, so we must adjust for that.
    """
def _get_periods_per_ymd(freq: BaseOffset) -> tuple[int, int, int]: ...
def _daily_finder(vmin: float, vmax: float, freq: BaseOffset) -> np.ndarray: ...
def _monthly_finder(vmin: float, vmax: float, freq: BaseOffset) -> np.ndarray: ...
def _quarterly_finder(vmin: float, vmax: float, freq: BaseOffset) -> np.ndarray: ...
def _annual_finder(vmin: float, vmax: float, freq: BaseOffset) -> np.ndarray: ...
def get_finder(freq: BaseOffset): ...

class TimeSeries_DateLocator(Locator):
    """
    Locates the ticks along an axis controlled by a :class:`Series`.

    Parameters
    ----------
    freq : BaseOffset
        Valid frequency specifier.
    minor_locator : {False, True}, optional
        Whether the locator is for minor ticks (True) or not.
    dynamic_mode : {True, False}, optional
        Whether the locator should work in dynamic mode.
    base : {int}, optional
    quarter : {int}, optional
    month : {int}, optional
    day : {int}, optional
    """
    axis: Axis
    freq: Incomplete
    base: Incomplete
    isminor: Incomplete
    isdynamic: Incomplete
    offset: int
    plot_obj: Incomplete
    finder: Incomplete
    def __init__(self, freq: BaseOffset, minor_locator: bool = False, dynamic_mode: bool = True, base: int = 1, quarter: int = 1, month: int = 1, day: int = 1, plot_obj: Incomplete | None = None) -> None: ...
    def _get_default_locs(self, vmin, vmax):
        """Returns the default locations of ticks."""
    def __call__(self):
        """Return the locations of the ticks."""
    def autoscale(self):
        """
        Sets the view limits to the nearest multiples of base that contain the
        data.
        """

class TimeSeries_DateFormatter(Formatter):
    """
    Formats the ticks along an axis controlled by a :class:`PeriodIndex`.

    Parameters
    ----------
    freq : BaseOffset
        Valid frequency specifier.
    minor_locator : bool, default False
        Whether the current formatter should apply to minor ticks (True) or
        major ticks (False).
    dynamic_mode : bool, default True
        Whether the formatter works in dynamic mode or not.
    """
    axis: Axis
    format: Incomplete
    freq: Incomplete
    locs: list[Any]
    formatdict: dict[Any, Any] | None
    isminor: Incomplete
    isdynamic: Incomplete
    offset: int
    plot_obj: Incomplete
    finder: Incomplete
    def __init__(self, freq: BaseOffset, minor_locator: bool = False, dynamic_mode: bool = True, plot_obj: Incomplete | None = None) -> None: ...
    def _set_default_format(self, vmin, vmax):
        """Returns the default ticks spacing."""
    def set_locs(self, locs) -> None:
        """Sets the locations of the ticks"""
    def __call__(self, x, pos: int | None = 0) -> str: ...

class TimeSeries_TimedeltaFormatter(Formatter):
    """
    Formats the ticks along an axis controlled by a :class:`TimedeltaIndex`.
    """
    axis: Axis
    @staticmethod
    def format_timedelta_ticks(x, pos, n_decimals: int) -> str:
        """
        Convert seconds to 'D days HH:MM:SS.F'
        """
    def __call__(self, x, pos: int | None = 0) -> str: ...
