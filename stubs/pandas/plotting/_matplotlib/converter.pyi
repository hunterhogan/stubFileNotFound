import functools
import matplotlib.dates
import matplotlib.ticker
import matplotlib.units
import munits
import np
import npt
import pandas._libs.lib as lib
import pandas.core.common as com
import pandas.core.tools.datetimes as tools
import pydt
from collections.abc import Generator
from datetime import datetime, tzinfo
from pandas._config.config import get_option as get_option
from pandas._libs.lib import is_float as is_float, is_integer as is_integer
from pandas._libs.tslibs.dtypes import FreqGroup as FreqGroup, periods_per_day as periods_per_day
from pandas._libs.tslibs.offsets import to_offset as to_offset
from pandas._libs.tslibs.period import Period as Period
from pandas._libs.tslibs.timestamps import Timestamp as Timestamp
from pandas._typing import F as F
from pandas.core.dtypes.common import is_float_dtype as is_float_dtype, is_integer_dtype as is_integer_dtype
from pandas.core.dtypes.inference import is_nested_list_like as is_nested_list_like
from pandas.core.indexes.base import Index as Index
from pandas.core.indexes.datetimes import date_range as date_range
from pandas.core.indexes.period import PeriodIndex as PeriodIndex, period_range as period_range
from pandas.core.series import Series as Series
from typing import ClassVar

TYPE_CHECKING: bool
npt: None
_mpl_units: dict
def get_pairs(): ...
def register_pandas_matplotlib_converters(func: F) -> F:
    """
    Decorator applying pandas_converters.
    """
def pandas_converters(*args, **kwds) -> Generator[None, None, None]:
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

class TimeConverter(matplotlib.units.ConversionInterface):
    @staticmethod
    def convert(value, unit, axis): ...
    @staticmethod
    def axisinfo(unit, axis) -> munits.AxisInfo | None: ...
    @staticmethod
    def default_units(x, axis) -> str: ...

class TimeFormatter(matplotlib.ticker.Formatter):
    def __init__(self, locs) -> None: ...
    def __call__(self, x, pos: int | None = ...) -> str:
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

class PeriodConverter(matplotlib.dates.DateConverter):
    @staticmethod
    def convert(values, units, axis): ...
    @staticmethod
    def _convert_1d(values, units, axis): ...
def get_datevalue(date, freq): ...

class DatetimeConverter(matplotlib.dates.DateConverter):
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

class PandasAutoDateFormatter(matplotlib.dates.AutoDateFormatter):
    def __init__(self, locator, tz, defaultfmt: str = ...) -> None: ...

class PandasAutoDateLocator(matplotlib.dates.AutoDateLocator):
    def get_locator(self, dmin, dmax):
        """Pick the best locator based on a distance."""
    def _get_unit(self): ...

class MilliSecondLocator(matplotlib.dates.DateLocator):
    UNIT: ClassVar[float] = ...
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
def _from_ordinal(x, tz: tzinfo | None) -> datetime: ...
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

_daily_finder: functools._lru_cache_wrapper
_monthly_finder: functools._lru_cache_wrapper
_quarterly_finder: functools._lru_cache_wrapper
_annual_finder: functools._lru_cache_wrapper
def get_finder(freq: BaseOffset): ...

class TimeSeries_DateLocator(matplotlib.ticker.Locator):
    def __init__(self, freq: BaseOffset, minor_locator: bool = ..., dynamic_mode: bool = ..., base: int = ..., quarter: int = ..., month: int = ..., day: int = ..., plot_obj) -> None: ...
    def _get_default_locs(self, vmin, vmax):
        """Returns the default locations of ticks."""
    def __call__(self):
        """Return the locations of the ticks."""
    def autoscale(self):
        """
        Sets the view limits to the nearest multiples of base that contain the
        data.
        """

class TimeSeries_DateFormatter(matplotlib.ticker.Formatter):
    def __init__(self, freq: BaseOffset, minor_locator: bool = ..., dynamic_mode: bool = ..., plot_obj) -> None: ...
    def _set_default_format(self, vmin, vmax):
        """Returns the default ticks spacing."""
    def set_locs(self, locs) -> None:
        """Sets the locations of the ticks"""
    def __call__(self, x, pos: int | None = ...) -> str: ...

class TimeSeries_TimedeltaFormatter(matplotlib.ticker.Formatter):
    @staticmethod
    def format_timedelta_ticks(x, pos, n_decimals: int) -> str:
        """
        Convert seconds to 'D days HH:MM:SS.F'
        """
    def __call__(self, x, pos: int | None = ...) -> str: ...
