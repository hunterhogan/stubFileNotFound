import _cython_3_0_11
import datetime
from _typeshed import Incomplete
from pandas._libs.properties import cache_readonly as cache_readonly
from pandas._libs.tslibs.timedeltas import Timedelta as Timedelta
from pandas._libs.tslibs.timestamps import Timestamp as Timestamp
from pandas.util._exceptions import find_stack_level as find_stack_level
from typing import Any, ClassVar, overload

INVALID_FREQ_ERR_MSG: str
MONTH_ALIASES: dict
__pyx_capi__: dict
__pyx_unpickle_BaseOffset: _cython_3_0_11.cython_function_or_method
__pyx_unpickle_RelativeDeltaOffset: _cython_3_0_11.cython_function_or_method
__test__: dict
_dont_uppercase: set
_get_offset: _cython_3_0_11.cython_function_or_method
_lite_rule_alias: dict
_offset_map: dict
_relativedelta_kwds: set
apply_wraps: _cython_3_0_11.cython_function_or_method
delta_to_tick: _cython_3_0_11.cython_function_or_method
int_to_weekday: dict
prefix_mapping: dict
roll_convention: _cython_3_0_11.cython_function_or_method
roll_qtrday: _cython_3_0_11.cython_function_or_method
shift_month: _cython_3_0_11.cython_function_or_method
shift_months: _cython_3_0_11.cython_function_or_method
to_offset: _cython_3_0_11.cython_function_or_method
weekday_to_int: dict

class ApplyTypeError(TypeError): ...

class BDay(BusinessMixin):
    _attributes: ClassVar[tuple] = ...
    _period_dtype_code: ClassVar[int] = ...
    _prefix: ClassVar[str] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def _apply(self, *args, **kwargs): ...
    def _apply_array(self, *args, **kwargs): ...
    def _offset_str(self, *args, **kwargs): ...
    def is_on_offset(self, *args, **kwargs): ...

class BMonthBegin(MonthOffset):
    _day_opt: ClassVar[str] = ...
    _prefix: ClassVar[str] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @overload
    @classmethod
    def __init__(cls) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""
    @overload
    @classmethod
    def __init__(cls) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""
    @overload
    @classmethod
    def __init__(cls) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""

class BMonthEnd(MonthOffset):
    _day_opt: ClassVar[str] = ...
    _prefix: ClassVar[str] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @overload
    @classmethod
    def __init__(cls) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""
    @overload
    @classmethod
    def __init__(cls) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""
    @overload
    @classmethod
    def __init__(cls) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""

class BQuarterBegin(QuarterOffset):
    _day_opt: ClassVar[str] = ...
    _default_starting_month: ClassVar[int] = ...
    _from_name_starting_month: ClassVar[int] = ...
    _output_name: ClassVar[str] = ...
    _prefix: ClassVar[str] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @overload
    @classmethod
    def __init__(cls) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""
    @overload
    @classmethod
    def __init__(cls, startingMonth=...) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""

class BQuarterEnd(QuarterOffset):
    _day_opt: ClassVar[str] = ...
    _default_starting_month: ClassVar[int] = ...
    _from_name_starting_month: ClassVar[int] = ...
    _output_name: ClassVar[str] = ...
    _prefix: ClassVar[str] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @overload
    @classmethod
    def __init__(cls) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""
    @overload
    @classmethod
    def __init__(cls, startingMonth=...) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""

class BYearBegin(YearOffset):
    _day_opt: ClassVar[str] = ...
    _default_month: ClassVar[int] = ...
    _outputName: ClassVar[str] = ...
    _prefix: ClassVar[str] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @overload
    @classmethod
    def __init__(cls) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""
    @overload
    @classmethod
    def __init__(cls) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""
    @overload
    @classmethod
    def __init__(cls, month=...) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""

class BYearEnd(YearOffset):
    _day_opt: ClassVar[str] = ...
    _default_month: ClassVar[int] = ...
    _outputName: ClassVar[str] = ...
    _prefix: ClassVar[str] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @overload
    @classmethod
    def __init__(cls) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""
    @overload
    @classmethod
    def __init__(cls) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""
    @overload
    @classmethod
    def __init__(cls, month=...) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""

class BaseOffset:
    _adjust_dst: ClassVar[bool] = ...
    _attributes: ClassVar[tuple] = ...
    _day_opt: ClassVar[None] = ...
    _use_relativedelta: ClassVar[bool] = ...
    __array_priority__: ClassVar[int] = ...
    _cache: Incomplete
    _params: Incomplete
    _prefix: Incomplete
    base: Incomplete
    freqstr: Incomplete
    kwds: Incomplete
    n: Incomplete
    name: Incomplete
    nanos: Incomplete
    normalize: Incomplete
    rule_code: Incomplete
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _apply(self, *args, **kwargs): ...
    def _apply_array(self, *args, **kwargs): ...
    def _get_offset_day(self, *args, **kwargs): ...
    def _offset_str(self, *args, **kwargs): ...
    def _repr_attrs(self, *args, **kwargs): ...
    @staticmethod
    def _validate_n(*args, **kwargs):
        """
        Require that `n` be an integer.

        Parameters
        ----------
        n : int

        Returns
        -------
        nint : int

        Raises
        ------
        TypeError if `int(n)` raises
        ValueError if n != int(n)
        """
    def copy(self) -> Any:
        """
        Return a copy of the frequency.

        Examples
        --------
        >>> freq = pd.DateOffset(1)
        >>> freq_copy = freq.copy()
        >>> freq is freq_copy
        False
        """
    @overload
    def is_anchored(self) -> Any:
        """
        Return boolean whether the frequency is a unit frequency (n=1).

        .. deprecated:: 2.2.0
            is_anchored is deprecated and will be removed in a future version.
            Use ``obj.n == 1`` instead.

        Examples
        --------
        >>> pd.DateOffset().is_anchored()
        True
        >>> pd.DateOffset(2).is_anchored()
        False
        """
    @overload
    def is_anchored(self) -> Any:
        """
        Return boolean whether the frequency is a unit frequency (n=1).

        .. deprecated:: 2.2.0
            is_anchored is deprecated and will be removed in a future version.
            Use ``obj.n == 1`` instead.

        Examples
        --------
        >>> pd.DateOffset().is_anchored()
        True
        >>> pd.DateOffset(2).is_anchored()
        False
        """
    def is_month_end(self, ts) -> Any:
        """
        Return boolean whether a timestamp occurs on the month end.

        Examples
        --------
        >>> ts = pd.Timestamp(2022, 1, 1)
        >>> freq = pd.offsets.Hour(5)
        >>> freq.is_month_end(ts)
        False
        """
    def is_month_start(self, ts) -> Any:
        """
        Return boolean whether a timestamp occurs on the month start.

        Examples
        --------
        >>> ts = pd.Timestamp(2022, 1, 1)
        >>> freq = pd.offsets.Hour(5)
        >>> freq.is_month_start(ts)
        True
        """
    @overload
    def is_on_offset(self, ts) -> Any:
        """
        Return boolean whether a timestamp intersects with this frequency.

        Parameters
        ----------
        dt : datetime.datetime
            Timestamp to check intersections with frequency.

        Examples
        --------
        >>> ts = pd.Timestamp(2022, 1, 1)
        >>> freq = pd.offsets.Day(1)
        >>> freq.is_on_offset(ts)
        True

        >>> ts = pd.Timestamp(2022, 8, 6)
        >>> ts.day_name()
        'Saturday'
        >>> freq = pd.offsets.BusinessDay(1)
        >>> freq.is_on_offset(ts)
        False
        """
    @overload
    def is_on_offset(self, ts) -> Any:
        """
        Return boolean whether a timestamp intersects with this frequency.

        Parameters
        ----------
        dt : datetime.datetime
            Timestamp to check intersections with frequency.

        Examples
        --------
        >>> ts = pd.Timestamp(2022, 1, 1)
        >>> freq = pd.offsets.Day(1)
        >>> freq.is_on_offset(ts)
        True

        >>> ts = pd.Timestamp(2022, 8, 6)
        >>> ts.day_name()
        'Saturday'
        >>> freq = pd.offsets.BusinessDay(1)
        >>> freq.is_on_offset(ts)
        False
        """
    def is_quarter_end(self, ts) -> Any:
        """
        Return boolean whether a timestamp occurs on the quarter end.

        Examples
        --------
        >>> ts = pd.Timestamp(2022, 1, 1)
        >>> freq = pd.offsets.Hour(5)
        >>> freq.is_quarter_end(ts)
        False
        """
    def is_quarter_start(self, ts) -> Any:
        """
        Return boolean whether a timestamp occurs on the quarter start.

        Examples
        --------
        >>> ts = pd.Timestamp(2022, 1, 1)
        >>> freq = pd.offsets.Hour(5)
        >>> freq.is_quarter_start(ts)
        True
        """
    def is_year_end(self, ts) -> Any:
        """
        Return boolean whether a timestamp occurs on the year end.

        Examples
        --------
        >>> ts = pd.Timestamp(2022, 1, 1)
        >>> freq = pd.offsets.Hour(5)
        >>> freq.is_year_end(ts)
        False
        """
    def is_year_start(self, ts) -> Any:
        """
        Return boolean whether a timestamp occurs on the year start.

        Examples
        --------
        >>> ts = pd.Timestamp(2022, 1, 1)
        >>> freq = pd.offsets.Hour(5)
        >>> freq.is_year_start(ts)
        True
        """
    def rollback(self, *args, **kwargs):
        """
        Roll provided date backward to next offset only if not on offset.

        Returns
        -------
        TimeStamp
            Rolled timestamp if not on offset, otherwise unchanged timestamp.
        """
    def rollforward(self, *args, **kwargs):
        """
        Roll provided date forward to next offset only if not on offset.

        Returns
        -------
        TimeStamp
            Rolled timestamp if not on offset, otherwise unchanged timestamp.
        """
    def __add__(self, other):
        """Return self+value."""
    def __eq__(self, other: object) -> bool:
        """Return self==value."""
    def __ge__(self, other: object) -> bool:
        """Return self>=value."""
    def __gt__(self, other: object) -> bool:
        """Return self>value."""
    def __hash__(self) -> int:
        """Return hash(self)."""
    def __le__(self, other: object) -> bool:
        """Return self<=value."""
    def __lt__(self, other: object) -> bool:
        """Return self<value."""
    def __mul__(self, other):
        """Return self*value."""
    def __ne__(self, other: object) -> bool:
        """Return self!=value."""
    def __neg__(self):
        """-self"""
    def __radd__(self, other): ...
    def __reduce_cython__(self, *args, **kwargs): ...
    def __rmul__(self, other): ...
    def __rsub__(self, other): ...
    def __setstate_cython__(self, *args, **kwargs): ...
    def __sub__(self, other):
        """Return self-value."""

class BusinessDay(BusinessMixin):
    _attributes: ClassVar[tuple] = ...
    _period_dtype_code: ClassVar[int] = ...
    _prefix: ClassVar[str] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @overload
    @classmethod
    def __init__(cls, n=...) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""
    @overload
    @classmethod
    def __init__(cls, normalize=...) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""
    def _apply(self, *args, **kwargs): ...
    def _apply_array(self, *args, **kwargs): ...
    def _offset_str(self, *args, **kwargs): ...
    def is_on_offset(self, *args, **kwargs): ...

class BusinessHour(BusinessMixin):
    _adjust_dst: ClassVar[bool] = ...
    _anchor: ClassVar[int] = ...
    _attributes: ClassVar[tuple] = ...
    _prefix: ClassVar[str] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    end: Incomplete
    next_bday: Incomplete
    start: Incomplete
    @overload
    def __init__(self, n=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @overload
    def __init__(self, start=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @overload
    def __init__(self, end=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @overload
    def __init__(self, normalize=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @overload
    def __init__(self, start=..., 
...end=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _apply(self, *args, **kwargs): ...
    def _get_business_hours_by_sec(self, *args, **kwargs):
        """
        Return business hours in a day by seconds.
        """
    def _get_closing_time(self, *args, **kwargs):
        """
        Get the closing time of a business hour interval by its opening time.

        Parameters
        ----------
        dt : datetime
            Opening time of a business hour interval.

        Returns
        -------
        result : datetime
            Corresponding closing time.
        """
    def _is_on_offset(self, *args, **kwargs):
        """
        Slight speedups using calculated values.
        """
    def _next_opening_time(self, *args, **kwargs):
        """
        If self.n and sign have the same sign, return the earliest opening time
        later than or equal to current time.
        Otherwise the latest opening time earlier than or equal to current
        time.

        Opening time always locates on BusinessDay.
        However, closing time may not if business hour extends over midnight.

        Parameters
        ----------
        other : datetime
            Current time.
        sign : int, default 1.
            Either 1 or -1. Going forward in time if it has the same sign as
            self.n. Going backward in time otherwise.

        Returns
        -------
        result : datetime
            Next opening time.
        """
    def _prev_opening_time(self, *args, **kwargs):
        """
        If n is positive, return the latest opening time earlier than or equal
        to current time.
        Otherwise the earliest opening time later than or equal to current
        time.

        Parameters
        ----------
        other : datetime
            Current time.

        Returns
        -------
        result : datetime
            Previous opening time.
        """
    def _repr_attrs(self, *args, **kwargs): ...
    def is_on_offset(self, *args, **kwargs): ...
    def rollback(self, *args, **kwargs):
        """
        Roll provided date backward to next offset only if not on offset.
        """
    def rollforward(self, *args, **kwargs):
        """
        Roll provided date forward to next offset only if not on offset.
        """

class BusinessMixin(SingleConstructorOffset):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    _offset: Incomplete
    calendar: Incomplete
    holidays: Incomplete
    offset: Incomplete
    weekmask: Incomplete
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _init_custom(self, *args, **kwargs):
        """
        Additional __init__ for Custom subclasses.
        """
    def _repr_attrs(self, *args, **kwargs): ...

class BusinessMonthBegin(MonthOffset):
    _day_opt: ClassVar[str] = ...
    _prefix: ClassVar[str] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

class BusinessMonthEnd(MonthOffset):
    _day_opt: ClassVar[str] = ...
    _prefix: ClassVar[str] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

class CBMonthBegin(_CustomBusinessMonth):
    _prefix: ClassVar[str] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

class CBMonthEnd(_CustomBusinessMonth):
    _prefix: ClassVar[str] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

class CDay(BusinessDay):
    _attributes: ClassVar[tuple] = ...
    _prefix: ClassVar[str] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    _period_dtype_code: Incomplete
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _apply(self, *args, **kwargs): ...
    def _apply_array(self, *args, **kwargs): ...
    def is_on_offset(self, *args, **kwargs): ...

class CustomBusinessDay(BusinessDay):
    _attributes: ClassVar[tuple] = ...
    _prefix: ClassVar[str] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    _period_dtype_code: Incomplete
    @overload
    def __init__(self) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @overload
    def __init__(self, weekmask=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @overload
    def __init__(self, calendar=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _apply(self, *args, **kwargs): ...
    def _apply_array(self, *args, **kwargs): ...
    def is_on_offset(self, *args, **kwargs): ...

class CustomBusinessHour(BusinessHour):
    _anchor: ClassVar[int] = ...
    _attributes: ClassVar[tuple] = ...
    _prefix: ClassVar[str] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @overload
    def __init__(self) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @overload
    def __init__(self, start=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @overload
    def __init__(self, end=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @overload
    def __init__(self, end=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @overload
    def __init__(self, start=..., 
...end=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @overload
    def __init__(self, weekmask=..., 
...start=..., end=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @overload
    def __init__(self, calendar=..., start=..., end=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""

class CustomBusinessMonthBegin(_CustomBusinessMonth):
    _prefix: ClassVar[str] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @overload
    @classmethod
    def __init__(cls) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""
    @overload
    @classmethod
    def __init__(cls, weekmask=...) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""
    @overload
    @classmethod
    def __init__(cls, calendar=...) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""

class CustomBusinessMonthEnd(_CustomBusinessMonth):
    _prefix: ClassVar[str] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @overload
    @classmethod
    def __init__(cls) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""
    @overload
    @classmethod
    def __init__(cls, weekmask=...) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""
    @overload
    @classmethod
    def __init__(cls, calendar=...) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""

class DateOffset(RelativeDeltaOffset):
    def __setattr__(self, name, value): ...

class Day(Tick):
    _creso: ClassVar[int] = ...
    _nanos_inc: ClassVar[int] = ...
    _period_dtype_code: ClassVar[int] = ...
    _prefix: ClassVar[str] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""

class Easter(SingleConstructorOffset):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""
    def _apply(self, *args, **kwargs): ...
    def is_on_offset(self, *args, **kwargs): ...

class FY5253(FY5253Mixin):
    _attributes: ClassVar[tuple] = ...
    _from_name: ClassVar[method] = ...
    _parse_suffix: ClassVar[method] = ...
    _prefix: ClassVar[str] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def _apply(self, *args, **kwargs): ...
    def get_year_end(self, *args, **kwargs): ...
    def is_on_offset(self, *args, **kwargs): ...

class FY5253Mixin(SingleConstructorOffset):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    rule_code: Incomplete
    startingMonth: Incomplete
    variation: Incomplete
    weekday: Incomplete
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _get_suffix_prefix(self, *args, **kwargs): ...
    def get_rule_code_suffix(self, *args, **kwargs): ...
    def is_anchored(self, *args, **kwargs): ...

class FY5253Quarter(FY5253Mixin):
    _attributes: ClassVar[tuple] = ...
    _from_name: ClassVar[method] = ...
    _prefix: ClassVar[str] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    _offset: Incomplete
    qtr_with_extra_week: Incomplete
    rule_code: Incomplete
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _apply(self, *args, **kwargs): ...
    def _rollback_to_year(self, *args, **kwargs):
        """
        Roll `other` back to the most recent date that was on a fiscal year
        end.

        Return the date of that year-end, the number of full quarters
        elapsed between that year-end and other, and the remaining Timedelta
        since the most recent quarter-end.

        Parameters
        ----------
        other : datetime or Timestamp

        Returns
        -------
        tuple of
        prev_year_end : Timestamp giving most recent fiscal year end
        num_qtrs : int
        tdelta : Timedelta
        """
    def get_weeks(self, *args, **kwargs): ...
    def is_on_offset(self, *args, **kwargs): ...
    def year_has_extra_week(self, *args, **kwargs): ...

class Hour(Tick):
    _creso: ClassVar[int] = ...
    _nanos_inc: ClassVar[int] = ...
    _period_dtype_code: ClassVar[int] = ...
    _prefix: ClassVar[str] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""

class LastWeekOfMonth(WeekOfMonthMixin):
    _attributes: ClassVar[tuple] = ...
    _from_name: ClassVar[method] = ...
    _prefix: ClassVar[str] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    def __init__(self) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _get_offset_day(self, *args, **kwargs):
        """
        Find the day in the same month as other that has the same
        weekday as self.weekday and is the last such day in the month.

        Parameters
        ----------
        other: datetime

        Returns
        -------
        day: int
        """

class Micro(Tick):
    _creso: ClassVar[int] = ...
    _nanos_inc: ClassVar[int] = ...
    _period_dtype_code: ClassVar[int] = ...
    _prefix: ClassVar[str] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @overload
    @classmethod
    def __init__(cls, n=...) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""
    @overload
    @classmethod
    def __init__(cls, n=...) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""
    @overload
    @classmethod
    def __init__(cls, n=...) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""

class Milli(Tick):
    _creso: ClassVar[int] = ...
    _nanos_inc: ClassVar[int] = ...
    _period_dtype_code: ClassVar[int] = ...
    _prefix: ClassVar[str] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @overload
    @classmethod
    def __init__(cls, n=...) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""
    @overload
    @classmethod
    def __init__(cls, n=...) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""
    @overload
    @classmethod
    def __init__(cls, n=...) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""

class Minute(Tick):
    _creso: ClassVar[int] = ...
    _nanos_inc: ClassVar[int] = ...
    _period_dtype_code: ClassVar[int] = ...
    _prefix: ClassVar[str] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @overload
    @classmethod
    def __init__(cls, n=...) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""
    @overload
    @classmethod
    def __init__(cls, n=...) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""
    @overload
    @classmethod
    def __init__(cls, n=...) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""

class MonthBegin(MonthOffset):
    _day_opt: ClassVar[str] = ...
    _prefix: ClassVar[str] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @overload
    @classmethod
    def __init__(cls) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""
    @overload
    @classmethod
    def __init__(cls) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""
    @overload
    @classmethod
    def __init__(cls) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""

class MonthEnd(MonthOffset):
    _day_opt: ClassVar[str] = ...
    _period_dtype_code: ClassVar[int] = ...
    _prefix: ClassVar[str] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @overload
    @classmethod
    def __init__(cls) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""
    @overload
    @classmethod
    def __init__(cls) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""
    @overload
    @classmethod
    def __init__(cls) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""

class MonthOffset(SingleConstructorOffset):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def _apply(self, *args, **kwargs): ...
    def _apply_array(self, *args, **kwargs): ...
    def is_on_offset(self, *args, **kwargs): ...

class Nano(Tick):
    _creso: ClassVar[int] = ...
    _nanos_inc: ClassVar[int] = ...
    _period_dtype_code: ClassVar[int] = ...
    _prefix: ClassVar[str] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @overload
    @classmethod
    def __init__(cls, n=...) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""
    @overload
    @classmethod
    def __init__(cls, n=...) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""
    @overload
    @classmethod
    def __init__(cls, n=...) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""

class OffsetMeta(type):
    __instancecheck__: ClassVar[method] = ...
    __subclasscheck__: ClassVar[method] = ...

class QuarterBegin(QuarterOffset):
    _day_opt: ClassVar[str] = ...
    _default_starting_month: ClassVar[int] = ...
    _from_name_starting_month: ClassVar[int] = ...
    _prefix: ClassVar[str] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""

class QuarterEnd(QuarterOffset):
    _day_opt: ClassVar[str] = ...
    _default_starting_month: ClassVar[int] = ...
    _prefix: ClassVar[str] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    _period_dtype_code: Incomplete
    def __init__(self) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""

class QuarterOffset(SingleConstructorOffset):
    _attributes: ClassVar[tuple] = ...
    _from_name: ClassVar[method] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    rule_code: Incomplete
    startingMonth: Incomplete
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _apply(self, *args, **kwargs): ...
    def _apply_array(self, *args, **kwargs): ...
    def is_anchored(self, *args, **kwargs): ...
    def is_on_offset(self, *args, **kwargs): ...

class RelativeDeltaOffset(BaseOffset):
    _adjust_dst: ClassVar[bool] = ...
    _attributes: ClassVar[tuple] = ...
    _pd_timedelta: Incomplete
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _apply(self, *args, **kwargs): ...
    def _apply_array(self, *args, **kwargs): ...
    def is_on_offset(self, *args, **kwargs): ...
    def __reduce_cython__(self, *args, **kwargs): ...
    def __setstate_cython__(self, *args, **kwargs): ...

class Second(Tick):
    _creso: ClassVar[int] = ...
    _nanos_inc: ClassVar[int] = ...
    _period_dtype_code: ClassVar[int] = ...
    _prefix: ClassVar[str] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @overload
    @classmethod
    def __init__(cls, n=...) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""
    @overload
    @classmethod
    def __init__(cls, n=...) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""
    @overload
    @classmethod
    def __init__(cls, n=...) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""

class SemiMonthBegin(SemiMonthOffset):
    _prefix: ClassVar[str] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def is_on_offset(self, *args, **kwargs): ...

class SemiMonthEnd(SemiMonthOffset):
    _min_day_of_month: ClassVar[int] = ...
    _prefix: ClassVar[str] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def is_on_offset(self, *args, **kwargs): ...

class SemiMonthOffset(SingleConstructorOffset):
    _attributes: ClassVar[tuple] = ...
    _default_day_of_month: ClassVar[int] = ...
    _from_name: ClassVar[method] = ...
    _min_day_of_month: ClassVar[int] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    day_of_month: Incomplete
    rule_code: Incomplete
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _apply(self, *args, **kwargs): ...
    def _apply_array(self, *args, **kwargs): ...

class SingleConstructorOffset(BaseOffset):
    _from_name: ClassVar[method] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __reduce__(self): ...

class Tick(SingleConstructorOffset):
    _adjust_dst: ClassVar[bool] = ...
    _attributes: ClassVar[tuple] = ...
    _prefix: ClassVar[str] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    _as_pd_timedelta: Incomplete
    delta: Incomplete
    nanos: Incomplete
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _apply(self, *args, **kwargs): ...
    def _next_higher_resolution(self, *args, **kwargs): ...
    def _repr_attrs(self, *args, **kwargs): ...
    @overload
    def is_anchored(self) -> Any:
        """
        Return False.

        .. deprecated:: 2.2.0
            is_anchored is deprecated and will be removed in a future version.
            Use ``False`` instead.

        Examples
        --------
        >>> pd.offsets.Hour().is_anchored()
        False
        >>> pd.offsets.Hour(2).is_anchored()
        False
        """
    @overload
    def is_anchored(self) -> Any:
        """
        Return False.

        .. deprecated:: 2.2.0
            is_anchored is deprecated and will be removed in a future version.
            Use ``False`` instead.

        Examples
        --------
        >>> pd.offsets.Hour().is_anchored()
        False
        >>> pd.offsets.Hour(2).is_anchored()
        False
        """
    def is_on_offset(self, *args, **kwargs): ...
    def __add__(self, other):
        """Return self+value."""
    def __eq__(self, other: object) -> bool:
        """Return self==value."""
    def __ge__(self, other: object) -> bool:
        """Return self>=value."""
    def __gt__(self, other: object) -> bool:
        """Return self>value."""
    def __hash__(self) -> int:
        """Return hash(self)."""
    def __le__(self, other: object) -> bool:
        """Return self<=value."""
    def __lt__(self, other: object) -> bool:
        """Return self<value."""
    def __mul__(self, other):
        """Return self*value."""
    def __ne__(self, other: object) -> bool:
        """Return self!=value."""
    def __radd__(self, other): ...
    def __rmul__(self, other): ...
    def __rtruediv__(self, other): ...
    def __truediv__(self, other):
        """Return self/value."""

class Week(SingleConstructorOffset):
    _attributes: ClassVar[tuple] = ...
    _from_name: ClassVar[method] = ...
    _inc: ClassVar[datetime.timedelta] = ...
    _prefix: ClassVar[str] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    _period_dtype_code: Incomplete
    rule_code: Incomplete
    weekday: Incomplete
    @overload
    def __init__(self, n=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @overload
    def __init__(self, weekday=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @overload
    def __init__(self, weekday=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _apply(self, *args, **kwargs): ...
    def _apply_array(self, *args, **kwargs): ...
    def is_anchored(self, *args, **kwargs): ...
    def is_on_offset(self, *args, **kwargs): ...

class WeekOfMonth(WeekOfMonthMixin):
    _attributes: ClassVar[tuple] = ...
    _from_name: ClassVar[method] = ...
    _prefix: ClassVar[str] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    def __init__(self) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _get_offset_day(self, *args, **kwargs):
        """
        Find the day in the same month as other that has the same
        weekday as self.weekday and is the self.week'th such day in the month.

        Parameters
        ----------
        other : datetime

        Returns
        -------
        day : int
        """

class WeekOfMonthMixin(SingleConstructorOffset):
    rule_code: Incomplete
    week: Incomplete
    weekday: Incomplete
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _apply(self, *args, **kwargs): ...
    def is_on_offset(self, *args, **kwargs): ...

class YearBegin(YearOffset):
    _day_opt: ClassVar[str] = ...
    _default_month: ClassVar[int] = ...
    _prefix: ClassVar[str] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @overload
    @classmethod
    def __init__(cls) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""
    @overload
    @classmethod
    def __init__(cls) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""
    @overload
    @classmethod
    def __init__(cls, month=...) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""
    @overload
    @classmethod
    def __init__(cls) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""

class YearEnd(YearOffset):
    _day_opt: ClassVar[str] = ...
    _default_month: ClassVar[int] = ...
    _prefix: ClassVar[str] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    _period_dtype_code: Incomplete
    @overload
    def __init__(self) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @overload
    def __init__(self) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @overload
    def __init__(self, month=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @overload
    def __init__(self) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""

class YearOffset(SingleConstructorOffset):
    _attributes: ClassVar[tuple] = ...
    _from_name: ClassVar[method] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    month: Incomplete
    rule_code: Incomplete
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _apply(self, *args, **kwargs): ...
    def _apply_array(self, *args, **kwargs): ...
    def _get_offset_day(self, *args, **kwargs): ...
    def is_on_offset(self, *args, **kwargs): ...

class _CustomBusinessMonth(BusinessMixin):
    _attributes: ClassVar[tuple] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    cbday_roll: Incomplete
    m_offset: Incomplete
    month_roll: Incomplete
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _apply(self, *args, **kwargs): ...
