import _cython_3_0_11
import numpy.dtypes
from _typeshed import Incomplete
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr as freq_to_period_freqstr
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime as OutOfBoundsDatetime
from pandas._libs.tslibs.offsets import BDay as BDay
from pandas._libs.tslibs.parsing import parse_datetime_string_with_reso as parse_datetime_string_with_reso
from pandas._libs.tslibs.timestamps import Timestamp as Timestamp
from typing import Any, ClassVar

DIFFERENT_FREQ: str
DT64NS_DTYPE: numpy.dtypes.DateTime64DType
INVALID_FREQ_ERR_MSG: str
__pyx_capi__: dict
__pyx_unpickle_PeriodMixin: _cython_3_0_11.cython_function_or_method
__test__: dict
extract_freq: _cython_3_0_11.cython_function_or_method
extract_ordinals: _cython_3_0_11.cython_function_or_method
freq_to_dtype_code: _cython_3_0_11.cython_function_or_method
from_ordinals: _cython_3_0_11.cython_function_or_method
get_period_field_arr: _cython_3_0_11.cython_function_or_method
period_array_strftime: _cython_3_0_11.cython_function_or_method
period_asfreq: _cython_3_0_11.cython_function_or_method
period_asfreq_arr: _cython_3_0_11.cython_function_or_method
period_ordinal: _cython_3_0_11.cython_function_or_method
periodarr_to_dt64arr: _cython_3_0_11.cython_function_or_method
validate_end_alias: _cython_3_0_11.cython_function_or_method

class IncompatibleFrequency(ValueError): ...

class Period(_Period):
    def __init__(self, *args, **kwargs) -> None: ...

class PeriodMixin:
    end_time: Incomplete
    start_time: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def _require_matching_freq(self, *args, **kwargs): ...
    def __reduce__(self): ...
    def __reduce_cython__(self, *args, **kwargs): ...
    def __setstate_cython__(self, *args, **kwargs): ...

class _Period(PeriodMixin):
    _from_ordinal: ClassVar[method] = ...
    _maybe_convert_freq: ClassVar[method] = ...
    now: ClassVar[method] = ...
    __array_priority__: ClassVar[int] = ...
    _dtype: Incomplete
    day: Incomplete
    day_of_week: Incomplete
    day_of_year: Incomplete
    dayofweek: Incomplete
    dayofyear: Incomplete
    days_in_month: Incomplete
    daysinmonth: Incomplete
    freq: Incomplete
    freqstr: Incomplete
    hour: Incomplete
    is_leap_year: Incomplete
    minute: Incomplete
    month: Incomplete
    ordinal: Incomplete
    quarter: Incomplete
    qyear: Incomplete
    second: Incomplete
    week: Incomplete
    weekday: Incomplete
    weekofyear: Incomplete
    year: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def _add_offset(self, *args, **kwargs): ...
    def _add_timedeltalike_scalar(self, *args, **kwargs): ...
    def asfreq(self, *args, **kwargs):
        """
        Convert Period to desired frequency, at the start or end of the interval.

        Parameters
        ----------
        freq : str, BaseOffset
            The desired frequency. If passing a `str`, it needs to be a
            valid :ref:`period alias <timeseries.period_aliases>`.
        how : {'E', 'S', 'end', 'start'}, default 'end'
            Start or end of the timespan.

        Returns
        -------
        resampled : Period

        Examples
        --------
        >>> period = pd.Period('2023-1-1', freq='D')
        >>> period.asfreq('h')
        Period('2023-01-01 23:00', 'h')
        """
    def strftime(self, *args, **kwargs):
        """
        Returns a formatted string representation of the :class:`Period`.

        ``fmt`` must be ``None`` or a string containing one or several directives.
        When ``None``, the format will be determined from the frequency of the Period.
        The method recognizes the same directives as the :func:`time.strftime`
        function of the standard Python distribution, as well as the specific
        additional directives ``%f``, ``%F``, ``%q``, ``%l``, ``%u``, ``%n``.
        (formatting & docs originally from scikits.timeries).

        +-----------+--------------------------------+-------+
        | Directive | Meaning                        | Notes |
        +===========+================================+=======+
        | ``%a``    | Locale's abbreviated weekday   |       |
        |           | name.                          |       |
        +-----------+--------------------------------+-------+
        | ``%A``    | Locale's full weekday name.    |       |
        +-----------+--------------------------------+-------+
        | ``%b``    | Locale's abbreviated month     |       |
        |           | name.                          |       |
        +-----------+--------------------------------+-------+
        | ``%B``    | Locale's full month name.      |       |
        +-----------+--------------------------------+-------+
        | ``%c``    | Locale's appropriate date and  |       |
        |           | time representation.           |       |
        +-----------+--------------------------------+-------+
        | ``%d``    | Day of the month as a decimal  |       |
        |           | number [01,31].                |       |
        +-----------+--------------------------------+-------+
        | ``%f``    | 'Fiscal' year without a        | \\(1)  |
        |           | century  as a decimal number   |       |
        |           | [00,99]                        |       |
        +-----------+--------------------------------+-------+
        | ``%F``    | 'Fiscal' year with a century   | \\(2)  |
        |           | as a decimal number            |       |
        +-----------+--------------------------------+-------+
        | ``%H``    | Hour (24-hour clock) as a      |       |
        |           | decimal number [00,23].        |       |
        +-----------+--------------------------------+-------+
        | ``%I``    | Hour (12-hour clock) as a      |       |
        |           | decimal number [01,12].        |       |
        +-----------+--------------------------------+-------+
        | ``%j``    | Day of the year as a decimal   |       |
        |           | number [001,366].              |       |
        +-----------+--------------------------------+-------+
        | ``%m``    | Month as a decimal number      |       |
        |           | [01,12].                       |       |
        +-----------+--------------------------------+-------+
        | ``%M``    | Minute as a decimal number     |       |
        |           | [00,59].                       |       |
        +-----------+--------------------------------+-------+
        | ``%p``    | Locale's equivalent of either  | \\(3)  |
        |           | AM or PM.                      |       |
        +-----------+--------------------------------+-------+
        | ``%q``    | Quarter as a decimal number    |       |
        |           | [1,4]                          |       |
        +-----------+--------------------------------+-------+
        | ``%S``    | Second as a decimal number     | \\(4)  |
        |           | [00,61].                       |       |
        +-----------+--------------------------------+-------+
        | ``%l``    | Millisecond as a decimal number|       |
        |           | [000,999].                     |       |
        +-----------+--------------------------------+-------+
        | ``%u``    | Microsecond as a decimal number|       |
        |           | [000000,999999].               |       |
        +-----------+--------------------------------+-------+
        | ``%n``    | Nanosecond as a decimal number |       |
        |           | [000000000,999999999].         |       |
        +-----------+--------------------------------+-------+
        | ``%U``    | Week number of the year        | \\(5)  |
        |           | (Sunday as the first day of    |       |
        |           | the week) as a decimal number  |       |
        |           | [00,53].  All days in a new    |       |
        |           | year preceding the first       |       |
        |           | Sunday are considered to be in |       |
        |           | week 0.                        |       |
        +-----------+--------------------------------+-------+
        | ``%w``    | Weekday as a decimal number    |       |
        |           | [0(Sunday),6].                 |       |
        +-----------+--------------------------------+-------+
        | ``%W``    | Week number of the year        | \\(5)  |
        |           | (Monday as the first day of    |       |
        |           | the week) as a decimal number  |       |
        |           | [00,53].  All days in a new    |       |
        |           | year preceding the first       |       |
        |           | Monday are considered to be in |       |
        |           | week 0.                        |       |
        +-----------+--------------------------------+-------+
        | ``%x``    | Locale's appropriate date      |       |
        |           | representation.                |       |
        +-----------+--------------------------------+-------+
        | ``%X``    | Locale's appropriate time      |       |
        |           | representation.                |       |
        +-----------+--------------------------------+-------+
        | ``%y``    | Year without century as a      |       |
        |           | decimal number [00,99].        |       |
        +-----------+--------------------------------+-------+
        | ``%Y``    | Year with century as a decimal |       |
        |           | number.                        |       |
        +-----------+--------------------------------+-------+
        | ``%Z``    | Time zone name (no characters  |       |
        |           | if no time zone exists).       |       |
        +-----------+--------------------------------+-------+
        | ``%%``    | A literal ``'%'`` character.   |       |
        +-----------+--------------------------------+-------+

        Notes
        -----

        (1)
            The ``%f`` directive is the same as ``%y`` if the frequency is
            not quarterly.
            Otherwise, it corresponds to the 'fiscal' year, as defined by
            the :attr:`qyear` attribute.

        (2)
            The ``%F`` directive is the same as ``%Y`` if the frequency is
            not quarterly.
            Otherwise, it corresponds to the 'fiscal' year, as defined by
            the :attr:`qyear` attribute.

        (3)
            The ``%p`` directive only affects the output hour field
            if the ``%I`` directive is used to parse the hour.

        (4)
            The range really is ``0`` to ``61``; this accounts for leap
            seconds and the (very rare) double leap seconds.

        (5)
            The ``%U`` and ``%W`` directives are only used in calculations
            when the day of the week and the year are specified.

        Examples
        --------

        >>> from pandas import Period
        >>> a = Period(freq='Q-JUL', year=2006, quarter=1)
        >>> a.strftime('%F-Q%q')
        '2006-Q1'
        >>> # Output the last month in the quarter of this date
        >>> a.strftime('%b-%Y')
        'Oct-2005'
        >>>
        >>> a = Period(freq='D', year=2001, month=1, day=1)
        >>> a.strftime('%d-%b-%Y')
        '01-Jan-2001'
        >>> a.strftime('%b. %d, %Y was a %A')
        'Jan. 01, 2001 was a Monday'
        """
    def to_timestamp(self) -> Any:
        """
        Return the Timestamp representation of the Period.

        Uses the target frequency specified at the part of the period specified
        by `how`, which is either `Start` or `Finish`.

        Parameters
        ----------
        freq : str or DateOffset
            Target frequency. Default is 'D' if self.freq is week or
            longer and 'S' otherwise.
        how : str, default 'S' (start)
            One of 'S', 'E'. Can be aliased as case insensitive
            'Start', 'Finish', 'Begin', 'End'.

        Returns
        -------
        Timestamp

        Examples
        --------
        >>> period = pd.Period('2023-1-1', freq='D')
        >>> timestamp = period.to_timestamp()
        >>> timestamp
        Timestamp('2023-01-01 00:00:00')
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
    def __ne__(self, other: object) -> bool:
        """Return self!=value."""
    def __radd__(self, other): ...
    def __reduce__(self): ...
    def __rsub__(self, other): ...
    def __sub__(self, other):
        """Return self-value."""
