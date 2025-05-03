import _abc
import dt
import lib as lib
import np
import npt
import pandas._libs.index as libindex
import pandas._libs.lib
import pandas._libs.tslibs.timezones as timezones
import pandas.core.arrays.datetimes
import pandas.core.common as com
import pandas.core.indexes.datetimelike
from _typeshed import Incomplete
from pandas._libs.lib import is_scalar as is_scalar
from pandas._libs.properties import cache_readonly as cache_readonly
from pandas._libs.tslibs.dtypes import Resolution as Resolution, periods_per_day as periods_per_day
from pandas._libs.tslibs.nattype import NaT as NaT
from pandas._libs.tslibs.offsets import Tick as Tick, to_offset as to_offset
from pandas._libs.tslibs.period import Period as Period
from pandas._libs.tslibs.timedeltas import Timedelta as Timedelta
from pandas._libs.tslibs.timestamps import Timestamp as Timestamp
from pandas.core.arrays.datetimes import DatetimeArray as DatetimeArray, tz_to_dtype as tz_to_dtype
from pandas.core.dtypes.dtypes import DatetimeTZDtype as DatetimeTZDtype
from pandas.core.dtypes.generic import ABCSeries as ABCSeries
from pandas.core.dtypes.missing import is_valid_na_for_dtype as is_valid_na_for_dtype
from pandas.core.indexes.base import Index as Index, maybe_extract_name as maybe_extract_name
from pandas.core.indexes.datetimelike import DatetimeTimedeltaMixin as DatetimeTimedeltaMixin
from pandas.core.indexes.extension import inherit_names as inherit_names
from pandas.core.tools.times import to_time as to_time
from pandas.util._decorators import doc as doc
from pandas.util._exceptions import find_stack_level as find_stack_level
from typing import ClassVar

TYPE_CHECKING: bool
prefix_mapping: dict
OFFSET_TO_PERIOD_FREQSTR: dict
def _new_DatetimeIndex(cls, d):
    """
    This is called upon unpickling, rather than the default which doesn't
    have arguments and breaks __new__
    """

class DatetimeIndex(pandas.core.indexes.datetimelike.DatetimeTimedeltaMixin):
    _typ: ClassVar[str] = ...
    _data_cls: ClassVar[type[pandas.core.arrays.datetimes.DatetimeArray]] = ...
    _supports_partial_string_indexing: ClassVar[bool] = ...
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    _resolution_obj: Incomplete
    _is_dates_only: Incomplete
    _formatter_func: Incomplete
    tz: Incomplete
    tzinfo: Incomplete
    dtype: Incomplete
    date: Incomplete
    time: Incomplete
    timetz: Incomplete
    is_month_start: Incomplete
    is_month_end: Incomplete
    is_quarter_start: Incomplete
    is_quarter_end: Incomplete
    is_year_start: Incomplete
    is_year_end: Incomplete
    is_leap_year: Incomplete
    is_normalized: Incomplete
    year: Incomplete
    month: Incomplete
    day: Incomplete
    hour: Incomplete
    minute: Incomplete
    second: Incomplete
    weekday: Incomplete
    dayofweek: Incomplete
    day_of_week: Incomplete
    dayofyear: Incomplete
    day_of_year: Incomplete
    quarter: Incomplete
    days_in_month: Incomplete
    daysinmonth: Incomplete
    microsecond: Incomplete
    nanosecond: Incomplete
    def strftime(self, date_format) -> Index:
        '''
        Convert to Index using specified date_format.

        Return an Index of formatted strings specified by date_format, which
        supports the same string format as the python standard library. Details
        of the string format can be found in `python string format
        doc <https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior>`__.

        Formats supported by the C `strftime` API but not by the python string format
        doc (such as `"%R"`, `"%r"`) are not officially supported and should be
        preferably replaced with their supported equivalents (such as `"%H:%M"`,
        `"%I:%M:%S %p"`).

        Note that `PeriodIndex` support additional directives, detailed in
        `Period.strftime`.

        Parameters
        ----------
        date_format : str
            Date format string (e.g. "%Y-%m-%d").

        Returns
        -------
        ndarray[object]
            NumPy ndarray of formatted strings.

        See Also
        --------
        to_datetime : Convert the given argument to datetime.
        DatetimeIndex.normalize : Return DatetimeIndex with times to midnight.
        DatetimeIndex.round : Round the DatetimeIndex to the specified freq.
        DatetimeIndex.floor : Floor the DatetimeIndex to the specified freq.
        Timestamp.strftime : Format a single Timestamp.
        Period.strftime : Format a single Period.

        Examples
        --------
        >>> rng = pd.date_range(pd.Timestamp("2018-03-10 09:00"),
        ...                     periods=3, freq=\'s\')
        >>> rng.strftime(\'%B %d, %Y, %r\')
        Index([\'March 10, 2018, 09:00:00 AM\', \'March 10, 2018, 09:00:01 AM\',
               \'March 10, 2018, 09:00:02 AM\'],
              dtype=\'object\')
        '''
    def tz_convert(self, tz) -> Self:
        """
        Convert tz-aware Datetime Array/Index from one time zone to another.

        Parameters
        ----------
        tz : str, pytz.timezone, dateutil.tz.tzfile, datetime.tzinfo or None
            Time zone for time. Corresponding timestamps would be converted
            to this time zone of the Datetime Array/Index. A `tz` of None will
            convert to UTC and remove the timezone information.

        Returns
        -------
        Array or Index

        Raises
        ------
        TypeError
            If Datetime Array/Index is tz-naive.

        See Also
        --------
        DatetimeIndex.tz : A timezone that has a variable offset from UTC.
        DatetimeIndex.tz_localize : Localize tz-naive DatetimeIndex to a
            given time zone, or remove timezone from a tz-aware DatetimeIndex.

        Examples
        --------
        With the `tz` parameter, we can change the DatetimeIndex
        to other time zones:

        >>> dti = pd.date_range(start='2014-08-01 09:00',
        ...                     freq='h', periods=3, tz='Europe/Berlin')

        >>> dti
        DatetimeIndex(['2014-08-01 09:00:00+02:00',
                       '2014-08-01 10:00:00+02:00',
                       '2014-08-01 11:00:00+02:00'],
                      dtype='datetime64[ns, Europe/Berlin]', freq='h')

        >>> dti.tz_convert('US/Central')
        DatetimeIndex(['2014-08-01 02:00:00-05:00',
                       '2014-08-01 03:00:00-05:00',
                       '2014-08-01 04:00:00-05:00'],
                      dtype='datetime64[ns, US/Central]', freq='h')

        With the ``tz=None``, we can remove the timezone (after converting
        to UTC if necessary):

        >>> dti = pd.date_range(start='2014-08-01 09:00', freq='h',
        ...                     periods=3, tz='Europe/Berlin')

        >>> dti
        DatetimeIndex(['2014-08-01 09:00:00+02:00',
                       '2014-08-01 10:00:00+02:00',
                       '2014-08-01 11:00:00+02:00'],
                        dtype='datetime64[ns, Europe/Berlin]', freq='h')

        >>> dti.tz_convert(None)
        DatetimeIndex(['2014-08-01 07:00:00',
                       '2014-08-01 08:00:00',
                       '2014-08-01 09:00:00'],
                        dtype='datetime64[ns]', freq='h')
        """
    def tz_localize(self, tz, ambiguous: TimeAmbiguous = ..., nonexistent: TimeNonexistent = ...) -> Self:
        """
        Localize tz-naive Datetime Array/Index to tz-aware Datetime Array/Index.

        This method takes a time zone (tz) naive Datetime Array/Index object
        and makes this time zone aware. It does not move the time to another
        time zone.

        This method can also be used to do the inverse -- to create a time
        zone unaware object from an aware object. To that end, pass `tz=None`.

        Parameters
        ----------
        tz : str, pytz.timezone, dateutil.tz.tzfile, datetime.tzinfo or None
            Time zone to convert timestamps to. Passing ``None`` will
            remove the time zone information preserving local time.
        ambiguous : 'infer', 'NaT', bool array, default 'raise'
            When clocks moved backward due to DST, ambiguous times may arise.
            For example in Central European Time (UTC+01), when going from
            03:00 DST to 02:00 non-DST, 02:30:00 local time occurs both at
            00:30:00 UTC and at 01:30:00 UTC. In such a situation, the
            `ambiguous` parameter dictates how ambiguous times should be
            handled.

            - 'infer' will attempt to infer fall dst-transition hours based on
              order
            - bool-ndarray where True signifies a DST time, False signifies a
              non-DST time (note that this flag is only applicable for
              ambiguous times)
            - 'NaT' will return NaT where there are ambiguous times
            - 'raise' will raise an AmbiguousTimeError if there are ambiguous
              times.

        nonexistent : 'shift_forward', 'shift_backward, 'NaT', timedelta, default 'raise'
            A nonexistent time does not exist in a particular timezone
            where clocks moved forward due to DST.

            - 'shift_forward' will shift the nonexistent time forward to the
              closest existing time
            - 'shift_backward' will shift the nonexistent time backward to the
              closest existing time
            - 'NaT' will return NaT where there are nonexistent times
            - timedelta objects will shift nonexistent times by the timedelta
            - 'raise' will raise an NonExistentTimeError if there are
              nonexistent times.

        Returns
        -------
        Same type as self
            Array/Index converted to the specified time zone.

        Raises
        ------
        TypeError
            If the Datetime Array/Index is tz-aware and tz is not None.

        See Also
        --------
        DatetimeIndex.tz_convert : Convert tz-aware DatetimeIndex from
            one time zone to another.

        Examples
        --------
        >>> tz_naive = pd.date_range('2018-03-01 09:00', periods=3)
        >>> tz_naive
        DatetimeIndex(['2018-03-01 09:00:00', '2018-03-02 09:00:00',
                       '2018-03-03 09:00:00'],
                      dtype='datetime64[ns]', freq='D')

        Localize DatetimeIndex in US/Eastern time zone:

        >>> tz_aware = tz_naive.tz_localize(tz='US/Eastern')
        >>> tz_aware
        DatetimeIndex(['2018-03-01 09:00:00-05:00',
                       '2018-03-02 09:00:00-05:00',
                       '2018-03-03 09:00:00-05:00'],
                      dtype='datetime64[ns, US/Eastern]', freq=None)

        With the ``tz=None``, we can remove the time zone information
        while keeping the local time (not converted to UTC):

        >>> tz_aware.tz_localize(None)
        DatetimeIndex(['2018-03-01 09:00:00', '2018-03-02 09:00:00',
                       '2018-03-03 09:00:00'],
                      dtype='datetime64[ns]', freq=None)

        Be careful with DST changes. When there is sequential data, pandas can
        infer the DST time:

        >>> s = pd.to_datetime(pd.Series(['2018-10-28 01:30:00',
        ...                               '2018-10-28 02:00:00',
        ...                               '2018-10-28 02:30:00',
        ...                               '2018-10-28 02:00:00',
        ...                               '2018-10-28 02:30:00',
        ...                               '2018-10-28 03:00:00',
        ...                               '2018-10-28 03:30:00']))
        >>> s.dt.tz_localize('CET', ambiguous='infer')
        0   2018-10-28 01:30:00+02:00
        1   2018-10-28 02:00:00+02:00
        2   2018-10-28 02:30:00+02:00
        3   2018-10-28 02:00:00+01:00
        4   2018-10-28 02:30:00+01:00
        5   2018-10-28 03:00:00+01:00
        6   2018-10-28 03:30:00+01:00
        dtype: datetime64[ns, CET]

        In some cases, inferring the DST is impossible. In such cases, you can
        pass an ndarray to the ambiguous parameter to set the DST explicitly

        >>> s = pd.to_datetime(pd.Series(['2018-10-28 01:20:00',
        ...                               '2018-10-28 02:36:00',
        ...                               '2018-10-28 03:46:00']))
        >>> s.dt.tz_localize('CET', ambiguous=np.array([True, True, False]))
        0   2018-10-28 01:20:00+02:00
        1   2018-10-28 02:36:00+02:00
        2   2018-10-28 03:46:00+01:00
        dtype: datetime64[ns, CET]

        If the DST transition causes nonexistent times, you can shift these
        dates forward or backwards with a timedelta object or `'shift_forward'`
        or `'shift_backwards'`.

        >>> s = pd.to_datetime(pd.Series(['2015-03-29 02:30:00',
        ...                               '2015-03-29 03:30:00']))
        >>> s.dt.tz_localize('Europe/Warsaw', nonexistent='shift_forward')
        0   2015-03-29 03:00:00+02:00
        1   2015-03-29 03:30:00+02:00
        dtype: datetime64[ns, Europe/Warsaw]

        >>> s.dt.tz_localize('Europe/Warsaw', nonexistent='shift_backward')
        0   2015-03-29 01:59:59.999999999+01:00
        1   2015-03-29 03:30:00+02:00
        dtype: datetime64[ns, Europe/Warsaw]

        >>> s.dt.tz_localize('Europe/Warsaw', nonexistent=pd.Timedelta('1h'))
        0   2015-03-29 03:30:00+02:00
        1   2015-03-29 03:30:00+02:00
        dtype: datetime64[ns, Europe/Warsaw]
        """
    def to_period(self, *args, **kwargs):
        '''
        Cast to PeriodArray/PeriodIndex at a particular frequency.

        Converts DatetimeArray/Index to PeriodArray/PeriodIndex.

        Parameters
        ----------
        freq : str or Period, optional
            One of pandas\' :ref:`period aliases <timeseries.period_aliases>`
            or an Period object. Will be inferred by default.

        Returns
        -------
        PeriodArray/PeriodIndex

        Raises
        ------
        ValueError
            When converting a DatetimeArray/Index with non-regular values,
            so that a frequency cannot be inferred.

        See Also
        --------
        PeriodIndex: Immutable ndarray holding ordinal values.
        DatetimeIndex.to_pydatetime: Return DatetimeIndex as object.

        Examples
        --------
        >>> df = pd.DataFrame({"y": [1, 2, 3]},
        ...                   index=pd.to_datetime(["2000-03-31 00:00:00",
        ...                                         "2000-05-31 00:00:00",
        ...                                         "2000-08-31 00:00:00"]))
        >>> df.index.to_period("M")
        PeriodIndex([\'2000-03\', \'2000-05\', \'2000-08\'],
                    dtype=\'period[M]\')

        Infer the daily frequency

        >>> idx = pd.date_range("2017-01-01", periods=2)
        >>> idx.to_period()
        PeriodIndex([\'2017-01-01\', \'2017-01-02\'],
                    dtype=\'period[D]\')
        '''
    def to_julian_date(self) -> Index:
        """
        Convert Datetime Array to float64 ndarray of Julian Dates.
        0 Julian date is noon January 1, 4713 BC.
        https://en.wikipedia.org/wiki/Julian_day
        """
    def isocalendar(self) -> DataFrame:
        """
        Calculate year, week, and day according to the ISO 8601 standard.

        Returns
        -------
        DataFrame
            With columns year, week and day.

        See Also
        --------
        Timestamp.isocalendar : Function return a 3-tuple containing ISO year,
            week number, and weekday for the given Timestamp object.
        datetime.date.isocalendar : Return a named tuple object with
            three components: year, week and weekday.

        Examples
        --------
        >>> idx = pd.date_range(start='2019-12-29', freq='D', periods=4)
        >>> idx.isocalendar()
                    year  week  day
        2019-12-29  2019    52    7
        2019-12-30  2020     1    1
        2019-12-31  2020     1    2
        2020-01-01  2020     1    3
        >>> idx.isocalendar().week
        2019-12-29    52
        2019-12-30     1
        2019-12-31     1
        2020-01-01     1
        Freq: D, Name: week, dtype: UInt32
        """
    @classmethod
    def __init__(cls, data, freq: Frequency | lib.NoDefault = ..., tz: pandas._libs.lib._NoDefault = ..., normalize: bool | lib.NoDefault = ..., closed: pandas._libs.lib._NoDefault = ..., ambiguous: TimeAmbiguous = ..., dayfirst: bool = ..., yearfirst: bool = ..., dtype: Dtype | None, copy: bool = ..., name: Hashable | None) -> Self: ...
    def __reduce__(self): ...
    def _is_comparable_dtype(self, dtype: DtypeObj) -> bool:
        """
        Can we compare values of the given dtype to our own?
        """
    def _can_range_setop(self, other) -> bool: ...
    def _get_time_micros(self) -> npt.NDArray[np.int64]:
        """
        Return the number of microseconds since midnight.

        Returns
        -------
        ndarray[int64_t]
        """
    def snap(self, freq: Frequency = ...) -> DatetimeIndex:
        """
        Snap time stamps to nearest occurring frequency.

        Returns
        -------
        DatetimeIndex

        Examples
        --------
        >>> idx = pd.DatetimeIndex(['2023-01-01', '2023-01-02',
        ...                        '2023-02-01', '2023-02-02'])
        >>> idx
        DatetimeIndex(['2023-01-01', '2023-01-02', '2023-02-01', '2023-02-02'],
        dtype='datetime64[ns]', freq=None)
        >>> idx.snap('MS')
        DatetimeIndex(['2023-01-01', '2023-01-01', '2023-02-01', '2023-02-01'],
        dtype='datetime64[ns]', freq=None)
        """
    def _parsed_string_to_bounds(self, reso: Resolution, parsed: dt.datetime):
        """
        Calculate datetime bounds for parsed time string and its resolution.

        Parameters
        ----------
        reso : Resolution
            Resolution provided by parsed string.
        parsed : datetime
            Datetime from parsed string.

        Returns
        -------
        lower, upper: pd.Timestamp
        """
    def _parse_with_reso(self, label: str): ...
    def _disallow_mismatched_indexing(self, key) -> None:
        """
        Check for mismatched-tzawareness indexing and re-raise as KeyError.
        """
    def get_loc(self, key):
        """
        Get integer location for requested label

        Returns
        -------
        loc : int
        """
    def _maybe_cast_slice_bound(self, label, side: str):
        """
        If label is a string, cast it to scalar type according to resolution.

        Parameters
        ----------
        label : object
        side : {'left', 'right'}

        Returns
        -------
        label : object

        Notes
        -----
        Value of `side` parameter should be validated in caller.
        """
    def slice_indexer(self, start, end, step):
        """
        Return indexer for specified label slice.
        Index.slice_indexer, customized to handle time slicing.

        In addition to functionality provided by Index.slice_indexer, does the
        following:

        - if both `start` and `end` are instances of `datetime.time`, it
          invokes `indexer_between_time`
        - if `start` and `end` are both either string or None perform
          value-based selection in non-monotonic cases.

        """
    def indexer_at_time(self, time, asof: bool = ...) -> npt.NDArray[np.intp]:
        '''
        Return index locations of values at particular time of day.

        Parameters
        ----------
        time : datetime.time or str
            Time passed in either as object (datetime.time) or as string in
            appropriate format ("%H:%M", "%H%M", "%I:%M%p", "%I%M%p",
            "%H:%M:%S", "%H%M%S", "%I:%M:%S%p", "%I%M%S%p").

        Returns
        -------
        np.ndarray[np.intp]

        See Also
        --------
        indexer_between_time : Get index locations of values between particular
            times of day.
        DataFrame.at_time : Select values at particular time of day.

        Examples
        --------
        >>> idx = pd.DatetimeIndex(["1/1/2020 10:00", "2/1/2020 11:00",
        ...                         "3/1/2020 10:00"])
        >>> idx.indexer_at_time("10:00")
        array([0, 2])
        '''
    def indexer_between_time(self, start_time, end_time, include_start: bool = ..., include_end: bool = ...) -> npt.NDArray[np.intp]:
        '''
        Return index locations of values between particular times of day.

        Parameters
        ----------
        start_time, end_time : datetime.time, str
            Time passed either as object (datetime.time) or as string in
            appropriate format ("%H:%M", "%H%M", "%I:%M%p", "%I%M%p",
            "%H:%M:%S", "%H%M%S", "%I:%M:%S%p","%I%M%S%p").
        include_start : bool, default True
        include_end : bool, default True

        Returns
        -------
        np.ndarray[np.intp]

        See Also
        --------
        indexer_at_time : Get index locations of values at particular time of day.
        DataFrame.between_time : Select values between particular times of day.

        Examples
        --------
        >>> idx = pd.date_range("2023-01-01", periods=4, freq="h")
        >>> idx
        DatetimeIndex([\'2023-01-01 00:00:00\', \'2023-01-01 01:00:00\',
                           \'2023-01-01 02:00:00\', \'2023-01-01 03:00:00\'],
                          dtype=\'datetime64[ns]\', freq=\'h\')
        >>> idx.indexer_between_time("00:00", "2:00", include_end=False)
        array([0, 1])
        '''
    def to_pydatetime(self, *args, **kwargs):
        """
        Return an ndarray of ``datetime.datetime`` objects.

        Returns
        -------
        numpy.ndarray

        Examples
        --------
        >>> idx = pd.date_range('2018-02-27', periods=3)
        >>> idx.to_pydatetime()
        array([datetime.datetime(2018, 2, 27, 0, 0),
               datetime.datetime(2018, 2, 28, 0, 0),
               datetime.datetime(2018, 3, 1, 0, 0)], dtype=object)
        """
    def std(self, *args, **kwargs):
        """
        Return sample standard deviation over requested axis.

        Normalized by `N-1` by default. This can be changed using ``ddof``.

        Parameters
        ----------
        axis : int, optional
            Axis for the function to be applied on. For :class:`pandas.Series`
            this parameter is unused and defaults to ``None``.
        ddof : int, default 1
            Degrees of Freedom. The divisor used in calculations is `N - ddof`,
            where `N` represents the number of elements.
        skipna : bool, default True
            Exclude NA/null values. If an entire row/column is ``NA``, the result
            will be ``NA``.

        Returns
        -------
        Timedelta

        See Also
        --------
        numpy.ndarray.std : Returns the standard deviation of the array elements
            along given axis.
        Series.std : Return sample standard deviation over requested axis.

        Examples
        --------
        For :class:`pandas.DatetimeIndex`:

        >>> idx = pd.date_range('2001-01-01 00:00', periods=3)
        >>> idx
        DatetimeIndex(['2001-01-01', '2001-01-02', '2001-01-03'],
                      dtype='datetime64[ns]', freq='D')
        >>> idx.std()
        Timedelta('1 days 00:00:00')
        """
    def normalize(self, *args, **kwargs):
        """
        Convert times to midnight.

        The time component of the date-time is converted to midnight i.e.
        00:00:00. This is useful in cases, when the time does not matter.
        Length is unaltered. The timezones are unaffected.

        This method is available on Series with datetime values under
        the ``.dt`` accessor, and directly on Datetime Array/Index.

        Returns
        -------
        DatetimeArray, DatetimeIndex or Series
            The same type as the original data. Series will have the same
            name and index. DatetimeIndex will have the same name.

        See Also
        --------
        floor : Floor the datetimes to the specified freq.
        ceil : Ceil the datetimes to the specified freq.
        round : Round the datetimes to the specified freq.

        Examples
        --------
        >>> idx = pd.date_range(start='2014-08-01 10:00', freq='h',
        ...                     periods=3, tz='Asia/Calcutta')
        >>> idx
        DatetimeIndex(['2014-08-01 10:00:00+05:30',
                       '2014-08-01 11:00:00+05:30',
                       '2014-08-01 12:00:00+05:30'],
                        dtype='datetime64[ns, Asia/Calcutta]', freq='h')
        >>> idx.normalize()
        DatetimeIndex(['2014-08-01 00:00:00+05:30',
                       '2014-08-01 00:00:00+05:30',
                       '2014-08-01 00:00:00+05:30'],
                       dtype='datetime64[ns, Asia/Calcutta]', freq=None)
        """
    def round(self, *args, **kwargs):
        '''
        Perform round operation on the data to the specified `freq`.

        Parameters
        ----------
        freq : str or Offset
            The frequency level to round the index to. Must be a fixed
            frequency like \'S\' (second) not \'ME\' (month end). See
            :ref:`frequency aliases <timeseries.offset_aliases>` for
            a list of possible `freq` values.
        ambiguous : \'infer\', bool-ndarray, \'NaT\', default \'raise\'
            Only relevant for DatetimeIndex:

            - \'infer\' will attempt to infer fall dst-transition hours based on
              order
            - bool-ndarray where True signifies a DST time, False designates
              a non-DST time (note that this flag is only applicable for
              ambiguous times)
            - \'NaT\' will return NaT where there are ambiguous times
            - \'raise\' will raise an AmbiguousTimeError if there are ambiguous
              times.

        nonexistent : \'shift_forward\', \'shift_backward\', \'NaT\', timedelta, default \'raise\'
            A nonexistent time does not exist in a particular timezone
            where clocks moved forward due to DST.

            - \'shift_forward\' will shift the nonexistent time forward to the
              closest existing time
            - \'shift_backward\' will shift the nonexistent time backward to the
              closest existing time
            - \'NaT\' will return NaT where there are nonexistent times
            - timedelta objects will shift nonexistent times by the timedelta
            - \'raise\' will raise an NonExistentTimeError if there are
              nonexistent times.

        Returns
        -------
        DatetimeIndex, TimedeltaIndex, or Series
            Index of the same type for a DatetimeIndex or TimedeltaIndex,
            or a Series with the same index for a Series.

        Raises
        ------
        ValueError if the `freq` cannot be converted.

        Notes
        -----
        If the timestamps have a timezone, rounding will take place relative to the
        local ("wall") time and re-localized to the same timezone. When rounding
        near daylight savings time, use ``nonexistent`` and ``ambiguous`` to
        control the re-localization behavior.

        Examples
        --------
        **DatetimeIndex**

        >>> rng = pd.date_range(\'1/1/2018 11:59:00\', periods=3, freq=\'min\')
        >>> rng
        DatetimeIndex([\'2018-01-01 11:59:00\', \'2018-01-01 12:00:00\',
                       \'2018-01-01 12:01:00\'],
                      dtype=\'datetime64[ns]\', freq=\'min\')
        >>> rng.round(\'h\')
        DatetimeIndex([\'2018-01-01 12:00:00\', \'2018-01-01 12:00:00\',
                       \'2018-01-01 12:00:00\'],
                      dtype=\'datetime64[ns]\', freq=None)

        **Series**

        >>> pd.Series(rng).dt.round("h")
        0   2018-01-01 12:00:00
        1   2018-01-01 12:00:00
        2   2018-01-01 12:00:00
        dtype: datetime64[ns]

        When rounding near a daylight savings time transition, use ``ambiguous`` or
        ``nonexistent`` to control how the timestamp should be re-localized.

        >>> rng_tz = pd.DatetimeIndex(["2021-10-31 03:30:00"], tz="Europe/Amsterdam")

        >>> rng_tz.floor("2h", ambiguous=False)
        DatetimeIndex([\'2021-10-31 02:00:00+01:00\'],
                      dtype=\'datetime64[ns, Europe/Amsterdam]\', freq=None)

        >>> rng_tz.floor("2h", ambiguous=True)
        DatetimeIndex([\'2021-10-31 02:00:00+02:00\'],
                      dtype=\'datetime64[ns, Europe/Amsterdam]\', freq=None)
        '''
    def floor(self, *args, **kwargs):
        '''
        Perform floor operation on the data to the specified `freq`.

        Parameters
        ----------
        freq : str or Offset
            The frequency level to floor the index to. Must be a fixed
            frequency like \'S\' (second) not \'ME\' (month end). See
            :ref:`frequency aliases <timeseries.offset_aliases>` for
            a list of possible `freq` values.
        ambiguous : \'infer\', bool-ndarray, \'NaT\', default \'raise\'
            Only relevant for DatetimeIndex:

            - \'infer\' will attempt to infer fall dst-transition hours based on
              order
            - bool-ndarray where True signifies a DST time, False designates
              a non-DST time (note that this flag is only applicable for
              ambiguous times)
            - \'NaT\' will return NaT where there are ambiguous times
            - \'raise\' will raise an AmbiguousTimeError if there are ambiguous
              times.

        nonexistent : \'shift_forward\', \'shift_backward\', \'NaT\', timedelta, default \'raise\'
            A nonexistent time does not exist in a particular timezone
            where clocks moved forward due to DST.

            - \'shift_forward\' will shift the nonexistent time forward to the
              closest existing time
            - \'shift_backward\' will shift the nonexistent time backward to the
              closest existing time
            - \'NaT\' will return NaT where there are nonexistent times
            - timedelta objects will shift nonexistent times by the timedelta
            - \'raise\' will raise an NonExistentTimeError if there are
              nonexistent times.

        Returns
        -------
        DatetimeIndex, TimedeltaIndex, or Series
            Index of the same type for a DatetimeIndex or TimedeltaIndex,
            or a Series with the same index for a Series.

        Raises
        ------
        ValueError if the `freq` cannot be converted.

        Notes
        -----
        If the timestamps have a timezone, flooring will take place relative to the
        local ("wall") time and re-localized to the same timezone. When flooring
        near daylight savings time, use ``nonexistent`` and ``ambiguous`` to
        control the re-localization behavior.

        Examples
        --------
        **DatetimeIndex**

        >>> rng = pd.date_range(\'1/1/2018 11:59:00\', periods=3, freq=\'min\')
        >>> rng
        DatetimeIndex([\'2018-01-01 11:59:00\', \'2018-01-01 12:00:00\',
                       \'2018-01-01 12:01:00\'],
                      dtype=\'datetime64[ns]\', freq=\'min\')
        >>> rng.floor(\'h\')
        DatetimeIndex([\'2018-01-01 11:00:00\', \'2018-01-01 12:00:00\',
                       \'2018-01-01 12:00:00\'],
                      dtype=\'datetime64[ns]\', freq=None)

        **Series**

        >>> pd.Series(rng).dt.floor("h")
        0   2018-01-01 11:00:00
        1   2018-01-01 12:00:00
        2   2018-01-01 12:00:00
        dtype: datetime64[ns]

        When rounding near a daylight savings time transition, use ``ambiguous`` or
        ``nonexistent`` to control how the timestamp should be re-localized.

        >>> rng_tz = pd.DatetimeIndex(["2021-10-31 03:30:00"], tz="Europe/Amsterdam")

        >>> rng_tz.floor("2h", ambiguous=False)
        DatetimeIndex([\'2021-10-31 02:00:00+01:00\'],
                     dtype=\'datetime64[ns, Europe/Amsterdam]\', freq=None)

        >>> rng_tz.floor("2h", ambiguous=True)
        DatetimeIndex([\'2021-10-31 02:00:00+02:00\'],
                      dtype=\'datetime64[ns, Europe/Amsterdam]\', freq=None)
        '''
    def ceil(self, *args, **kwargs):
        '''
        Perform ceil operation on the data to the specified `freq`.

        Parameters
        ----------
        freq : str or Offset
            The frequency level to ceil the index to. Must be a fixed
            frequency like \'S\' (second) not \'ME\' (month end). See
            :ref:`frequency aliases <timeseries.offset_aliases>` for
            a list of possible `freq` values.
        ambiguous : \'infer\', bool-ndarray, \'NaT\', default \'raise\'
            Only relevant for DatetimeIndex:

            - \'infer\' will attempt to infer fall dst-transition hours based on
              order
            - bool-ndarray where True signifies a DST time, False designates
              a non-DST time (note that this flag is only applicable for
              ambiguous times)
            - \'NaT\' will return NaT where there are ambiguous times
            - \'raise\' will raise an AmbiguousTimeError if there are ambiguous
              times.

        nonexistent : \'shift_forward\', \'shift_backward\', \'NaT\', timedelta, default \'raise\'
            A nonexistent time does not exist in a particular timezone
            where clocks moved forward due to DST.

            - \'shift_forward\' will shift the nonexistent time forward to the
              closest existing time
            - \'shift_backward\' will shift the nonexistent time backward to the
              closest existing time
            - \'NaT\' will return NaT where there are nonexistent times
            - timedelta objects will shift nonexistent times by the timedelta
            - \'raise\' will raise an NonExistentTimeError if there are
              nonexistent times.

        Returns
        -------
        DatetimeIndex, TimedeltaIndex, or Series
            Index of the same type for a DatetimeIndex or TimedeltaIndex,
            or a Series with the same index for a Series.

        Raises
        ------
        ValueError if the `freq` cannot be converted.

        Notes
        -----
        If the timestamps have a timezone, ceiling will take place relative to the
        local ("wall") time and re-localized to the same timezone. When ceiling
        near daylight savings time, use ``nonexistent`` and ``ambiguous`` to
        control the re-localization behavior.

        Examples
        --------
        **DatetimeIndex**

        >>> rng = pd.date_range(\'1/1/2018 11:59:00\', periods=3, freq=\'min\')
        >>> rng
        DatetimeIndex([\'2018-01-01 11:59:00\', \'2018-01-01 12:00:00\',
                       \'2018-01-01 12:01:00\'],
                      dtype=\'datetime64[ns]\', freq=\'min\')
        >>> rng.ceil(\'h\')
        DatetimeIndex([\'2018-01-01 12:00:00\', \'2018-01-01 12:00:00\',
                       \'2018-01-01 13:00:00\'],
                      dtype=\'datetime64[ns]\', freq=None)

        **Series**

        >>> pd.Series(rng).dt.ceil("h")
        0   2018-01-01 12:00:00
        1   2018-01-01 12:00:00
        2   2018-01-01 13:00:00
        dtype: datetime64[ns]

        When rounding near a daylight savings time transition, use ``ambiguous`` or
        ``nonexistent`` to control how the timestamp should be re-localized.

        >>> rng_tz = pd.DatetimeIndex(["2021-10-31 01:30:00"], tz="Europe/Amsterdam")

        >>> rng_tz.ceil("h", ambiguous=False)
        DatetimeIndex([\'2021-10-31 02:00:00+01:00\'],
                      dtype=\'datetime64[ns, Europe/Amsterdam]\', freq=None)

        >>> rng_tz.ceil("h", ambiguous=True)
        DatetimeIndex([\'2021-10-31 02:00:00+02:00\'],
                      dtype=\'datetime64[ns, Europe/Amsterdam]\', freq=None)
        '''
    def month_name(self, *args, **kwargs):
        """
        Return the month names with specified locale.

        Parameters
        ----------
        locale : str, optional
            Locale determining the language in which to return the month name.
            Default is English locale (``'en_US.utf8'``). Use the command
            ``locale -a`` on your terminal on Unix systems to find your locale
            language code.

        Returns
        -------
        Series or Index
            Series or Index of month names.

        Examples
        --------
        >>> s = pd.Series(pd.date_range(start='2018-01', freq='ME', periods=3))
        >>> s
        0   2018-01-31
        1   2018-02-28
        2   2018-03-31
        dtype: datetime64[ns]
        >>> s.dt.month_name()
        0     January
        1    February
        2       March
        dtype: object

        >>> idx = pd.date_range(start='2018-01', freq='ME', periods=3)
        >>> idx
        DatetimeIndex(['2018-01-31', '2018-02-28', '2018-03-31'],
                      dtype='datetime64[ns]', freq='ME')
        >>> idx.month_name()
        Index(['January', 'February', 'March'], dtype='object')

        Using the ``locale`` parameter you can set a different locale language,
        for example: ``idx.month_name(locale='pt_BR.utf8')`` will return month
        names in Brazilian Portuguese language.

        >>> idx = pd.date_range(start='2018-01', freq='ME', periods=3)
        >>> idx
        DatetimeIndex(['2018-01-31', '2018-02-28', '2018-03-31'],
                      dtype='datetime64[ns]', freq='ME')
        >>> idx.month_name(locale='pt_BR.utf8')  # doctest: +SKIP
        Index(['Janeiro', 'Fevereiro', 'Março'], dtype='object')
        """
    def day_name(self, *args, **kwargs):
        """
        Return the day names with specified locale.

        Parameters
        ----------
        locale : str, optional
            Locale determining the language in which to return the day name.
            Default is English locale (``'en_US.utf8'``). Use the command
            ``locale -a`` on your terminal on Unix systems to find your locale
            language code.

        Returns
        -------
        Series or Index
            Series or Index of day names.

        Examples
        --------
        >>> s = pd.Series(pd.date_range(start='2018-01-01', freq='D', periods=3))
        >>> s
        0   2018-01-01
        1   2018-01-02
        2   2018-01-03
        dtype: datetime64[ns]
        >>> s.dt.day_name()
        0       Monday
        1      Tuesday
        2    Wednesday
        dtype: object

        >>> idx = pd.date_range(start='2018-01-01', freq='D', periods=3)
        >>> idx
        DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03'],
                      dtype='datetime64[ns]', freq='D')
        >>> idx.day_name()
        Index(['Monday', 'Tuesday', 'Wednesday'], dtype='object')

        Using the ``locale`` parameter you can set a different locale language,
        for example: ``idx.day_name(locale='pt_BR.utf8')`` will return day
        names in Brazilian Portuguese language.

        >>> idx = pd.date_range(start='2018-01-01', freq='D', periods=3)
        >>> idx
        DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03'],
                      dtype='datetime64[ns]', freq='D')
        >>> idx.day_name(locale='pt_BR.utf8') # doctest: +SKIP
        Index(['Segunda', 'Terça', 'Quarta'], dtype='object')
        """
    def as_unit(self, *args, **kwargs): ...
    @property
    def _engine_type(self): ...
    @property
    def inferred_type(self): ...
def date_range(start, end, periods, freq, tz, normalize: bool = ..., name: Hashable | None, inclusive: IntervalClosedType = ..., *, unit: str | None, **kwargs) -> DatetimeIndex:
    '''
    Return a fixed frequency DatetimeIndex.

    Returns the range of equally spaced time points (where the difference between any
    two adjacent points is specified by the given frequency) such that they all
    satisfy `start <[=] x <[=] end`, where the first one and the last one are, resp.,
    the first and last time points in that range that fall on the boundary of ``freq``
    (if given as a frequency string) or that are valid for ``freq`` (if given as a
    :class:`pandas.tseries.offsets.DateOffset`). (If exactly one of ``start``,
    ``end``, or ``freq`` is *not* specified, this missing parameter can be computed
    given ``periods``, the number of timesteps in the range. See the note below.)

    Parameters
    ----------
    start : str or datetime-like, optional
        Left bound for generating dates.
    end : str or datetime-like, optional
        Right bound for generating dates.
    periods : int, optional
        Number of periods to generate.
    freq : str, Timedelta, datetime.timedelta, or DateOffset, default \'D\'
        Frequency strings can have multiples, e.g. \'5h\'. See
        :ref:`here <timeseries.offset_aliases>` for a list of
        frequency aliases.
    tz : str or tzinfo, optional
        Time zone name for returning localized DatetimeIndex, for example
        \'Asia/Hong_Kong\'. By default, the resulting DatetimeIndex is
        timezone-naive unless timezone-aware datetime-likes are passed.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    name : str, default None
        Name of the resulting DatetimeIndex.
    inclusive : {"both", "neither", "left", "right"}, default "both"
        Include boundaries; Whether to set each bound as closed or open.

        .. versionadded:: 1.4.0
    unit : str, default None
        Specify the desired resolution of the result.

        .. versionadded:: 2.0.0
    **kwargs
        For compatibility. Has no effect on the result.

    Returns
    -------
    DatetimeIndex

    See Also
    --------
    DatetimeIndex : An immutable container for datetimes.
    timedelta_range : Return a fixed frequency TimedeltaIndex.
    period_range : Return a fixed frequency PeriodIndex.
    interval_range : Return a fixed frequency IntervalIndex.

    Notes
    -----
    Of the four parameters ``start``, ``end``, ``periods``, and ``freq``,
    exactly three must be specified. If ``freq`` is omitted, the resulting
    ``DatetimeIndex`` will have ``periods`` linearly spaced elements between
    ``start`` and ``end`` (closed on both sides).

    To learn more about the frequency strings, please see `this link
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__.

    Examples
    --------
    **Specifying the values**

    The next four examples generate the same `DatetimeIndex`, but vary
    the combination of `start`, `end` and `periods`.

    Specify `start` and `end`, with the default daily frequency.

    >>> pd.date_range(start=\'1/1/2018\', end=\'1/08/2018\')
    DatetimeIndex([\'2018-01-01\', \'2018-01-02\', \'2018-01-03\', \'2018-01-04\',
                   \'2018-01-05\', \'2018-01-06\', \'2018-01-07\', \'2018-01-08\'],
                  dtype=\'datetime64[ns]\', freq=\'D\')

    Specify timezone-aware `start` and `end`, with the default daily frequency.

    >>> pd.date_range(
    ...     start=pd.to_datetime("1/1/2018").tz_localize("Europe/Berlin"),
    ...     end=pd.to_datetime("1/08/2018").tz_localize("Europe/Berlin"),
    ... )
    DatetimeIndex([\'2018-01-01 00:00:00+01:00\', \'2018-01-02 00:00:00+01:00\',
                   \'2018-01-03 00:00:00+01:00\', \'2018-01-04 00:00:00+01:00\',
                   \'2018-01-05 00:00:00+01:00\', \'2018-01-06 00:00:00+01:00\',
                   \'2018-01-07 00:00:00+01:00\', \'2018-01-08 00:00:00+01:00\'],
                  dtype=\'datetime64[ns, Europe/Berlin]\', freq=\'D\')

    Specify `start` and `periods`, the number of periods (days).

    >>> pd.date_range(start=\'1/1/2018\', periods=8)
    DatetimeIndex([\'2018-01-01\', \'2018-01-02\', \'2018-01-03\', \'2018-01-04\',
                   \'2018-01-05\', \'2018-01-06\', \'2018-01-07\', \'2018-01-08\'],
                  dtype=\'datetime64[ns]\', freq=\'D\')

    Specify `end` and `periods`, the number of periods (days).

    >>> pd.date_range(end=\'1/1/2018\', periods=8)
    DatetimeIndex([\'2017-12-25\', \'2017-12-26\', \'2017-12-27\', \'2017-12-28\',
                   \'2017-12-29\', \'2017-12-30\', \'2017-12-31\', \'2018-01-01\'],
                  dtype=\'datetime64[ns]\', freq=\'D\')

    Specify `start`, `end`, and `periods`; the frequency is generated
    automatically (linearly spaced).

    >>> pd.date_range(start=\'2018-04-24\', end=\'2018-04-27\', periods=3)
    DatetimeIndex([\'2018-04-24 00:00:00\', \'2018-04-25 12:00:00\',
                   \'2018-04-27 00:00:00\'],
                  dtype=\'datetime64[ns]\', freq=None)

    **Other Parameters**

    Changed the `freq` (frequency) to ``\'ME\'`` (month end frequency).

    >>> pd.date_range(start=\'1/1/2018\', periods=5, freq=\'ME\')
    DatetimeIndex([\'2018-01-31\', \'2018-02-28\', \'2018-03-31\', \'2018-04-30\',
                   \'2018-05-31\'],
                  dtype=\'datetime64[ns]\', freq=\'ME\')

    Multiples are allowed

    >>> pd.date_range(start=\'1/1/2018\', periods=5, freq=\'3ME\')
    DatetimeIndex([\'2018-01-31\', \'2018-04-30\', \'2018-07-31\', \'2018-10-31\',
                   \'2019-01-31\'],
                  dtype=\'datetime64[ns]\', freq=\'3ME\')

    `freq` can also be specified as an Offset object.

    >>> pd.date_range(start=\'1/1/2018\', periods=5, freq=pd.offsets.MonthEnd(3))
    DatetimeIndex([\'2018-01-31\', \'2018-04-30\', \'2018-07-31\', \'2018-10-31\',
                   \'2019-01-31\'],
                  dtype=\'datetime64[ns]\', freq=\'3ME\')

    Specify `tz` to set the timezone.

    >>> pd.date_range(start=\'1/1/2018\', periods=5, tz=\'Asia/Tokyo\')
    DatetimeIndex([\'2018-01-01 00:00:00+09:00\', \'2018-01-02 00:00:00+09:00\',
                   \'2018-01-03 00:00:00+09:00\', \'2018-01-04 00:00:00+09:00\',
                   \'2018-01-05 00:00:00+09:00\'],
                  dtype=\'datetime64[ns, Asia/Tokyo]\', freq=\'D\')

    `inclusive` controls whether to include `start` and `end` that are on the
    boundary. The default, "both", includes boundary points on either end.

    >>> pd.date_range(start=\'2017-01-01\', end=\'2017-01-04\', inclusive="both")
    DatetimeIndex([\'2017-01-01\', \'2017-01-02\', \'2017-01-03\', \'2017-01-04\'],
                  dtype=\'datetime64[ns]\', freq=\'D\')

    Use ``inclusive=\'left\'`` to exclude `end` if it falls on the boundary.

    >>> pd.date_range(start=\'2017-01-01\', end=\'2017-01-04\', inclusive=\'left\')
    DatetimeIndex([\'2017-01-01\', \'2017-01-02\', \'2017-01-03\'],
                  dtype=\'datetime64[ns]\', freq=\'D\')

    Use ``inclusive=\'right\'`` to exclude `start` if it falls on the boundary, and
    similarly ``inclusive=\'neither\'`` will exclude both `start` and `end`.

    >>> pd.date_range(start=\'2017-01-01\', end=\'2017-01-04\', inclusive=\'right\')
    DatetimeIndex([\'2017-01-02\', \'2017-01-03\', \'2017-01-04\'],
                  dtype=\'datetime64[ns]\', freq=\'D\')

    **Specify a unit**

    >>> pd.date_range(start="2017-01-01", periods=10, freq="100YS", unit="s")
    DatetimeIndex([\'2017-01-01\', \'2117-01-01\', \'2217-01-01\', \'2317-01-01\',
                   \'2417-01-01\', \'2517-01-01\', \'2617-01-01\', \'2717-01-01\',
                   \'2817-01-01\', \'2917-01-01\'],
                  dtype=\'datetime64[s]\', freq=\'100YS-JAN\')
    '''
def bdate_range(start, end, periods: int | None, freq: Frequency | dt.timedelta = ..., tz, normalize: bool = ..., name: Hashable | None, weekmask, holidays, inclusive: IntervalClosedType = ..., **kwargs) -> DatetimeIndex:
    '''
    Return a fixed frequency DatetimeIndex with business day as the default.

    Parameters
    ----------
    start : str or datetime-like, default None
        Left bound for generating dates.
    end : str or datetime-like, default None
        Right bound for generating dates.
    periods : int, default None
        Number of periods to generate.
    freq : str, Timedelta, datetime.timedelta, or DateOffset, default \'B\'
        Frequency strings can have multiples, e.g. \'5h\'. The default is
        business daily (\'B\').
    tz : str or None
        Time zone name for returning localized DatetimeIndex, for example
        Asia/Beijing.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    name : str, default None
        Name of the resulting DatetimeIndex.
    weekmask : str or None, default None
        Weekmask of valid business days, passed to ``numpy.busdaycalendar``,
        only used when custom frequency strings are passed.  The default
        value None is equivalent to \'Mon Tue Wed Thu Fri\'.
    holidays : list-like or None, default None
        Dates to exclude from the set of valid business days, passed to
        ``numpy.busdaycalendar``, only used when custom frequency strings
        are passed.
    inclusive : {"both", "neither", "left", "right"}, default "both"
        Include boundaries; Whether to set each bound as closed or open.

        .. versionadded:: 1.4.0
    **kwargs
        For compatibility. Has no effect on the result.

    Returns
    -------
    DatetimeIndex

    Notes
    -----
    Of the four parameters: ``start``, ``end``, ``periods``, and ``freq``,
    exactly three must be specified.  Specifying ``freq`` is a requirement
    for ``bdate_range``.  Use ``date_range`` if specifying ``freq`` is not
    desired.

    To learn more about the frequency strings, please see `this link
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__.

    Examples
    --------
    Note how the two weekend days are skipped in the result.

    >>> pd.bdate_range(start=\'1/1/2018\', end=\'1/08/2018\')
    DatetimeIndex([\'2018-01-01\', \'2018-01-02\', \'2018-01-03\', \'2018-01-04\',
               \'2018-01-05\', \'2018-01-08\'],
              dtype=\'datetime64[ns]\', freq=\'B\')
    '''
def _time_to_micros(time_obj: dt.time) -> int: ...
