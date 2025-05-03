import np
import pandas._libs.lib as lib
import pandas.core.accessor
import pandas.core.base
from _typeshed import Incomplete
from pandas._libs.lib import is_list_like as is_list_like
from pandas.core.accessor import PandasDelegate as PandasDelegate, delegate_names as delegate_names
from pandas.core.arrays.arrow.array import ArrowExtensionArray as ArrowExtensionArray
from pandas.core.arrays.datetimes import DatetimeArray as DatetimeArray
from pandas.core.arrays.period import PeriodArray as PeriodArray
from pandas.core.arrays.timedeltas import TimedeltaArray as TimedeltaArray
from pandas.core.base import NoNewAttributesMixin as NoNewAttributesMixin, PandasObject as PandasObject
from pandas.core.dtypes.common import is_integer_dtype as is_integer_dtype
from pandas.core.dtypes.dtypes import ArrowDtype as ArrowDtype, CategoricalDtype as CategoricalDtype, DatetimeTZDtype as DatetimeTZDtype, PeriodDtype as PeriodDtype
from pandas.core.dtypes.generic import ABCSeries as ABCSeries
from pandas.core.indexes.datetimes import DatetimeIndex as DatetimeIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex as TimedeltaIndex
from pandas.util._exceptions import find_stack_level as find_stack_level
from typing import ClassVar

TYPE_CHECKING: bool

class Properties(pandas.core.accessor.PandasDelegate, pandas.core.base.PandasObject, pandas.core.base.NoNewAttributesMixin):
    _hidden_attrs: ClassVar[frozenset] = ...
    def __init__(self, data: Series, orig) -> None: ...
    def _get_values(self): ...
    def _delegate_property_get(self, name: str): ...
    def _delegate_property_set(self, name: str, value, *args, **kwargs): ...
    def _delegate_method(self, name: str, *args, **kwargs): ...

class ArrowTemporalProperties(pandas.core.accessor.PandasDelegate, pandas.core.base.PandasObject, pandas.core.base.NoNewAttributesMixin):
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
    tz: Incomplete
    is_month_start: Incomplete
    is_month_end: Incomplete
    is_quarter_start: Incomplete
    is_quarter_end: Incomplete
    is_year_start: Incomplete
    is_year_end: Incomplete
    is_leap_year: Incomplete
    date: Incomplete
    time: Incomplete
    unit: Incomplete
    days: Incomplete
    seconds: Incomplete
    microseconds: Incomplete
    nanoseconds: Incomplete
    def __init__(self, data: Series, orig) -> None: ...
    def _delegate_property_get(self, name: str): ...
    def _delegate_method(self, name: str, *args, **kwargs): ...
    def to_pytimedelta(self): ...
    def to_pydatetime(self): ...
    def isocalendar(self) -> DataFrame: ...
    def tz_localize(self, *args, **kwargs): ...
    def tz_convert(self, *args, **kwargs): ...
    def normalize(self, *args, **kwargs): ...
    def strftime(self, *args, **kwargs): ...
    def round(self, *args, **kwargs): ...
    def floor(self, *args, **kwargs): ...
    def ceil(self, *args, **kwargs): ...
    def month_name(self, *args, **kwargs): ...
    def day_name(self, *args, **kwargs): ...
    def as_unit(self, *args, **kwargs): ...
    def total_seconds(self, *args, **kwargs): ...
    @property
    def components(self): ...

class DatetimeProperties(Properties):
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
    tz: Incomplete
    is_month_start: Incomplete
    is_month_end: Incomplete
    is_quarter_start: Incomplete
    is_quarter_end: Incomplete
    is_year_start: Incomplete
    is_year_end: Incomplete
    is_leap_year: Incomplete
    date: Incomplete
    time: Incomplete
    timetz: Incomplete
    unit: Incomplete
    def to_pydatetime(self) -> np.ndarray:
        """
        Return the data as an array of :class:`datetime.datetime` objects.

        .. deprecated:: 2.1.0

            The current behavior of dt.to_pydatetime is deprecated.
            In a future version this will return a Series containing python
            datetime objects instead of a ndarray.

        Timezone information is retained if present.

        .. warning::

           Python's datetime uses microsecond resolution, which is lower than
           pandas (nanosecond). The values are truncated.

        Returns
        -------
        numpy.ndarray
            Object dtype array containing native Python datetime objects.

        See Also
        --------
        datetime.datetime : Standard library value for a datetime.

        Examples
        --------
        >>> s = pd.Series(pd.date_range('20180310', periods=2))
        >>> s
        0   2018-03-10
        1   2018-03-11
        dtype: datetime64[ns]

        >>> s.dt.to_pydatetime()
        array([datetime.datetime(2018, 3, 10, 0, 0),
               datetime.datetime(2018, 3, 11, 0, 0)], dtype=object)

        pandas' nanosecond precision is truncated to microseconds.

        >>> s = pd.Series(pd.date_range('20180310', periods=2, freq='ns'))
        >>> s
        0   2018-03-10 00:00:00.000000000
        1   2018-03-10 00:00:00.000000001
        dtype: datetime64[ns]

        >>> s.dt.to_pydatetime()
        array([datetime.datetime(2018, 3, 10, 0, 0),
               datetime.datetime(2018, 3, 10, 0, 0)], dtype=object)
        """
    def isocalendar(self) -> DataFrame:
        '''
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
        >>> ser = pd.to_datetime(pd.Series(["2010-01-01", pd.NaT]))
        >>> ser.dt.isocalendar()
           year  week  day
        0  2009    53     5
        1  <NA>  <NA>  <NA>
        >>> ser.dt.isocalendar().week
        0      53
        1    <NA>
        Name: week, dtype: UInt32
        '''
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
    def tz_localize(self, *args, **kwargs):
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
    def tz_convert(self, *args, **kwargs):
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
    def strftime(self, *args, **kwargs):
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
    def freq(self): ...

class TimedeltaProperties(Properties):
    days: Incomplete
    seconds: Incomplete
    microseconds: Incomplete
    nanoseconds: Incomplete
    unit: Incomplete
    def to_pytimedelta(self) -> np.ndarray:
        '''
        Return an array of native :class:`datetime.timedelta` objects.

        Python\'s standard `datetime` library uses a different representation
        timedelta\'s. This method converts a Series of pandas Timedeltas
        to `datetime.timedelta` format with the same length as the original
        Series.

        Returns
        -------
        numpy.ndarray
            Array of 1D containing data with `datetime.timedelta` type.

        See Also
        --------
        datetime.timedelta : A duration expressing the difference
            between two date, time, or datetime.

        Examples
        --------
        >>> s = pd.Series(pd.to_timedelta(np.arange(5), unit="d"))
        >>> s
        0   0 days
        1   1 days
        2   2 days
        3   3 days
        4   4 days
        dtype: timedelta64[ns]

        >>> s.dt.to_pytimedelta()
        array([datetime.timedelta(0), datetime.timedelta(days=1),
        datetime.timedelta(days=2), datetime.timedelta(days=3),
        datetime.timedelta(days=4)], dtype=object)
        '''
    def total_seconds(self, *args, **kwargs):
        """
        Return total duration of each element expressed in seconds.

        This method is available directly on TimedeltaArray, TimedeltaIndex
        and on Series containing timedelta values under the ``.dt`` namespace.

        Returns
        -------
        ndarray, Index or Series
            When the calling object is a TimedeltaArray, the return type
            is ndarray.  When the calling object is a TimedeltaIndex,
            the return type is an Index with a float64 dtype. When the calling object
            is a Series, the return type is Series of type `float64` whose
            index is the same as the original.

        See Also
        --------
        datetime.timedelta.total_seconds : Standard library version
            of this method.
        TimedeltaIndex.components : Return a DataFrame with components of
            each Timedelta.

        Examples
        --------
        **Series**

        >>> s = pd.Series(pd.to_timedelta(np.arange(5), unit='d'))
        >>> s
        0   0 days
        1   1 days
        2   2 days
        3   3 days
        4   4 days
        dtype: timedelta64[ns]

        >>> s.dt.total_seconds()
        0         0.0
        1     86400.0
        2    172800.0
        3    259200.0
        4    345600.0
        dtype: float64

        **TimedeltaIndex**

        >>> idx = pd.to_timedelta(np.arange(5), unit='d')
        >>> idx
        TimedeltaIndex(['0 days', '1 days', '2 days', '3 days', '4 days'],
                       dtype='timedelta64[ns]', freq=None)

        >>> idx.total_seconds()
        Index([0.0, 86400.0, 172800.0, 259200.0, 345600.0], dtype='float64')
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
    def as_unit(self, *args, **kwargs): ...
    @property
    def components(self): ...
    @property
    def freq(self): ...

class PeriodProperties(Properties):
    year: Incomplete
    month: Incomplete
    day: Incomplete
    hour: Incomplete
    minute: Incomplete
    second: Incomplete
    weekofyear: Incomplete
    weekday: Incomplete
    week: Incomplete
    dayofweek: Incomplete
    day_of_week: Incomplete
    dayofyear: Incomplete
    day_of_year: Incomplete
    quarter: Incomplete
    qyear: Incomplete
    days_in_month: Incomplete
    daysinmonth: Incomplete
    start_time: Incomplete
    end_time: Incomplete
    freq: Incomplete
    is_leap_year: Incomplete
    def strftime(self, *args, **kwargs):
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
    def to_timestamp(self, *args, **kwargs):
        '''
        Cast to DatetimeArray/Index.

        Parameters
        ----------
        freq : str or DateOffset, optional
            Target frequency. The default is \'D\' for week or longer,
            \'s\' otherwise.
        how : {\'s\', \'e\', \'start\', \'end\'}
            Whether to use the start or end of the time period being converted.

        Returns
        -------
        DatetimeArray/Index

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023-01", "2023-02", "2023-03"], freq="M")
        >>> idx.to_timestamp()
        DatetimeIndex([\'2023-01-01\', \'2023-02-01\', \'2023-03-01\'],
        dtype=\'datetime64[ns]\', freq=\'MS\')
        '''
    def asfreq(self, *args, **kwargs):
        """
        Convert the PeriodArray to the specified frequency `freq`.

        Equivalent to applying :meth:`pandas.Period.asfreq` with the given arguments
        to each :class:`~pandas.Period` in this PeriodArray.

        Parameters
        ----------
        freq : str
            A frequency.
        how : str {'E', 'S'}, default 'E'
            Whether the elements should be aligned to the end
            or start within pa period.

            * 'E', 'END', or 'FINISH' for end,
            * 'S', 'START', or 'BEGIN' for start.

            January 31st ('END') vs. January 1st ('START') for example.

        Returns
        -------
        PeriodArray
            The transformed PeriodArray with the new frequency.

        See Also
        --------
        PeriodIndex.asfreq: Convert each Period in a PeriodIndex to the given frequency.
        Period.asfreq : Convert a :class:`~pandas.Period` object to the given frequency.

        Examples
        --------
        >>> pidx = pd.period_range('2010-01-01', '2015-01-01', freq='Y')
        >>> pidx
        PeriodIndex(['2010', '2011', '2012', '2013', '2014', '2015'],
        dtype='period[Y-DEC]')

        >>> pidx.asfreq('M')
        PeriodIndex(['2010-12', '2011-12', '2012-12', '2013-12', '2014-12',
        '2015-12'], dtype='period[M]')

        >>> pidx.asfreq('M', how='S')
        PeriodIndex(['2010-01', '2011-01', '2012-01', '2013-01', '2014-01',
        '2015-01'], dtype='period[M]')
        """

class CombinedDatetimelikeProperties(DatetimeProperties, TimedeltaProperties, PeriodProperties):
    @classmethod
    def __init__(cls, data: Series) -> None: ...
