import _cython_3_0_11
import datetime
from _typeshed import Incomplete
from typing import Any, ClassVar, overload

NaT: NaTType
__nat_unpickle: _cython_3_0_11.cython_function_or_method
__pyx_capi__: dict
__pyx_unpickle__NaT: _cython_3_0_11.cython_function_or_method
__test__: dict
_make_error_func: _cython_3_0_11.cython_function_or_method
_make_nan_func: _cython_3_0_11.cython_function_or_method
_make_nat_func: _cython_3_0_11.cython_function_or_method
iNaT: int
nat_strings: set

class NaTType(_NaT):
    def __init__(self, *args, **kwargs) -> None: ...
    def as_unit(self, *args, **kwargs):
        '''
        Convert the underlying int64 representaton to the given unit.

        Parameters
        ----------
        unit : {"ns", "us", "ms", "s"}
        round_ok : bool, default True
            If False and the conversion requires rounding, raise.

        Returns
        -------
        Timestamp

        Examples
        --------
        >>> ts = pd.Timestamp(\'2023-01-01 00:00:00.01\')
        >>> ts
        Timestamp(\'2023-01-01 00:00:00.010000\')
        >>> ts.unit
        \'ms\'
        >>> ts = ts.as_unit(\'s\')
        >>> ts
        Timestamp(\'2023-01-01 00:00:00\')
        >>> ts.unit
        \'s\'
        '''
    def astimezone(self, tz=...) -> Any:
        """
        Convert timezone-aware Timestamp to another time zone.

        Parameters
        ----------
        tz : str, pytz.timezone, dateutil.tz.tzfile or None
            Time zone for time which Timestamp will be converted to.
            None will remove timezone holding UTC time.

        Returns
        -------
        converted : Timestamp

        Raises
        ------
        TypeError
            If Timestamp is tz-naive.

        Examples
        --------
        Create a timestamp object with UTC timezone:

        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651', tz='UTC')
        >>> ts
        Timestamp('2020-03-14 15:32:52.192548651+0000', tz='UTC')

        Change to Tokyo timezone:

        >>> ts.tz_convert(tz='Asia/Tokyo')
        Timestamp('2020-03-15 00:32:52.192548651+0900', tz='Asia/Tokyo')

        Can also use ``astimezone``:

        >>> ts.astimezone(tz='Asia/Tokyo')
        Timestamp('2020-03-15 00:32:52.192548651+0900', tz='Asia/Tokyo')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.tz_convert(tz='Asia/Tokyo')
        NaT
        """
    def ceil(self, *args, **kwargs):
        '''
        Return a new Timestamp ceiled to this resolution.

        Parameters
        ----------
        freq : str
            Frequency string indicating the ceiling resolution.
        ambiguous : bool or {\'raise\', \'NaT\'}, default \'raise\'
            The behavior is as follows:

            * bool contains flags to determine if time is dst or not (note
              that this flag is only applicable for ambiguous fall dst dates).
            * \'NaT\' will return NaT for an ambiguous time.
            * \'raise\' will raise an AmbiguousTimeError for an ambiguous time.

        nonexistent : {\'raise\', \'shift_forward\', \'shift_backward, \'NaT\', timedelta}, default \'raise\'
            A nonexistent time does not exist in a particular timezone
            where clocks moved forward due to DST.

            * \'shift_forward\' will shift the nonexistent time forward to the
              closest existing time.
            * \'shift_backward\' will shift the nonexistent time backward to the
              closest existing time.
            * \'NaT\' will return NaT where there are nonexistent times.
            * timedelta objects will shift nonexistent times by the timedelta.
            * \'raise\' will raise an NonExistentTimeError if there are
              nonexistent times.

        Raises
        ------
        ValueError if the freq cannot be converted.

        Notes
        -----
        If the Timestamp has a timezone, ceiling will take place relative to the
        local ("wall") time and re-localized to the same timezone. When ceiling
        near daylight savings time, use ``nonexistent`` and ``ambiguous`` to
        control the re-localization behavior.

        Examples
        --------
        Create a timestamp object:

        >>> ts = pd.Timestamp(\'2020-03-14T15:32:52.192548651\')

        A timestamp can be ceiled using multiple frequency units:

        >>> ts.ceil(freq=\'h\') # hour
        Timestamp(\'2020-03-14 16:00:00\')

        >>> ts.ceil(freq=\'min\') # minute
        Timestamp(\'2020-03-14 15:33:00\')

        >>> ts.ceil(freq=\'s\') # seconds
        Timestamp(\'2020-03-14 15:32:53\')

        >>> ts.ceil(freq=\'us\') # microseconds
        Timestamp(\'2020-03-14 15:32:52.192549\')

        ``freq`` can also be a multiple of a single unit, like \'5min\' (i.e.  5 minutes):

        >>> ts.ceil(freq=\'5min\')
        Timestamp(\'2020-03-14 15:35:00\')

        or a combination of multiple units, like \'1h30min\' (i.e. 1 hour and 30 minutes):

        >>> ts.ceil(freq=\'1h30min\')
        Timestamp(\'2020-03-14 16:30:00\')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.ceil()
        NaT

        When rounding near a daylight savings time transition, use ``ambiguous`` or
        ``nonexistent`` to control how the timestamp should be re-localized.

        >>> ts_tz = pd.Timestamp("2021-10-31 01:30:00").tz_localize("Europe/Amsterdam")

        >>> ts_tz.ceil("h", ambiguous=False)
        Timestamp(\'2021-10-31 02:00:00+0100\', tz=\'Europe/Amsterdam\')

        >>> ts_tz.ceil("h", ambiguous=True)
        Timestamp(\'2021-10-31 02:00:00+0200\', tz=\'Europe/Amsterdam\')
        '''
    def combine(self, date, time) -> Any:
        """
        Timestamp.combine(date, time)

        Combine date, time into datetime with same date and time fields.

        Examples
        --------
        >>> from datetime import date, time
        >>> pd.Timestamp.combine(date(2020, 3, 14), time(15, 30, 15))
        Timestamp('2020-03-14 15:30:15')
        """
    @overload
    def ctime(self) -> Any:
        """
        Return ctime() style string.

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01 10:00:00.00')
        >>> ts
        Timestamp('2023-01-01 10:00:00')
        >>> ts.ctime()
        'Sun Jan  1 10:00:00 2023'
        """
    @overload
    def ctime(self) -> Any:
        """
        Return ctime() style string.

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01 10:00:00.00')
        >>> ts
        Timestamp('2023-01-01 10:00:00')
        >>> ts.ctime()
        'Sun Jan  1 10:00:00 2023'
        """
    def date(self) -> Any:
        """
        Return date object with same year, month and day.

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01 10:00:00.00')
        >>> ts
        Timestamp('2023-01-01 10:00:00')
        >>> ts.date()
        datetime.date(2023, 1, 1)
        """
    @overload
    def day_name(self) -> Any:
        """
        Return the day name of the Timestamp with specified locale.

        Parameters
        ----------
        locale : str, default None (English locale)
            Locale determining the language in which to return the day name.

        Returns
        -------
        str

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')
        >>> ts.day_name()
        'Saturday'

        Analogous for ``pd.NaT``:

        >>> pd.NaT.day_name()
        nan
        """
    @overload
    def day_name(self) -> Any:
        """
        Return the day name of the Timestamp with specified locale.

        Parameters
        ----------
        locale : str, default None (English locale)
            Locale determining the language in which to return the day name.

        Returns
        -------
        str

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')
        >>> ts.day_name()
        'Saturday'

        Analogous for ``pd.NaT``:

        >>> pd.NaT.day_name()
        nan
        """
    def dst(self) -> Any:
        """
        Return the daylight saving time (DST) adjustment.

        Examples
        --------
        >>> ts = pd.Timestamp('2000-06-01 00:00:00', tz='Europe/Brussels')
        >>> ts
        Timestamp('2000-06-01 00:00:00+0200', tz='Europe/Brussels')
        >>> ts.dst()
        datetime.timedelta(seconds=3600)
        """
    def floor(self, *args, **kwargs):
        '''
        Return a new Timestamp floored to this resolution.

        Parameters
        ----------
        freq : str
            Frequency string indicating the flooring resolution.
        ambiguous : bool or {\'raise\', \'NaT\'}, default \'raise\'
            The behavior is as follows:

            * bool contains flags to determine if time is dst or not (note
              that this flag is only applicable for ambiguous fall dst dates).
            * \'NaT\' will return NaT for an ambiguous time.
            * \'raise\' will raise an AmbiguousTimeError for an ambiguous time.

        nonexistent : {\'raise\', \'shift_forward\', \'shift_backward, \'NaT\', timedelta}, default \'raise\'
            A nonexistent time does not exist in a particular timezone
            where clocks moved forward due to DST.

            * \'shift_forward\' will shift the nonexistent time forward to the
              closest existing time.
            * \'shift_backward\' will shift the nonexistent time backward to the
              closest existing time.
            * \'NaT\' will return NaT where there are nonexistent times.
            * timedelta objects will shift nonexistent times by the timedelta.
            * \'raise\' will raise an NonExistentTimeError if there are
              nonexistent times.

        Raises
        ------
        ValueError if the freq cannot be converted.

        Notes
        -----
        If the Timestamp has a timezone, flooring will take place relative to the
        local ("wall") time and re-localized to the same timezone. When flooring
        near daylight savings time, use ``nonexistent`` and ``ambiguous`` to
        control the re-localization behavior.

        Examples
        --------
        Create a timestamp object:

        >>> ts = pd.Timestamp(\'2020-03-14T15:32:52.192548651\')

        A timestamp can be floored using multiple frequency units:

        >>> ts.floor(freq=\'h\') # hour
        Timestamp(\'2020-03-14 15:00:00\')

        >>> ts.floor(freq=\'min\') # minute
        Timestamp(\'2020-03-14 15:32:00\')

        >>> ts.floor(freq=\'s\') # seconds
        Timestamp(\'2020-03-14 15:32:52\')

        >>> ts.floor(freq=\'ns\') # nanoseconds
        Timestamp(\'2020-03-14 15:32:52.192548651\')

        ``freq`` can also be a multiple of a single unit, like \'5min\' (i.e.  5 minutes):

        >>> ts.floor(freq=\'5min\')
        Timestamp(\'2020-03-14 15:30:00\')

        or a combination of multiple units, like \'1h30min\' (i.e. 1 hour and 30 minutes):

        >>> ts.floor(freq=\'1h30min\')
        Timestamp(\'2020-03-14 15:00:00\')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.floor()
        NaT

        When rounding near a daylight savings time transition, use ``ambiguous`` or
        ``nonexistent`` to control how the timestamp should be re-localized.

        >>> ts_tz = pd.Timestamp("2021-10-31 03:30:00").tz_localize("Europe/Amsterdam")

        >>> ts_tz.floor("2h", ambiguous=False)
        Timestamp(\'2021-10-31 02:00:00+0100\', tz=\'Europe/Amsterdam\')

        >>> ts_tz.floor("2h", ambiguous=True)
        Timestamp(\'2021-10-31 02:00:00+0200\', tz=\'Europe/Amsterdam\')
        '''
    def fromisocalendar(self, *args, **kwargs):
        """int, int, int -> Construct a date from the ISO year, week number and weekday.

        This is the inverse of the date.isocalendar() function"""
    def fromordinal(self, *args, **kwargs):
        """
        Construct a timestamp from a a proleptic Gregorian ordinal.

        Parameters
        ----------
        ordinal : int
            Date corresponding to a proleptic Gregorian ordinal.
        tz : str, pytz.timezone, dateutil.tz.tzfile or None
            Time zone for the Timestamp.

        Notes
        -----
        By definition there cannot be any tz info on the ordinal itself.

        Examples
        --------
        >>> pd.Timestamp.fromordinal(737425)
        Timestamp('2020-01-01 00:00:00')
        """
    def fromtimestamp(self, ts) -> Any:
        """
        Timestamp.fromtimestamp(ts)

        Transform timestamp[, tz] to tz's local time from POSIX timestamp.

        Examples
        --------
        >>> pd.Timestamp.fromtimestamp(1584199972)  # doctest: +SKIP
        Timestamp('2020-03-14 15:32:52')

        Note that the output may change depending on your local time.
        """
    def isocalendar(self) -> Any:
        """
        Return a named tuple containing ISO year, week number, and weekday.

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01 10:00:00')
        >>> ts
        Timestamp('2023-01-01 10:00:00')
        >>> ts.isocalendar()
        datetime.IsoCalendarDate(year=2022, week=52, weekday=7)
        """
    def isoweekday(self) -> Any:
        """
        Return the day of the week represented by the date.

        Monday == 1 ... Sunday == 7.

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01 10:00:00')
        >>> ts
        Timestamp('2023-01-01 10:00:00')
        >>> ts.isoweekday()
        7
        """
    @overload
    def month_name(self) -> Any:
        """
        Return the month name of the Timestamp with specified locale.

        Parameters
        ----------
        locale : str, default None (English locale)
            Locale determining the language in which to return the month name.

        Returns
        -------
        str

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')
        >>> ts.month_name()
        'March'

        Analogous for ``pd.NaT``:

        >>> pd.NaT.month_name()
        nan
        """
    @overload
    def month_name(self) -> Any:
        """
        Return the month name of the Timestamp with specified locale.

        Parameters
        ----------
        locale : str, default None (English locale)
            Locale determining the language in which to return the month name.

        Returns
        -------
        str

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')
        >>> ts.month_name()
        'March'

        Analogous for ``pd.NaT``:

        >>> pd.NaT.month_name()
        nan
        """
    @overload
    def now(self) -> Any:
        """
        Return new Timestamp object representing current time local to tz.

        Parameters
        ----------
        tz : str or timezone object, default None
            Timezone to localize to.

        Examples
        --------
        >>> pd.Timestamp.now()  # doctest: +SKIP
        Timestamp('2020-11-16 22:06:16.378782')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.now()
        NaT
        """
    @overload
    def now(self) -> Any:
        """
        Return new Timestamp object representing current time local to tz.

        Parameters
        ----------
        tz : str or timezone object, default None
            Timezone to localize to.

        Examples
        --------
        >>> pd.Timestamp.now()  # doctest: +SKIP
        Timestamp('2020-11-16 22:06:16.378782')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.now()
        NaT
        """
    @overload
    def replace(self, year=..., hour=...) -> Any:
        """
        Implements datetime.replace, handles nanoseconds.

        Parameters
        ----------
        year : int, optional
        month : int, optional
        day : int, optional
        hour : int, optional
        minute : int, optional
        second : int, optional
        microsecond : int, optional
        nanosecond : int, optional
        tzinfo : tz-convertible, optional
        fold : int, optional

        Returns
        -------
        Timestamp with fields replaced

        Examples
        --------
        Create a timestamp object:

        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651', tz='UTC')
        >>> ts
        Timestamp('2020-03-14 15:32:52.192548651+0000', tz='UTC')

        Replace year and the hour:

        >>> ts.replace(year=1999, hour=10)
        Timestamp('1999-03-14 10:32:52.192548651+0000', tz='UTC')

        Replace timezone (not a conversion):

        >>> import pytz
        >>> ts.replace(tzinfo=pytz.timezone('US/Pacific'))
        Timestamp('2020-03-14 15:32:52.192548651-0700', tz='US/Pacific')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.replace(tzinfo=pytz.timezone('US/Pacific'))
        NaT
        """
    @overload
    def replace(self, tzinfo=...) -> Any:
        """
        Implements datetime.replace, handles nanoseconds.

        Parameters
        ----------
        year : int, optional
        month : int, optional
        day : int, optional
        hour : int, optional
        minute : int, optional
        second : int, optional
        microsecond : int, optional
        nanosecond : int, optional
        tzinfo : tz-convertible, optional
        fold : int, optional

        Returns
        -------
        Timestamp with fields replaced

        Examples
        --------
        Create a timestamp object:

        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651', tz='UTC')
        >>> ts
        Timestamp('2020-03-14 15:32:52.192548651+0000', tz='UTC')

        Replace year and the hour:

        >>> ts.replace(year=1999, hour=10)
        Timestamp('1999-03-14 10:32:52.192548651+0000', tz='UTC')

        Replace timezone (not a conversion):

        >>> import pytz
        >>> ts.replace(tzinfo=pytz.timezone('US/Pacific'))
        Timestamp('2020-03-14 15:32:52.192548651-0700', tz='US/Pacific')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.replace(tzinfo=pytz.timezone('US/Pacific'))
        NaT
        """
    @overload
    def replace(self, tzinfo=...) -> Any:
        """
        Implements datetime.replace, handles nanoseconds.

        Parameters
        ----------
        year : int, optional
        month : int, optional
        day : int, optional
        hour : int, optional
        minute : int, optional
        second : int, optional
        microsecond : int, optional
        nanosecond : int, optional
        tzinfo : tz-convertible, optional
        fold : int, optional

        Returns
        -------
        Timestamp with fields replaced

        Examples
        --------
        Create a timestamp object:

        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651', tz='UTC')
        >>> ts
        Timestamp('2020-03-14 15:32:52.192548651+0000', tz='UTC')

        Replace year and the hour:

        >>> ts.replace(year=1999, hour=10)
        Timestamp('1999-03-14 10:32:52.192548651+0000', tz='UTC')

        Replace timezone (not a conversion):

        >>> import pytz
        >>> ts.replace(tzinfo=pytz.timezone('US/Pacific'))
        Timestamp('2020-03-14 15:32:52.192548651-0700', tz='US/Pacific')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.replace(tzinfo=pytz.timezone('US/Pacific'))
        NaT
        """
    def round(self, *args, **kwargs):
        '''
        Round the Timestamp to the specified resolution.

        Parameters
        ----------
        freq : str
            Frequency string indicating the rounding resolution.
        ambiguous : bool or {\'raise\', \'NaT\'}, default \'raise\'
            The behavior is as follows:

            * bool contains flags to determine if time is dst or not (note
              that this flag is only applicable for ambiguous fall dst dates).
            * \'NaT\' will return NaT for an ambiguous time.
            * \'raise\' will raise an AmbiguousTimeError for an ambiguous time.

        nonexistent : {\'raise\', \'shift_forward\', \'shift_backward, \'NaT\', timedelta}, default \'raise\'
            A nonexistent time does not exist in a particular timezone
            where clocks moved forward due to DST.

            * \'shift_forward\' will shift the nonexistent time forward to the
              closest existing time.
            * \'shift_backward\' will shift the nonexistent time backward to the
              closest existing time.
            * \'NaT\' will return NaT where there are nonexistent times.
            * timedelta objects will shift nonexistent times by the timedelta.
            * \'raise\' will raise an NonExistentTimeError if there are
              nonexistent times.

        Returns
        -------
        a new Timestamp rounded to the given resolution of `freq`

        Raises
        ------
        ValueError if the freq cannot be converted

        Notes
        -----
        If the Timestamp has a timezone, rounding will take place relative to the
        local ("wall") time and re-localized to the same timezone. When rounding
        near daylight savings time, use ``nonexistent`` and ``ambiguous`` to
        control the re-localization behavior.

        Examples
        --------
        Create a timestamp object:

        >>> ts = pd.Timestamp(\'2020-03-14T15:32:52.192548651\')

        A timestamp can be rounded using multiple frequency units:

        >>> ts.round(freq=\'h\') # hour
        Timestamp(\'2020-03-14 16:00:00\')

        >>> ts.round(freq=\'min\') # minute
        Timestamp(\'2020-03-14 15:33:00\')

        >>> ts.round(freq=\'s\') # seconds
        Timestamp(\'2020-03-14 15:32:52\')

        >>> ts.round(freq=\'ms\') # milliseconds
        Timestamp(\'2020-03-14 15:32:52.193000\')

        ``freq`` can also be a multiple of a single unit, like \'5min\' (i.e.  5 minutes):

        >>> ts.round(freq=\'5min\')
        Timestamp(\'2020-03-14 15:35:00\')

        or a combination of multiple units, like \'1h30min\' (i.e. 1 hour and 30 minutes):

        >>> ts.round(freq=\'1h30min\')
        Timestamp(\'2020-03-14 15:00:00\')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.round()
        NaT

        When rounding near a daylight savings time transition, use ``ambiguous`` or
        ``nonexistent`` to control how the timestamp should be re-localized.

        >>> ts_tz = pd.Timestamp("2021-10-31 01:30:00").tz_localize("Europe/Amsterdam")

        >>> ts_tz.round("h", ambiguous=False)
        Timestamp(\'2021-10-31 02:00:00+0100\', tz=\'Europe/Amsterdam\')

        >>> ts_tz.round("h", ambiguous=True)
        Timestamp(\'2021-10-31 02:00:00+0200\', tz=\'Europe/Amsterdam\')
        '''
    def strftime(self, *args, **kwargs):
        """
        Return a formatted string of the Timestamp.

        Parameters
        ----------
        format : str
            Format string to convert Timestamp to string.
            See strftime documentation for more information on the format string:
            https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior.

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')
        >>> ts.strftime('%Y-%m-%d %X')
        '2020-03-14 15:32:52'
        """
    def strptime(self, string, format) -> Any:
        '''
        Timestamp.strptime(string, format)

        Function is not implemented. Use pd.to_datetime().

        Examples
        --------
        >>> pd.Timestamp.strptime("2023-01-01", "%d/%m/%y")
        Traceback (most recent call last):
        NotImplementedError
        '''
    def time(self) -> Any:
        """
        Return time object with same time but with tzinfo=None.

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01 10:00:00')
        >>> ts
        Timestamp('2023-01-01 10:00:00')
        >>> ts.time()
        datetime.time(10, 0)
        """
    def timestamp(self) -> Any:
        """
        Return POSIX timestamp as float.

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548')
        >>> ts.timestamp()
        1584199972.192548
        """
    def timetuple(self) -> Any:
        """
        Return time tuple, compatible with time.localtime().

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01 10:00:00')
        >>> ts
        Timestamp('2023-01-01 10:00:00')
        >>> ts.timetuple()
        time.struct_time(tm_year=2023, tm_mon=1, tm_mday=1,
        tm_hour=10, tm_min=0, tm_sec=0, tm_wday=6, tm_yday=1, tm_isdst=-1)
        """
    def timetz(self) -> Any:
        """
        Return time object with same time and tzinfo.

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01 10:00:00', tz='Europe/Brussels')
        >>> ts
        Timestamp('2023-01-01 10:00:00+0100', tz='Europe/Brussels')
        >>> ts.timetz()
        datetime.time(10, 0, tzinfo=<DstTzInfo 'Europe/Brussels' CET+1:00:00 STD>)
        """
    @overload
    def to_pydatetime(self) -> Any:
        """
        Convert a Timestamp object to a native Python datetime object.

        If warn=True, issue a warning if nanoseconds is nonzero.

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548')
        >>> ts.to_pydatetime()
        datetime.datetime(2020, 3, 14, 15, 32, 52, 192548)

        Analogous for ``pd.NaT``:

        >>> pd.NaT.to_pydatetime()
        NaT
        """
    @overload
    def to_pydatetime(self) -> Any:
        """
        Convert a Timestamp object to a native Python datetime object.

        If warn=True, issue a warning if nanoseconds is nonzero.

        Examples
        --------
        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548')
        >>> ts.to_pydatetime()
        datetime.datetime(2020, 3, 14, 15, 32, 52, 192548)

        Analogous for ``pd.NaT``:

        >>> pd.NaT.to_pydatetime()
        NaT
        """
    @overload
    def today(self) -> Any:
        """
        Return the current time in the local timezone.

        This differs from datetime.today() in that it can be localized to a
        passed timezone.

        Parameters
        ----------
        tz : str or timezone object, default None
            Timezone to localize to.

        Examples
        --------
        >>> pd.Timestamp.today()    # doctest: +SKIP
        Timestamp('2020-11-16 22:37:39.969883')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.today()
        NaT
        """
    @overload
    def today(self) -> Any:
        """
        Return the current time in the local timezone.

        This differs from datetime.today() in that it can be localized to a
        passed timezone.

        Parameters
        ----------
        tz : str or timezone object, default None
            Timezone to localize to.

        Examples
        --------
        >>> pd.Timestamp.today()    # doctest: +SKIP
        Timestamp('2020-11-16 22:37:39.969883')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.today()
        NaT
        """
    @overload
    def today(self) -> Any:
        """
        Return the current time in the local timezone.

        This differs from datetime.today() in that it can be localized to a
        passed timezone.

        Parameters
        ----------
        tz : str or timezone object, default None
            Timezone to localize to.

        Examples
        --------
        >>> pd.Timestamp.today()    # doctest: +SKIP
        Timestamp('2020-11-16 22:37:39.969883')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.today()
        NaT
        """
    def toordinal(self) -> Any:
        """
        Return proleptic Gregorian ordinal. January 1 of year 1 is day 1.

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01 10:00:50')
        >>> ts
        Timestamp('2023-01-01 10:00:50')
        >>> ts.toordinal()
        738521
        """
    def total_seconds(self) -> Any:
        """
        Total seconds in the duration.

        Examples
        --------
        >>> td = pd.Timedelta('1min')
        >>> td
        Timedelta('0 days 00:01:00')
        >>> td.total_seconds()
        60.0
        """
    @overload
    def tz_convert(self, tz=...) -> Any:
        """
        Convert timezone-aware Timestamp to another time zone.

        Parameters
        ----------
        tz : str, pytz.timezone, dateutil.tz.tzfile or None
            Time zone for time which Timestamp will be converted to.
            None will remove timezone holding UTC time.

        Returns
        -------
        converted : Timestamp

        Raises
        ------
        TypeError
            If Timestamp is tz-naive.

        Examples
        --------
        Create a timestamp object with UTC timezone:

        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651', tz='UTC')
        >>> ts
        Timestamp('2020-03-14 15:32:52.192548651+0000', tz='UTC')

        Change to Tokyo timezone:

        >>> ts.tz_convert(tz='Asia/Tokyo')
        Timestamp('2020-03-15 00:32:52.192548651+0900', tz='Asia/Tokyo')

        Can also use ``astimezone``:

        >>> ts.astimezone(tz='Asia/Tokyo')
        Timestamp('2020-03-15 00:32:52.192548651+0900', tz='Asia/Tokyo')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.tz_convert(tz='Asia/Tokyo')
        NaT
        """
    @overload
    def tz_convert(self, tz=...) -> Any:
        """
        Convert timezone-aware Timestamp to another time zone.

        Parameters
        ----------
        tz : str, pytz.timezone, dateutil.tz.tzfile or None
            Time zone for time which Timestamp will be converted to.
            None will remove timezone holding UTC time.

        Returns
        -------
        converted : Timestamp

        Raises
        ------
        TypeError
            If Timestamp is tz-naive.

        Examples
        --------
        Create a timestamp object with UTC timezone:

        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651', tz='UTC')
        >>> ts
        Timestamp('2020-03-14 15:32:52.192548651+0000', tz='UTC')

        Change to Tokyo timezone:

        >>> ts.tz_convert(tz='Asia/Tokyo')
        Timestamp('2020-03-15 00:32:52.192548651+0900', tz='Asia/Tokyo')

        Can also use ``astimezone``:

        >>> ts.astimezone(tz='Asia/Tokyo')
        Timestamp('2020-03-15 00:32:52.192548651+0900', tz='Asia/Tokyo')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.tz_convert(tz='Asia/Tokyo')
        NaT
        """
    def tz_localize(self, *args, **kwargs):
        """
        Localize the Timestamp to a timezone.

        Convert naive Timestamp to local time zone or remove
        timezone from timezone-aware Timestamp.

        Parameters
        ----------
        tz : str, pytz.timezone, dateutil.tz.tzfile or None
            Time zone for time which Timestamp will be converted to.
            None will remove timezone holding local time.

        ambiguous : bool, 'NaT', default 'raise'
            When clocks moved backward due to DST, ambiguous times may arise.
            For example in Central European Time (UTC+01), when going from
            03:00 DST to 02:00 non-DST, 02:30:00 local time occurs both at
            00:30:00 UTC and at 01:30:00 UTC. In such a situation, the
            `ambiguous` parameter dictates how ambiguous times should be
            handled.

            The behavior is as follows:

            * bool contains flags to determine if time is dst or not (note
              that this flag is only applicable for ambiguous fall dst dates).
            * 'NaT' will return NaT for an ambiguous time.
            * 'raise' will raise an AmbiguousTimeError for an ambiguous time.

        nonexistent : 'shift_forward', 'shift_backward, 'NaT', timedelta, default 'raise'
            A nonexistent time does not exist in a particular timezone
            where clocks moved forward due to DST.

            The behavior is as follows:

            * 'shift_forward' will shift the nonexistent time forward to the
              closest existing time.
            * 'shift_backward' will shift the nonexistent time backward to the
              closest existing time.
            * 'NaT' will return NaT where there are nonexistent times.
            * timedelta objects will shift nonexistent times by the timedelta.
            * 'raise' will raise an NonExistentTimeError if there are
              nonexistent times.

        Returns
        -------
        localized : Timestamp

        Raises
        ------
        TypeError
            If the Timestamp is tz-aware and tz is not None.

        Examples
        --------
        Create a naive timestamp object:

        >>> ts = pd.Timestamp('2020-03-14T15:32:52.192548651')
        >>> ts
        Timestamp('2020-03-14 15:32:52.192548651')

        Add 'Europe/Stockholm' as timezone:

        >>> ts.tz_localize(tz='Europe/Stockholm')
        Timestamp('2020-03-14 15:32:52.192548651+0100', tz='Europe/Stockholm')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.tz_localize()
        NaT
        """
    def tzname(self) -> Any:
        """
        Return time zone name.

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01 10:00:00', tz='Europe/Brussels')
        >>> ts
        Timestamp('2023-01-01 10:00:00+0100', tz='Europe/Brussels')
        >>> ts.tzname()
        'CET'
        """
    def utcfromtimestamp(self, ts) -> Any:
        """
        Timestamp.utcfromtimestamp(ts)

        Construct a timezone-aware UTC datetime from a POSIX timestamp.

        Notes
        -----
        Timestamp.utcfromtimestamp behavior differs from datetime.utcfromtimestamp
        in returning a timezone-aware object.

        Examples
        --------
        >>> pd.Timestamp.utcfromtimestamp(1584199972)
        Timestamp('2020-03-14 15:32:52+0000', tz='UTC')
        """
    @overload
    def utcnow(self) -> Any:
        """
        Timestamp.utcnow()

        Return a new Timestamp representing UTC day and time.

        Examples
        --------
        >>> pd.Timestamp.utcnow()   # doctest: +SKIP
        Timestamp('2020-11-16 22:50:18.092888+0000', tz='UTC')
        """
    @overload
    def utcnow(self) -> Any:
        """
        Timestamp.utcnow()

        Return a new Timestamp representing UTC day and time.

        Examples
        --------
        >>> pd.Timestamp.utcnow()   # doctest: +SKIP
        Timestamp('2020-11-16 22:50:18.092888+0000', tz='UTC')
        """
    def utcoffset(self) -> Any:
        """
        Return utc offset.

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01 10:00:00', tz='Europe/Brussels')
        >>> ts
        Timestamp('2023-01-01 10:00:00+0100', tz='Europe/Brussels')
        >>> ts.utcoffset()
        datetime.timedelta(seconds=3600)
        """
    def utctimetuple(self) -> Any:
        """
        Return UTC time tuple, compatible with time.localtime().

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01 10:00:00', tz='Europe/Brussels')
        >>> ts
        Timestamp('2023-01-01 10:00:00+0100', tz='Europe/Brussels')
        >>> ts.utctimetuple()
        time.struct_time(tm_year=2023, tm_mon=1, tm_mday=1, tm_hour=9,
        tm_min=0, tm_sec=0, tm_wday=6, tm_yday=1, tm_isdst=0)
        """
    def weekday(self) -> Any:
        """
        Return the day of the week represented by the date.

        Monday == 0 ... Sunday == 6.

        Examples
        --------
        >>> ts = pd.Timestamp('2023-01-01')
        >>> ts
        Timestamp('2023-01-01  00:00:00')
        >>> ts.weekday()
        6
        """
    def __reduce__(self): ...
    def __reduce_ex__(self, protocol): ...
    def __rfloordiv__(self, other): ...
    def __rmul__(self, other): ...
    def __rtruediv__(self, other): ...
    @property
    def day(self): ...
    @property
    def day_of_week(self): ...
    @property
    def day_of_year(self): ...
    @property
    def dayofweek(self): ...
    @property
    def dayofyear(self): ...
    @property
    def days(self): ...
    @property
    def days_in_month(self): ...
    @property
    def daysinmonth(self): ...
    @property
    def hour(self): ...
    @property
    def microsecond(self): ...
    @property
    def microseconds(self): ...
    @property
    def millisecond(self): ...
    @property
    def minute(self): ...
    @property
    def month(self): ...
    @property
    def nanosecond(self): ...
    @property
    def nanoseconds(self): ...
    @property
    def quarter(self): ...
    @property
    def qyear(self): ...
    @property
    def second(self): ...
    @property
    def seconds(self): ...
    @property
    def tz(self): ...
    @property
    def tzinfo(self): ...
    @property
    def value(self): ...
    @property
    def week(self): ...
    @property
    def weekofyear(self): ...
    @property
    def year(self): ...

class _NaT(datetime.datetime):
    __array_priority__: ClassVar[int] = ...
    _value: Incomplete
    asm8: Incomplete
    is_leap_year: Incomplete
    is_month_end: Incomplete
    is_month_start: Incomplete
    is_quarter_end: Incomplete
    is_quarter_start: Incomplete
    is_year_end: Incomplete
    is_year_start: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def isoformat(self, *args, **kwargs): ...
    def to_datetime64(self) -> Any:
        """
        Return a numpy.datetime64 object with same precision.

        Examples
        --------
        >>> ts = pd.Timestamp(year=2023, month=1, day=1,
        ...                   hour=10, second=15)
        >>> ts
        Timestamp('2023-01-01 10:00:15')
        >>> ts.to_datetime64()
        numpy.datetime64('2023-01-01T10:00:15.000000')
        """
    @overload
    def to_numpy(self) -> Any:
        '''
        Convert the Timestamp to a NumPy datetime64 or timedelta64.

        With the default \'dtype\', this is an alias method for `NaT.to_datetime64()`.

        The copy parameter is available here only for compatibility. Its value
        will not affect the return value.

        Returns
        -------
        numpy.datetime64 or numpy.timedelta64

        See Also
        --------
        DatetimeIndex.to_numpy : Similar method for DatetimeIndex.

        Examples
        --------
        >>> ts = pd.Timestamp(\'2020-03-14T15:32:52.192548651\')
        >>> ts.to_numpy()
        numpy.datetime64(\'2020-03-14T15:32:52.192548651\')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.to_numpy()
        numpy.datetime64(\'NaT\')

        >>> pd.NaT.to_numpy("m8[ns]")
        numpy.timedelta64(\'NaT\',\'ns\')
        '''
    @overload
    def to_numpy(self) -> Any:
        '''
        Convert the Timestamp to a NumPy datetime64 or timedelta64.

        With the default \'dtype\', this is an alias method for `NaT.to_datetime64()`.

        The copy parameter is available here only for compatibility. Its value
        will not affect the return value.

        Returns
        -------
        numpy.datetime64 or numpy.timedelta64

        See Also
        --------
        DatetimeIndex.to_numpy : Similar method for DatetimeIndex.

        Examples
        --------
        >>> ts = pd.Timestamp(\'2020-03-14T15:32:52.192548651\')
        >>> ts.to_numpy()
        numpy.datetime64(\'2020-03-14T15:32:52.192548651\')

        Analogous for ``pd.NaT``:

        >>> pd.NaT.to_numpy()
        numpy.datetime64(\'NaT\')

        >>> pd.NaT.to_numpy("m8[ns]")
        numpy.timedelta64(\'NaT\',\'ns\')
        '''
    def __add__(self, other):
        """Return self+value."""
    def __eq__(self, other: object) -> bool:
        """Return self==value."""
    def __floordiv__(self, other):
        """Return self//value."""
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
    def __pos__(self):
        """+self"""
    def __radd__(self, other): ...
    def __reduce_cython__(self, *args, **kwargs): ...
    def __rfloordiv__(self, other):
        """Return value//self."""
    def __rmul__(self, other):
        """Return value*self."""
    def __rsub__(self, other): ...
    def __rtruediv__(self, other):
        """Return value/self."""
    def __setstate_cython__(self, *args, **kwargs): ...
    def __sub__(self, other):
        """Return self-value."""
    def __truediv__(self, other):
        """Return self/value."""
