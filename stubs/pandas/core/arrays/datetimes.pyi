import numpy as np
from _typeshed import Incomplete
from collections.abc import Iterator
from datetime import tzinfo
from pandas import DataFrame as DataFrame
from pandas._libs import lib as lib, tslib as tslib
from pandas._libs.tslibs import BaseOffset as BaseOffset, NaT as NaT, NaTType as NaTType, Resolution as Resolution, Timestamp as Timestamp, astype_overflowsafe as astype_overflowsafe, fields as fields, get_resolution as get_resolution, get_supported_dtype as get_supported_dtype, get_unit_from_dtype as get_unit_from_dtype, ints_to_pydatetime as ints_to_pydatetime, is_date_array_normalized as is_date_array_normalized, is_supported_dtype as is_supported_dtype, is_unitless as is_unitless, normalize_i8_timestamps as normalize_i8_timestamps, timezones as timezones, to_offset as to_offset, tz_convert_from_utc as tz_convert_from_utc, tzconversion as tzconversion
from pandas._typing import ArrayLike as ArrayLike, DateTimeErrorChoices as DateTimeErrorChoices, DtypeObj as DtypeObj, IntervalClosedType as IntervalClosedType, Self as Self, TimeAmbiguous as TimeAmbiguous, TimeNonexistent as TimeNonexistent, npt as npt
from pandas.core.arrays import PeriodArray as PeriodArray, datetimelike as dtl
from pandas.core.dtypes.common import DT64NS_DTYPE as DT64NS_DTYPE, INT64_DTYPE as INT64_DTYPE, is_bool_dtype as is_bool_dtype, is_float_dtype as is_float_dtype, is_string_dtype as is_string_dtype, pandas_dtype as pandas_dtype
from pandas.core.dtypes.dtypes import DatetimeTZDtype as DatetimeTZDtype, ExtensionDtype as ExtensionDtype, PeriodDtype as PeriodDtype
from pandas.tseries.offsets import Day as Day, Tick as Tick
from typing import overload

_ITER_CHUNKSIZE: int

@overload
def tz_to_dtype(tz: tzinfo, unit: str = ...) -> DatetimeTZDtype: ...
@overload
def tz_to_dtype(tz: None, unit: str = ...) -> np.dtype[np.datetime64]: ...
def _field_accessor(name: str, field: str, docstring: str | None = None): ...

class DatetimeArray(dtl.TimelikeOps, dtl.DatelikeOps):
    """
    Pandas ExtensionArray for tz-naive or tz-aware datetime data.

    .. warning::

       DatetimeArray is currently experimental, and its API may change
       without warning. In particular, :attr:`DatetimeArray.dtype` is
       expected to change to always be an instance of an ``ExtensionDtype``
       subclass.

    Parameters
    ----------
    values : Series, Index, DatetimeArray, ndarray
        The datetime data.

        For DatetimeArray `values` (or a Series or Index boxing one),
        `dtype` and `freq` will be extracted from `values`.

    dtype : numpy.dtype or DatetimeTZDtype
        Note that the only NumPy dtype allowed is 'datetime64[ns]'.
    freq : str or Offset, optional
        The frequency.
    copy : bool, default False
        Whether to copy the underlying array of values.

    Attributes
    ----------
    None

    Methods
    -------
    None

    Examples
    --------
    >>> pd.arrays.DatetimeArray._from_sequence(
    ...    pd.DatetimeIndex(['2023-01-01', '2023-01-02'], freq='D'))
    <DatetimeArray>
    ['2023-01-01 00:00:00', '2023-01-02 00:00:00']
    Length: 2, dtype: datetime64[ns]
    """
    _typ: str
    _internal_fill_value: Incomplete
    _recognized_scalars: Incomplete
    _is_recognized_dtype: Incomplete
    _infer_matches: Incomplete
    @property
    def _scalar_type(self) -> type[Timestamp]: ...
    _bool_ops: list[str]
    _object_ops: list[str]
    _field_ops: list[str]
    _other_ops: list[str]
    _datetimelike_ops: list[str]
    _datetimelike_methods: list[str]
    __array_priority__: int
    _dtype: np.dtype[np.datetime64] | DatetimeTZDtype
    _freq: BaseOffset | None
    _default_dtype = DT64NS_DTYPE
    @classmethod
    def _from_scalars(cls, scalars, *, dtype: DtypeObj) -> Self: ...
    @classmethod
    def _validate_dtype(cls, values, dtype): ...
    @classmethod
    def _simple_new(cls, values: npt.NDArray[np.datetime64], freq: BaseOffset | None = None, dtype: np.dtype[np.datetime64] | DatetimeTZDtype = ...) -> Self: ...
    @classmethod
    def _from_sequence(cls, scalars, *, dtype: Incomplete | None = None, copy: bool = False): ...
    @classmethod
    def _from_sequence_not_strict(cls, data, *, dtype: Incomplete | None = None, copy: bool = False, tz=..., freq: str | BaseOffset | lib.NoDefault | None = ..., dayfirst: bool = False, yearfirst: bool = False, ambiguous: TimeAmbiguous = 'raise') -> Self:
        """
        A non-strict version of _from_sequence, called from DatetimeIndex.__new__.
        """
    @classmethod
    def _generate_range(cls, start, end, periods: int | None, freq, tz: Incomplete | None = None, normalize: bool = False, ambiguous: TimeAmbiguous = 'raise', nonexistent: TimeNonexistent = 'raise', inclusive: IntervalClosedType = 'both', *, unit: str | None = None) -> Self: ...
    def _unbox_scalar(self, value) -> np.datetime64: ...
    def _scalar_from_string(self, value) -> Timestamp | NaTType: ...
    def _check_compatible_with(self, other) -> None: ...
    def _box_func(self, x: np.datetime64) -> Timestamp | NaTType: ...
    @property
    def dtype(self) -> np.dtype[np.datetime64] | DatetimeTZDtype:
        """
        The dtype for the DatetimeArray.

        .. warning::

           A future version of pandas will change dtype to never be a
           ``numpy.dtype``. Instead, :attr:`DatetimeArray.dtype` will
           always be an instance of an ``ExtensionDtype`` subclass.

        Returns
        -------
        numpy.dtype or DatetimeTZDtype
            If the values are tz-naive, then ``np.dtype('datetime64[ns]')``
            is returned.

            If the values are tz-aware, then the ``DatetimeTZDtype``
            is returned.
        """
    @property
    def tz(self) -> tzinfo | None:
        '''
        Return the timezone.

        Returns
        -------
        datetime.tzinfo, pytz.tzinfo.BaseTZInfo, dateutil.tz.tz.tzfile, or None
            Returns None when the array is tz-naive.

        Examples
        --------
        For Series:

        >>> s = pd.Series(["1/1/2020 10:00:00+00:00", "2/1/2020 11:00:00+00:00"])
        >>> s = pd.to_datetime(s)
        >>> s
        0   2020-01-01 10:00:00+00:00
        1   2020-02-01 11:00:00+00:00
        dtype: datetime64[ns, UTC]
        >>> s.dt.tz
        datetime.timezone.utc

        For DatetimeIndex:

        >>> idx = pd.DatetimeIndex(["1/1/2020 10:00:00+00:00",
        ...                         "2/1/2020 11:00:00+00:00"])
        >>> idx.tz
        datetime.timezone.utc
        '''
    @tz.setter
    def tz(self, value) -> None: ...
    @property
    def tzinfo(self) -> tzinfo | None:
        """
        Alias for tz attribute
        """
    @property
    def is_normalized(self) -> bool:
        '''
        Returns True if all of the dates are at midnight ("no time")
        '''
    @property
    def _resolution_obj(self) -> Resolution: ...
    def __array__(self, dtype: Incomplete | None = None, copy: Incomplete | None = None) -> np.ndarray: ...
    def __iter__(self) -> Iterator:
        """
        Return an iterator over the boxed values

        Yields
        ------
        tstamp : Timestamp
        """
    def astype(self, dtype, copy: bool = True): ...
    def _format_native_types(self, *, na_rep: str | float = 'NaT', date_format: Incomplete | None = None, **kwargs) -> npt.NDArray[np.object_]: ...
    def _has_same_tz(self, other) -> bool: ...
    def _assert_tzawareness_compat(self, other) -> None: ...
    def _add_offset(self, offset: BaseOffset) -> Self: ...
    def _local_timestamps(self) -> npt.NDArray[np.int64]:
        """
        Convert to an i8 (unix-like nanosecond timestamp) representation
        while keeping the local timezone and not using UTC.
        This is used to calculate time-of-day information as if the timestamps
        were timezone-naive.
        """
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
    def tz_localize(self, tz, ambiguous: TimeAmbiguous = 'raise', nonexistent: TimeNonexistent = 'raise') -> Self:
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
    def to_pydatetime(self) -> npt.NDArray[np.object_]:
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
    def normalize(self) -> Self:
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
    def to_period(self, freq: Incomplete | None = None) -> PeriodArray:
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
    def month_name(self, locale: Incomplete | None = None) -> npt.NDArray[np.object_]:
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
    def day_name(self, locale: Incomplete | None = None) -> npt.NDArray[np.object_]:
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
    @property
    def time(self) -> npt.NDArray[np.object_]:
        '''
        Returns numpy array of :class:`datetime.time` objects.

        The time part of the Timestamps.

        Examples
        --------
        For Series:

        >>> s = pd.Series(["1/1/2020 10:00:00+00:00", "2/1/2020 11:00:00+00:00"])
        >>> s = pd.to_datetime(s)
        >>> s
        0   2020-01-01 10:00:00+00:00
        1   2020-02-01 11:00:00+00:00
        dtype: datetime64[ns, UTC]
        >>> s.dt.time
        0    10:00:00
        1    11:00:00
        dtype: object

        For DatetimeIndex:

        >>> idx = pd.DatetimeIndex(["1/1/2020 10:00:00+00:00",
        ...                         "2/1/2020 11:00:00+00:00"])
        >>> idx.time
        array([datetime.time(10, 0), datetime.time(11, 0)], dtype=object)
        '''
    @property
    def timetz(self) -> npt.NDArray[np.object_]:
        '''
        Returns numpy array of :class:`datetime.time` objects with timezones.

        The time part of the Timestamps.

        Examples
        --------
        For Series:

        >>> s = pd.Series(["1/1/2020 10:00:00+00:00", "2/1/2020 11:00:00+00:00"])
        >>> s = pd.to_datetime(s)
        >>> s
        0   2020-01-01 10:00:00+00:00
        1   2020-02-01 11:00:00+00:00
        dtype: datetime64[ns, UTC]
        >>> s.dt.timetz
        0    10:00:00+00:00
        1    11:00:00+00:00
        dtype: object

        For DatetimeIndex:

        >>> idx = pd.DatetimeIndex(["1/1/2020 10:00:00+00:00",
        ...                         "2/1/2020 11:00:00+00:00"])
        >>> idx.timetz
        array([datetime.time(10, 0, tzinfo=datetime.timezone.utc),
        datetime.time(11, 0, tzinfo=datetime.timezone.utc)], dtype=object)
        '''
    @property
    def date(self) -> npt.NDArray[np.object_]:
        '''
        Returns numpy array of python :class:`datetime.date` objects.

        Namely, the date part of Timestamps without time and
        timezone information.

        Examples
        --------
        For Series:

        >>> s = pd.Series(["1/1/2020 10:00:00+00:00", "2/1/2020 11:00:00+00:00"])
        >>> s = pd.to_datetime(s)
        >>> s
        0   2020-01-01 10:00:00+00:00
        1   2020-02-01 11:00:00+00:00
        dtype: datetime64[ns, UTC]
        >>> s.dt.date
        0    2020-01-01
        1    2020-02-01
        dtype: object

        For DatetimeIndex:

        >>> idx = pd.DatetimeIndex(["1/1/2020 10:00:00+00:00",
        ...                         "2/1/2020 11:00:00+00:00"])
        >>> idx.date
        array([datetime.date(2020, 1, 1), datetime.date(2020, 2, 1)], dtype=object)
        '''
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
    year: Incomplete
    month: Incomplete
    day: Incomplete
    hour: Incomplete
    minute: Incomplete
    second: Incomplete
    microsecond: Incomplete
    nanosecond: Incomplete
    _dayofweek_doc: str
    day_of_week: Incomplete
    dayofweek = day_of_week
    weekday = day_of_week
    day_of_year: Incomplete
    dayofyear = day_of_year
    quarter: Incomplete
    days_in_month: Incomplete
    daysinmonth = days_in_month
    _is_month_doc: str
    is_month_start: Incomplete
    is_month_end: Incomplete
    is_quarter_start: Incomplete
    is_quarter_end: Incomplete
    is_year_start: Incomplete
    is_year_end: Incomplete
    is_leap_year: Incomplete
    def to_julian_date(self) -> npt.NDArray[np.float64]:
        """
        Convert Datetime Array to float64 ndarray of Julian Dates.
        0 Julian date is noon January 1, 4713 BC.
        https://en.wikipedia.org/wiki/Julian_day
        """
    def std(self, axis: Incomplete | None = None, dtype: Incomplete | None = None, out: Incomplete | None = None, ddof: int = 1, keepdims: bool = False, skipna: bool = True):
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

def _sequence_to_dt64(data: ArrayLike, *, copy: bool = False, tz: tzinfo | None = None, dayfirst: bool = False, yearfirst: bool = False, ambiguous: TimeAmbiguous = 'raise', out_unit: str | None = None):
    '''
    Parameters
    ----------
    data : np.ndarray or ExtensionArray
        dtl.ensure_arraylike_for_datetimelike has already been called.
    copy : bool, default False
    tz : tzinfo or None, default None
    dayfirst : bool, default False
    yearfirst : bool, default False
    ambiguous : str, bool, or arraylike, default \'raise\'
        See pandas._libs.tslibs.tzconversion.tz_localize_to_utc.
    out_unit : str or None, default None
        Desired output resolution.

    Returns
    -------
    result : numpy.ndarray
        The sequence converted to a numpy array with dtype ``datetime64[unit]``.
        Where `unit` is "ns" unless specified otherwise by `out_unit`.
    tz : tzinfo or None
        Either the user-provided tzinfo or one inferred from the data.

    Raises
    ------
    TypeError : PeriodDType data is passed
    '''
def _construct_from_dt64_naive(data: np.ndarray, *, tz: tzinfo | None, copy: bool, ambiguous: TimeAmbiguous) -> tuple[np.ndarray, bool]:
    """
    Convert datetime64 data to a supported dtype, localizing if necessary.
    """
def objects_to_datetime64(data: np.ndarray, dayfirst, yearfirst, utc: bool = False, errors: DateTimeErrorChoices = 'raise', allow_object: bool = False, out_unit: str = 'ns'):
    '''
    Convert data to array of timestamps.

    Parameters
    ----------
    data : np.ndarray[object]
    dayfirst : bool
    yearfirst : bool
    utc : bool, default False
        Whether to convert/localize timestamps to UTC.
    errors : {\'raise\', \'ignore\', \'coerce\'}
    allow_object : bool
        Whether to return an object-dtype ndarray instead of raising if the
        data contains more than one timezone.
    out_unit : str, default "ns"

    Returns
    -------
    result : ndarray
        np.datetime64[out_unit] if returned values represent wall times or UTC
        timestamps.
        object if mixed timezones
    inferred_tz : tzinfo or None
        If not None, then the datetime64 values in `result` denote UTC timestamps.

    Raises
    ------
    ValueError : if data cannot be converted to datetimes
    TypeError  : When a type cannot be converted to datetime
    '''
def maybe_convert_dtype(data, copy: bool, tz: tzinfo | None = None):
    """
    Convert data based on dtype conventions, issuing
    errors where appropriate.

    Parameters
    ----------
    data : np.ndarray or pd.Index
    copy : bool
    tz : tzinfo or None, default None

    Returns
    -------
    data : np.ndarray or pd.Index
    copy : bool

    Raises
    ------
    TypeError : PeriodDType data is passed
    """
def _maybe_infer_tz(tz: tzinfo | None, inferred_tz: tzinfo | None) -> tzinfo | None:
    """
    If a timezone is inferred from data, check that it is compatible with
    the user-provided timezone, if any.

    Parameters
    ----------
    tz : tzinfo or None
    inferred_tz : tzinfo or None

    Returns
    -------
    tz : tzinfo or None

    Raises
    ------
    TypeError : if both timezones are present but do not match
    """
def _validate_dt64_dtype(dtype):
    """
    Check that a dtype, if passed, represents either a numpy datetime64[ns]
    dtype or a pandas DatetimeTZDtype.

    Parameters
    ----------
    dtype : object

    Returns
    -------
    dtype : None, numpy.dtype, or DatetimeTZDtype

    Raises
    ------
    ValueError : invalid dtype

    Notes
    -----
    Unlike _validate_tz_from_dtype, this does _not_ allow non-existent
    tz errors to go through
    """
def _validate_tz_from_dtype(dtype, tz: tzinfo | None, explicit_tz_none: bool = False) -> tzinfo | None:
    """
    If the given dtype is a DatetimeTZDtype, extract the implied
    tzinfo object from it and check that it does not conflict with the given
    tz.

    Parameters
    ----------
    dtype : dtype, str
    tz : None, tzinfo
    explicit_tz_none : bool, default False
        Whether tz=None was passed explicitly, as opposed to lib.no_default.

    Returns
    -------
    tz : consensus tzinfo

    Raises
    ------
    ValueError : on tzinfo mismatch
    """
def _infer_tz_from_endpoints(start: Timestamp, end: Timestamp, tz: tzinfo | None) -> tzinfo | None:
    """
    If a timezone is not explicitly given via `tz`, see if one can
    be inferred from the `start` and `end` endpoints.  If more than one
    of these inputs provides a timezone, require that they all agree.

    Parameters
    ----------
    start : Timestamp
    end : Timestamp
    tz : tzinfo or None

    Returns
    -------
    tz : tzinfo or None

    Raises
    ------
    TypeError : if start and end timezones do not agree
    """
def _maybe_normalize_endpoints(start: Timestamp | None, end: Timestamp | None, normalize: bool): ...
def _maybe_localize_point(ts: Timestamp | None, freq, tz, ambiguous, nonexistent) -> Timestamp | None:
    """
    Localize a start or end Timestamp to the timezone of the corresponding
    start or end Timestamp

    Parameters
    ----------
    ts : start or end Timestamp to potentially localize
    freq : Tick, DateOffset, or None
    tz : str, timezone object or None
    ambiguous: str, localization behavior for ambiguous times
    nonexistent: str, localization behavior for nonexistent times

    Returns
    -------
    ts : Timestamp
    """
def _generate_range(start: Timestamp | None, end: Timestamp | None, periods: int | None, offset: BaseOffset, *, unit: str):
    """
    Generates a sequence of dates corresponding to the specified time
    offset. Similar to dateutil.rrule except uses pandas DateOffset
    objects to represent time increments.

    Parameters
    ----------
    start : Timestamp or None
    end : Timestamp or None
    periods : int or None
    offset : DateOffset
    unit : str

    Notes
    -----
    * This method is faster for generating weekdays than dateutil.rrule
    * At least two of (start, end, periods) must be specified.
    * If both start and end are specified, the returned dates will
    satisfy start <= date <= end.

    Returns
    -------
    dates : generator object
    """
