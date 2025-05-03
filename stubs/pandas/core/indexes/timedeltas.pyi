import _abc
import pandas._libs.index as libindex
import pandas._libs.lib
import pandas._libs.lib as lib
import pandas.core.arrays.timedeltas
import pandas.core.common as com
import pandas.core.indexes.datetimelike
from _typeshed import Incomplete
from pandas._libs.lib import is_scalar as is_scalar
from pandas._libs.tslibs.dtypes import Resolution as Resolution
from pandas._libs.tslibs.offsets import to_offset as to_offset
from pandas._libs.tslibs.timedeltas import Timedelta as Timedelta, disallow_ambiguous_unit as disallow_ambiguous_unit
from pandas.core.arrays.timedeltas import TimedeltaArray as TimedeltaArray
from pandas.core.dtypes.common import pandas_dtype as pandas_dtype
from pandas.core.dtypes.generic import ABCSeries as ABCSeries
from pandas.core.indexes.base import Index as Index, maybe_extract_name as maybe_extract_name
from pandas.core.indexes.datetimelike import DatetimeTimedeltaMixin as DatetimeTimedeltaMixin
from pandas.core.indexes.extension import inherit_names as inherit_names
from pandas.util._exceptions import find_stack_level as find_stack_level
from typing import ClassVar

TYPE_CHECKING: bool

class TimedeltaIndex(pandas.core.indexes.datetimelike.DatetimeTimedeltaMixin):
    _typ: ClassVar[str] = ...
    _data_cls: ClassVar[type[pandas.core.arrays.timedeltas.TimedeltaArray]] = ...
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    components: Incomplete
    days: Incomplete
    seconds: Incomplete
    microseconds: Incomplete
    nanoseconds: Incomplete
    def _get_string_slice(self, key: str_t): ...
    @classmethod
    def __init__(cls, data, unit: pandas._libs.lib._NoDefault = ..., freq: pandas._libs.lib._NoDefault = ..., closed: pandas._libs.lib._NoDefault = ..., dtype, copy: bool = ..., name) -> None: ...
    def _is_comparable_dtype(self, dtype: DtypeObj) -> bool:
        """
        Can we compare values of the given dtype to our own?
        """
    def get_loc(self, key):
        """
        Get integer location for requested label

        Returns
        -------
        loc : int, slice, or ndarray[int]
        """
    def _parse_with_reso(self, label: str): ...
    def _parsed_string_to_bounds(self, reso, parsed: Timedelta): ...
    def to_pytimedelta(self, *args, **kwargs):
        """
        Return an ndarray of datetime.timedelta objects.

        Returns
        -------
        numpy.ndarray

        Examples
        --------
        >>> tdelta_idx = pd.to_timedelta([1, 2, 3], unit='D')
        >>> tdelta_idx
        TimedeltaIndex(['1 days', '2 days', '3 days'],
                        dtype='timedelta64[ns]', freq=None)
        >>> tdelta_idx.to_pytimedelta()
        array([datetime.timedelta(days=1), datetime.timedelta(days=2),
               datetime.timedelta(days=3)], dtype=object)
        """
    def sum(self, *args, **kwargs): ...
    def std(self, *args, **kwargs): ...
    def median(self, *args, **kwargs): ...
    def __neg__(self, *args, **kwargs): ...
    def __pos__(self, *args, **kwargs): ...
    def __abs__(self, *args, **kwargs): ...
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
    @property
    def _engine_type(self): ...
    @property
    def _resolution_obj(self): ...
    @property
    def inferred_type(self): ...
def timedelta_range(start, end, periods: int | None, freq, name, closed, *, unit: str | None) -> TimedeltaIndex:
    '''
    Return a fixed frequency TimedeltaIndex with day as the default.

    Parameters
    ----------
    start : str or timedelta-like, default None
        Left bound for generating timedeltas.
    end : str or timedelta-like, default None
        Right bound for generating timedeltas.
    periods : int, default None
        Number of periods to generate.
    freq : str, Timedelta, datetime.timedelta, or DateOffset, default \'D\'
        Frequency strings can have multiples, e.g. \'5h\'.
    name : str, default None
        Name of the resulting TimedeltaIndex.
    closed : str, default None
        Make the interval closed with respect to the given frequency to
        the \'left\', \'right\', or both sides (None).
    unit : str, default None
        Specify the desired resolution of the result.

        .. versionadded:: 2.0.0

    Returns
    -------
    TimedeltaIndex

    Notes
    -----
    Of the four parameters ``start``, ``end``, ``periods``, and ``freq``,
    exactly three must be specified. If ``freq`` is omitted, the resulting
    ``TimedeltaIndex`` will have ``periods`` linearly spaced elements between
    ``start`` and ``end`` (closed on both sides).

    To learn more about the frequency strings, please see `this link
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__.

    Examples
    --------
    >>> pd.timedelta_range(start=\'1 day\', periods=4)
    TimedeltaIndex([\'1 days\', \'2 days\', \'3 days\', \'4 days\'],
                   dtype=\'timedelta64[ns]\', freq=\'D\')

    The ``closed`` parameter specifies which endpoint is included.  The default
    behavior is to include both endpoints.

    >>> pd.timedelta_range(start=\'1 day\', periods=4, closed=\'right\')
    TimedeltaIndex([\'2 days\', \'3 days\', \'4 days\'],
                   dtype=\'timedelta64[ns]\', freq=\'D\')

    The ``freq`` parameter specifies the frequency of the TimedeltaIndex.
    Only fixed frequencies can be passed, non-fixed frequencies such as
    \'M\' (month end) will raise.

    >>> pd.timedelta_range(start=\'1 day\', end=\'2 days\', freq=\'6h\')
    TimedeltaIndex([\'1 days 00:00:00\', \'1 days 06:00:00\', \'1 days 12:00:00\',
                    \'1 days 18:00:00\', \'2 days 00:00:00\'],
                   dtype=\'timedelta64[ns]\', freq=\'6h\')

    Specify ``start``, ``end``, and ``periods``; the frequency is generated
    automatically (linearly spaced).

    >>> pd.timedelta_range(start=\'1 day\', end=\'5 days\', periods=4)
    TimedeltaIndex([\'1 days 00:00:00\', \'2 days 08:00:00\', \'3 days 16:00:00\',
                    \'5 days 00:00:00\'],
                   dtype=\'timedelta64[ns]\', freq=None)

    **Specify a unit**

    >>> pd.timedelta_range("1 Day", periods=3, freq="100000D", unit="s")
    TimedeltaIndex([\'1 days\', \'100001 days\', \'200001 days\'],
                   dtype=\'timedelta64[s]\', freq=\'100000D\')
    '''
