import numpy as np
from _typeshed import Incomplete
from collections.abc import Hashable
from datetime import datetime
from pandas._libs import index as libindex
from pandas._libs.tslibs import BaseOffset as BaseOffset, NaT as NaT, Period as Period, Resolution as Resolution, Tick as Tick
from pandas._typing import Dtype as Dtype, DtypeObj as DtypeObj, Self as Self, npt as npt
from pandas.core.arrays.period import PeriodArray as PeriodArray, period_array as period_array, raise_on_incompatible as raise_on_incompatible, validate_dtype_freq as validate_dtype_freq
from pandas.core.dtypes.dtypes import PeriodDtype as PeriodDtype
from pandas.core.indexes.datetimelike import DatetimeIndexOpsMixin as DatetimeIndexOpsMixin
from pandas.core.indexes.datetimes import DatetimeIndex as DatetimeIndex, Index as Index
from pandas.util._decorators import cache_readonly as cache_readonly, doc as doc

_index_doc_kwargs: Incomplete
_shared_doc_kwargs: Incomplete

def _new_PeriodIndex(cls, **d): ...

class PeriodIndex(DatetimeIndexOpsMixin):
    """
    Immutable ndarray holding ordinal values indicating regular periods in time.

    Index keys are boxed to Period objects which carries the metadata (eg,
    frequency information).

    Parameters
    ----------
    data : array-like (1d int np.ndarray or PeriodArray), optional
        Optional period-like data to construct index with.
    copy : bool
        Make a copy of input ndarray.
    freq : str or period object, optional
        One of pandas period strings or corresponding objects.
    year : int, array, or Series, default None

        .. deprecated:: 2.2.0
           Use PeriodIndex.from_fields instead.
    month : int, array, or Series, default None

        .. deprecated:: 2.2.0
           Use PeriodIndex.from_fields instead.
    quarter : int, array, or Series, default None

        .. deprecated:: 2.2.0
           Use PeriodIndex.from_fields instead.
    day : int, array, or Series, default None

        .. deprecated:: 2.2.0
           Use PeriodIndex.from_fields instead.
    hour : int, array, or Series, default None

        .. deprecated:: 2.2.0
           Use PeriodIndex.from_fields instead.
    minute : int, array, or Series, default None

        .. deprecated:: 2.2.0
           Use PeriodIndex.from_fields instead.
    second : int, array, or Series, default None

        .. deprecated:: 2.2.0
           Use PeriodIndex.from_fields instead.
    dtype : str or PeriodDtype, default None

    Attributes
    ----------
    day
    dayofweek
    day_of_week
    dayofyear
    day_of_year
    days_in_month
    daysinmonth
    end_time
    freq
    freqstr
    hour
    is_leap_year
    minute
    month
    quarter
    qyear
    second
    start_time
    week
    weekday
    weekofyear
    year

    Methods
    -------
    asfreq
    strftime
    to_timestamp
    from_fields
    from_ordinals

    See Also
    --------
    Index : The base pandas Index type.
    Period : Represents a period of time.
    DatetimeIndex : Index with datetime64 data.
    TimedeltaIndex : Index of timedelta64 data.
    period_range : Create a fixed-frequency PeriodIndex.

    Examples
    --------
    >>> idx = pd.PeriodIndex.from_fields(year=[2000, 2002], quarter=[1, 3])
    >>> idx
    PeriodIndex(['2000Q1', '2002Q3'], dtype='period[Q-DEC]')
    """
    _typ: str
    _data: PeriodArray
    freq: BaseOffset
    dtype: PeriodDtype
    _data_cls = PeriodArray
    _supports_partial_string_indexing: bool
    @property
    def _engine_type(self) -> type[libindex.PeriodEngine]: ...
    def _resolution_obj(self) -> Resolution: ...
    def asfreq(self, freq: Incomplete | None = None, how: str = 'E') -> Self: ...
    def to_timestamp(self, freq: Incomplete | None = None, how: str = 'start') -> DatetimeIndex: ...
    @property
    def hour(self) -> Index: ...
    @property
    def minute(self) -> Index: ...
    @property
    def second(self) -> Index: ...
    def __new__(cls, data: Incomplete | None = None, ordinal: Incomplete | None = None, freq: Incomplete | None = None, dtype: Dtype | None = None, copy: bool = False, name: Hashable | None = None, **fields) -> Self: ...
    @classmethod
    def from_fields(cls, *, year: Incomplete | None = None, quarter: Incomplete | None = None, month: Incomplete | None = None, day: Incomplete | None = None, hour: Incomplete | None = None, minute: Incomplete | None = None, second: Incomplete | None = None, freq: Incomplete | None = None) -> Self: ...
    @classmethod
    def from_ordinals(cls, ordinals, *, freq, name: Incomplete | None = None) -> Self: ...
    @property
    def values(self) -> npt.NDArray[np.object_]: ...
    def _maybe_convert_timedelta(self, other) -> int | npt.NDArray[np.int64]:
        """
        Convert timedelta-like input to an integer multiple of self.freq

        Parameters
        ----------
        other : timedelta, np.timedelta64, DateOffset, int, np.ndarray

        Returns
        -------
        converted : int, np.ndarray[int64]

        Raises
        ------
        IncompatibleFrequency : if the input cannot be written as a multiple
            of self.freq.  Note IncompatibleFrequency subclasses ValueError.
        """
    def _is_comparable_dtype(self, dtype: DtypeObj) -> bool:
        """
        Can we compare values of the given dtype to our own?
        """
    def asof_locs(self, where: Index, mask: npt.NDArray[np.bool_]) -> np.ndarray:
        """
        where : array of timestamps
        mask : np.ndarray[bool]
            Array of booleans where data is not NA.
        """
    @property
    def is_full(self) -> bool:
        """
        Returns True if this PeriodIndex is range-like in that all Periods
        between start and end are present, in order.
        """
    @property
    def inferred_type(self) -> str: ...
    def _convert_tolerance(self, tolerance, target): ...
    def get_loc(self, key):
        """
        Get integer location for requested label.

        Parameters
        ----------
        key : Period, NaT, str, or datetime
            String or datetime key must be parsable as Period.

        Returns
        -------
        loc : int or ndarray[int64]

        Raises
        ------
        KeyError
            Key is not present in the index.
        TypeError
            If key is listlike or otherwise not hashable.
        """
    def _disallow_mismatched_indexing(self, key: Period) -> None: ...
    def _cast_partial_indexing_scalar(self, label: datetime) -> Period: ...
    def _maybe_cast_slice_bound(self, label, side: str): ...
    def _parsed_string_to_bounds(self, reso: Resolution, parsed: datetime): ...
    def shift(self, periods: int = 1, freq: Incomplete | None = None) -> Self: ...

def period_range(start: Incomplete | None = None, end: Incomplete | None = None, periods: int | None = None, freq: Incomplete | None = None, name: Hashable | None = None) -> PeriodIndex:
    '''
    Return a fixed frequency PeriodIndex.

    The day (calendar) is the default frequency.

    Parameters
    ----------
    start : str, datetime, date, pandas.Timestamp, or period-like, default None
        Left bound for generating periods.
    end : str, datetime, date, pandas.Timestamp, or period-like, default None
        Right bound for generating periods.
    periods : int, default None
        Number of periods to generate.
    freq : str or DateOffset, optional
        Frequency alias. By default the freq is taken from `start` or `end`
        if those are Period objects. Otherwise, the default is ``"D"`` for
        daily frequency.
    name : str, default None
        Name of the resulting PeriodIndex.

    Returns
    -------
    PeriodIndex

    Notes
    -----
    Of the three parameters: ``start``, ``end``, and ``periods``, exactly two
    must be specified.

    To learn more about the frequency strings, please see `this link
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__.

    Examples
    --------
    >>> pd.period_range(start=\'2017-01-01\', end=\'2018-01-01\', freq=\'M\')
    PeriodIndex([\'2017-01\', \'2017-02\', \'2017-03\', \'2017-04\', \'2017-05\', \'2017-06\',
             \'2017-07\', \'2017-08\', \'2017-09\', \'2017-10\', \'2017-11\', \'2017-12\',
             \'2018-01\'],
            dtype=\'period[M]\')

    If ``start`` or ``end`` are ``Period`` objects, they will be used as anchor
    endpoints for a ``PeriodIndex`` with frequency matching that of the
    ``period_range`` constructor.

    >>> pd.period_range(start=pd.Period(\'2017Q1\', freq=\'Q\'),
    ...                 end=pd.Period(\'2017Q2\', freq=\'Q\'), freq=\'M\')
    PeriodIndex([\'2017-03\', \'2017-04\', \'2017-05\', \'2017-06\'],
                dtype=\'period[M]\')
    '''
