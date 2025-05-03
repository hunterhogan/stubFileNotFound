import _abc
import np
import npt
import pandas._libs.index as libindex
import pandas.core.arrays.period
import pandas.core.common as com
import pandas.core.indexes.base as ibase
import pandas.core.indexes.datetimelike
from _typeshed import Incomplete
from datetime import datetime
from pandas._libs.lib import is_integer as is_integer
from pandas._libs.properties import cache_readonly as cache_readonly
from pandas._libs.tslibs.dtypes import Resolution as Resolution
from pandas._libs.tslibs.nattype import NaT as NaT
from pandas._libs.tslibs.offsets import BaseOffset as BaseOffset, Tick as Tick
from pandas._libs.tslibs.period import Period as Period
from pandas.core.arrays.period import PeriodArray as PeriodArray, period_array as period_array, raise_on_incompatible as raise_on_incompatible, validate_dtype_freq as validate_dtype_freq
from pandas.core.dtypes.dtypes import PeriodDtype as PeriodDtype
from pandas.core.dtypes.generic import ABCSeries as ABCSeries
from pandas.core.dtypes.missing import is_valid_na_for_dtype as is_valid_na_for_dtype
from pandas.core.indexes.base import Index as Index, maybe_extract_name as maybe_extract_name
from pandas.core.indexes.datetimelike import DatetimeIndexOpsMixin as DatetimeIndexOpsMixin
from pandas.core.indexes.datetimes import DatetimeIndex as DatetimeIndex
from pandas.core.indexes.extension import inherit_names as inherit_names
from pandas.util._decorators import doc as doc
from pandas.util._exceptions import find_stack_level as find_stack_level
from typing import ClassVar

TYPE_CHECKING: bool
OFFSET_TO_PERIOD_FREQSTR: dict
_index_doc_kwargs: dict
_shared_doc_kwargs: dict
def _new_PeriodIndex(cls, **d): ...

class PeriodIndex(pandas.core.indexes.datetimelike.DatetimeIndexOpsMixin):
    _typ: ClassVar[str] = ...
    _data_cls: ClassVar[type[pandas.core.arrays.period.PeriodArray]] = ...
    _supports_partial_string_indexing: ClassVar[bool] = ...
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    _resolution_obj: Incomplete
    hour: Incomplete
    minute: Incomplete
    second: Incomplete
    is_leap_year: Incomplete
    start_time: Incomplete
    end_time: Incomplete
    year: Incomplete
    month: Incomplete
    day: Incomplete
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
    def asfreq(self, freq, how: str = ...) -> Self:
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
        pandas.arrays.PeriodArray.asfreq: Convert each Period in a PeriodArray to the given frequency.
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
    def to_timestamp(self, freq, how: str = ...) -> DatetimeIndex:
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
    @classmethod
    def __init__(cls, data, ordinal, freq, dtype: Dtype | None, copy: bool = ..., name: Hashable | None, **fields) -> Self: ...
    @classmethod
    def from_fields(cls, *, year, quarter, month, day, hour, minute, second, freq) -> Self: ...
    @classmethod
    def from_ordinals(cls, ordinals, *, freq, name) -> Self: ...
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
    def _parsed_string_to_bounds(self, reso: Resolution, parsed: datetime): ...
    def shift(self, periods: int = ..., freq) -> Self:
        """
        Shift index by desired number of time frequency increments.

        This method is for shifting the values of datetime-like indexes
        by a specified time increment a given number of times.

        Parameters
        ----------
        periods : int, default 1
            Number of periods (or increments) to shift by,
            can be positive or negative.
        freq : pandas.DateOffset, pandas.Timedelta or string, optional
            Frequency increment to shift by.
            If None, the index is shifted by its own `freq` attribute.
            Offset aliases are valid strings, e.g., 'D', 'W', 'M' etc.

        Returns
        -------
        pandas.DatetimeIndex
            Shifted index.

        See Also
        --------
        Index.shift : Shift values of Index.
        PeriodIndex.shift : Shift values of PeriodIndex.
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
    @property
    def _engine_type(self): ...
    @property
    def values(self): ...
    @property
    def is_full(self): ...
    @property
    def inferred_type(self): ...
def period_range(start, end, periods: int | None, freq, name: Hashable | None) -> PeriodIndex:
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
