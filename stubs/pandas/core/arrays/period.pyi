import np
import npt
import numpy
import pandas._libs.algos as libalgos
import pandas._libs.lib as lib
import pandas._libs.tslibs.parsing as parsing
import pandas._libs.tslibs.period
import pandas._libs.tslibs.period as libperiod
import pandas.core.arrays.datetimelike
import pandas.core.arrays.datetimelike as dtl
import pandas.core.common as com
import typing
from _typeshed import Incomplete
from datetime import timedelta
from pandas._libs.algos import ensure_object as ensure_object
from pandas._libs.arrays import NDArrayBacked as NDArrayBacked
from pandas._libs.properties import cache_readonly as cache_readonly
from pandas._libs.tslibs.dtypes import FreqGroup as FreqGroup, PeriodDtypeBase as PeriodDtypeBase, freq_to_period_freqstr as freq_to_period_freqstr
from pandas._libs.tslibs.fields import isleapyear_arr as isleapyear_arr
from pandas._libs.tslibs.nattype import NaT as NaT, NaTType as NaTType
from pandas._libs.tslibs.np_datetime import add_overflowsafe as add_overflowsafe, astype_overflowsafe as astype_overflowsafe, get_unit_from_dtype as get_unit_from_dtype
from pandas._libs.tslibs.offsets import BaseOffset as BaseOffset, Tick as Tick, delta_to_tick as delta_to_tick, to_offset as to_offset
from pandas._libs.tslibs.period import IncompatibleFrequency as IncompatibleFrequency, Period as Period, get_period_field_arr as get_period_field_arr, period_asfreq_arr as period_asfreq_arr
from pandas._libs.tslibs.timedeltas import Timedelta as Timedelta
from pandas._libs.tslibs.vectorized import c_dt64arr_to_periodarr as c_dt64arr_to_periodarr
from pandas.core.dtypes.common import pandas_dtype as pandas_dtype
from pandas.core.dtypes.dtypes import DatetimeTZDtype as DatetimeTZDtype, PeriodDtype as PeriodDtype
from pandas.core.dtypes.generic import ABCIndex as ABCIndex, ABCPeriodIndex as ABCPeriodIndex, ABCSeries as ABCSeries, ABCTimedeltaArray as ABCTimedeltaArray
from pandas.core.dtypes.missing import isna as isna
from pandas.util._decorators import doc as doc
from pandas.util._exceptions import find_stack_level as find_stack_level
from typing import Any, Callable, ClassVar, Literal

TYPE_CHECKING: bool
iNaT: int
DIFFERENT_FREQ: str
BaseOffsetT: typing.TypeVar
_shared_doc_kwargs: dict
def _field_accessor(name: str, docstring: str | None): ...

class PeriodArray(pandas.core.arrays.datetimelike.DatelikeOps, pandas._libs.tslibs.period.PeriodMixin):
    __array_priority__: ClassVar[int] = ...
    _typ: ClassVar[str] = ...
    _internal_fill_value: ClassVar[numpy.int64] = ...
    _recognized_scalars: ClassVar[tuple] = ...
    _infer_matches: ClassVar[tuple] = ...
    _other_ops: ClassVar[list] = ...
    _bool_ops: ClassVar[list] = ...
    _object_ops: ClassVar[list] = ...
    _field_ops: ClassVar[list] = ...
    _datetimelike_ops: ClassVar[list] = ...
    _datetimelike_methods: ClassVar[list] = ...
    dtype: Incomplete
    def _is_recognized_dtype(self, x): ...
    def __init__(self, values, dtype: Dtype | None, freq, copy: bool = ...) -> None: ...
    @classmethod
    def _simple_new(cls, values: npt.NDArray[np.int64], dtype: PeriodDtype) -> Self: ...
    @classmethod
    def _from_sequence(cls, scalars, *, dtype: Dtype | None, copy: bool = ...) -> Self: ...
    @classmethod
    def _from_sequence_of_strings(cls, strings, *, dtype: Dtype | None, copy: bool = ...) -> Self: ...
    @classmethod
    def _from_datetime64(cls, data, freq, tz) -> Self:
        """
        Construct a PeriodArray from a datetime64 array

        Parameters
        ----------
        data : ndarray[datetime64[ns], datetime64[ns, tz]]
        freq : str or Tick
        tz : tzinfo, optional

        Returns
        -------
        PeriodArray[freq]
        """
    @classmethod
    def _generate_range(cls, start, end, periods, freq): ...
    @classmethod
    def _from_fields(cls, *, fields: dict, freq) -> Self: ...
    def _unbox_scalar(self, value: Period | NaTType) -> np.int64: ...
    def _scalar_from_string(self, value: str) -> Period: ...
    def _check_compatible_with(self, other: Period | NaTType | PeriodArray) -> None: ...
    def __array__(self, dtype: NpDtype | None, copy: bool | None) -> np.ndarray: ...
    def __arrow_array__(self, type):
        """
        Convert myself into a pyarrow Array.
        """
    def to_timestamp(self, freq, how: str = ...) -> DatetimeArray:
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
    def _box_func(self, x) -> Period | NaTType: ...
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
    def _formatter(self, boxed: bool = ...): ...
    def _format_native_types(self, *, na_rep: str | float = ..., date_format, **kwargs) -> npt.NDArray[np.object_]:
        """
        actually format my specific types
        """
    def astype(self, dtype, copy: bool = ...): ...
    def searchsorted(self, value: NumpyValueArrayLike | ExtensionArray, side: Literal['left', 'right'] = ..., sorter: NumpySorter | None) -> npt.NDArray[np.intp] | np.intp: ...
    def _pad_or_backfill(self, *, method: FillnaOptions, limit: int | None, limit_area: Literal['inside', 'outside'] | None, copy: bool = ...) -> Self: ...
    def fillna(self, value, method, limit: int | None, copy: bool = ...) -> Self: ...
    def _addsub_int_array_or_scalar(self, other: np.ndarray | int, op: Callable[[Any, Any], Any]) -> Self:
        """
        Add or subtract array of integers.

        Parameters
        ----------
        other : np.ndarray[int64] or int
        op : {operator.add, operator.sub}

        Returns
        -------
        result : PeriodArray
        """
    def _add_offset(self, other: BaseOffset): ...
    def _add_timedeltalike_scalar(self, other):
        """
        Parameters
        ----------
        other : timedelta, Tick, np.timedelta64

        Returns
        -------
        PeriodArray
        """
    def _add_timedelta_arraylike(self, other: TimedeltaArray | npt.NDArray[np.timedelta64]) -> Self:
        """
        Parameters
        ----------
        other : TimedeltaArray or ndarray[timedelta64]

        Returns
        -------
        PeriodArray
        """
    def _check_timedeltalike_freq_compat(self, other):
        """
        Arithmetic operations with timedelta-like scalars or array `other`
        are only valid if `other` is an integer multiple of `self.freq`.
        If the operation is valid, find that integer multiple.  Otherwise,
        raise because the operation is invalid.

        Parameters
        ----------
        other : timedelta, np.timedelta64, Tick,
                ndarray[timedelta64], TimedeltaArray, TimedeltaIndex

        Returns
        -------
        multiple : int or ndarray[int64]

        Raises
        ------
        IncompatibleFrequency
        """
    @property
    def _scalar_type(self): ...
    @property
    def freq(self): ...
    @property
    def freqstr(self): ...
    @property
    def year(self): ...
    @property
    def month(self): ...
    @property
    def day(self): ...
    @property
    def hour(self): ...
    @property
    def minute(self): ...
    @property
    def second(self): ...
    @property
    def weekofyear(self): ...
    @property
    def week(self): ...
    @property
    def day_of_week(self): ...
    @property
    def dayofweek(self): ...
    @property
    def weekday(self): ...
    @property
    def dayofyear(self): ...
    @property
    def day_of_year(self): ...
    @property
    def quarter(self): ...
    @property
    def qyear(self): ...
    @property
    def days_in_month(self): ...
    @property
    def daysinmonth(self): ...
    @property
    def is_leap_year(self): ...
def raise_on_incompatible(left, right) -> IncompatibleFrequency:
    """
    Helper function to render a consistent error message when raising
    IncompatibleFrequency.

    Parameters
    ----------
    left : PeriodArray
    right : None, DateOffset, Period, ndarray, or timedelta-like

    Returns
    -------
    IncompatibleFrequency
        Exception to be raised by the caller.
    """
def period_array(data: Sequence[Period | str | None] | AnyArrayLike, freq: str | Tick | BaseOffset | None, copy: bool = ...) -> PeriodArray:
    """
    Construct a new PeriodArray from a sequence of Period scalars.

    Parameters
    ----------
    data : Sequence of Period objects
        A sequence of Period objects. These are required to all have
        the same ``freq.`` Missing values can be indicated by ``None``
        or ``pandas.NaT``.
    freq : str, Tick, or Offset
        The frequency of every element of the array. This can be specified
        to avoid inferring the `freq` from `data`.
    copy : bool, default False
        Whether to ensure a copy of the data is made.

    Returns
    -------
    PeriodArray

    See Also
    --------
    PeriodArray
    pandas.PeriodIndex

    Examples
    --------
    >>> period_array([pd.Period('2017', freq='Y'),
    ...               pd.Period('2018', freq='Y')])
    <PeriodArray>
    ['2017', '2018']
    Length: 2, dtype: period[Y-DEC]

    >>> period_array([pd.Period('2017', freq='Y'),
    ...               pd.Period('2018', freq='Y'),
    ...               pd.NaT])
    <PeriodArray>
    ['2017', '2018', 'NaT']
    Length: 3, dtype: period[Y-DEC]

    Integers that look like years are handled

    >>> period_array([2000, 2001, 2002], freq='D')
    <PeriodArray>
    ['2000-01-01', '2001-01-01', '2002-01-01']
    Length: 3, dtype: period[D]

    Datetime-like strings may also be passed

    >>> period_array(['2000-Q1', '2000-Q2', '2000-Q3', '2000-Q4'], freq='Q')
    <PeriodArray>
    ['2000Q1', '2000Q2', '2000Q3', '2000Q4']
    Length: 4, dtype: period[Q-DEC]
    """
def validate_dtype_freq(dtype, freq: BaseOffsetT | BaseOffset | timedelta | str | None) -> BaseOffsetT:
    """
    If both a dtype and a freq are available, ensure they match.  If only
    dtype is available, extract the implied freq.

    Parameters
    ----------
    dtype : dtype
    freq : DateOffset or None

    Returns
    -------
    freq : DateOffset

    Raises
    ------
    ValueError : non-period dtype
    IncompatibleFrequency : mismatch between dtype and freq
    """
def dt64arr_to_periodarr(data, freq, tz) -> tuple[npt.NDArray[np.int64], BaseOffset]:
    """
    Convert an datetime-like array to values Period ordinals.

    Parameters
    ----------
    data : Union[Series[datetime64[ns]], DatetimeIndex, ndarray[datetime64ns]]
    freq : Optional[Union[str, Tick]]
        Must match the `freq` on the `data` if `data` is a DatetimeIndex
        or Series.
    tz : Optional[tzinfo]

    Returns
    -------
    ordinals : ndarray[int64]
    freq : Tick
        The frequency extracted from the Series or DatetimeIndex if that's
        used.

    """
def _get_ordinal_range(start, end, periods, freq, mult: int = ...): ...
def _range_from_fields(year, month, quarter, day, hour, minute, second, freq) -> tuple[np.ndarray, BaseOffset]: ...
def _make_field_arrays(*fields) -> list[np.ndarray]: ...
