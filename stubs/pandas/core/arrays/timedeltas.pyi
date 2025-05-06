import numpy as np
from _typeshed import Incomplete
from collections.abc import Iterator
from pandas import DataFrame as DataFrame
from pandas._libs import lib as lib, tslibs as tslibs
from pandas._libs.tslibs import NaT as NaT, NaTType as NaTType, Tick as Tick, Timedelta as Timedelta, astype_overflowsafe as astype_overflowsafe, get_supported_dtype as get_supported_dtype, iNaT as iNaT, is_supported_dtype as is_supported_dtype, periods_per_second as periods_per_second
from pandas._libs.tslibs.fields import get_timedelta_days as get_timedelta_days, get_timedelta_field as get_timedelta_field
from pandas._libs.tslibs.timedeltas import array_to_timedelta64 as array_to_timedelta64, floordiv_object_array as floordiv_object_array, ints_to_pytimedelta as ints_to_pytimedelta, parse_timedelta_unit as parse_timedelta_unit, truediv_object_array as truediv_object_array
from pandas._typing import AxisInt as AxisInt, DateTimeErrorChoices as DateTimeErrorChoices, DtypeObj as DtypeObj, NpDtype as NpDtype, Self as Self, npt as npt
from pandas.core import nanops as nanops, roperator as roperator
from pandas.core.arrays import datetimelike as dtl
from pandas.core.dtypes.common import TD64NS_DTYPE as TD64NS_DTYPE, is_float_dtype as is_float_dtype, is_integer_dtype as is_integer_dtype, is_object_dtype as is_object_dtype, is_scalar as is_scalar, is_string_dtype as is_string_dtype, pandas_dtype as pandas_dtype

def _field_accessor(name: str, alias: str, docstring: str): ...

class TimedeltaArray(dtl.TimelikeOps):
    '''
    Pandas ExtensionArray for timedelta data.

    .. warning::

       TimedeltaArray is currently experimental, and its API may change
       without warning. In particular, :attr:`TimedeltaArray.dtype` is
       expected to change to be an instance of an ``ExtensionDtype``
       subclass.

    Parameters
    ----------
    values : array-like
        The timedelta data.

    dtype : numpy.dtype
        Currently, only ``numpy.dtype("timedelta64[ns]")`` is accepted.
    freq : Offset, optional
    copy : bool, default False
        Whether to copy the underlying array of data.

    Attributes
    ----------
    None

    Methods
    -------
    None

    Examples
    --------
    >>> pd.arrays.TimedeltaArray._from_sequence(pd.TimedeltaIndex([\'1h\', \'2h\']))
    <TimedeltaArray>
    [\'0 days 01:00:00\', \'0 days 02:00:00\']
    Length: 2, dtype: timedelta64[ns]
    '''
    _typ: str
    _internal_fill_value: Incomplete
    _recognized_scalars: Incomplete
    _is_recognized_dtype: Incomplete
    _infer_matches: Incomplete
    @property
    def _scalar_type(self) -> type[Timedelta]: ...
    __array_priority__: int
    _other_ops: list[str]
    _bool_ops: list[str]
    _object_ops: list[str]
    _field_ops: list[str]
    _datetimelike_ops: list[str]
    _datetimelike_methods: list[str]
    def _box_func(self, x: np.timedelta64) -> Timedelta | NaTType: ...
    @property
    def dtype(self) -> np.dtype[np.timedelta64]:
        """
        The dtype for the TimedeltaArray.

        .. warning::

           A future version of pandas will change dtype to be an instance
           of a :class:`pandas.api.extensions.ExtensionDtype` subclass,
           not a ``numpy.dtype``.

        Returns
        -------
        numpy.dtype
        """
    _freq: Incomplete
    _default_dtype = TD64NS_DTYPE
    @classmethod
    def _validate_dtype(cls, values, dtype): ...
    @classmethod
    def _simple_new(cls, values: npt.NDArray[np.timedelta64], freq: Tick | None = None, dtype: np.dtype[np.timedelta64] = ...) -> Self: ...
    @classmethod
    def _from_sequence(cls, data, *, dtype: Incomplete | None = None, copy: bool = False) -> Self: ...
    @classmethod
    def _from_sequence_not_strict(cls, data, *, dtype: Incomplete | None = None, copy: bool = False, freq=..., unit: Incomplete | None = None) -> Self:
        """
        _from_sequence_not_strict but without responsibility for finding the
        result's `freq`.
        """
    @classmethod
    def _generate_range(cls, start, end, periods, freq, closed: Incomplete | None = None, *, unit: str | None = None) -> Self: ...
    def _unbox_scalar(self, value) -> np.timedelta64: ...
    def _scalar_from_string(self, value) -> Timedelta | NaTType: ...
    def _check_compatible_with(self, other) -> None: ...
    def astype(self, dtype, copy: bool = True): ...
    def __iter__(self) -> Iterator: ...
    def sum(self, *, axis: AxisInt | None = None, dtype: NpDtype | None = None, out: Incomplete | None = None, keepdims: bool = False, initial: Incomplete | None = None, skipna: bool = True, min_count: int = 0): ...
    def std(self, *, axis: AxisInt | None = None, dtype: NpDtype | None = None, out: Incomplete | None = None, ddof: int = 1, keepdims: bool = False, skipna: bool = True): ...
    def _accumulate(self, name: str, *, skipna: bool = True, **kwargs): ...
    def _formatter(self, boxed: bool = False): ...
    def _format_native_types(self, *, na_rep: str | float = 'NaT', date_format: Incomplete | None = None, **kwargs) -> npt.NDArray[np.object_]: ...
    def _add_offset(self, other) -> None: ...
    def __mul__(self, other) -> Self: ...
    __rmul__ = __mul__
    def _scalar_divlike_op(self, other, op):
        """
        Shared logic for __truediv__, __rtruediv__, __floordiv__, __rfloordiv__
        with scalar 'other'.
        """
    def _cast_divlike_op(self, other): ...
    def _vector_divlike_op(self, other, op) -> np.ndarray | Self:
        """
        Shared logic for __truediv__, __floordiv__, and their reversed versions
        with timedelta64-dtype ndarray other.
        """
    def __truediv__(self, other): ...
    def __rtruediv__(self, other): ...
    def __floordiv__(self, other): ...
    def __rfloordiv__(self, other): ...
    def __mod__(self, other): ...
    def __rmod__(self, other): ...
    def __divmod__(self, other): ...
    def __rdivmod__(self, other): ...
    def __neg__(self) -> TimedeltaArray: ...
    def __pos__(self) -> TimedeltaArray: ...
    def __abs__(self) -> TimedeltaArray: ...
    def total_seconds(self) -> npt.NDArray[np.float64]:
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
    def to_pytimedelta(self) -> npt.NDArray[np.object_]:
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
    days_docstring: Incomplete
    days: Incomplete
    seconds_docstring: Incomplete
    seconds: Incomplete
    microseconds_docstring: Incomplete
    microseconds: Incomplete
    nanoseconds_docstring: Incomplete
    nanoseconds: Incomplete
    @property
    def components(self) -> DataFrame:
        """
        Return a DataFrame of the individual resolution components of the Timedeltas.

        The components (days, hours, minutes seconds, milliseconds, microseconds,
        nanoseconds) are returned as columns in a DataFrame.

        Returns
        -------
        DataFrame

        Examples
        --------
        >>> tdelta_idx = pd.to_timedelta(['1 day 3 min 2 us 42 ns'])
        >>> tdelta_idx
        TimedeltaIndex(['1 days 00:03:00.000002042'],
                       dtype='timedelta64[ns]', freq=None)
        >>> tdelta_idx.components
           days  hours  minutes  seconds  milliseconds  microseconds  nanoseconds
        0     1      0        3        0             0             2           42
        """

def sequence_to_td64ns(data, copy: bool = False, unit: Incomplete | None = None, errors: DateTimeErrorChoices = 'raise') -> tuple[np.ndarray, Tick | None]:
    '''
    Parameters
    ----------
    data : list-like
    copy : bool, default False
    unit : str, optional
        The timedelta unit to treat integers as multiples of. For numeric
        data this defaults to ``\'ns\'``.
        Must be un-specified if the data contains a str and ``errors=="raise"``.
    errors : {"raise", "coerce", "ignore"}, default "raise"
        How to handle elements that cannot be converted to timedelta64[ns].
        See ``pandas.to_timedelta`` for details.

    Returns
    -------
    converted : numpy.ndarray
        The sequence converted to a numpy array with dtype ``timedelta64[ns]``.
    inferred_freq : Tick or None
        The inferred frequency of the sequence.

    Raises
    ------
    ValueError : Data cannot be converted to timedelta64[ns].

    Notes
    -----
    Unlike `pandas.to_timedelta`, if setting ``errors=ignore`` will not cause
    errors to be ignored; they are caught and subsequently ignored at a
    higher level.
    '''
def _ints_to_td64ns(data, unit: str = 'ns'):
    '''
    Convert an ndarray with integer-dtype to timedelta64[ns] dtype, treating
    the integers as multiples of the given timedelta unit.

    Parameters
    ----------
    data : numpy.ndarray with integer-dtype
    unit : str, default "ns"
        The timedelta unit to treat integers as multiples of.

    Returns
    -------
    numpy.ndarray : timedelta64[ns] array converted from data
    bool : whether a copy was made
    '''
def _objects_to_td64ns(data, unit: Incomplete | None = None, errors: DateTimeErrorChoices = 'raise'):
    '''
    Convert a object-dtyped or string-dtyped array into an
    timedelta64[ns]-dtyped array.

    Parameters
    ----------
    data : ndarray or Index
    unit : str, default "ns"
        The timedelta unit to treat integers as multiples of.
        Must not be specified if the data contains a str.
    errors : {"raise", "coerce", "ignore"}, default "raise"
        How to handle elements that cannot be converted to timedelta64[ns].
        See ``pandas.to_timedelta`` for details.

    Returns
    -------
    numpy.ndarray : timedelta64[ns] array converted from data

    Raises
    ------
    ValueError : Data cannot be converted to timedelta64[ns].

    Notes
    -----
    Unlike `pandas.to_timedelta`, if setting `errors=ignore` will not cause
    errors to be ignored; they are caught and subsequently ignored at a
    higher level.
    '''
def _validate_td64_dtype(dtype) -> DtypeObj: ...
