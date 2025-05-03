import np
import npt
import numpy
import numpy.dtypes
import pandas._libs.lib
import pandas._libs.lib as lib
import pandas._libs.tslibs as tslibs
import pandas.compat.numpy.function as nv
import pandas.core.array_algos.datetimelike_accumulations as datetimelike_accumulations
import pandas.core.arrays.datetimelike
import pandas.core.arrays.datetimelike as dtl
import pandas.core.common as com
import pandas.core.nanops as nanops
import pandas.core.roperator as roperator
from pandas._libs.lib import is_scalar as is_scalar
from pandas._libs.tslibs.conversion import cast_from_unit_vectorized as cast_from_unit_vectorized
from pandas._libs.tslibs.dtypes import periods_per_second as periods_per_second
from pandas._libs.tslibs.fields import get_timedelta_days as get_timedelta_days, get_timedelta_field as get_timedelta_field
from pandas._libs.tslibs.nattype import NaT as NaT, NaTType as NaTType
from pandas._libs.tslibs.np_datetime import astype_overflowsafe as astype_overflowsafe, get_supported_dtype as get_supported_dtype, is_supported_dtype as is_supported_dtype
from pandas._libs.tslibs.offsets import Tick as Tick
from pandas._libs.tslibs.timedeltas import Timedelta as Timedelta, array_to_timedelta64 as array_to_timedelta64, floordiv_object_array as floordiv_object_array, ints_to_pytimedelta as ints_to_pytimedelta, parse_timedelta_unit as parse_timedelta_unit, truediv_object_array as truediv_object_array
from pandas.core.arrays._ranges import generate_regular_range as generate_regular_range
from pandas.core.dtypes.base import ExtensionDtype as ExtensionDtype
from pandas.core.dtypes.common import is_float_dtype as is_float_dtype, is_integer_dtype as is_integer_dtype, is_object_dtype as is_object_dtype, is_string_dtype as is_string_dtype, pandas_dtype as pandas_dtype
from pandas.core.dtypes.missing import isna as isna
from pandas.core.ops.common import unpack_zerodim_and_defer as unpack_zerodim_and_defer
from pandas.util._validators import validate_endpoints as validate_endpoints
from typing import ClassVar

TYPE_CHECKING: bool
iNaT: int
TD64NS_DTYPE: numpy.dtypes.TimeDelta64DType
def _field_accessor(name: str, alias: str, docstring: str): ...

class TimedeltaArray(pandas.core.arrays.datetimelike.TimelikeOps):
    _typ: ClassVar[str] = ...
    _internal_fill_value: ClassVar[numpy.timedelta64] = ...
    _recognized_scalars: ClassVar[tuple] = ...
    _infer_matches: ClassVar[tuple] = ...
    __array_priority__: ClassVar[int] = ...
    _other_ops: ClassVar[list] = ...
    _bool_ops: ClassVar[list] = ...
    _object_ops: ClassVar[list] = ...
    _field_ops: ClassVar[list] = ...
    _datetimelike_ops: ClassVar[list] = ...
    _datetimelike_methods: ClassVar[list] = ...
    _freq: ClassVar[None] = ...
    _default_dtype: ClassVar[numpy.dtypes.TimeDelta64DType] = ...
    days_docstring: ClassVar[str] = ...
    seconds_docstring: ClassVar[str] = ...
    microseconds_docstring: ClassVar[str] = ...
    nanoseconds_docstring: ClassVar[str] = ...
    def _is_recognized_dtype(self, x): ...
    def _box_func(self, x: np.timedelta64) -> Timedelta | NaTType: ...
    @classmethod
    def _validate_dtype(cls, values, dtype): ...
    @classmethod
    def _simple_new(cls, values: npt.NDArray[np.timedelta64], freq: Tick | None, dtype: np.dtype[np.timedelta64] = ...) -> Self: ...
    @classmethod
    def _from_sequence(cls, data, *, dtype, copy: bool = ...) -> Self: ...
    @classmethod
    def _from_sequence_not_strict(cls, data, *, dtype, copy: bool = ..., freq: pandas._libs.lib._NoDefault = ..., unit) -> Self:
        """
        _from_sequence_not_strict but without responsibility for finding the
        result's `freq`.
        """
    @classmethod
    def _generate_range(cls, start, end, periods, freq, closed, *, unit: str | None) -> Self: ...
    def _unbox_scalar(self, value) -> np.timedelta64: ...
    def _scalar_from_string(self, value) -> Timedelta | NaTType: ...
    def _check_compatible_with(self, other) -> None: ...
    def astype(self, dtype, copy: bool = ...): ...
    def __iter__(self) -> Iterator: ...
    def sum(self, *, axis: AxisInt | None, dtype: NpDtype | None, out, keepdims: bool = ..., initial, skipna: bool = ..., min_count: int = ...): ...
    def std(self, *, axis: AxisInt | None, dtype: NpDtype | None, out, ddof: int = ..., keepdims: bool = ..., skipna: bool = ...): ...
    def _accumulate(self, name: str, *, skipna: bool = ..., **kwargs): ...
    def _formatter(self, boxed: bool = ...): ...
    def _format_native_types(self, *, na_rep: str | float = ..., date_format, **kwargs) -> npt.NDArray[np.object_]: ...
    def _add_offset(self, other): ...
    def __mul__(self, other) -> Self: ...
    def __rmul__(self, other) -> Self: ...
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
    @property
    def _scalar_type(self): ...
    @property
    def dtype(self): ...
    @property
    def days(self): ...
    @property
    def seconds(self): ...
    @property
    def microseconds(self): ...
    @property
    def nanoseconds(self): ...
    @property
    def components(self): ...
def sequence_to_td64ns(data, copy: bool = ..., unit, errors: DateTimeErrorChoices = ...) -> tuple[np.ndarray, Tick | None]:
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
def _ints_to_td64ns(data, unit: str = ...):
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
def _objects_to_td64ns(data, unit, errors: DateTimeErrorChoices = ...):
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
