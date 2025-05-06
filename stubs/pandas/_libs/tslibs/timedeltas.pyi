import _cython_3_0_11
import datetime
from _typeshed import Incomplete
from pandas._libs.tslibs.fields import RoundTo as RoundTo, round_nsint64 as round_nsint64
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime as OutOfBoundsDatetime, OutOfBoundsTimedelta as OutOfBoundsTimedelta
from typing import Any, ClassVar, overload

__pyx_capi__: dict
__pyx_unpickle__Timedelta: _cython_3_0_11.cython_function_or_method
__test__: dict
_binary_op_method_timedeltalike: _cython_3_0_11.cython_function_or_method
_no_input: object
_op_unary_method: _cython_3_0_11.cython_function_or_method
_timedelta_unpickle: _cython_3_0_11.cython_function_or_method
array_to_timedelta64: _cython_3_0_11.cython_function_or_method
delta_to_nanoseconds: _cython_3_0_11.cython_function_or_method
disallow_ambiguous_unit: _cython_3_0_11.cython_function_or_method
floordiv_object_array: _cython_3_0_11.cython_function_or_method
get_unit_for_round: _cython_3_0_11.cython_function_or_method
ints_to_pytimedelta: _cython_3_0_11.cython_function_or_method
parse_timedelta_unit: _cython_3_0_11.cython_function_or_method
truediv_object_array: _cython_3_0_11.cython_function_or_method

class MinMaxReso:
    def __init__(self, *args, **kwargs) -> None: ...
    def __get__(self, instance, owner): ...
    def __set__(self, instance, value): ...

class Timedelta(_Timedelta):
    _req_any_kwargs_new: ClassVar[set] = ...
    def __init__(self, *args, **kwargs) -> None: ...
    def _round(self, *args, **kwargs): ...
    def ceil(self, *args, **kwargs):
        """
        Return a new Timedelta ceiled to this resolution.

        Parameters
        ----------
        freq : str
            Frequency string indicating the ceiling resolution.
            It uses the same units as class constructor :class:`~pandas.Timedelta`.

        Examples
        --------
        >>> td = pd.Timedelta('1001ms')
        >>> td
        Timedelta('0 days 00:00:01.001000')
        >>> td.ceil('s')
        Timedelta('0 days 00:00:02')
        """
    def floor(self, *args, **kwargs):
        """
        Return a new Timedelta floored to this resolution.

        Parameters
        ----------
        freq : str
            Frequency string indicating the flooring resolution.
            It uses the same units as class constructor :class:`~pandas.Timedelta`.

        Examples
        --------
        >>> td = pd.Timedelta('1001ms')
        >>> td
        Timedelta('0 days 00:00:01.001000')
        >>> td.floor('s')
        Timedelta('0 days 00:00:01')
        """
    def round(self, *args, **kwargs):
        """
        Round the Timedelta to the specified resolution.

        Parameters
        ----------
        freq : str
            Frequency string indicating the rounding resolution.
            It uses the same units as class constructor :class:`~pandas.Timedelta`.

        Returns
        -------
        a new Timedelta rounded to the given resolution of `freq`

        Raises
        ------
        ValueError if the freq cannot be converted

        Examples
        --------
        >>> td = pd.Timedelta('1001ms')
        >>> td
        Timedelta('0 days 00:00:01.001000')
        >>> td.round('s')
        Timedelta('0 days 00:00:01')
        """
    def __abs__(self): ...
    def __add__(self, other): ...
    def __divmod__(self, other): ...
    def __floordiv__(self, other): ...
    def __mod__(self, other): ...
    def __mul__(self, other): ...
    def __neg__(self): ...
    def __pos__(self): ...
    def __radd__(self, other): ...
    def __rdivmod__(self, other): ...
    def __reduce__(self): ...
    def __rfloordiv__(self, other): ...
    def __rmod__(self, other): ...
    def __rmul__(self, other): ...
    def __rsub__(self, other): ...
    def __rtruediv__(self, other): ...
    def __sub__(self, other): ...
    def __truediv__(self, other): ...

class _Timedelta(datetime.timedelta):
    _from_value_and_reso: ClassVar[method] = ...
    __array_priority__: ClassVar[int] = ...
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    _creso: Incomplete
    _d: Incomplete
    _h: Incomplete
    _is_populated: Incomplete
    _m: Incomplete
    _ms: Incomplete
    _ns: Incomplete
    _s: Incomplete
    _unit: Incomplete
    _us: Incomplete
    _value: Incomplete
    asm8: Incomplete
    components: Incomplete
    days: Incomplete
    max: Incomplete
    microseconds: Incomplete
    min: Incomplete
    nanoseconds: Any
    resolution: Incomplete
    resolution_string: Incomplete
    seconds: Incomplete
    unit: Incomplete
    value: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def _maybe_cast_to_matching_resos(self, *args, **kwargs):
        """
        If _resos do not match, cast to the higher resolution, raising on overflow.
        """
    def _repr_base(self, *args, **kwargs):
        """

        Parameters
        ----------
        format : None|all|sub_day|long

        Returns
        -------
        converted : string of a Timedelta

        """
    def as_unit(self, *args, **kwargs):
        '''
        Convert the underlying int64 representation to the given unit.

        Parameters
        ----------
        unit : {"ns", "us", "ms", "s"}
        round_ok : bool, default True
            If False and the conversion requires rounding, raise.

        Returns
        -------
        Timedelta

        Examples
        --------
        >>> td = pd.Timedelta(\'1001ms\')
        >>> td
        Timedelta(\'0 days 00:00:01.001000\')
        >>> td.as_unit(\'s\')
        Timedelta(\'0 days 00:00:01\')
        '''
    def isoformat(self, *args, **kwargs):
        """
        Format the Timedelta as ISO 8601 Duration.

        ``P[n]Y[n]M[n]DT[n]H[n]M[n]S``, where the ``[n]`` s are replaced by the
        values. See https://en.wikipedia.org/wiki/ISO_8601#Durations.

        Returns
        -------
        str

        See Also
        --------
        Timestamp.isoformat : Function is used to convert the given
            Timestamp object into the ISO format.

        Notes
        -----
        The longest component is days, whose value may be larger than
        365.
        Every component is always included, even if its value is 0.
        Pandas uses nanosecond precision, so up to 9 decimal places may
        be included in the seconds component.
        Trailing 0's are removed from the seconds component after the decimal.
        We do not 0 pad components, so it's `...T5H...`, not `...T05H...`

        Examples
        --------
        >>> td = pd.Timedelta(days=6, minutes=50, seconds=3,
        ...                   milliseconds=10, microseconds=10, nanoseconds=12)

        >>> td.isoformat()
        'P6DT0H50M3.010010012S'
        >>> pd.Timedelta(hours=1, seconds=10).isoformat()
        'P0DT1H0M10S'
        >>> pd.Timedelta(days=500.5).isoformat()
        'P500DT12H0M0S'
        """
    def to_numpy(self) -> Any:
        """
        Convert the Timedelta to a NumPy timedelta64.

        This is an alias method for `Timedelta.to_timedelta64()`. The dtype and
        copy parameters are available here only for compatibility. Their values
        will not affect the return value.

        Returns
        -------
        numpy.timedelta64

        See Also
        --------
        Series.to_numpy : Similar method for Series.

        Examples
        --------
        >>> td = pd.Timedelta('3D')
        >>> td
        Timedelta('3 days 00:00:00')
        >>> td.to_numpy()
        numpy.timedelta64(259200000000000,'ns')
        """
    @overload
    def to_pytimedelta(self) -> Any:
        """
        Convert a pandas Timedelta object into a python ``datetime.timedelta`` object.

        Timedelta objects are internally saved as numpy datetime64[ns] dtype.
        Use to_pytimedelta() to convert to object dtype.

        Returns
        -------
        datetime.timedelta or numpy.array of datetime.timedelta

        See Also
        --------
        to_timedelta : Convert argument to Timedelta type.

        Notes
        -----
        Any nanosecond resolution will be lost.

        Examples
        --------
        >>> td = pd.Timedelta('3D')
        >>> td
        Timedelta('3 days 00:00:00')
        >>> td.to_pytimedelta()
        datetime.timedelta(days=3)
        """
    @overload
    def to_pytimedelta(self) -> Any:
        """
        Convert a pandas Timedelta object into a python ``datetime.timedelta`` object.

        Timedelta objects are internally saved as numpy datetime64[ns] dtype.
        Use to_pytimedelta() to convert to object dtype.

        Returns
        -------
        datetime.timedelta or numpy.array of datetime.timedelta

        See Also
        --------
        to_timedelta : Convert argument to Timedelta type.

        Notes
        -----
        Any nanosecond resolution will be lost.

        Examples
        --------
        >>> td = pd.Timedelta('3D')
        >>> td
        Timedelta('3 days 00:00:00')
        >>> td.to_pytimedelta()
        datetime.timedelta(days=3)
        """
    def to_timedelta64(self) -> Any:
        """
        Return a numpy.timedelta64 object with 'ns' precision.

        Examples
        --------
        >>> td = pd.Timedelta('3D')
        >>> td
        Timedelta('3 days 00:00:00')
        >>> td.to_timedelta64()
        numpy.timedelta64(259200000000000,'ns')
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
    def view(self, int) -> Any:
        """
        Array view compatibility.

        Parameters
        ----------
        dtype : str or dtype
            The dtype to view the underlying data as.

        Examples
        --------
        >>> td = pd.Timedelta('3D')
        >>> td
        Timedelta('3 days 00:00:00')
        >>> td.view(int)
        259200000000000
        """
    def __bool__(self) -> bool:
        """True if self else False"""
    def __eq__(self, other: object) -> bool:
        """Return self==value."""
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
    def __ne__(self, other: object) -> bool:
        """Return self!=value."""
    def __reduce_cython__(self, *args, **kwargs): ...
    def __setstate_cython__(self, *args, **kwargs): ...
