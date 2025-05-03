import np
import npt
import pandas._libs.algos as algos
import pandas._libs.lib
import pandas._libs.lib as lib
import pandas.compat.numpy.function as nv
import pandas.core.algorithms as algorithms
import pandas.core.array_algos.datetimelike_accumulations as datetimelike_accumulations
import pandas.core.arraylike
import pandas.core.arrays._mixins
import pandas.core.common as com
import pandas.core.missing as missing
import pandas.core.nanops as nanops
import pandas.core.ops as ops
import pandas.tseries.frequencies as frequencies
from _typeshed import Incomplete
from builtins import AxisInt
from datetime import datetime
from pandas._libs.arrays import NDArrayBacked as NDArrayBacked
from pandas._libs.lib import is_list_like as is_list_like
from pandas._libs.properties import cache_readonly as cache_readonly
from pandas._libs.tslibs.dtypes import Resolution as Resolution, periods_per_day as periods_per_day
from pandas._libs.tslibs.fields import RoundTo as RoundTo, round_nsint64 as round_nsint64
from pandas._libs.tslibs.nattype import NaT as NaT, NaTType as NaTType
from pandas._libs.tslibs.np_datetime import add_overflowsafe as add_overflowsafe, astype_overflowsafe as astype_overflowsafe, compare_mismatched_resolutions as compare_mismatched_resolutions, get_unit_from_dtype as get_unit_from_dtype
from pandas._libs.tslibs.offsets import BaseOffset as BaseOffset, Tick as Tick, to_offset as to_offset
from pandas._libs.tslibs.period import IncompatibleFrequency as IncompatibleFrequency, Period as Period
from pandas._libs.tslibs.timedeltas import Timedelta as Timedelta, get_unit_for_round as get_unit_for_round, ints_to_pytimedelta as ints_to_pytimedelta
from pandas._libs.tslibs.timestamps import Timestamp as Timestamp, integer_op_not_supported as integer_op_not_supported
from pandas._libs.tslibs.vectorized import ints_to_pydatetime as ints_to_pydatetime
from pandas._typing import F as F
from pandas.core.algorithms import isin as isin, map_array as map_array, unique1d as unique1d
from pandas.core.arraylike import OpsMixin as OpsMixin
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray as NDArrayBackedExtensionArray, ravel_compat as ravel_compat
from pandas.core.arrays.arrow.array import ArrowExtensionArray as ArrowExtensionArray
from pandas.core.arrays.base import ExtensionArray as ExtensionArray
from pandas.core.arrays.integer import IntegerArray as IntegerArray
from pandas.core.construction import ensure_wrapped_if_datetimelike as ensure_wrapped_if_datetimelike, extract_array as extract_array, pd_array as pd_array
from pandas.core.dtypes.base import ExtensionDtype as ExtensionDtype
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike as construct_1d_object_array_from_listlike
from pandas.core.dtypes.common import is_all_strings as is_all_strings, is_integer_dtype as is_integer_dtype, is_object_dtype as is_object_dtype, is_string_dtype as is_string_dtype, pandas_dtype as pandas_dtype
from pandas.core.dtypes.dtypes import ArrowDtype as ArrowDtype, CategoricalDtype as CategoricalDtype, DatetimeTZDtype as DatetimeTZDtype, PeriodDtype as PeriodDtype
from pandas.core.dtypes.generic import ABCCategorical as ABCCategorical, ABCMultiIndex as ABCMultiIndex
from pandas.core.dtypes.missing import is_valid_na_for_dtype as is_valid_na_for_dtype, isna as isna
from pandas.core.indexers.utils import check_array_indexer as check_array_indexer, check_setitem_lengths as check_setitem_lengths
from pandas.core.ops.common import unpack_zerodim_and_defer as unpack_zerodim_and_defer
from pandas.core.ops.invalid import invalid_comparison as invalid_comparison, make_invalid_op as make_invalid_op
from pandas.errors import AbstractMethodError as AbstractMethodError, InvalidComparison as InvalidComparison, PerformanceWarning as PerformanceWarning
from pandas.util._decorators import Appender as Appender, Substitution as Substitution
from pandas.util._exceptions import find_stack_level as find_stack_level
from typing import Any, ArrayLike, DTScalarOrNaT, Dtype, InterpolateOptions, NpDtype, PositionalIndexer2D, TimeAmbiguous, TimeNonexistent

TYPE_CHECKING: bool
iNaT: int
Self: None
npt: None
def _make_unpacked_invalid_op(op_name: str): ...
def _period_dispatch(meth: F) -> F:
    """
    For PeriodArray methods, dispatch to DatetimeArray and re-wrap the results
    in PeriodArray.  We cannot use ._ndarray directly for the affected
    methods because the i8 data has different semantics on NaT values.
    """

class DatetimeLikeArrayMixin(pandas.core.arraylike.OpsMixin, pandas.core.arrays._mixins.NDArrayBackedExtensionArray):
    _can_hold_na: Incomplete
    def __init__(self, data, dtype: Dtype | None, freq, copy: bool = ...) -> None: ...
    def _scalar_from_string(self, value: str) -> DTScalarOrNaT:
        """
        Construct a scalar type from a string.

        Parameters
        ----------
        value : str

        Returns
        -------
        Period, Timestamp, or Timedelta, or NaT
            Whatever the type of ``self._scalar_type`` is.

        Notes
        -----
        This should call ``self._check_compatible_with`` before
        unboxing the result.
        """
    def _unbox_scalar(self, value: DTScalarOrNaT) -> np.int64 | np.datetime64 | np.timedelta64:
        """
        Unbox the integer value of a scalar `value`.

        Parameters
        ----------
        value : Period, Timestamp, Timedelta, or NaT
            Depending on subclass.

        Returns
        -------
        int

        Examples
        --------
        >>> arr = pd.array(np.array(['1970-01-01'], 'datetime64[ns]'))
        >>> arr._unbox_scalar(arr[0])
        numpy.datetime64('1970-01-01T00:00:00.000000000')
        """
    def _check_compatible_with(self, other: DTScalarOrNaT) -> None:
        """
        Verify that `self` and `other` are compatible.

        * DatetimeArray verifies that the timezones (if any) match
        * PeriodArray verifies that the freq matches
        * Timedelta has no verification

        In each case, NaT is considered compatible.

        Parameters
        ----------
        other

        Raises
        ------
        Exception
        """
    def _box_func(self, x):
        """
        box function to get object from internal representation
        """
    def _box_values(self, values) -> np.ndarray:
        """
        apply box func to passed values
        """
    def __iter__(self) -> Iterator: ...
    def _format_native_types(self, *, na_rep: str | float = ..., date_format) -> npt.NDArray[np.object_]:
        """
        Helper method for astype when converting to strings.

        Returns
        -------
        ndarray[str]
        """
    def _formatter(self, boxed: bool = ...): ...
    def __array__(self, dtype: NpDtype | None, copy: bool | None) -> np.ndarray: ...
    def __getitem__(self, key: PositionalIndexer2D) -> Self | DTScalarOrNaT:
        """
        This getitem defers to the underlying array, which by-definition can
        only handle list-likes, slices, and integer scalars
        """
    def _get_getitem_freq(self, key) -> BaseOffset | None:
        """
        Find the `freq` attribute to assign to the result of a __getitem__ lookup.
        """
    def __setitem__(self, key: int | Sequence[int] | Sequence[bool] | slice, value: NaTType | Any | Sequence[Any]) -> None: ...
    def _maybe_clear_freq(self) -> None: ...
    def astype(self, dtype, copy: bool = ...): ...
    def view(self, dtype: Dtype | None) -> ArrayLike: ...
    def _validate_comparison_value(self, other): ...
    def _validate_scalar(self, value, *, allow_listlike: bool = ..., unbox: bool = ...):
        """
        Validate that the input value can be cast to our scalar_type.

        Parameters
        ----------
        value : object
        allow_listlike: bool, default False
            When raising an exception, whether the message should say
            listlike inputs are allowed.
        unbox : bool, default True
            Whether to unbox the result before returning.  Note: unbox=False
            skips the setitem compatibility check.

        Returns
        -------
        self._scalar_type or NaT
        """
    def _validation_error_message(self, value, allow_listlike: bool = ...) -> str:
        """
        Construct an exception message on validation error.

        Some methods allow only scalar inputs, while others allow either scalar
        or listlike.

        Parameters
        ----------
        allow_listlike: bool, default False

        Returns
        -------
        str
        """
    def _validate_listlike(self, value, allow_object: bool = ...): ...
    def _validate_setitem_value(self, value): ...
    def _unbox(self, other) -> np.int64 | np.datetime64 | np.timedelta64 | np.ndarray:
        """
        Unbox either a scalar with _unbox_scalar or an instance of our own type.
        """
    def map(self, *args, **kwargs): ...
    def isin(self, values: ArrayLike) -> npt.NDArray[np.bool_]:
        """
        Compute boolean array of whether each value is found in the
        passed set of values.

        Parameters
        ----------
        values : np.ndarray or ExtensionArray

        Returns
        -------
        ndarray[bool]
        """
    def isna(self) -> npt.NDArray[np.bool_]: ...
    def _maybe_mask_results(self, result: np.ndarray, fill_value: int = ..., convert) -> np.ndarray:
        """
        Parameters
        ----------
        result : np.ndarray
        fill_value : object, default iNaT
        convert : str, dtype or None

        Returns
        -------
        result : ndarray with values replace by the fill_value

        mask the result if needed, convert to the provided dtype if its not
        None

        This is an internal routine.
        """
    def _cmp_method(self, other, op): ...
    def __pow__(self, other): ...
    def __rpow__(self, other): ...
    def __mul__(self, other): ...
    def __rmul__(self, other): ...
    def __truediv__(self, other): ...
    def __rtruediv__(self, other): ...
    def __floordiv__(self, other): ...
    def __rfloordiv__(self, other): ...
    def __mod__(self, other): ...
    def __rmod__(self, other): ...
    def __divmod__(self, other): ...
    def __rdivmod__(self, other): ...
    def _get_i8_values_and_mask(self, other) -> tuple[int | npt.NDArray[np.int64], None | npt.NDArray[np.bool_]]:
        """
        Get the int64 values and b_mask to pass to add_overflowsafe.
        """
    def _get_arithmetic_result_freq(self, other) -> BaseOffset | None:
        """
        Check if we can preserve self.freq in addition or subtraction.
        """
    def _add_datetimelike_scalar(self, other) -> DatetimeArray: ...
    def _add_datetime_arraylike(self, other: DatetimeArray) -> DatetimeArray: ...
    def _sub_datetimelike_scalar(self, other: datetime | np.datetime64) -> TimedeltaArray: ...
    def _sub_datetime_arraylike(self, other: DatetimeArray) -> TimedeltaArray: ...
    def _sub_datetimelike(self, other: Timestamp | DatetimeArray) -> TimedeltaArray: ...
    def _add_period(self, other: Period) -> PeriodArray: ...
    def _add_offset(self, offset): ...
    def _add_timedeltalike_scalar(self, other):
        """
        Add a delta of a timedeltalike

        Returns
        -------
        Same type as self
        """
    def _add_timedelta_arraylike(self, other: TimedeltaArray):
        """
        Add a delta of a TimedeltaIndex

        Returns
        -------
        Same type as self
        """
    def _add_timedeltalike(self, other: Timedelta | TimedeltaArray): ...
    def _add_nat(self):
        """
        Add pd.NaT to self
        """
    def _sub_nat(self):
        """
        Subtract pd.NaT from self
        """
    def _sub_periodlike(self, other: Period | PeriodArray) -> npt.NDArray[np.object_]: ...
    def _addsub_object_array(self, other: npt.NDArray[np.object_], op):
        """
        Add or subtract array-like of DateOffset objects

        Parameters
        ----------
        other : np.ndarray[object]
        op : {operator.add, operator.sub}

        Returns
        -------
        np.ndarray[object]
            Except in fastpath case with length 1 where we operate on the
            contained scalar.
        """
    def _accumulate(self, name: str, *, skipna: bool = ..., **kwargs) -> Self: ...
    def __add__(self, other): ...
    def __radd__(self, other): ...
    def __sub__(self, other): ...
    def __rsub__(self, other): ...
    def __iadd__(self, other) -> Self: ...
    def __isub__(self, other) -> Self: ...
    def _quantile(self, *args, **kwargs) -> Self: ...
    def min(self, *args, **kwargs):
        """
        Return the minimum value of the Array or minimum along
        an axis.

        See Also
        --------
        numpy.ndarray.min
        Index.min : Return the minimum value in an Index.
        Series.min : Return the minimum value in a Series.
        """
    def max(self, *args, **kwargs):
        """
        Return the maximum value of the Array or maximum along
        an axis.

        See Also
        --------
        numpy.ndarray.max
        Index.max : Return the maximum value in an Index.
        Series.max : Return the maximum value in a Series.
        """
    def mean(self, *, skipna: bool = ..., axis: AxisInt | None = ...):
        """
        Return the mean value of the Array.

        Parameters
        ----------
        skipna : bool, default True
            Whether to ignore any NaT elements.
        axis : int, optional, default 0

        Returns
        -------
        scalar
            Timestamp or Timedelta.

        See Also
        --------
        numpy.ndarray.mean : Returns the average of array elements along a given axis.
        Series.mean : Return the mean value in a Series.

        Notes
        -----
        mean is only defined for Datetime and Timedelta dtypes, not for Period.

        Examples
        --------
        For :class:`pandas.DatetimeIndex`:

        >>> idx = pd.date_range('2001-01-01 00:00', periods=3)
        >>> idx
        DatetimeIndex(['2001-01-01', '2001-01-02', '2001-01-03'],
                      dtype='datetime64[ns]', freq='D')
        >>> idx.mean()
        Timestamp('2001-01-02 00:00:00')

        For :class:`pandas.TimedeltaIndex`:

        >>> tdelta_idx = pd.to_timedelta([1, 2, 3], unit='D')
        >>> tdelta_idx
        TimedeltaIndex(['1 days', '2 days', '3 days'],
                        dtype='timedelta64[ns]', freq=None)
        >>> tdelta_idx.mean()
        Timedelta('2 days 00:00:00')
        """
    def median(self, *args, **kwargs): ...
    def _mode(self, dropna: bool = ...): ...
    def _groupby_op(self, *, how: str, has_dropped_na: bool, min_count: int, ngroups: int, ids: npt.NDArray[np.intp], **kwargs): ...
    @property
    def _scalar_type(self): ...
    @property
    def asi8(self): ...
    @property
    def _isnan(self): ...
    @property
    def _hasna(self): ...
    @property
    def freqstr(self): ...
    @property
    def inferred_freq(self): ...
    @property
    def _resolution_obj(self): ...
    @property
    def resolution(self): ...
    @property
    def _is_monotonic_increasing(self): ...
    @property
    def _is_monotonic_decreasing(self): ...
    @property
    def _is_unique(self): ...

class DatelikeOps(DatetimeLikeArrayMixin):
    def strftime(self, date_format: str) -> npt.NDArray[np.object_]:
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
_round_doc: str
_round_example: str
_floor_example: str
_ceil_example: str

class TimelikeOps(DatetimeLikeArrayMixin):
    freq: Incomplete
    _creso: Incomplete
    unit: Incomplete
    def __init__(self, values, dtype, freq: pandas._libs.lib._NoDefault = ..., copy: bool = ...) -> None: ...
    @classmethod
    def _validate_dtype(cls, values, dtype): ...
    def _maybe_pin_freq(self, freq, validate_kwds: dict):
        """
        Constructor helper to pin the appropriate `freq` attribute.  Assumes
        that self._freq is currently set to any freq inferred in
        _from_sequence_not_strict.
        """
    @classmethod
    def _validate_frequency(cls, index, freq: BaseOffset, **kwargs):
        """
        Validate that a frequency is compatible with the values of a given
        Datetime Array/Index or Timedelta Array/Index

        Parameters
        ----------
        index : DatetimeIndex or TimedeltaIndex
            The index on which to determine if the given frequency is valid
        freq : DateOffset
            The frequency to validate
        """
    @classmethod
    def _generate_range(cls, start, end, periods: int | None, freq, *args, **kwargs) -> Self: ...
    def as_unit(self, unit: str, round_ok: bool = ...) -> Self: ...
    def _ensure_matching_resos(self, other): ...
    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs): ...
    def _round(self, freq, mode, ambiguous, nonexistent): ...
    def round(self, freq, ambiguous: TimeAmbiguous = ..., nonexistent: TimeNonexistent = ...) -> Self:
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
    def floor(self, freq, ambiguous: TimeAmbiguous = ..., nonexistent: TimeNonexistent = ...) -> Self:
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
    def ceil(self, freq, ambiguous: TimeAmbiguous = ..., nonexistent: TimeNonexistent = ...) -> Self:
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
    def any(self, *, axis: AxisInt | None, skipna: bool = ...) -> bool: ...
    def all(self, *, axis: AxisInt | None, skipna: bool = ...) -> bool: ...
    def _maybe_clear_freq(self) -> None: ...
    def _with_freq(self, freq) -> Self:
        '''
        Helper to get a view on the same data, with a new freq.

        Parameters
        ----------
        freq : DateOffset, None, or "infer"

        Returns
        -------
        Same type as self
        '''
    def _values_for_json(self) -> np.ndarray: ...
    def factorize(self, use_na_sentinel: bool = ..., sort: bool = ...): ...
    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[Self], axis: AxisInt = ...) -> Self: ...
    def copy(self, order: str = ...) -> Self: ...
    def interpolate(self, *, method: InterpolateOptions, axis: int, index: Index, limit, limit_direction, limit_area, copy: bool, **kwargs) -> Self:
        """
        See NDFrame.interpolate.__doc__.
        """
    @property
    def _is_dates_only(self): ...
def ensure_arraylike_for_datetimelike(data, copy: bool, cls_name: str) -> tuple[ArrayLike, bool]: ...
def validate_periods(periods: int | float | None) -> int | None:
    """
    If a `periods` argument is passed to the Datetime/Timedelta Array/Index
    constructor, cast it to an integer.

    Parameters
    ----------
    periods : None, float, int

    Returns
    -------
    periods : None or int

    Raises
    ------
    TypeError
        if periods is None, float, or int
    """
def _validate_inferred_freq(freq: BaseOffset | None, inferred_freq: BaseOffset | None) -> BaseOffset | None:
    """
    If the user passes a freq and another freq is inferred from passed data,
    require that they match.

    Parameters
    ----------
    freq : DateOffset or None
    inferred_freq : DateOffset or None

    Returns
    -------
    freq : DateOffset or None
    """
def dtype_to_unit(dtype: DatetimeTZDtype | np.dtype | ArrowDtype) -> str:
    """
    Return the unit str corresponding to the dtype's resolution.

    Parameters
    ----------
    dtype : DatetimeTZDtype or np.dtype
        If np.dtype, we assume it is a datetime64 dtype.

    Returns
    -------
    str
    """
