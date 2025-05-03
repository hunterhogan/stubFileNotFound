import np
import npt
import pandas._libs.lib as lib
import pandas._libs.missing as libmissing
import pandas.compat.numpy.function as nv
import pandas.core.algorithms as algos
import pandas.core.array_algos.masked_accumulations as masked_accumulations
import pandas.core.array_algos.masked_reductions as masked_reductions
import pandas.core.arraylike
import pandas.core.arraylike as arraylike
import pandas.core.arrays.base
import pandas.core.missing as missing
import pandas.core.nanops as nanops
import pandas.core.ops as ops
import typing
from builtins import AxisInt, Shape
from pandas._libs.lib import is_bool as is_bool, is_list_like as is_list_like, is_scalar as is_scalar
from pandas._libs.tslibs.np_datetime import is_supported_dtype as is_supported_dtype
from pandas.compat import is_platform_windows as is_platform_windows
from pandas.core.algorithms import factorize_array as factorize_array, isin as isin, map_array as map_array, mode as mode, take as take
from pandas.core.array_algos.quantile import quantile_with_mask as quantile_with_mask
from pandas.core.arraylike import OpsMixin as OpsMixin
from pandas.core.arrays._utils import to_numpy_dtype_inference as to_numpy_dtype_inference
from pandas.core.arrays.base import ExtensionArray as ExtensionArray
from pandas.core.construction import ensure_wrapped_if_datetimelike as ensure_wrapped_if_datetimelike, extract_array as extract_array, pd_array as pd_array
from pandas.core.dtypes.base import ExtensionDtype as ExtensionDtype
from pandas.core.dtypes.common import is_integer_dtype as is_integer_dtype, is_string_dtype as is_string_dtype, pandas_dtype as pandas_dtype
from pandas.core.dtypes.dtypes import BaseMaskedDtype as BaseMaskedDtype
from pandas.core.dtypes.missing import array_equivalent as array_equivalent, is_valid_na_for_dtype as is_valid_na_for_dtype, isna as isna, notna as notna
from pandas.core.indexers.utils import check_array_indexer as check_array_indexer
from pandas.core.ops.invalid import invalid_comparison as invalid_comparison
from pandas.core.util.hashing import hash_array as hash_array
from pandas.errors import AbstractMethodError as AbstractMethodError
from pandas.util._decorators import doc as doc
from pandas.util._validators import validate_fillna_kwargs as validate_fillna_kwargs
from typing import Any, ArrayLike, AstypeArg, Callable, ClassVar, DtypeObj, FillnaOptions, InterpolateOptions, Literal, NpDtype, PositionalIndexer, Scalar

TYPE_CHECKING: bool
Self: None
npt: None
IS64: bool

class BaseMaskedArray(pandas.core.arraylike.OpsMixin, pandas.core.arrays.base.ExtensionArray):
    _truthy_value: ClassVar[typing._UnionGenericAlias] = ...
    _falsey_value: ClassVar[typing._UnionGenericAlias] = ...
    __array_priority__: ClassVar[int] = ...
    @classmethod
    def _simple_new(cls, values: np.ndarray, mask: npt.NDArray[np.bool_]) -> Self: ...
    def __init__(self, values: np.ndarray, mask: npt.NDArray[np.bool_], copy: bool = ...) -> None: ...
    @classmethod
    def _from_sequence(cls, scalars, *, dtype, copy: bool = ...) -> Self: ...
    @classmethod
    def _empty(cls, shape: Shape, dtype: ExtensionDtype):
        """
        Create an ExtensionArray with the given shape and dtype.

        See also
        --------
        ExtensionDtype.empty
            ExtensionDtype.empty is the 'official' public version of this API.
        """
    def _formatter(self, boxed: bool = ...) -> Callable[[Any], str | None]: ...
    def __getitem__(self, item: PositionalIndexer) -> Self | Any: ...
    def _pad_or_backfill(self, *, method: FillnaOptions, limit: int | None, limit_area: Literal['inside', 'outside'] | None, copy: bool = ...) -> Self: ...
    def fillna(self, value, method, limit: int | None, copy: bool = ...) -> Self:
        '''
        Fill NA/NaN values using the specified method.

        Parameters
        ----------
        value : scalar, array-like
            If a scalar value is passed it is used to fill all missing values.
            Alternatively, an array-like "value" can be given. It\'s expected
            that the array-like have the same length as \'self\'.
        method : {\'backfill\', \'bfill\', \'pad\', \'ffill\', None}, default None
            Method to use for filling holes in reindexed Series:

            * pad / ffill: propagate last valid observation forward to next valid.
            * backfill / bfill: use NEXT valid observation to fill gap.

            .. deprecated:: 2.1.0

        limit : int, default None
            If method is specified, this is the maximum number of consecutive
            NaN values to forward/backward fill. In other words, if there is
            a gap with more than this number of consecutive NaNs, it will only
            be partially filled. If method is not specified, this is the
            maximum number of entries along the entire axis where NaNs will be
            filled.

            .. deprecated:: 2.1.0

        copy : bool, default True
            Whether to make a copy of the data before filling. If False, then
            the original should be modified and no new memory should be allocated.
            For ExtensionArray subclasses that cannot do this, it is at the
            author\'s discretion whether to ignore "copy=False" or to raise.
            The base class implementation ignores the keyword in pad/backfill
            cases.

        Returns
        -------
        ExtensionArray
            With NA/NaN filled.

        Examples
        --------
        >>> arr = pd.array([np.nan, np.nan, 2, 3, np.nan, np.nan])
        >>> arr.fillna(0)
        <IntegerArray>
        [0, 0, 2, 3, 0, 0]
        Length: 6, dtype: Int64
        '''
    @classmethod
    def _coerce_to_array(cls, values, *, dtype: DtypeObj, copy: bool = ...) -> tuple[np.ndarray, np.ndarray]: ...
    def _validate_setitem_value(self, value):
        """
        Check if we have a scalar that we can cast losslessly.

        Raises
        ------
        TypeError
        """
    def __setitem__(self, key, value) -> None: ...
    def __contains__(self, key) -> bool: ...
    def __iter__(self) -> Iterator: ...
    def __len__(self) -> int: ...
    def swapaxes(self, axis1, axis2) -> Self: ...
    def delete(self, loc, axis: AxisInt = ...) -> Self: ...
    def reshape(self, *args, **kwargs) -> Self: ...
    def ravel(self, *args, **kwargs) -> Self: ...
    def round(self, decimals: int = ..., *args, **kwargs):
        """
        Round each value in the array a to the given number of decimals.

        Parameters
        ----------
        decimals : int, default 0
            Number of decimal places to round to. If decimals is negative,
            it specifies the number of positions to the left of the decimal point.
        *args, **kwargs
            Additional arguments and keywords have no effect but might be
            accepted for compatibility with NumPy.

        Returns
        -------
        NumericArray
            Rounded values of the NumericArray.

        See Also
        --------
        numpy.around : Round values of an np.array.
        DataFrame.round : Round values of a DataFrame.
        Series.round : Round values of a Series.
        """
    def __invert__(self) -> Self: ...
    def __neg__(self) -> Self: ...
    def __pos__(self) -> Self: ...
    def __abs__(self) -> Self: ...
    def _values_for_json(self) -> np.ndarray: ...
    def to_numpy(self, dtype: npt.DTypeLike | None, copy: bool = ..., na_value: object = ...) -> np.ndarray:
        '''
        Convert to a NumPy Array.

        By default converts to an object-dtype NumPy array. Specify the `dtype` and
        `na_value` keywords to customize the conversion.

        Parameters
        ----------
        dtype : dtype, default object
            The numpy dtype to convert to.
        copy : bool, default False
            Whether to ensure that the returned value is a not a view on
            the array. Note that ``copy=False`` does not *ensure* that
            ``to_numpy()`` is no-copy. Rather, ``copy=True`` ensure that
            a copy is made, even if not strictly necessary. This is typically
            only possible when no missing values are present and `dtype`
            is the equivalent numpy dtype.
        na_value : scalar, optional
             Scalar missing value indicator to use in numpy array. Defaults
             to the native missing value indicator of this array (pd.NA).

        Returns
        -------
        numpy.ndarray

        Examples
        --------
        An object-dtype is the default result

        >>> a = pd.array([True, False, pd.NA], dtype="boolean")
        >>> a.to_numpy()
        array([True, False, <NA>], dtype=object)

        When no missing values are present, an equivalent dtype can be used.

        >>> pd.array([True, False], dtype="boolean").to_numpy(dtype="bool")
        array([ True, False])
        >>> pd.array([1, 2], dtype="Int64").to_numpy("int64")
        array([1, 2])

        However, requesting such dtype will raise a ValueError if
        missing values are present and the default missing value :attr:`NA`
        is used.

        >>> a = pd.array([True, False, pd.NA], dtype="boolean")
        >>> a
        <BooleanArray>
        [True, False, <NA>]
        Length: 3, dtype: boolean

        >>> a.to_numpy(dtype="bool")
        Traceback (most recent call last):
        ...
        ValueError: cannot convert to bool numpy array in presence of missing values

        Specify a valid `na_value` instead

        >>> a.to_numpy(dtype="bool", na_value=False)
        array([ True, False, False])
        '''
    def tolist(self):
        """
        Return a list of the values.

        These are each a scalar type, which is a Python scalar
        (for str, int, float) or a pandas scalar
        (for Timestamp/Timedelta/Interval/Period)

        Returns
        -------
        list

        Examples
        --------
        >>> arr = pd.array([1, 2, 3])
        >>> arr.tolist()
        [1, 2, 3]
        """
    def astype(self, dtype: AstypeArg, copy: bool = ...) -> ArrayLike: ...
    def __array__(self, dtype: NpDtype | None, copy: bool | None) -> np.ndarray:
        """
        the array interface, return my values
        We return an object array here to preserve our scalar values
        """
    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs): ...
    def __arrow_array__(self, type):
        """
        Convert myself into a pyarrow Array.
        """
    def _propagate_mask(self, mask: npt.NDArray[np.bool_] | None, other) -> npt.NDArray[np.bool_]: ...
    def _arith_method(self, other, op): ...
    def _logical_method(self, other, op): ...
    def _cmp_method(self, other, op) -> BooleanArray: ...
    def _maybe_mask_result(self, result: np.ndarray | tuple[np.ndarray, np.ndarray], mask: np.ndarray):
        """
        Parameters
        ----------
        result : array-like or tuple[array-like]
        mask : array-like bool
        """
    def isna(self) -> np.ndarray: ...
    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[Self], axis: AxisInt = ...) -> Self: ...
    def _hash_pandas_object(self, *, encoding: str, hash_key: str, categorize: bool) -> npt.NDArray[np.uint64]: ...
    def take(self, indexer, *, allow_fill: bool = ..., fill_value: Scalar | None, axis: AxisInt = ...) -> Self: ...
    def isin(self, values: ArrayLike) -> BooleanArray: ...
    def copy(self) -> Self: ...
    def duplicated(self, keep: Literal['first', 'last', False] = ...) -> npt.NDArray[np.bool_]:
        '''
        Return boolean ndarray denoting duplicate values.

        Parameters
        ----------
        keep : {\'first\', \'last\', False}, default \'first\'
            - ``first`` : Mark duplicates as ``True`` except for the first occurrence.
            - ``last`` : Mark duplicates as ``True`` except for the last occurrence.
            - False : Mark all duplicates as ``True``.

        Returns
        -------
        ndarray[bool]

        Examples
        --------
        >>> pd.array([1, 1, 2, 3, 3], dtype="Int64").duplicated()
        array([False,  True, False, False,  True])
        '''
    def unique(self) -> Self:
        """
        Compute the BaseMaskedArray of unique values.

        Returns
        -------
        uniques : BaseMaskedArray
        """
    def searchsorted(self, value: NumpyValueArrayLike | ExtensionArray, side: Literal['left', 'right'] = ..., sorter: NumpySorter | None) -> npt.NDArray[np.intp] | np.intp:
        """
        Find indices where elements should be inserted to maintain order.

        Find the indices into a sorted array `self` (a) such that, if the
        corresponding elements in `value` were inserted before the indices,
        the order of `self` would be preserved.

        Assuming that `self` is sorted:

        ======  ================================
        `side`  returned index `i` satisfies
        ======  ================================
        left    ``self[i-1] < value <= self[i]``
        right   ``self[i-1] <= value < self[i]``
        ======  ================================

        Parameters
        ----------
        value : array-like, list or scalar
            Value(s) to insert into `self`.
        side : {'left', 'right'}, optional
            If 'left', the index of the first suitable location found is given.
            If 'right', return the last such index.  If there is no suitable
            index, return either 0 or N (where N is the length of `self`).
        sorter : 1-D array-like, optional
            Optional array of integer indices that sort array a into ascending
            order. They are typically the result of argsort.

        Returns
        -------
        array of ints or int
            If value is array-like, array of insertion points.
            If value is scalar, a single integer.

        See Also
        --------
        numpy.searchsorted : Similar method from NumPy.

        Examples
        --------
        >>> arr = pd.array([1, 2, 3, 5])
        >>> arr.searchsorted([4])
        array([3])
        """
    def factorize(self, use_na_sentinel: bool = ...) -> tuple[np.ndarray, ExtensionArray]:
        '''
        Encode the extension array as an enumerated type.

        Parameters
        ----------
        use_na_sentinel : bool, default True
            If True, the sentinel -1 will be used for NaN values. If False,
            NaN values will be encoded as non-negative integers and will not drop the
            NaN from the uniques of the values.

            .. versionadded:: 1.5.0

        Returns
        -------
        codes : ndarray
            An integer NumPy array that\'s an indexer into the original
            ExtensionArray.
        uniques : ExtensionArray
            An ExtensionArray containing the unique values of `self`.

            .. note::

               uniques will *not* contain an entry for the NA value of
               the ExtensionArray if there are any missing values present
               in `self`.

        See Also
        --------
        factorize : Top-level factorize method that dispatches here.

        Notes
        -----
        :meth:`pandas.factorize` offers a `sort` keyword as well.

        Examples
        --------
        >>> idx1 = pd.PeriodIndex(["2014-01", "2014-01", "2014-02", "2014-02",
        ...                       "2014-03", "2014-03"], freq="M")
        >>> arr, idx = idx1.factorize()
        >>> arr
        array([0, 0, 1, 1, 2, 2])
        >>> idx
        PeriodIndex([\'2014-01\', \'2014-02\', \'2014-03\'], dtype=\'period[M]\')
        '''
    def _values_for_argsort(self) -> np.ndarray:
        """
        Return values for sorting.

        Returns
        -------
        ndarray
            The transformed values should maintain the ordering between values
            within the array.

        See Also
        --------
        ExtensionArray.argsort : Return the indices that would sort this array.

        Notes
        -----
        The caller is responsible for *not* modifying these values in-place, so
        it is safe for implementers to give views on ``self``.

        Functions that use this (e.g. ``ExtensionArray.argsort``) should ignore
        entries with missing values in the original array (according to
        ``self.isna()``). This means that the corresponding entries in the returned
        array don't need to be modified to sort correctly.

        Examples
        --------
        In most cases, this is the underlying Numpy array of the ``ExtensionArray``:

        >>> arr = pd.array([1, 2, 3])
        >>> arr._values_for_argsort()
        array([1, 2, 3])
        """
    def value_counts(self, dropna: bool = ...) -> Series:
        """
        Returns a Series containing counts of each unique value.

        Parameters
        ----------
        dropna : bool, default True
            Don't include counts of missing values.

        Returns
        -------
        counts : Series

        See Also
        --------
        Series.value_counts
        """
    def _mode(self, dropna: bool = ...) -> Self: ...
    def equals(self, other) -> bool:
        """
        Return if another array is equivalent to this array.

        Equivalent means that both arrays have the same shape and dtype, and
        all values compare equal. Missing values in the same location are
        considered equal (in contrast with normal equality).

        Parameters
        ----------
        other : ExtensionArray
            Array to compare to this Array.

        Returns
        -------
        boolean
            Whether the arrays are equivalent.

        Examples
        --------
        >>> arr1 = pd.array([1, 2, np.nan])
        >>> arr2 = pd.array([1, 2, np.nan])
        >>> arr1.equals(arr2)
        True
        """
    def _quantile(self, qs: npt.NDArray[np.float64], interpolation: str) -> BaseMaskedArray:
        """
        Dispatch to quantile_with_mask, needed because we do not have
        _from_factorized.

        Notes
        -----
        We assume that all impacted cases are 1D-only.
        """
    def _reduce(self, name: str, *, skipna: bool = ..., keepdims: bool = ..., **kwargs): ...
    def _wrap_reduction_result(self, name: str, result, *, skipna, axis): ...
    def _wrap_na_result(self, *, name, axis, mask_size): ...
    def _wrap_min_count_reduction_result(self, name: str, result, *, skipna, min_count, axis): ...
    def sum(self, *, skipna: bool = ..., min_count: int = ..., axis: AxisInt | None = ..., **kwargs): ...
    def prod(self, *, skipna: bool = ..., min_count: int = ..., axis: AxisInt | None = ..., **kwargs): ...
    def mean(self, *, skipna: bool = ..., axis: AxisInt | None = ..., **kwargs): ...
    def var(self, *, skipna: bool = ..., axis: AxisInt | None = ..., ddof: int = ..., **kwargs): ...
    def std(self, *, skipna: bool = ..., axis: AxisInt | None = ..., ddof: int = ..., **kwargs): ...
    def min(self, *, skipna: bool = ..., axis: AxisInt | None = ..., **kwargs): ...
    def max(self, *, skipna: bool = ..., axis: AxisInt | None = ..., **kwargs): ...
    def map(self, mapper, na_action): ...
    def any(self, *, skipna: bool = ..., axis: AxisInt | None = ..., **kwargs):
        '''
        Return whether any element is truthy.

        Returns False unless there is at least one element that is truthy.
        By default, NAs are skipped. If ``skipna=False`` is specified and
        missing values are present, similar :ref:`Kleene logic <boolean.kleene>`
        is used as for logical operations.

        .. versionchanged:: 1.4.0

        Parameters
        ----------
        skipna : bool, default True
            Exclude NA values. If the entire array is NA and `skipna` is
            True, then the result will be False, as for an empty array.
            If `skipna` is False, the result will still be True if there is
            at least one element that is truthy, otherwise NA will be returned
            if there are NA\'s present.
        axis : int, optional, default 0
        **kwargs : any, default None
            Additional keywords have no effect but might be accepted for
            compatibility with NumPy.

        Returns
        -------
        bool or :attr:`pandas.NA`

        See Also
        --------
        numpy.any : Numpy version of this method.
        BaseMaskedArray.all : Return whether all elements are truthy.

        Examples
        --------
        The result indicates whether any element is truthy (and by default
        skips NAs):

        >>> pd.array([True, False, True]).any()
        True
        >>> pd.array([True, False, pd.NA]).any()
        True
        >>> pd.array([False, False, pd.NA]).any()
        False
        >>> pd.array([], dtype="boolean").any()
        False
        >>> pd.array([pd.NA], dtype="boolean").any()
        False
        >>> pd.array([pd.NA], dtype="Float64").any()
        False

        With ``skipna=False``, the result can be NA if this is logically
        required (whether ``pd.NA`` is True or False influences the result):

        >>> pd.array([True, False, pd.NA]).any(skipna=False)
        True
        >>> pd.array([1, 0, pd.NA]).any(skipna=False)
        True
        >>> pd.array([False, False, pd.NA]).any(skipna=False)
        <NA>
        >>> pd.array([0, 0, pd.NA]).any(skipna=False)
        <NA>
        '''
    def all(self, *, skipna: bool = ..., axis: AxisInt | None = ..., **kwargs):
        '''
        Return whether all elements are truthy.

        Returns True unless there is at least one element that is falsey.
        By default, NAs are skipped. If ``skipna=False`` is specified and
        missing values are present, similar :ref:`Kleene logic <boolean.kleene>`
        is used as for logical operations.

        .. versionchanged:: 1.4.0

        Parameters
        ----------
        skipna : bool, default True
            Exclude NA values. If the entire array is NA and `skipna` is
            True, then the result will be True, as for an empty array.
            If `skipna` is False, the result will still be False if there is
            at least one element that is falsey, otherwise NA will be returned
            if there are NA\'s present.
        axis : int, optional, default 0
        **kwargs : any, default None
            Additional keywords have no effect but might be accepted for
            compatibility with NumPy.

        Returns
        -------
        bool or :attr:`pandas.NA`

        See Also
        --------
        numpy.all : Numpy version of this method.
        BooleanArray.any : Return whether any element is truthy.

        Examples
        --------
        The result indicates whether all elements are truthy (and by default
        skips NAs):

        >>> pd.array([True, True, pd.NA]).all()
        True
        >>> pd.array([1, 1, pd.NA]).all()
        True
        >>> pd.array([True, False, pd.NA]).all()
        False
        >>> pd.array([], dtype="boolean").all()
        True
        >>> pd.array([pd.NA], dtype="boolean").all()
        True
        >>> pd.array([pd.NA], dtype="Float64").all()
        True

        With ``skipna=False``, the result can be NA if this is logically
        required (whether ``pd.NA`` is True or False influences the result):

        >>> pd.array([True, True, pd.NA]).all(skipna=False)
        <NA>
        >>> pd.array([1, 1, pd.NA]).all(skipna=False)
        <NA>
        >>> pd.array([True, False, pd.NA]).all(skipna=False)
        False
        >>> pd.array([1, 0, pd.NA]).all(skipna=False)
        False
        '''
    def interpolate(self, *, method: InterpolateOptions, axis: int, index, limit, limit_direction, limit_area, copy: bool, **kwargs) -> FloatingArray:
        """
        See NDFrame.interpolate.__doc__.
        """
    def _accumulate(self, name: str, *, skipna: bool = ..., **kwargs) -> BaseMaskedArray: ...
    def _groupby_op(self, *, how: str, has_dropped_na: bool, min_count: int, ngroups: int, ids: npt.NDArray[np.intp], **kwargs): ...
    @property
    def dtype(self): ...
    @property
    def shape(self): ...
    @property
    def ndim(self): ...
    @property
    def T(self): ...
    @property
    def _hasna(self): ...
    @property
    def _na_value(self): ...
    @property
    def nbytes(self): ...
def transpose_homogeneous_masked_arrays(masked_arrays: Sequence[BaseMaskedArray]) -> list[BaseMaskedArray]:
    """Transpose masked arrays in a list, but faster.

    Input should be a list of 1-dim masked arrays of equal length and all have the
    same dtype. The caller is responsible for ensuring validity of input data.
    """
