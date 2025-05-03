import numpy as np
from _typeshed import Incomplete
from collections.abc import Iterator, Sequence
from pandas import Series as Series
from pandas._libs import lib as lib
from pandas._libs.tslibs import is_supported_dtype as is_supported_dtype
from pandas._typing import ArrayLike as ArrayLike, AstypeArg as AstypeArg, AxisInt as AxisInt, DtypeObj as DtypeObj, FillnaOptions as FillnaOptions, InterpolateOptions as InterpolateOptions, NpDtype as NpDtype, NumpySorter as NumpySorter, NumpyValueArrayLike as NumpyValueArrayLike, PositionalIndexer as PositionalIndexer, Scalar as Scalar, ScalarIndexer as ScalarIndexer, Self as Self, SequenceIndexer as SequenceIndexer, Shape as Shape, npt as npt
from pandas.compat import IS64 as IS64, is_platform_windows as is_platform_windows
from pandas.core import arraylike as arraylike, missing as missing, nanops as nanops, ops as ops
from pandas.core.algorithms import factorize_array as factorize_array, isin as isin, map_array as map_array, mode as mode, take as take
from pandas.core.array_algos import masked_accumulations as masked_accumulations, masked_reductions as masked_reductions
from pandas.core.array_algos.quantile import quantile_with_mask as quantile_with_mask
from pandas.core.arraylike import OpsMixin as OpsMixin
from pandas.core.arrays import BooleanArray as BooleanArray, FloatingArray as FloatingArray
from pandas.core.arrays._utils import to_numpy_dtype_inference as to_numpy_dtype_inference
from pandas.core.arrays.base import ExtensionArray as ExtensionArray
from pandas.core.construction import ensure_wrapped_if_datetimelike as ensure_wrapped_if_datetimelike, extract_array as extract_array
from pandas.core.dtypes.base import ExtensionDtype as ExtensionDtype
from pandas.core.dtypes.common import is_bool as is_bool, is_integer_dtype as is_integer_dtype, is_list_like as is_list_like, is_scalar as is_scalar, is_string_dtype as is_string_dtype, pandas_dtype as pandas_dtype
from pandas.core.dtypes.dtypes import BaseMaskedDtype as BaseMaskedDtype
from pandas.core.dtypes.missing import array_equivalent as array_equivalent, is_valid_na_for_dtype as is_valid_na_for_dtype, isna as isna, notna as notna
from pandas.core.indexers import check_array_indexer as check_array_indexer
from pandas.core.ops import invalid_comparison as invalid_comparison
from pandas.core.util.hashing import hash_array as hash_array
from pandas.errors import AbstractMethodError as AbstractMethodError
from pandas.util._decorators import doc as doc
from pandas.util._validators import validate_fillna_kwargs as validate_fillna_kwargs
from typing import Any, Callable, Literal, overload

class BaseMaskedArray(OpsMixin, ExtensionArray):
    """
    Base class for masked arrays (which use _data and _mask to store the data).

    numpy based
    """
    _internal_fill_value: Scalar
    _data: np.ndarray
    _mask: npt.NDArray[np.bool_]
    _truthy_value = Scalar
    _falsey_value = Scalar
    @classmethod
    def _simple_new(cls, values: np.ndarray, mask: npt.NDArray[np.bool_]) -> Self: ...
    def __init__(self, values: np.ndarray, mask: npt.NDArray[np.bool_], copy: bool = False) -> None: ...
    @classmethod
    def _from_sequence(cls, scalars, *, dtype: Incomplete | None = None, copy: bool = False) -> Self: ...
    @classmethod
    def _empty(cls, shape: Shape, dtype: ExtensionDtype): ...
    def _formatter(self, boxed: bool = False) -> Callable[[Any], str | None]: ...
    @property
    def dtype(self) -> BaseMaskedDtype: ...
    @overload
    def __getitem__(self, item: ScalarIndexer) -> Any: ...
    @overload
    def __getitem__(self, item: SequenceIndexer) -> Self: ...
    def _pad_or_backfill(self, *, method: FillnaOptions, limit: int | None = None, limit_area: Literal['inside', 'outside'] | None = None, copy: bool = True) -> Self: ...
    def fillna(self, value: Incomplete | None = None, method: Incomplete | None = None, limit: int | None = None, copy: bool = True) -> Self: ...
    @classmethod
    def _coerce_to_array(cls, values, *, dtype: DtypeObj, copy: bool = False) -> tuple[np.ndarray, np.ndarray]: ...
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
    @property
    def shape(self) -> Shape: ...
    @property
    def ndim(self) -> int: ...
    def swapaxes(self, axis1, axis2) -> Self: ...
    def delete(self, loc, axis: AxisInt = 0) -> Self: ...
    def reshape(self, *args, **kwargs) -> Self: ...
    def ravel(self, *args, **kwargs) -> Self: ...
    @property
    def T(self) -> Self: ...
    def round(self, decimals: int = 0, *args, **kwargs):
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
    def to_numpy(self, dtype: npt.DTypeLike | None = None, copy: bool = False, na_value: object = ...) -> np.ndarray:
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
    def tolist(self): ...
    @overload
    def astype(self, dtype: npt.DTypeLike, copy: bool = ...) -> np.ndarray: ...
    @overload
    def astype(self, dtype: ExtensionDtype, copy: bool = ...) -> ExtensionArray: ...
    @overload
    def astype(self, dtype: AstypeArg, copy: bool = ...) -> ArrayLike: ...
    __array_priority__: int
    def __array__(self, dtype: NpDtype | None = None, copy: bool | None = None) -> np.ndarray:
        """
        the array interface, return my values
        We return an object array here to preserve our scalar values
        """
    _HANDLED_TYPES: tuple[type, ...]
    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs): ...
    def __arrow_array__(self, type: Incomplete | None = None):
        """
        Convert myself into a pyarrow Array.
        """
    @property
    def _hasna(self) -> bool: ...
    def _propagate_mask(self, mask: npt.NDArray[np.bool_] | None, other) -> npt.NDArray[np.bool_]: ...
    def _arith_method(self, other, op): ...
    _logical_method = _arith_method
    def _cmp_method(self, other, op) -> BooleanArray: ...
    def _maybe_mask_result(self, result: np.ndarray | tuple[np.ndarray, np.ndarray], mask: np.ndarray):
        """
        Parameters
        ----------
        result : array-like or tuple[array-like]
        mask : array-like bool
        """
    def isna(self) -> np.ndarray: ...
    @property
    def _na_value(self): ...
    @property
    def nbytes(self) -> int: ...
    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[Self], axis: AxisInt = 0) -> Self: ...
    def _hash_pandas_object(self, *, encoding: str, hash_key: str, categorize: bool) -> npt.NDArray[np.uint64]: ...
    def take(self, indexer, *, allow_fill: bool = False, fill_value: Scalar | None = None, axis: AxisInt = 0) -> Self: ...
    def isin(self, values: ArrayLike) -> BooleanArray: ...
    def copy(self) -> Self: ...
    def duplicated(self, keep: Literal['first', 'last', False] = 'first') -> npt.NDArray[np.bool_]: ...
    def unique(self) -> Self:
        """
        Compute the BaseMaskedArray of unique values.

        Returns
        -------
        uniques : BaseMaskedArray
        """
    def searchsorted(self, value: NumpyValueArrayLike | ExtensionArray, side: Literal['left', 'right'] = 'left', sorter: NumpySorter | None = None) -> npt.NDArray[np.intp] | np.intp: ...
    def factorize(self, use_na_sentinel: bool = True) -> tuple[np.ndarray, ExtensionArray]: ...
    def _values_for_argsort(self) -> np.ndarray: ...
    def value_counts(self, dropna: bool = True) -> Series:
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
    def _mode(self, dropna: bool = True) -> Self: ...
    def equals(self, other) -> bool: ...
    def _quantile(self, qs: npt.NDArray[np.float64], interpolation: str) -> BaseMaskedArray:
        """
        Dispatch to quantile_with_mask, needed because we do not have
        _from_factorized.

        Notes
        -----
        We assume that all impacted cases are 1D-only.
        """
    def _reduce(self, name: str, *, skipna: bool = True, keepdims: bool = False, **kwargs): ...
    def _wrap_reduction_result(self, name: str, result, *, skipna, axis): ...
    def _wrap_na_result(self, *, name, axis, mask_size): ...
    def _wrap_min_count_reduction_result(self, name: str, result, *, skipna, min_count, axis): ...
    def sum(self, *, skipna: bool = True, min_count: int = 0, axis: AxisInt | None = 0, **kwargs): ...
    def prod(self, *, skipna: bool = True, min_count: int = 0, axis: AxisInt | None = 0, **kwargs): ...
    def mean(self, *, skipna: bool = True, axis: AxisInt | None = 0, **kwargs): ...
    def var(self, *, skipna: bool = True, axis: AxisInt | None = 0, ddof: int = 1, **kwargs): ...
    def std(self, *, skipna: bool = True, axis: AxisInt | None = 0, ddof: int = 1, **kwargs): ...
    def min(self, *, skipna: bool = True, axis: AxisInt | None = 0, **kwargs): ...
    def max(self, *, skipna: bool = True, axis: AxisInt | None = 0, **kwargs): ...
    def map(self, mapper, na_action: Incomplete | None = None): ...
    def any(self, *, skipna: bool = True, axis: AxisInt | None = 0, **kwargs):
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
    def all(self, *, skipna: bool = True, axis: AxisInt | None = 0, **kwargs):
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
    def _accumulate(self, name: str, *, skipna: bool = True, **kwargs) -> BaseMaskedArray: ...
    def _groupby_op(self, *, how: str, has_dropped_na: bool, min_count: int, ngroups: int, ids: npt.NDArray[np.intp], **kwargs): ...

def transpose_homogeneous_masked_arrays(masked_arrays: Sequence[BaseMaskedArray]) -> list[BaseMaskedArray]:
    """Transpose masked arrays in a list, but faster.

    Input should be a list of 1-dim masked arrays of equal length and all have the
    same dtype. The caller is responsible for ensuring validity of input data.
    """
