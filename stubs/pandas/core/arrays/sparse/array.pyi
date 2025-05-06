import numpy as np
from _typeshed import Incomplete
from collections.abc import Sequence
from enum import Enum
from pandas import Series as Series
from pandas._libs.sparse import BlockIndex as BlockIndex, IntIndex as IntIndex, SparseIndex as SparseIndex
from pandas._typing import ArrayLike as ArrayLike, AstypeArg as AstypeArg, Axis as Axis, AxisInt as AxisInt, Dtype as Dtype, FillnaOptions as FillnaOptions, NpDtype as NpDtype, NumpySorter as NumpySorter, PositionalIndexer as PositionalIndexer, Scalar as Scalar, ScalarIndexer as ScalarIndexer, Self as Self, SequenceIndexer as SequenceIndexer, npt as npt
from pandas.core.arraylike import OpsMixin as OpsMixin
from pandas.core.arrays import ExtensionArray as ExtensionArray
from pandas.core.base import PandasObject as PandasObject
from pandas.core.construction import ensure_wrapped_if_datetimelike as ensure_wrapped_if_datetimelike, extract_array as extract_array, sanitize_array as sanitize_array
from pandas.core.dtypes.cast import construct_1d_arraylike_from_scalar as construct_1d_arraylike_from_scalar, find_common_type as find_common_type, maybe_box_datetimelike as maybe_box_datetimelike
from pandas.core.dtypes.common import is_bool_dtype as is_bool_dtype, is_integer as is_integer, is_list_like as is_list_like, is_object_dtype as is_object_dtype, is_scalar as is_scalar, is_string_dtype as is_string_dtype, pandas_dtype as pandas_dtype
from pandas.core.dtypes.dtypes import DatetimeTZDtype as DatetimeTZDtype, SparseDtype as SparseDtype
from pandas.core.dtypes.generic import ABCIndex as ABCIndex, ABCSeries as ABCSeries
from pandas.core.dtypes.missing import isna as isna, na_value_for_dtype as na_value_for_dtype, notna as notna
from pandas.core.indexers import check_array_indexer as check_array_indexer, unpack_tuple_and_ellipses as unpack_tuple_and_ellipses
from pandas.util._validators import validate_bool_kwarg as validate_bool_kwarg, validate_insert_loc as validate_insert_loc
from scipy.sparse import spmatrix
from typing import Any, Literal, overload

from collections.abc import Callable

class ellipsis(Enum):
    Ellipsis = '...'

Ellipsis: Incomplete
SparseIndexKind: Incomplete
_sparray_doc_kwargs: Incomplete

def _get_fill(arr: SparseArray) -> np.ndarray:
    """
    Create a 0-dim ndarray containing the fill value

    Parameters
    ----------
    arr : SparseArray

    Returns
    -------
    fill_value : ndarray
        0-dim ndarray with just the fill value.

    Notes
    -----
    coerce fill_value to arr dtype if possible
    int64 SparseArray can have NaN as fill_value if there is no missing
    """
def _sparse_array_op(left: SparseArray, right: SparseArray, op: Callable, name: str) -> SparseArray:
    """
    Perform a binary operation between two arrays.

    Parameters
    ----------
    left : Union[SparseArray, ndarray]
    right : Union[SparseArray, ndarray]
    op : Callable
        The binary operation to perform
    name str
        Name of the callable.

    Returns
    -------
    SparseArray
    """
def _wrap_result(name: str, data, sparse_index, fill_value, dtype: Dtype | None = None) -> SparseArray:
    """
    wrap op result to have correct dtype
    """

class SparseArray(OpsMixin, PandasObject, ExtensionArray):
    """
    An ExtensionArray for storing sparse data.

    Parameters
    ----------
    data : array-like or scalar
        A dense array of values to store in the SparseArray. This may contain
        `fill_value`.
    sparse_index : SparseIndex, optional
    fill_value : scalar, optional
        Elements in data that are ``fill_value`` are not stored in the
        SparseArray. For memory savings, this should be the most common value
        in `data`. By default, `fill_value` depends on the dtype of `data`:

        =========== ==========
        data.dtype  na_value
        =========== ==========
        float       ``np.nan``
        int         ``0``
        bool        False
        datetime64  ``pd.NaT``
        timedelta64 ``pd.NaT``
        =========== ==========

        The fill value is potentially specified in three ways. In order of
        precedence, these are

        1. The `fill_value` argument
        2. ``dtype.fill_value`` if `fill_value` is None and `dtype` is
           a ``SparseDtype``
        3. ``data.dtype.fill_value`` if `fill_value` is None and `dtype`
           is not a ``SparseDtype`` and `data` is a ``SparseArray``.

    kind : str
        Can be 'integer' or 'block', default is 'integer'.
        The type of storage for sparse locations.

        * 'block': Stores a `block` and `block_length` for each
          contiguous *span* of sparse values. This is best when
          sparse data tends to be clumped together, with large
          regions of ``fill-value`` values between sparse values.
        * 'integer': uses an integer to store the location of
          each sparse value.

    dtype : np.dtype or SparseDtype, optional
        The dtype to use for the SparseArray. For numpy dtypes, this
        determines the dtype of ``self.sp_values``. For SparseDtype,
        this determines ``self.sp_values`` and ``self.fill_value``.
    copy : bool, default False
        Whether to explicitly copy the incoming `data` array.

    Attributes
    ----------
    None

    Methods
    -------
    None

    Examples
    --------
    >>> from pandas.arrays import SparseArray
    >>> arr = SparseArray([0, 0, 1, 2])
    >>> arr
    [0, 0, 1, 2]
    Fill: 0
    IntIndex
    Indices: array([2, 3], dtype=int32)
    """
    _subtyp: str
    _hidden_attrs: Incomplete
    _sparse_index: SparseIndex
    _sparse_values: np.ndarray
    _dtype: SparseDtype
    def __init__(self, data, sparse_index: Incomplete | None = None, fill_value: Incomplete | None = None, kind: SparseIndexKind = 'integer', dtype: Dtype | None = None, copy: bool = False) -> None: ...
    @classmethod
    def _simple_new(cls, sparse_array: np.ndarray, sparse_index: SparseIndex, dtype: SparseDtype) -> Self: ...
    @classmethod
    def from_spmatrix(cls, data: spmatrix) -> Self:
        """
        Create a SparseArray from a scipy.sparse matrix.

        Parameters
        ----------
        data : scipy.sparse.sp_matrix
            This should be a SciPy sparse matrix where the size
            of the second dimension is 1. In other words, a
            sparse matrix with a single column.

        Returns
        -------
        SparseArray

        Examples
        --------
        >>> import scipy.sparse
        >>> mat = scipy.sparse.coo_matrix((4, 1))
        >>> pd.arrays.SparseArray.from_spmatrix(mat)
        [0.0, 0.0, 0.0, 0.0]
        Fill: 0.0
        IntIndex
        Indices: array([], dtype=int32)
        """
    def __array__(self, dtype: NpDtype | None = None, copy: bool | None = None) -> np.ndarray: ...
    def __setitem__(self, key, value) -> None: ...
    @classmethod
    def _from_sequence(cls, scalars, *, dtype: Dtype | None = None, copy: bool = False): ...
    @classmethod
    def _from_factorized(cls, values, original): ...
    @property
    def sp_index(self) -> SparseIndex:
        """
        The SparseIndex containing the location of non- ``fill_value`` points.
        """
    @property
    def sp_values(self) -> np.ndarray:
        """
        An ndarray containing the non- ``fill_value`` values.

        Examples
        --------
        >>> from pandas.arrays import SparseArray
        >>> s = SparseArray([0, 0, 1, 0, 2], fill_value=0)
        >>> s.sp_values
        array([1, 2])
        """
    @property
    def dtype(self) -> SparseDtype: ...
    @property
    def fill_value(self):
        '''
        Elements in `data` that are `fill_value` are not stored.

        For memory savings, this should be the most common value in the array.

        Examples
        --------
        >>> ser = pd.Series([0, 0, 2, 2, 2], dtype="Sparse[int]")
        >>> ser.sparse.fill_value
        0
        >>> spa_dtype = pd.SparseDtype(dtype=np.int32, fill_value=2)
        >>> ser = pd.Series([0, 0, 2, 2, 2], dtype=spa_dtype)
        >>> ser.sparse.fill_value
        2
        '''
    @fill_value.setter
    def fill_value(self, value) -> None: ...
    @property
    def kind(self) -> SparseIndexKind:
        """
        The kind of sparse index for this array. One of {'integer', 'block'}.
        """
    @property
    def _valid_sp_values(self) -> np.ndarray: ...
    def __len__(self) -> int: ...
    @property
    def _null_fill_value(self) -> bool: ...
    def _fill_value_matches(self, fill_value) -> bool: ...
    @property
    def nbytes(self) -> int: ...
    @property
    def density(self) -> float:
        """
        The percent of non- ``fill_value`` points, as decimal.

        Examples
        --------
        >>> from pandas.arrays import SparseArray
        >>> s = SparseArray([0, 0, 1, 1, 1], fill_value=0)
        >>> s.density
        0.6
        """
    @property
    def npoints(self) -> int:
        """
        The number of non- ``fill_value`` points.

        Examples
        --------
        >>> from pandas.arrays import SparseArray
        >>> s = SparseArray([0, 0, 1, 1, 1], fill_value=0)
        >>> s.npoints
        3
        """
    def isna(self) -> Self: ...
    def _pad_or_backfill(self, *, method: FillnaOptions, limit: int | None = None, limit_area: Literal['inside', 'outside'] | None = None, copy: bool = True) -> Self: ...
    def fillna(self, value: Incomplete | None = None, method: FillnaOptions | None = None, limit: int | None = None, copy: bool = True) -> Self:
        """
        Fill missing values with `value`.

        Parameters
        ----------
        value : scalar, optional
        method : str, optional

            .. warning::

               Using 'method' will result in high memory use,
               as all `fill_value` methods will be converted to
               an in-memory ndarray

        limit : int, optional

        copy: bool, default True
            Ignored for SparseArray.

        Returns
        -------
        SparseArray

        Notes
        -----
        When `value` is specified, the result's ``fill_value`` depends on
        ``self.fill_value``. The goal is to maintain low-memory use.

        If ``self.fill_value`` is NA, the result dtype will be
        ``SparseDtype(self.dtype, fill_value=value)``. This will preserve
        amount of memory used before and after filling.

        When ``self.fill_value`` is not NA, the result dtype will be
        ``self.dtype``. Again, this preserves the amount of memory used.
        """
    def shift(self, periods: int = 1, fill_value: Incomplete | None = None) -> Self: ...
    def _first_fill_value_loc(self):
        """
        Get the location of the first fill value.

        Returns
        -------
        int
        """
    def duplicated(self, keep: Literal['first', 'last', False] = 'first') -> npt.NDArray[np.bool_]: ...
    def unique(self) -> Self: ...
    def _values_for_factorize(self): ...
    def factorize(self, use_na_sentinel: bool = True) -> tuple[np.ndarray, SparseArray]: ...
    def value_counts(self, dropna: bool = True) -> Series:
        """
        Returns a Series containing counts of unique values.

        Parameters
        ----------
        dropna : bool, default True
            Don't include counts of NaN, even if NaN is in sp_values.

        Returns
        -------
        counts : Series
        """
    @overload
    def __getitem__(self, key: ScalarIndexer) -> Any: ...
    @overload
    def __getitem__(self, key: SequenceIndexer | tuple[int | ellipsis, ...]) -> Self: ...
    def _get_val_at(self, loc): ...
    def take(self, indices, *, allow_fill: bool = False, fill_value: Incomplete | None = None) -> Self: ...
    def _take_with_fill(self, indices, fill_value: Incomplete | None = None) -> np.ndarray: ...
    def _take_without_fill(self, indices) -> Self: ...
    def searchsorted(self, v: ArrayLike | object, side: Literal['left', 'right'] = 'left', sorter: NumpySorter | None = None) -> npt.NDArray[np.intp] | np.intp: ...
    def copy(self) -> Self: ...
    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[Self]) -> Self: ...
    def astype(self, dtype: AstypeArg | None = None, copy: bool = True):
        '''
        Change the dtype of a SparseArray.

        The output will always be a SparseArray. To convert to a dense
        ndarray with a certain dtype, use :meth:`numpy.asarray`.

        Parameters
        ----------
        dtype : np.dtype or ExtensionDtype
            For SparseDtype, this changes the dtype of
            ``self.sp_values`` and the ``self.fill_value``.

            For other dtypes, this only changes the dtype of
            ``self.sp_values``.

        copy : bool, default True
            Whether to ensure a copy is made, even if not necessary.

        Returns
        -------
        SparseArray

        Examples
        --------
        >>> arr = pd.arrays.SparseArray([0, 0, 1, 2])
        >>> arr
        [0, 0, 1, 2]
        Fill: 0
        IntIndex
        Indices: array([2, 3], dtype=int32)

        >>> arr.astype(SparseDtype(np.dtype(\'int32\')))
        [0, 0, 1, 2]
        Fill: 0
        IntIndex
        Indices: array([2, 3], dtype=int32)

        Using a NumPy dtype with a different kind (e.g. float) will coerce
        just ``self.sp_values``.

        >>> arr.astype(SparseDtype(np.dtype(\'float64\')))
        ... # doctest: +NORMALIZE_WHITESPACE
        [nan, nan, 1.0, 2.0]
        Fill: nan
        IntIndex
        Indices: array([2, 3], dtype=int32)

        Using a SparseDtype, you can also change the fill value as well.

        >>> arr.astype(SparseDtype("float64", fill_value=0.0))
        ... # doctest: +NORMALIZE_WHITESPACE
        [0.0, 0.0, 1.0, 2.0]
        Fill: 0.0
        IntIndex
        Indices: array([2, 3], dtype=int32)
        '''
    def map(self, mapper, na_action: Incomplete | None = None) -> Self:
        """
        Map categories using an input mapping or function.

        Parameters
        ----------
        mapper : dict, Series, callable
            The correspondence from old values to new.
        na_action : {None, 'ignore'}, default None
            If 'ignore', propagate NA values, without passing them to the
            mapping correspondence.

        Returns
        -------
        SparseArray
            The output array will have the same density as the input.
            The output fill value will be the result of applying the
            mapping to ``self.fill_value``

        Examples
        --------
        >>> arr = pd.arrays.SparseArray([0, 1, 2])
        >>> arr.map(lambda x: x + 10)
        [10, 11, 12]
        Fill: 10
        IntIndex
        Indices: array([1, 2], dtype=int32)

        >>> arr.map({0: 10, 1: 11, 2: 12})
        [10, 11, 12]
        Fill: 10
        IntIndex
        Indices: array([1, 2], dtype=int32)

        >>> arr.map(pd.Series([10, 11, 12], index=[0, 1, 2]))
        [10, 11, 12]
        Fill: 10
        IntIndex
        Indices: array([1, 2], dtype=int32)
        """
    def to_dense(self) -> np.ndarray:
        """
        Convert SparseArray to a NumPy array.

        Returns
        -------
        arr : NumPy array
        """
    def _where(self, mask, value): ...
    def __setstate__(self, state) -> None:
        """Necessary for making this object picklable"""
    def nonzero(self) -> tuple[npt.NDArray[np.int32]]: ...
    def _reduce(self, name: str, *, skipna: bool = True, keepdims: bool = False, **kwargs): ...
    def all(self, axis: Incomplete | None = None, *args, **kwargs):
        """
        Tests whether all elements evaluate True

        Returns
        -------
        all : bool

        See Also
        --------
        numpy.all
        """
    def any(self, axis: AxisInt = 0, *args, **kwargs) -> bool:
        """
        Tests whether at least one of elements evaluate True

        Returns
        -------
        any : bool

        See Also
        --------
        numpy.any
        """
    def sum(self, axis: AxisInt = 0, min_count: int = 0, skipna: bool = True, *args, **kwargs) -> Scalar:
        """
        Sum of non-NA/null values

        Parameters
        ----------
        axis : int, default 0
            Not Used. NumPy compatibility.
        min_count : int, default 0
            The required number of valid values to perform the summation. If fewer
            than ``min_count`` valid values are present, the result will be the missing
            value indicator for subarray type.
        *args, **kwargs
            Not Used. NumPy compatibility.

        Returns
        -------
        scalar
        """
    def cumsum(self, axis: AxisInt = 0, *args, **kwargs) -> SparseArray:
        """
        Cumulative sum of non-NA/null values.

        When performing the cumulative summation, any non-NA/null values will
        be skipped. The resulting SparseArray will preserve the locations of
        NaN values, but the fill value will be `np.nan` regardless.

        Parameters
        ----------
        axis : int or None
            Axis over which to perform the cumulative summation. If None,
            perform cumulative summation over flattened array.

        Returns
        -------
        cumsum : SparseArray
        """
    def mean(self, axis: Axis = 0, *args, **kwargs):
        """
        Mean of non-NA/null values

        Returns
        -------
        mean : float
        """
    def max(self, *, axis: AxisInt | None = None, skipna: bool = True):
        """
        Max of array values, ignoring NA values if specified.

        Parameters
        ----------
        axis : int, default 0
            Not Used. NumPy compatibility.
        skipna : bool, default True
            Whether to ignore NA values.

        Returns
        -------
        scalar
        """
    def min(self, *, axis: AxisInt | None = None, skipna: bool = True):
        """
        Min of array values, ignoring NA values if specified.

        Parameters
        ----------
        axis : int, default 0
            Not Used. NumPy compatibility.
        skipna : bool, default True
            Whether to ignore NA values.

        Returns
        -------
        scalar
        """
    def _min_max(self, kind: Literal['min', 'max'], skipna: bool) -> Scalar:
        '''
        Min/max of non-NA/null values

        Parameters
        ----------
        kind : {"min", "max"}
        skipna : bool

        Returns
        -------
        scalar
        '''
    def _argmin_argmax(self, kind: Literal['argmin', 'argmax']) -> int: ...
    def argmax(self, skipna: bool = True) -> int: ...
    def argmin(self, skipna: bool = True) -> int: ...
    _HANDLED_TYPES: Incomplete
    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs): ...
    def _arith_method(self, other, op): ...
    def _cmp_method(self, other, op) -> SparseArray: ...
    _logical_method = _cmp_method
    def _unary_method(self, op) -> SparseArray: ...
    def __pos__(self) -> SparseArray: ...
    def __neg__(self) -> SparseArray: ...
    def __invert__(self) -> SparseArray: ...
    def __abs__(self) -> SparseArray: ...
    def __repr__(self) -> str: ...
    def _formatter(self, boxed: bool = False): ...

def _make_sparse(arr: np.ndarray, kind: SparseIndexKind = 'block', fill_value: Incomplete | None = None, dtype: np.dtype | None = None):
    """
    Convert ndarray to sparse format

    Parameters
    ----------
    arr : ndarray
    kind : {'block', 'integer'}
    fill_value : NaN or another value
    dtype : np.dtype, optional
    copy : bool, default False

    Returns
    -------
    (sparse_values, index, fill_value) : (ndarray, SparseIndex, Scalar)
    """
@overload
def make_sparse_index(length: int, indices, kind: Literal['block']) -> BlockIndex: ...
@overload
def make_sparse_index(length: int, indices, kind: Literal['integer']) -> IntIndex: ...
