import np
import npt
import pandas._libs.lib as lib
import pandas._libs.sparse as splib
import pandas.compat.numpy.function as nv
import pandas.core.algorithms as algos
import pandas.core.arraylike
import pandas.core.arraylike as arraylike
import pandas.core.arrays.base
import pandas.core.base
import pandas.core.common as com
import pandas.io.formats.printing as printing
from _typeshed import Incomplete
from builtins import ellipsis
from pandas._libs.lib import is_integer as is_integer, is_list_like as is_list_like, is_scalar as is_scalar
from pandas._libs.sparse import BlockIndex as BlockIndex, IntIndex as IntIndex, SparseIndex as SparseIndex
from pandas._libs.tslibs.nattype import NaT as NaT
from pandas.core.arraylike import OpsMixin as OpsMixin
from pandas.core.arrays.base import ExtensionArray as ExtensionArray
from pandas.core.base import PandasObject as PandasObject
from pandas.core.construction import ensure_wrapped_if_datetimelike as ensure_wrapped_if_datetimelike, extract_array as extract_array, sanitize_array as sanitize_array
from pandas.core.dtypes.astype import astype_array as astype_array
from pandas.core.dtypes.cast import construct_1d_arraylike_from_scalar as construct_1d_arraylike_from_scalar, find_common_type as find_common_type, maybe_box_datetimelike as maybe_box_datetimelike
from pandas.core.dtypes.common import is_bool_dtype as is_bool_dtype, is_object_dtype as is_object_dtype, is_string_dtype as is_string_dtype, pandas_dtype as pandas_dtype
from pandas.core.dtypes.dtypes import DatetimeTZDtype as DatetimeTZDtype, SparseDtype as SparseDtype
from pandas.core.dtypes.generic import ABCIndex as ABCIndex, ABCSeries as ABCSeries
from pandas.core.dtypes.missing import isna as isna, na_value_for_dtype as na_value_for_dtype, notna as notna
from pandas.core.indexers.utils import check_array_indexer as check_array_indexer, unpack_tuple_and_ellipses as unpack_tuple_and_ellipses
from pandas.core.nanops import check_below_min_count as check_below_min_count
from pandas.errors import PerformanceWarning as PerformanceWarning
from pandas.util._decorators import doc as doc
from pandas.util._exceptions import find_stack_level as find_stack_level
from pandas.util._validators import validate_bool_kwarg as validate_bool_kwarg, validate_insert_loc as validate_insert_loc
from typing import Any, Callable, ClassVar, Literal

TYPE_CHECKING: bool
_sparray_doc_kwargs: dict
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
def _wrap_result(name: str, data, sparse_index, fill_value, dtype: Dtype | None) -> SparseArray:
    """
    wrap op result to have correct dtype
    """

class SparseArray(pandas.core.arraylike.OpsMixin, pandas.core.base.PandasObject, pandas.core.arrays.base.ExtensionArray):
    _subtyp: ClassVar[str] = ...
    _hidden_attrs: ClassVar[frozenset] = ...
    _HANDLED_TYPES: ClassVar[tuple] = ...
    fill_value: Incomplete
    def __init__(self, data, sparse_index, fill_value, kind: SparseIndexKind = ..., dtype: Dtype | None, copy: bool = ...) -> None: ...
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
    def __array__(self, dtype: NpDtype | None, copy: bool | None) -> np.ndarray: ...
    def __setitem__(self, key, value) -> None: ...
    @classmethod
    def _from_sequence(cls, scalars, *, dtype: Dtype | None, copy: bool = ...): ...
    @classmethod
    def _from_factorized(cls, values, original): ...
    def __len__(self) -> int: ...
    def _fill_value_matches(self, fill_value) -> bool: ...
    def isna(self) -> Self: ...
    def _pad_or_backfill(self, *, method: FillnaOptions, limit: int | None, limit_area: Literal['inside', 'outside'] | None, copy: bool = ...) -> Self: ...
    def fillna(self, value, method: FillnaOptions | None, limit: int | None, copy: bool = ...) -> Self:
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
    def shift(self, periods: int = ..., fill_value) -> Self: ...
    def _first_fill_value_loc(self):
        """
        Get the location of the first fill value.

        Returns
        -------
        int
        """
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
    def unique(self) -> Self: ...
    def _values_for_factorize(self): ...
    def factorize(self, use_na_sentinel: bool = ...) -> tuple[np.ndarray, SparseArray]: ...
    def value_counts(self, dropna: bool = ...) -> Series:
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
    def __getitem__(self, key: PositionalIndexer | tuple[int | ellipsis, ...]) -> Self | Any: ...
    def _get_val_at(self, loc): ...
    def take(self, indices, *, allow_fill: bool = ..., fill_value) -> Self: ...
    def _take_with_fill(self, indices, fill_value) -> np.ndarray: ...
    def _take_without_fill(self, indices) -> Self: ...
    def searchsorted(self, v: ArrayLike | object, side: Literal['left', 'right'] = ..., sorter: NumpySorter | None) -> npt.NDArray[np.intp] | np.intp: ...
    def copy(self) -> Self: ...
    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[Self]) -> Self: ...
    def astype(self, dtype: AstypeArg | None, copy: bool = ...):
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
    def map(self, mapper, na_action) -> Self:
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
    def nonzero(self) -> tuple[npt.NDArray[np.int32]]: ...
    def _reduce(self, name: str, *, skipna: bool = ..., keepdims: bool = ..., **kwargs): ...
    def all(self, axis, *args, **kwargs):
        """
        Tests whether all elements evaluate True

        Returns
        -------
        all : bool

        See Also
        --------
        numpy.all
        """
    def any(self, axis: AxisInt = ..., *args, **kwargs) -> bool:
        """
        Tests whether at least one of elements evaluate True

        Returns
        -------
        any : bool

        See Also
        --------
        numpy.any
        """
    def sum(self, axis: AxisInt = ..., min_count: int = ..., skipna: bool = ..., *args, **kwargs) -> Scalar:
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
    def cumsum(self, axis: AxisInt = ..., *args, **kwargs) -> SparseArray:
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
    def mean(self, axis: Axis = ..., *args, **kwargs):
        """
        Mean of non-NA/null values

        Returns
        -------
        mean : float
        """
    def max(self, *, axis: AxisInt | None, skipna: bool = ...):
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
    def min(self, *, axis: AxisInt | None, skipna: bool = ...):
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
    def argmax(self, skipna: bool = ...) -> int: ...
    def argmin(self, skipna: bool = ...) -> int: ...
    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs): ...
    def _arith_method(self, other, op): ...
    def _cmp_method(self, other, op) -> SparseArray: ...
    def _logical_method(self, other, op) -> SparseArray: ...
    def _unary_method(self, op) -> SparseArray: ...
    def __pos__(self) -> SparseArray: ...
    def __neg__(self) -> SparseArray: ...
    def __invert__(self) -> SparseArray: ...
    def __abs__(self) -> SparseArray: ...
    def _formatter(self, boxed: bool = ...): ...
    @property
    def sp_index(self): ...
    @property
    def sp_values(self): ...
    @property
    def dtype(self): ...
    @property
    def kind(self): ...
    @property
    def _valid_sp_values(self): ...
    @property
    def _null_fill_value(self): ...
    @property
    def nbytes(self): ...
    @property
    def density(self): ...
    @property
    def npoints(self): ...
def _make_sparse(arr: np.ndarray, kind: SparseIndexKind = ..., fill_value, dtype: np.dtype | None):
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
def make_sparse_index(length: int, indices, kind: SparseIndexKind) -> SparseIndex: ...
