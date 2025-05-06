import numpy as np
from _typeshed import Incomplete
from collections.abc import Hashable, Iterator
from pandas._libs import index as libindex, lib as lib
from pandas._typing import Axis as Axis, Dtype as Dtype, NaPosition as NaPosition, Self as Self, npt as npt
from pandas.core.dtypes.common import ensure_platform_int as ensure_platform_int, ensure_python_int as ensure_python_int, is_float as is_float, is_integer as is_integer, is_scalar as is_scalar, is_signed_integer_dtype as is_signed_integer_dtype
from pandas.core.indexes.base import Index as Index, maybe_extract_name as maybe_extract_name
from pandas.util._decorators import cache_readonly as cache_readonly, deprecate_nonkeyword_arguments as deprecate_nonkeyword_arguments, doc as doc
from typing import Any, Literal, overload

from collections.abc import Callable

_empty_range: Incomplete
_dtype_int64: Incomplete

class RangeIndex(Index):
    '''
    Immutable Index implementing a monotonic integer range.

    RangeIndex is a memory-saving special case of an Index limited to representing
    monotonic ranges with a 64-bit dtype. Using RangeIndex may in some instances
    improve computing speed.

    This is the default index type used
    by DataFrame and Series when no explicit index is provided by the user.

    Parameters
    ----------
    start : int (default: 0), range, or other RangeIndex instance
        If int and "stop" is not given, interpreted as "stop" instead.
    stop : int (default: 0)
    step : int (default: 1)
    dtype : np.int64
        Unused, accepted for homogeneity with other index types.
    copy : bool, default False
        Unused, accepted for homogeneity with other index types.
    name : object, optional
        Name to be stored in the index.

    Attributes
    ----------
    start
    stop
    step

    Methods
    -------
    from_range

    See Also
    --------
    Index : The base pandas Index type.

    Examples
    --------
    >>> list(pd.RangeIndex(5))
    [0, 1, 2, 3, 4]

    >>> list(pd.RangeIndex(-2, 4))
    [-2, -1, 0, 1, 2, 3]

    >>> list(pd.RangeIndex(0, 10, 2))
    [0, 2, 4, 6, 8]

    >>> list(pd.RangeIndex(2, -10, -3))
    [2, -1, -4, -7]

    >>> list(pd.RangeIndex(0))
    []

    >>> list(pd.RangeIndex(1, 0))
    []
    '''
    _typ: str
    _dtype_validation_metadata: Incomplete
    _range: range
    _values: np.ndarray
    @property
    def _engine_type(self) -> type[libindex.Int64Engine]: ...
    def __new__(cls, start: Incomplete | None = None, stop: Incomplete | None = None, step: Incomplete | None = None, dtype: Dtype | None = None, copy: bool = False, name: Hashable | None = None) -> Self: ...
    @classmethod
    def from_range(cls, data: range, name: Incomplete | None = None, dtype: Dtype | None = None) -> Self:
        """
        Create :class:`pandas.RangeIndex` from a ``range`` object.

        Returns
        -------
        RangeIndex

        Examples
        --------
        >>> pd.RangeIndex.from_range(range(5))
        RangeIndex(start=0, stop=5, step=1)

        >>> pd.RangeIndex.from_range(range(2, -10, -3))
        RangeIndex(start=2, stop=-10, step=-3)
        """
    @classmethod
    def _simple_new(cls, values: range, name: Hashable | None = None) -> Self: ...
    @classmethod
    def _validate_dtype(cls, dtype: Dtype | None) -> None: ...
    def _constructor(self) -> type[Index]:
        """return the class to use for construction"""
    def _data(self) -> np.ndarray:
        """
        An int array that for performance reasons is created only when needed.

        The constructed array is saved in ``_cache``.
        """
    def _get_data_as_items(self) -> list[tuple[str, int]]:
        """return a list of tuples of start, stop, step"""
    def __reduce__(self): ...
    def _format_attrs(self):
        """
        Return a list of tuples of the (attr, formatted_value)
        """
    def _format_with_header(self, *, header: list[str], na_rep: str) -> list[str]: ...
    @property
    def start(self) -> int:
        """
        The value of the `start` parameter (``0`` if this was not supplied).

        Examples
        --------
        >>> idx = pd.RangeIndex(5)
        >>> idx.start
        0

        >>> idx = pd.RangeIndex(2, -10, -3)
        >>> idx.start
        2
        """
    @property
    def stop(self) -> int:
        """
        The value of the `stop` parameter.

        Examples
        --------
        >>> idx = pd.RangeIndex(5)
        >>> idx.stop
        5

        >>> idx = pd.RangeIndex(2, -10, -3)
        >>> idx.stop
        -10
        """
    @property
    def step(self) -> int:
        """
        The value of the `step` parameter (``1`` if this was not supplied).

        Examples
        --------
        >>> idx = pd.RangeIndex(5)
        >>> idx.step
        1

        >>> idx = pd.RangeIndex(2, -10, -3)
        >>> idx.step
        -3

        Even if :class:`pandas.RangeIndex` is empty, ``step`` is still ``1`` if
        not supplied.

        >>> idx = pd.RangeIndex(1, 0)
        >>> idx.step
        1
        """
    def nbytes(self) -> int:
        """
        Return the number of bytes in the underlying data.
        """
    def memory_usage(self, deep: bool = False) -> int:
        """
        Memory usage of my values

        Parameters
        ----------
        deep : bool
            Introspect the data deeply, interrogate
            `object` dtypes for system-level memory consumption

        Returns
        -------
        bytes used

        Notes
        -----
        Memory usage does not include memory consumed by elements that
        are not components of the array if deep=False

        See Also
        --------
        numpy.ndarray.nbytes
        """
    @property
    def dtype(self) -> np.dtype: ...
    @property
    def is_unique(self) -> bool:
        """return if the index has unique values"""
    def is_monotonic_increasing(self) -> bool: ...
    def is_monotonic_decreasing(self) -> bool: ...
    def __contains__(self, key: Any) -> bool: ...
    @property
    def inferred_type(self) -> str: ...
    def get_loc(self, key) -> int: ...
    def _get_indexer(self, target: Index, method: str | None = None, limit: int | None = None, tolerance: Incomplete | None = None) -> npt.NDArray[np.intp]: ...
    def _should_fallback_to_positional(self) -> bool:
        """
        Should an integer key be treated as positional?
        """
    def tolist(self) -> list[int]: ...
    def __iter__(self) -> Iterator[int]: ...
    def _shallow_copy(self, values, name: Hashable = ...): ...
    def _view(self) -> Self: ...
    def copy(self, name: Hashable | None = None, deep: bool = False) -> Self: ...
    def _minmax(self, meth: str): ...
    def min(self, axis: Incomplete | None = None, skipna: bool = True, *args, **kwargs) -> int:
        """The minimum value of the RangeIndex"""
    def max(self, axis: Incomplete | None = None, skipna: bool = True, *args, **kwargs) -> int:
        """The maximum value of the RangeIndex"""
    def argsort(self, *args, **kwargs) -> npt.NDArray[np.intp]:
        """
        Returns the indices that would sort the index and its
        underlying data.

        Returns
        -------
        np.ndarray[np.intp]

        See Also
        --------
        numpy.ndarray.argsort
        """
    def factorize(self, sort: bool = False, use_na_sentinel: bool = True) -> tuple[npt.NDArray[np.intp], RangeIndex]: ...
    def equals(self, other: object) -> bool:
        """
        Determines if two Index objects contain the same elements.
        """
    @overload
    def sort_values(self, *, return_indexer: Literal[False] = False, ascending: bool = ..., na_position: NaPosition = ..., key: Callable | None = ...) -> Self: ...
    @overload
    def sort_values(self, *, return_indexer: Literal[True], ascending: bool = ..., na_position: NaPosition = ..., key: Callable | None = ...) -> tuple[Self, np.ndarray | RangeIndex]: ...
    @overload
    def sort_values(self, *, return_indexer: bool = ..., ascending: bool = ..., na_position: NaPosition = ..., key: Callable | None = ...) -> Self | tuple[Self, np.ndarray | RangeIndex]: ...
    def _intersection(self, other: Index, sort: bool = False): ...
    def _min_fitting_element(self, lower_limit: int) -> int:
        """Returns the smallest element greater than or equal to the limit"""
    def _extended_gcd(self, a: int, b: int) -> tuple[int, int, int]:
        """
        Extended Euclidean algorithms to solve Bezout's identity:
           a*x + b*y = gcd(x, y)
        Finds one particular solution for x, y: s, t
        Returns: gcd, s, t
        """
    def _range_in_self(self, other: range) -> bool:
        """Check if other range is contained in self"""
    def _union(self, other: Index, sort: bool | None):
        """
        Form the union of two Index objects and sorts if possible

        Parameters
        ----------
        other : Index or array-like

        sort : bool or None, default None
            Whether to sort (monotonically increasing) the resulting index.
            ``sort=None|True`` returns a ``RangeIndex`` if possible or a sorted
            ``Index`` with a int64 dtype if not.
            ``sort=False`` can return a ``RangeIndex`` if self is monotonically
            increasing and other is fully contained in self. Otherwise, returns
            an unsorted ``Index`` with an int64 dtype.

        Returns
        -------
        union : Index
        """
    def _difference(self, other, sort: Incomplete | None = None): ...
    def symmetric_difference(self, other, result_name: Hashable | None = None, sort: Incomplete | None = None): ...
    def delete(self, loc) -> Index: ...
    def insert(self, loc: int, item) -> Index: ...
    def _concat(self, indexes: list[Index], name: Hashable) -> Index:
        '''
        Overriding parent method for the case of all RangeIndex instances.

        When all members of "indexes" are of type RangeIndex: result will be
        RangeIndex if possible, Index with a int64 dtype otherwise. E.g.:
        indexes = [RangeIndex(3), RangeIndex(3, 6)] -> RangeIndex(6)
        indexes = [RangeIndex(3), RangeIndex(4, 6)] -> Index([0,1,2,4,5], dtype=\'int64\')
        '''
    def __len__(self) -> int:
        """
        return the length of the RangeIndex
        """
    @property
    def size(self) -> int: ...
    def __getitem__(self, key):
        """
        Conserve RangeIndex type for scalar and slice keys.
        """
    def _getitem_slice(self, slobj: slice) -> Self:
        """
        Fastpath for __getitem__ when we know we have a slice.
        """
    def __floordiv__(self, other): ...
    def all(self, *args, **kwargs) -> bool: ...
    def any(self, *args, **kwargs) -> bool: ...
    def _cmp_method(self, other, op): ...
    def _arith_method(self, other, op):
        """
        Parameters
        ----------
        other : Any
        op : callable that accepts 2 params
            perform the binary op
        """
    def take(self, indices, axis: Axis = 0, allow_fill: bool = True, fill_value: Incomplete | None = None, **kwargs) -> Index: ...
