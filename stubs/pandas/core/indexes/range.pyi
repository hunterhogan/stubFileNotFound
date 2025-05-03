import np
import npt
import numpy.dtypes
import pandas._libs.index as libindex
import pandas._libs.lib as lib
import pandas.compat.numpy.function as nv
import pandas.core.common as com
import pandas.core.indexes.base
import pandas.core.indexes.base as ibase
import pandas.core.ops as ops
from _typeshed import Incomplete
from collections.abc import Hashable, Iterator
from pandas._libs.algos import ensure_platform_int as ensure_platform_int, unique_deltas as unique_deltas
from pandas._libs.lib import is_float as is_float, is_integer as is_integer, is_scalar as is_scalar, no_default as no_default
from pandas._libs.properties import cache_readonly as cache_readonly
from pandas.core.construction import extract_array as extract_array
from pandas.core.dtypes.common import ensure_python_int as ensure_python_int, is_signed_integer_dtype as is_signed_integer_dtype
from pandas.core.dtypes.generic import ABCTimedeltaIndex as ABCTimedeltaIndex
from pandas.core.indexes.base import Index as Index, maybe_extract_name as maybe_extract_name
from pandas.core.ops.common import unpack_zerodim_and_defer as unpack_zerodim_and_defer
from pandas.util._decorators import deprecate_nonkeyword_arguments as deprecate_nonkeyword_arguments, doc as doc
from typing import Any, Callable, ClassVar

TYPE_CHECKING: bool
_empty_range: range
_dtype_int64: numpy.dtypes.Int64DType

class RangeIndex(pandas.core.indexes.base.Index):
    _typ: ClassVar[str] = ...
    _dtype_validation_metadata: ClassVar[tuple] = ...
    _constructor: Incomplete
    _data: Incomplete
    nbytes: Incomplete
    is_monotonic_increasing: Incomplete
    is_monotonic_decreasing: Incomplete
    _should_fallback_to_positional: Incomplete
    @classmethod
    def __init__(cls, start, stop, step, dtype: Dtype | None, copy: bool = ..., name: Hashable | None) -> Self: ...
    @classmethod
    def from_range(cls, data: range, name, dtype: Dtype | None) -> Self:
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
    def _simple_new(cls, values: range, name: Hashable | None) -> Self: ...
    @classmethod
    def _validate_dtype(cls, dtype: Dtype | None) -> None: ...
    def _get_data_as_items(self) -> list[tuple[str, int]]:
        """return a list of tuples of start, stop, step"""
    def __reduce__(self): ...
    def _format_attrs(self):
        """
        Return a list of tuples of the (attr, formatted_value)
        """
    def _format_with_header(self, *, header: list[str], na_rep: str) -> list[str]: ...
    def memory_usage(self, deep: bool = ...) -> int:
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
    def __contains__(self, key: Any) -> bool: ...
    def get_loc(self, key) -> int:
        """
        Get integer location, slice or boolean mask for requested label.

        Parameters
        ----------
        key : label

        Returns
        -------
        int if unique index, slice if monotonic index, else mask

        Examples
        --------
        >>> unique_index = pd.Index(list('abc'))
        >>> unique_index.get_loc('b')
        1

        >>> monotonic_index = pd.Index(list('abbc'))
        >>> monotonic_index.get_loc('b')
        slice(1, 3, None)

        >>> non_monotonic_index = pd.Index(list('abcb'))
        >>> non_monotonic_index.get_loc('b')
        array([False,  True, False,  True])
        """
    def _get_indexer(self, target: Index, method: str | None, limit: int | None, tolerance) -> npt.NDArray[np.intp]: ...
    def tolist(self) -> list[int]: ...
    def __iter__(self) -> Iterator[int]:
        """
        Return an iterator of the values.

        These are each a scalar type, which is a Python scalar
        (for str, int, float) or a pandas scalar
        (for Timestamp/Timedelta/Interval/Period)

        Returns
        -------
        iterator

        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> for x in s:
        ...     print(x)
        1
        2
        3
        """
    def _shallow_copy(self, values, name: Hashable = ...):
        """
        Create a new Index with the same class as the caller, don't copy the
        data, use the same object attributes with passed in attributes taking
        precedence.

        *this is an internal non-public method*

        Parameters
        ----------
        values : the values to create the new Index, optional
        name : Label, defaults to self.name
        """
    def _view(self) -> Self: ...
    def copy(self, name: Hashable | None, deep: bool = ...) -> Self:
        """
        Make a copy of this object.

        Name is set on the new object.

        Parameters
        ----------
        name : Label, optional
            Set name for new object.
        deep : bool, default False

        Returns
        -------
        Index
            Index refer to new object which is a copy of this object.

        Notes
        -----
        In most cases, there should be no functional difference from using
        ``deep``, but if ``deep`` is passed it will attempt to deepcopy.

        Examples
        --------
        >>> idx = pd.Index(['a', 'b', 'c'])
        >>> new_idx = idx.copy()
        >>> idx is new_idx
        False
        """
    def _minmax(self, meth: str): ...
    def min(self, axis, skipna: bool = ..., *args, **kwargs) -> int:
        """The minimum value of the RangeIndex"""
    def max(self, axis, skipna: bool = ..., *args, **kwargs) -> int:
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
    def factorize(self, sort: bool = ..., use_na_sentinel: bool = ...) -> tuple[npt.NDArray[np.intp], RangeIndex]: ...
    def equals(self, other: object) -> bool:
        """
        Determines if two Index objects contain the same elements.
        """
    def sort_values(self, *, return_indexer: bool = ..., ascending: bool = ..., na_position: NaPosition = ..., key: Callable | None) -> Self | tuple[Self, np.ndarray | RangeIndex]: ...
    def _intersection(self, other: Index, sort: bool = ...): ...
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
    def _difference(self, other, sort): ...
    def symmetric_difference(self, other, result_name: Hashable | None, sort): ...
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
    def take(self, indices, axis: Axis = ..., allow_fill: bool = ..., fill_value, **kwargs) -> Index: ...
    @property
    def _engine_type(self): ...
    @property
    def start(self): ...
    @property
    def stop(self): ...
    @property
    def step(self): ...
    @property
    def dtype(self): ...
    @property
    def is_unique(self): ...
    @property
    def inferred_type(self): ...
    @property
    def size(self): ...
