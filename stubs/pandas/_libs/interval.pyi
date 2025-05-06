import _cython_3_0_11
from _typeshed import Incomplete
from pandas._libs.algos import is_monotonic as is_monotonic
from typing import Any, ClassVar, overload

NODE_CLASSES: dict
VALID_CLOSED: frozenset
__pyx_unpickle_Float64ClosedBothIntervalNode: _cython_3_0_11.cython_function_or_method
__pyx_unpickle_Float64ClosedLeftIntervalNode: _cython_3_0_11.cython_function_or_method
__pyx_unpickle_Float64ClosedNeitherIntervalNode: _cython_3_0_11.cython_function_or_method
__pyx_unpickle_Float64ClosedRightIntervalNode: _cython_3_0_11.cython_function_or_method
__pyx_unpickle_Int64ClosedBothIntervalNode: _cython_3_0_11.cython_function_or_method
__pyx_unpickle_Int64ClosedLeftIntervalNode: _cython_3_0_11.cython_function_or_method
__pyx_unpickle_Int64ClosedNeitherIntervalNode: _cython_3_0_11.cython_function_or_method
__pyx_unpickle_Int64ClosedRightIntervalNode: _cython_3_0_11.cython_function_or_method
__pyx_unpickle_IntervalMixin: _cython_3_0_11.cython_function_or_method
__pyx_unpickle_IntervalNode: _cython_3_0_11.cython_function_or_method
__pyx_unpickle_IntervalTree: _cython_3_0_11.cython_function_or_method
__pyx_unpickle_Uint64ClosedBothIntervalNode: _cython_3_0_11.cython_function_or_method
__pyx_unpickle_Uint64ClosedLeftIntervalNode: _cython_3_0_11.cython_function_or_method
__pyx_unpickle_Uint64ClosedNeitherIntervalNode: _cython_3_0_11.cython_function_or_method
__pyx_unpickle_Uint64ClosedRightIntervalNode: _cython_3_0_11.cython_function_or_method
__test__: dict
intervals_to_interval_bounds: _cython_3_0_11.cython_function_or_method

class Interval(IntervalMixin):
    _typ: ClassVar[str] = ...
    __array_priority__: ClassVar[int] = ...
    closed: Incomplete
    left: Incomplete
    right: Incomplete
    @overload
    def __init__(self, left=..., right=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    @overload
    def __init__(self, left=..., right=..., closed=...) -> Any:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _validate_endpoint(self, *args, **kwargs): ...
    @overload
    def overlaps(self, i2) -> Any:
        """
        Check whether two Interval objects overlap.

        Two intervals overlap if they share a common point, including closed
        endpoints. Intervals that only have an open endpoint in common do not
        overlap.

        Parameters
        ----------
        other : Interval
            Interval to check against for an overlap.

        Returns
        -------
        bool
            True if the two intervals overlap.

        See Also
        --------
        IntervalArray.overlaps : The corresponding method for IntervalArray.
        IntervalIndex.overlaps : The corresponding method for IntervalIndex.

        Examples
        --------
        >>> i1 = pd.Interval(0, 2)
        >>> i2 = pd.Interval(1, 3)
        >>> i1.overlaps(i2)
        True
        >>> i3 = pd.Interval(4, 5)
        >>> i1.overlaps(i3)
        False

        Intervals that share closed endpoints overlap:

        >>> i4 = pd.Interval(0, 1, closed='both')
        >>> i5 = pd.Interval(1, 2, closed='both')
        >>> i4.overlaps(i5)
        True

        Intervals that only have an open endpoint in common do not overlap:

        >>> i6 = pd.Interval(1, 2, closed='neither')
        >>> i4.overlaps(i6)
        False
        """
    @overload
    def overlaps(self, i3) -> Any:
        """
        Check whether two Interval objects overlap.

        Two intervals overlap if they share a common point, including closed
        endpoints. Intervals that only have an open endpoint in common do not
        overlap.

        Parameters
        ----------
        other : Interval
            Interval to check against for an overlap.

        Returns
        -------
        bool
            True if the two intervals overlap.

        See Also
        --------
        IntervalArray.overlaps : The corresponding method for IntervalArray.
        IntervalIndex.overlaps : The corresponding method for IntervalIndex.

        Examples
        --------
        >>> i1 = pd.Interval(0, 2)
        >>> i2 = pd.Interval(1, 3)
        >>> i1.overlaps(i2)
        True
        >>> i3 = pd.Interval(4, 5)
        >>> i1.overlaps(i3)
        False

        Intervals that share closed endpoints overlap:

        >>> i4 = pd.Interval(0, 1, closed='both')
        >>> i5 = pd.Interval(1, 2, closed='both')
        >>> i4.overlaps(i5)
        True

        Intervals that only have an open endpoint in common do not overlap:

        >>> i6 = pd.Interval(1, 2, closed='neither')
        >>> i4.overlaps(i6)
        False
        """
    @overload
    def overlaps(self, i5) -> Any:
        """
        Check whether two Interval objects overlap.

        Two intervals overlap if they share a common point, including closed
        endpoints. Intervals that only have an open endpoint in common do not
        overlap.

        Parameters
        ----------
        other : Interval
            Interval to check against for an overlap.

        Returns
        -------
        bool
            True if the two intervals overlap.

        See Also
        --------
        IntervalArray.overlaps : The corresponding method for IntervalArray.
        IntervalIndex.overlaps : The corresponding method for IntervalIndex.

        Examples
        --------
        >>> i1 = pd.Interval(0, 2)
        >>> i2 = pd.Interval(1, 3)
        >>> i1.overlaps(i2)
        True
        >>> i3 = pd.Interval(4, 5)
        >>> i1.overlaps(i3)
        False

        Intervals that share closed endpoints overlap:

        >>> i4 = pd.Interval(0, 1, closed='both')
        >>> i5 = pd.Interval(1, 2, closed='both')
        >>> i4.overlaps(i5)
        True

        Intervals that only have an open endpoint in common do not overlap:

        >>> i6 = pd.Interval(1, 2, closed='neither')
        >>> i4.overlaps(i6)
        False
        """
    @overload
    def overlaps(self, i6) -> Any:
        """
        Check whether two Interval objects overlap.

        Two intervals overlap if they share a common point, including closed
        endpoints. Intervals that only have an open endpoint in common do not
        overlap.

        Parameters
        ----------
        other : Interval
            Interval to check against for an overlap.

        Returns
        -------
        bool
            True if the two intervals overlap.

        See Also
        --------
        IntervalArray.overlaps : The corresponding method for IntervalArray.
        IntervalIndex.overlaps : The corresponding method for IntervalIndex.

        Examples
        --------
        >>> i1 = pd.Interval(0, 2)
        >>> i2 = pd.Interval(1, 3)
        >>> i1.overlaps(i2)
        True
        >>> i3 = pd.Interval(4, 5)
        >>> i1.overlaps(i3)
        False

        Intervals that share closed endpoints overlap:

        >>> i4 = pd.Interval(0, 1, closed='both')
        >>> i5 = pd.Interval(1, 2, closed='both')
        >>> i4.overlaps(i5)
        True

        Intervals that only have an open endpoint in common do not overlap:

        >>> i6 = pd.Interval(1, 2, closed='neither')
        >>> i4.overlaps(i6)
        False
        """
    def __add__(self, other):
        """Return self+value."""
    def __contains__(self, other) -> bool:
        """Return bool(key in self)."""
    def __eq__(self, other: object) -> bool:
        """Return self==value."""
    def __floordiv__(self, other):
        """Return self//value."""
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
    def __mul__(self, other):
        """Return self*value."""
    def __ne__(self, other: object) -> bool:
        """Return self!=value."""
    def __radd__(self, other): ...
    def __reduce__(self): ...
    def __rfloordiv__(self, other):
        """Return value//self."""
    def __rmul__(self, other): ...
    def __rsub__(self, other):
        """Return value-self."""
    def __rtruediv__(self, other):
        """Return value/self."""
    def __sub__(self, other):
        """Return self-value."""
    def __truediv__(self, other):
        """Return self/value."""

class IntervalMixin:
    closed_left: Incomplete
    closed_right: Incomplete
    is_empty: Incomplete
    length: Incomplete
    mid: Incomplete
    open_left: Incomplete
    open_right: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def _check_closed_matches(self, *args, **kwargs):
        """
        Check if the closed attribute of `other` matches.

        Note that 'left' and 'right' are considered different from 'both'.

        Parameters
        ----------
        other : Interval, IntervalIndex, IntervalArray
        name : str
            Name to use for 'other' in the error message.

        Raises
        ------
        ValueError
            When `other` is not closed exactly the same as self.
        """
    def __reduce__(self): ...
    def __reduce_cython__(self, *args, **kwargs): ...
    def __setstate_cython__(self, *args, **kwargs): ...

class IntervalTree(IntervalMixin):
    _is_overlapping: Incomplete
    _left_sorter: Incomplete
    _na_count: Incomplete
    _right_sorter: Incomplete
    closed: Incomplete
    dtype: Incomplete
    is_monotonic_increasing: Incomplete
    is_overlapping: Incomplete
    left: Incomplete
    left_sorter: Incomplete
    right: Incomplete
    right_sorter: Incomplete
    root: Incomplete
    def __init__(self, *args, **kwargs) -> None:
        """
        Parameters
        ----------
        left, right : np.ndarray[ndim=1]
            Left and right bounds for each interval. Assumed to contain no
            NaNs.
        closed : {'left', 'right', 'both', 'neither'}, optional
            Whether the intervals are closed on the left-side, right-side, both
            or neither. Defaults to 'right'.
        leaf_size : int, optional
            Parameter that controls when the tree switches from creating nodes
            to brute-force search. Tune this parameter to optimize query
            performance.
        """
    def __pyx_fuse_0get_indexer(self, *args, **kwargs):
        """Return the positions corresponding to unique intervals that overlap
                with the given array of scalar targets.
        """
    def __pyx_fuse_0get_indexer_non_unique(self, *args, **kwargs):
        """Return the positions corresponding to intervals that overlap with
                the given array of scalar targets. Non-unique positions are repeated.
        """
    def __pyx_fuse_1get_indexer(self, *args, **kwargs):
        """Return the positions corresponding to unique intervals that overlap
                with the given array of scalar targets.
        """
    def __pyx_fuse_1get_indexer_non_unique(self, *args, **kwargs):
        """Return the positions corresponding to intervals that overlap with
                the given array of scalar targets. Non-unique positions are repeated.
        """
    def __pyx_fuse_2get_indexer(self, *args, **kwargs):
        """Return the positions corresponding to unique intervals that overlap
                with the given array of scalar targets.
        """
    def __pyx_fuse_2get_indexer_non_unique(self, *args, **kwargs):
        """Return the positions corresponding to intervals that overlap with
                the given array of scalar targets. Non-unique positions are repeated.
        """
    def clear_mapping(self, *args, **kwargs): ...
    def get_indexer(self, *args, **kwargs):
        """Return the positions corresponding to unique intervals that overlap
                with the given array of scalar targets.
        """
    def get_indexer_non_unique(self, *args, **kwargs):
        """Return the positions corresponding to intervals that overlap with
                the given array of scalar targets. Non-unique positions are repeated.
        """
    def __reduce__(self): ...
    def __reduce_cython__(self, *args, **kwargs): ...
    def __setstate_cython__(self, *args, **kwargs): ...
