from _typeshed import Incomplete
from python_toolbox import (
	caching as caching, cute_iter_tools as cute_iter_tools, misc_tools as misc_tools,
	nifty_collections as nifty_collections, sequence_tools as sequence_tools)
from typing import Any
import abc
import collections
import collections.abc

infinity: Incomplete

class _BasePermView(metaclass=abc.ABCMeta):
    """Abstract base class for viewers on Perm."""

    perm: Incomplete
    def __init__(self, perm: Any) -> None: ...
    __repr__: Incomplete
    @abc.abstractmethod
    def __getitem__(self, i: Any) -> Any: ...

class PermItems(sequence_tools.CuteSequenceMixin, _BasePermView, collections.abc.Sequence[Any], metaclass=abc.ABCMeta):
    """
    A viewer of a perm's items, similar to `dict.items()`.

    This is useful for dapplied perms; it lets you view the perm (both index
    access and iteration) as a sequence where each item is a 2-tuple, where the
    first item is from the domain and the second item is its corresponding item
    from the sequence.
    """

    def __getitem__(self, i: Any) -> Any: ...

class PermAsDictoid(sequence_tools.CuteSequenceMixin, _BasePermView, collections.abc.Mapping[Any, Any], metaclass=abc.ABCMeta):
    """A dict-like interface to a `Perm`."""

    def __getitem__(self, key: Any) -> Any: ...
    def __iter__(self) -> Any: ...

class PermType(abc.ABCMeta):
    """
    Metaclass for `Perm` and `Comb`.

    The functionality provided is: If someone tries to create a `Perm` with a
    `CombSpace`, we automatically use `Comb`.
    """

    def __call__(cls, item: Any, perm_space: Any=None) -> Any: ...

class Perm(sequence_tools.CuteSequenceMixin, collections.abc.Sequence[Any], metaclass=PermType):
    """
    A permutation of items from a `PermSpace`.

    In combinatorics, a permutation is a sequence of items taken from the
    original sequence.

    Example:

        >>> perm_space = PermSpace('abcd')
        >>> perm = Perm('dcba', perm_space)
        >>> perm
        <Perm: ('d', 'c', 'b', 'a')>
        >>> perm_space.index(perm)
        23

    """

    @classmethod
    def coerce(cls, item: Any, perm_space: Any=None) -> Any:
        """Coerce item into a perm, optionally of a specified `PermSpace`."""
    nominal_perm_space: Incomplete
    is_rapplied: Incomplete
    is_recurrent: Incomplete
    is_partial: Incomplete
    is_combination: Incomplete
    is_dapplied: Incomplete
    is_pure: Incomplete
    undapplied: Incomplete
    uncombinationed: Incomplete
    _perm_sequence: Incomplete
    def __init__(self, perm_sequence: Any, perm_space: Any=None) -> None:
        """
        Create the `Perm`.

        If `perm_space` is not supplied, we assume that this is a pure
        permutation, i.e. a permutation on `range(len(perm_sequence))`.
        """
    _reduced: Incomplete
    __iter__: Incomplete
    def __eq__(self, other: object) -> Any: ...
    __ne__: Incomplete
    __hash__: Incomplete
    __bool__: Incomplete
    def __contains__(self, item: Any) -> bool: ...
    def index(self, member: Any) -> Any:
        """
        Get the index number of `member` in the permutation.

        Example:

            >>> perm = PermSpace(5)[10]
            >>> perm
            <Perm: (0, 2, 4, 1, 3)>
            >>> perm.index(3)
            4

        """
    @caching.CachedProperty
    def inverse(self) -> Any:
        """
        The inverse of this permutation.

        i.e. the permutation that we need to multiply this permutation by to
        get the identity permutation.

        This is also accessible as `~perm`.

        Example:

            >>> perm = PermSpace(5)[10]
            >>> perm
            <Perm: (0, 2, 4, 1, 3)>
            >>> ~perm
            <Perm: (0, 3, 1, 4, 2)>
            >>> perm * ~perm
            <Perm: (0, 1, 2, 3, 4)>

        """
    __invert__: Incomplete
    domain: Incomplete
    @caching.CachedProperty
    def unrapplied(self) -> Any:
        """An unrapplied version of this permutation."""
    def __getitem__(self, i: Any) -> Any: ...
    length: Incomplete
    def apply(self, sequence: Any, result_type: Any=None) -> Any:
        """
        Apply the perm to a sequence, choosing items from it.

        This can also be used as `sequence * perm`. Example:

            >>> perm = PermSpace(5)[10]
            >>> perm
            <Perm: (0, 2, 4, 1, 3)>
            >>> perm.apply('growl')
            'golrw'
            >>> 'growl' * perm
            'golrw'

        Specify `result_type` to determine the type of the result returned. If
        `result_type=None`, will use `tuple`, except when `other` is a `str` or
        `Perm`, in which case that same type would be used.
        """
    __rmul__ = apply
    __mul__: Incomplete
    def __pow__(self, exponent: Any) -> Any:
        """Raise the perm by the power of `exponent`."""
    @caching.CachedProperty
    def degree(self) -> Any:
        """
        The permutation's degree.

        You can think of a permutation's degree like this: Imagine that you're
        starting with the identity permutation, and you want to make this
        permutation, by switching two items with each other over and over again
        until you get this permutation. The degree is the number of such
        switches you'll have to make.
        """
    @caching.CachedProperty
    def n_cycles(self) -> Any:
        """
        The number of cycles in this permutation.

        If item 1 points at item 7, and item 7 points at item 3, and item 3
        points at item 1 again, then that's one cycle. `n_cycles` is the total
        number of cycles in this permutation.
        """
    def get_neighbors(self, *, degrees: Any=(1,), perm_space: Any=None) -> Any:
        """
        Get the neighbor permutations of this permutation.

        This means, get the permutations that are close to this permutation. By
        default, this means permutations that are one transformation (switching
        a pair of items) away from this permutation. You can specify a custom
        sequence of integers to the `degrees` argument to get different degrees
        of relation. (e.g. specify `degrees=(1, 2)` to get both the closest
        neighbors and the second-closest neighbors.)
        """
    def __lt__(self, other: Any) -> Any: ...
    __reversed__: Incomplete
    items: Incomplete
    as_dictoid: Incomplete

class UnrecurrentedMixin:
    """Mixin for a permutation in a space that's been unrecurrented."""

    def __getitem__(self, i: Any) -> Any: ...
    def __iter__(self) -> Any: ...
    index: Incomplete

class UnrecurrentedPerm(UnrecurrentedMixin, Perm, metaclass=abc.ABCMeta):
    """A permutation in a space that's been unrecurrented."""



