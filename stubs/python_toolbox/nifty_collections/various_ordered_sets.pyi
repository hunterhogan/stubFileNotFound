from _typeshed import Incomplete
from collections.abc import Generator
from python_toolbox import (
	caching as caching, comparison_tools as comparison_tools, context_management as context_management,
	freezing as freezing)
from typing import Any
import collections

KEY: Incomplete
PREV: Incomplete
NEXT: Incomplete

class BaseOrderedSet(collections.abc.Set, collections.abc.Sequence[Any]):
    """
    Base class for `OrderedSet` and `FrozenOrderedSet`, i.e. set with an order.

    This behaves like a `set` except items have an order. (By default they're
    ordered by insertion order, but that order can be changed.)
    """

    def __init__(self, iterable: Any=()) -> None: ...
    def __getitem__(self, index: Any) -> Any: ...
    def __len__(self) -> int: ...
    def __contains__(self, key: Any) -> bool: ...
    def __iter__(self) -> Any: ...
    def __reversed__(self) -> Generator[Incomplete]: ...
    def __eq__(self, other: object) -> Any: ...
    _end: Incomplete
    _map: Incomplete
    def __clear(self) -> None:
        """Clear the ordered set, removing all items."""
    def __add(self, key: Any, last: bool = True) -> None:
        """
        Add an element to a set.

        This has no effect if the element is already present.

        Specify `last=False` to add the item at the start of the ordered set.
        """

class FrozenOrderedSet(BaseOrderedSet):
    """
    A `frozenset` with an order.

    This behaves like a `frozenset` (i.e. a set that can't be changed after
    creation) except items have an order. (By default they're ordered by
    insertion order, but that order can be changed.)
    """

    def __hash__(self) -> Any: ...

class OrderedSet(BaseOrderedSet, collections.abc.MutableSet[Any]):
    """
    A `set` with an order.

    This behaves like a `set` except items have an order. (By default they're
    ordered by insertion order, but that order can be changed.)
    """

    add: Incomplete
    clear: Incomplete
    def move_to_end(self, key: Any, last: bool = True) -> None:
        """Move an existing element to the end (or start if `last=False`.)."""
    def sort(self, key: Any=None, reverse: bool = False) -> None:
        """
        Sort the items according to their keys, changing the order in-place.

        The optional `key` argument will be passed to the `sorted` function as
        a key function.
        """
    def discard(self, key: Any) -> None:
        """
        Remove an element from a set if it is a member.

        If the element is not a member, do nothing.
        """
    def pop(self, last: bool = True) -> Any:
        """Remove and return an arbitrary set element."""
    def get_frozen(self) -> Any:
        """Get a frozen version of this ordered set."""

class EmittingOrderedSet(OrderedSet):
    """An ordered set that emits to `.emitter` every time it's modified."""

    emitter: Incomplete
    def __init__(self, iterable: Any=(), *, emitter: Any=None) -> None: ...
    def add(self, key: Any, last: bool = True) -> None:
        """
        Add an element to a set.

        This has no effect if the element is already present.
        """
    def discard(self, key: Any) -> None:
        """
        Remove an element from a set if it is a member.

        If the element is not a member, do nothing.
        """
    def clear(self) -> None:
        """Clear the ordered set, removing all items."""
    def set_emitter(self, emitter: Any) -> None:
        """Set `emitter` to be emitted with on every modification."""
    def _emit(self) -> None: ...
    def move_to_end(self, key: Any, last: bool = True) -> None:
        """Move an existing element to the end (or start if `last=False`.)."""
    _emitter_freezer: Incomplete
    def __eq__(self, other: object) -> Any: ...
    def get_without_emitter(self) -> Any:
        """Get a version of this ordered set without an emitter attached."""



