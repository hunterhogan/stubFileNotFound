from _typeshed import Incomplete
from python_toolbox import (
	comparison_tools as comparison_tools, decorator_tools as decorator_tools, misc_tools as misc_tools)
from python_toolbox.third_party.decorator import decorator as decorator
from typing import Any
import collections

infinity: Incomplete

class _SENTINEL(misc_tools.NonInstantiable):
    """Sentinel used to detect the end of an iterable."""

def _convert_index_to_exhaustion_point(index: Any) -> Any:
    """
    Convert an index to an "exhaustion point".

    The index may be either an integer or infinity.

    "Exhaustion point" means "until which index do we need to exhaust the
    internal iterator." If an index of `3` was requested, we need to exhaust it
    to index `3`, but if `-7` was requested, we have no choice but to exhaust
    the iterator completely (i.e. to `infinity`, actually the last element,)
    because only then we could know which member is the seventh-to-last.
    """
@decorator
def _with_lock(method: Any, *args: Any, **kwargs: Any) -> Any:
    """Decorator for using the `LazyTuple`'s lock."""

class LazyTuple(collections.abc.Sequence[Any]):
    """
    A lazy tuple which requests as few values as possible from its iterator.

    Wrap your iterators with `LazyTuple` and enjoy tuple-ish features like
    indexed access, comparisons, length measuring, element counting and more.

    Example:

        def my_generator():
            yield from ('hello', 'world', 'have', 'fun')

        lazy_tuple = LazyTuple(my_generator())

        assert lazy_tuple[2] == 'have'
        assert len(lazy_tuple) == 4

    `LazyTuple` holds the given iterable and pulls items out of it. It pulls as
    few items as it possibly can. For example, if you ask for the third
    element, it will pull exactly three elements and then return the third one.

    Some actions require exhausting the entire iterator. For example, checking
    the `LazyTuple` length, or doing indexex access with a negative index.
    (e.g. asking for the seventh-to-last element.)

    If you're passing in an iterator you definitely know to be infinite,
    specify `definitely_infinite=True`.
    """

    is_exhausted: Incomplete
    collected_data: Incomplete
    _iterator: Incomplete
    definitely_infinite: Incomplete
    lock: Incomplete
    def __init__(self, iterable: Any, definitely_infinite: bool = False) -> None: ...
    @classmethod
    @decorator_tools.helpful_decorator_builder
    def factory(cls, definitely_infinite: bool = False) -> Any:
        """
        Decorator to make generators return a `LazyTuple`.

        Example:

            @LazyTuple.factory()
            def my_generator():
                yield from ['hello', 'world', 'have', 'fun']

        This works on any function that returns an iterator. todo: Make it work
        on iterator classes.
        """
    @property
    def known_length(self) -> Any:
        """The number of items which have been taken from the internal iterator."""
    def exhaust(self, i: Any=...) -> None:
        """
        Take items from the internal iterators and save them.

        This will take enough items so we will have `i` items in total,
        including the items we had before.
        """
    def __getitem__(self, i: Any) -> Any:
        """Get item by index, either an integer index or a slice."""
    def __len__(self) -> int: ...
    def __eq__(self, other: object) -> Any: ...
    def __ne__(self, other: object) -> Any: ...
    def __bool__(self) -> bool: ...
    def __lt__(self, other: Any) -> Any: ...
    def __add__(self, other: Any) -> Any: ...
    def __radd__(self, other: Any) -> Any: ...
    def __mul__(self, other: Any) -> Any: ...
    def __rmul__(self, other: Any) -> Any: ...
    def __hash__(self) -> Any:
        """
        Get the `LazyTuple`'s hash.

        Note: Hashing the `LazyTuple` will completely exhaust it.
        """



