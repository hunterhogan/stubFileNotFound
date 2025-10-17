from _typeshed import Incomplete
from collections.abc import Generator
from python_toolbox import math_tools as math_tools, misc_tools as misc_tools, sequence_tools as sequence_tools
from typing import Any

infinity: Incomplete

class _EMPTY_SENTINEL(misc_tools.NonInstantiable): ...

def iterate_overlapping_subsequences(iterable: Any, length: int = 2, wrap_around: bool = False, lazy_tuple: bool = False) -> Any:
    """
    Iterate over overlapping subsequences from the iterable.

    Example: if the iterable is [0, 1, 2, 3], then the result would be
    `[(0, 1), (1, 2), (2, 3)]`. (Except it would be an iterator and not an
    actual list.)

    With a length of 3, the result would be an iterator of `[(0, 1, 2), (1,
    2, 3)]`.

    If `wrap_around=True`, the result would be `[(0, 1, 2), (1,
    2, 3), (2, 3, 0), (3, 0, 1)]`.

    If `lazy_tuple=True`, returns a `LazyTuple` rather than an iterator.
    """
def _iterate_overlapping_subsequences(iterable: Any, length: Any, wrap_around: Any) -> Generator[Incomplete, Incomplete]: ...
def shorten(iterable: Any, length: Any, lazy_tuple: bool = False) -> Any:
    """
    Shorten an iterable to `length`.

    Iterate over the given iterable, but stop after `n` iterations (Or when the
    iterable stops iteration by itself.)

    `n` may be infinite.

    If `lazy_tuple=True`, returns a `LazyTuple` rather than an iterator.
    """
def _shorten(iterable: Any, length: Any) -> Generator[Incomplete, Incomplete]: ...
def enumerate(iterable: Any, reverse_index: bool = False, lazy_tuple: bool = False) -> Any:
    """
    Iterate over `(i, item)` pairs, where `i` is the index number of `item`.

    This is an extension of the builtin `enumerate`. What it allows is to get a
    reverse index, by specifying `reverse_index=True`. This causes `i` to count
    down to zero instead of up from zero, so the `i` of the last member will be
    zero.

    If `lazy_tuple=True`, returns a `LazyTuple` rather than an iterator.
    """
def _enumerate(iterable: Any, reverse_index: Any) -> Any: ...
def is_iterable(thing: Any) -> Any:
    """Return whether an object is iterable."""
def get_length(iterable: Any) -> Any:
    """
    Get the length of an iterable.

    If given an iterator, it will be exhausted.
    """
def iter_with(iterable: Any, context_manager: Any, lazy_tuple: bool = False) -> Any:
    """
    Iterate on `iterable`, `with`ing the context manager on every `next`.

    If `lazy_tuple=True`, returns a `LazyTuple` rather than an iterator.
    """
def _iter_with(iterable: Any, context_manager: Any) -> Generator[Incomplete]: ...
def get_items(iterable: Any, n_items: Any, container_type: Any=...) -> Any:
    """
    Get the next `n_items` items from the iterable as a `tuple`.

    If there are less than `n` items, no exception will be raised. Whatever
    items are there will be returned.

    If you pass in a different kind of container than `tuple` as
    `container_type`, it'll be used to wrap the results.
    """
def double_filter(filter_function: Any, iterable: Any, lazy_tuple: bool = False) -> Any:
    """
    Filter an `iterable` into two iterables according to a `filter_function`.

    This is similar to the builtin `filter`, except it returns a tuple of two
    iterators, the first iterating on items that passed the filter function,
    and the second iterating on items that didn't.

    Note that this function is not thread-safe. (You may not consume the two
    iterators on two separate threads.)

    If `lazy_tuple=True`, returns two `LazyTuple` objects rather than two
    iterator.
    """
def get_ratio(filter_function: Any, iterable: Any) -> Any:
    """Get the ratio of `iterable` items that pass `filter_function`."""
def fill(iterable: Any, fill_value: Any=None, fill_value_maker: Any=None, length: Any=..., sequence_type: Any=None, lazy_tuple: bool = False) -> Any:
    """
    Iterate on `iterable`, and after it's exhaused, yield fill values.

    If `fill_value_maker` is given, it's used to create fill values
    dynamically. (Useful if your fill value is `[]` and you don't want to use
    many copies of the same list.)

    If `length` is given, shortens the iterator to that length.

    If `sequence_type` is given, instead of returning an iterator, this
    function will return a sequence of that type. If `lazy_tuple=True`, uses a
    `LazyTuple`. (Can't use both options together.)
    """
def _fill(iterable: Any, fill_value: Any, fill_value_maker: Any, length: Any) -> Generator[Incomplete, None, Incomplete]: ...
def call_until_exception(function: Any, exception: Any, lazy_tuple: bool = False) -> Any:
    """
    Iterate on values returned from `function` until getting `exception`.

    If `lazy_tuple=True`, returns a `LazyTuple` rather than an iterator.
    """
def _call_until_exception(function: Any, exception: Any) -> Generator[Incomplete]: ...
def get_single_if_any(iterable: Any, *, exception_on_multiple: bool = True, none_on_multiple: bool = False) -> Any:
    """
    Get the single item of `iterable`, if any.

    Default behavior: Get the first item from `iterable`, and ensure it doesn't
    have any more items (raise an exception if it does.)

    If you pass in `exception_on_multiple=False`: If `iterable` has more than
    one item, an exception won't be raised. The first value will be returned.

    If you pass in `none_on_multiple=True`: If `iterable` has more than one
    item, `None` will be returned regardless of the value of the first item.
    Note that passing `none_on_multiple=True` causes the
    `exception_on_multiple` argument to be ignored. (This is a bit ugly but I
    made it that way so you wouldn't have to manually pass
    `exception_on_multiple=False` in this case.)
    """
def are_equal(*sequences: Any, easy_types: Any=...) -> Any:
    """
    Are the given sequences equal?

    This tries to make a cheap comparison between the sequences if possible,
    but if not, it goes over the sequences in parallel item-by-item and checks
    whether the items are all equal. A cheap comparison is attempted only if
    the sequences are all of the same type, and that type is in `easy_types`.
    (It's important to restrict `easy_types` only to types where equality
    between the sequences is the same as equality between every item in the
    sequences.)
    """
def is_sorted(iterable: Any, *, rising: bool = True, strict: bool = False, key: Any=None) -> Any:
    """
    Is `iterable` sorted?

    Goes over the iterable item by item and checks whether it's sorted. If one
    item breaks the order, returns `False` and stops iterating. If after going
    over all the items, they were all sorted, returns `True`.

    You may specify `rising=False` to check for a reverse ordering. (i.e. each
    item should be lower or equal than the last one.)

    You may specify `strict=True` to check for a strict order. (i.e. each item
    must be strictly bigger than the last one, or strictly smaller if
    `rising=False`.)

    You may specify a key function as the `key` argument.
    """

class _PUSHBACK_SENTINEL(misc_tools.NonInstantiable):
    """Sentinel used by `PushbackIterator` to say nothing was pushed back."""

class PushbackIterator:
    """
    Iterator allowing to push back the last item so it'll be yielded next time.

    Initialize `PushbackIterator` with your favorite iterator as the argument
    and it'll create an iterator wrapping it on which you can call
    `.push_back()` to have it take the recently yielded item and yield it again
    next time.

    Only one item may be pushed back at any time.
    """

    iterator: Incomplete
    last_item: Incomplete
    just_pushed_back: bool
    def __init__(self, iterable: Any) -> None: ...
    def __next__(self) -> Any: ...
    __iter__: Incomplete
    def push_back(self) -> None:
        """
        Push the last item back, so it'll come up in the next iteration.

        You can't push back twice without iterating, because we only save the
        last item and not any previous items.
        """

def iterate_pop(poppable: Any, lazy_tuple: bool = False) -> Any:
    """Iterate by doing `.pop()` until no more items."""
def iterate_popleft(left_poppable: Any, lazy_tuple: bool = False) -> Any:
    """Iterate by doing `.popleft()` until no more items."""
def iterate_popitem(item_poppable: Any, lazy_tuple: bool = False) -> Any:
    """Iterate by doing `.popitem()` until no more items."""
def zip_non_equal(iterables: Any, lazy_tuple: bool = False) -> Any:
    """Zip the iterables, but only yield the tuples where the items aren't equal."""



