from _typeshed import Incomplete
from python_toolbox import caching as caching, math_tools as math_tools, misc_tools as misc_tools
from typing import Any
import abc
import collections

infinity: Incomplete

def are_equal_regardless_of_order(seq1: Any, seq2: Any) -> Any:
    """
    Do `seq1` and `seq2` contain the same elements, same number of times?

    Disregards order of elements.

    Currently will fail for items that have problems with comparing.
    """
def flatten(iterable: Any) -> Any:
    """
    Flatten a sequence, returning a sequence of all its items' items.

    For example, `flatten([[1, 2], [3], [4, 'meow']]) == [1, 2, 3, 4, 'meow']`.
    """

class NO_FILL_VALUE(misc_tools.NonInstantiable):
    """Sentinel that means: Don't fill last partition with default fill values."""

def partitions(sequence: Any, partition_size: Any=None, *, n_partitions: Any=None, allow_remainder: bool = True, larger_on_remainder: bool = False, fill_value: Any=...) -> Any:
    """
    Partition `sequence` into equal partitions of size `partition_size`, or
    determine size automatically given the number of partitions as
    `n_partitions`.

    If the sequence can't be divided into precisely equal partitions, the last
    partition will contain less members than all the other partitions.

    Example:

        >>> partitions([0, 1, 2, 3, 4], 2)
        [[0, 1], [2, 3], [4]]

    (You need to give *either* a `partition_size` *or* an `n_partitions`
    argument, not both.)

    Specify `allow_remainder=False` to enforce that the all the partition sizes
    be equal; if there's a remainder while `allow_remainder=False`, an
    exception will be raised.

    By default, if there's a remainder, the last partition will be smaller than
    the others. (e.g. a sequence of 7 items, when partitioned into pairs, will
    have 3 pairs and then a partition with only 1 element.) Specify
    `larger_on_remainder=True` to make the last partition be a bigger partition
    in case there's a remainder. (e.g. a sequence of a 7 items divided into
    pairs would result in 2 pairs and one triplet.)

    If you want the remainder partition to be of equal size with the other
    partitions, you can specify `fill_value` as the padding for the last
    partition. A specified value for `fill_value` implies
    `allow_remainder=True` and will cause an exception to be raised if
    specified with `allow_remainder=False`.

    Example:

        >>> partitions([0, 1, 2, 3, 4], 3, fill_value='meow')
        [[0, 1, 2], [3, 4, 'meow']]

    """
def is_immutable_sequence(thing: Any) -> Any:
    """Is `thing` an immutable sequence, like `tuple`?"""
def to_tuple(single_or_sequence: Any, item_type: Any=None, item_test: Any=None) -> Any:
    """
    Convert an item or a sequence of items into a tuple of items.

    This is typically used in functions that request a sequence of items but
    are considerate enough to accept a single item and wrap it in a tuple
    `(item,)` themselves.

    This function figures out whether the user entered a sequence of items, in
    which case it will only be converted to a tuple and returned; or the user
    entered a single item, in which case a tuple `(item,)` will be returned.

    To aid this function in parsing, you may optionally specify `item_type`
    which is the type of the items, or alternatively `item_test` which is a
    callable that takes an object and returns whether it's a valid item. These
    are necessary only when your items might be sequences themselves.

    You may optionally put multiple types in `item_type`, and each object would
    be required to match to at least one of them.
    """
def pop_until(sequence: Any, condition: Any=...) -> Any:
    """
    Look for item in `sequence` that passes `condition`, popping away others.

    When sequence is empty, propagates the `IndexError`.
    """
def get_recurrences(sequence: Any) -> Any:
    """
    Get a `dict` of all items that repeat at least twice.

    The values of the dict are the numbers of repititions of each item.
    """
def ensure_iterable_is_immutable_sequence(iterable: Any, default_type: Any=..., unallowed_types: Any=...) -> Any:
    """
    Return a version of `iterable` that is an immutable sequence.

    If `iterable` is already an immutable sequence, it returns it as is;
    otherwise, it makes it into a `tuple`, or into any other data type
    specified in `default_type`.
    """
def ensure_iterable_is_sequence(iterable: Any, default_type: Any=..., unallowed_types: Any=...) -> Any:
    """
    Return a version of `iterable` that is a sequence.

    If `iterable` is already a sequence, it returns it as is; otherwise, it
    makes it into a `tuple`, or into any other data type specified in
    `default_type`.
    """

class CuteSequenceMixin(misc_tools.AlternativeLengthMixin):
    """A sequence mixin that adds extra functionality."""

    def take_random(self) -> Any:
        """Take a random item from the sequence."""
    def __contains__(self, item: Any) -> bool: ...

class CuteSequence(CuteSequenceMixin, collections.abc.Sequence[Any], metaclass=abc.ABCMeta):
    """A sequence type that adds extra functionality."""

def get_length(sequence: Any) -> Any:
    """Get the length of a sequence."""
def divide_to_slices(sequence: Any, n_slices: Any) -> Any:
    """
    Divide a sequence to slices.

    Example:

        >>> divide_to_slices(range(10), 3)
        [range(0, 4), range(4, 7), range(7, 10)]

    """
def is_subsequence(big_sequence: Any, small_sequence: Any) -> Any:
    """
    Check whether `small_sequence` is a subsequence of `big_sequence`.

    For example:

        >>> is_subsequence([1, 2, 3, 4], [2, 3])
        True
        >>> is_subsequence([1, 2, 3, 4], [4, 5])
        False

    This can be used on any kind of sequence, including tuples, lists and
    strings.
    """



