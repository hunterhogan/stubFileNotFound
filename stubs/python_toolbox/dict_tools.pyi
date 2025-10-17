from _typeshed import Incomplete
from collections.abc import Generator
from python_toolbox import comparison_tools as comparison_tools, cute_iter_tools as cute_iter_tools
from typing import Any

def filter_items(d: Any, condition: Any, double: bool = False, force_dict_type: Any=None) -> Any:
    """
    Get new dict with items from `d` that satisfy the `condition` functions.

    `condition` is a function that takes a key and a value.

    The newly created dict will be of the same class as `d`, e.g. if you passed
    an ordered dict as `d`, the result will be an ordered dict, using the
    correct order.

    Specify `double=True` to get a tuple of two dicts instead of one. The
    second dict will have all the rejected items.
    """
def get_tuple(d: Any, iterable: Any) -> Any:
    """Get a tuple of values corresponding to an `iterable` of keys."""
def get_contained(d: Any, container: Any) -> Any:
    """Get a list of the values in the dict whose keys are in `container`."""
def fancy_string(d: Any, indent: int = 0) -> Any:
    """Show a dict as a string, slightly nicer than dict.__repr__."""
def devour_items(d: Any) -> Generator[Incomplete]:
    """Iterator that pops (key, value) pairs from `d` until it's empty."""
def devour_keys(d: Any) -> Generator[Incomplete]:
    """Iterator that pops keys from `d` until it's exhaused (i.e. empty)."""
def sum_dicts(dicts: Any) -> Any:
    """
    Return the sum of a bunch of dicts i.e. all the dicts merged into one.

    If there are any collisions, the latest dicts in the sequence win.
    """
def remove_keys(d: Any, keys_to_remove: Any) -> Any:
    """
    Remove keys from a dict.

    `keys_to_remove` is allowed to be either an iterable (in which case it will
    be iterated on and keys with the same name will be removed), a container
    (in which case this function will iterate over the keys of the dict, and if
    they're contained they'll be removed), or a filter function (in which case
    this function will iterate over the keys of the dict, and if they pass the
    filter function they'll be removed.)

    If key doesn't exist, doesn't raise an exception.
    """
def get_sorted_values(d: Any, key: Any=None) -> Any:
    """Get the values of dict `d` as a `tuple` sorted by their respective keys."""
def reverse(d: Any) -> Any:
    """
    Reverse a `dict`, creating a new `dict` where keys and values are switched.

    Example:

        >>> reverse({'one': 1, 'two': 2, 'three': 3})
        {1: 'one', 2: 'two', 3: 'three'})

    This function requires that:

      1. The values will be distinct, i.e. no value will appear more than once.
      2. All the values be hashable.

    """



