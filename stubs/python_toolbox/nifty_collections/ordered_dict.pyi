from collections import OrderedDict as StdlibOrderedDict
from typing import Any

class OrderedDict(StdlibOrderedDict):
    """
    A dictionary with an order.

    This is a subclass of `collections.OrderedDict` with a couple of
    improvements.
    """

    def sort(self, key: Any=None, reverse: bool = False) -> None:
        """
        Sort the items according to their keys, changing the order in-place.

        The optional `key` argument, (not to be confused with the dictionary
        keys,) will be passed to the `sorted` function as a key function.
        """
    def index(self, key: Any) -> Any:
        """Get the index number of `key`."""
    @property
    def reversed(self) -> Any:
        """Get a version of this `OrderedDict` with key order reversed."""



