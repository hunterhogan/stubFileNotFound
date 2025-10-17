from _typeshed import Incomplete
from python_toolbox import caching as caching, nifty_collections as nifty_collections, sequence_tools as sequence_tools
from typing import Any
import abc
import collections
import collections.abc

infinity: Incomplete

class MapSpace(sequence_tools.CuteSequenceMixin, collections.abc.Sequence[Any], metaclass=abc.ABCMeta):
    """
    A space of a function applied to a sequence.

    This is similar to Python's builtin `map`, except that it behaves like a
    sequence rather than an iterable. (Though it's also iterable.) You can
    access any item by its index number.

    Example:

        >>> map_space = MapSpace(lambda x: x ** 2, range(7))
        >>> map_space
        MapSpace(<function <lambda> at 0x00000000030C1510>, range(0, 7))
        >>> len(map_space)
        7
        >>> map_space[3]
        9
        >>> tuple(map_space)
        (0, 1, 4, 9, 16, 25, 36)

    """

    function: Incomplete
    sequence: Incomplete
    def __init__(self, function: Any, sequence: Any) -> None: ...
    length: Incomplete
    def __getitem__(self, i: Any) -> Any: ...
    def __iter__(self) -> Any: ...
    _reduced: Incomplete
    __eq__: Incomplete
    __hash__: Incomplete
    __bool__: Incomplete



