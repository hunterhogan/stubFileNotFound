from _typeshed import Incomplete
from collections.abc import Generator
from python_toolbox import (
	binary_search as binary_search, caching as caching, nifty_collections as nifty_collections,
	sequence_tools as sequence_tools)
from typing import Any
import abc
import collections
import collections.abc

infinity: Incomplete

class ChainSpace(sequence_tools.CuteSequenceMixin, collections.abc.Sequence[Any], metaclass=abc.ABCMeta):
    """
    A space of sequences chained together.

    This is similar to `itertools.chain`, except that items can be fetched by
    index number rather than just iteration.

    Example:

        >>> chain_space = ChainSpace(('abc', (1, 2, 3)))
        >>> chain_space
        <ChainSpace: 3+3>
        >>> chain_space[4]
        2
        >>> tuple(chain_space)
        ('a', 'b', 'c', 1, 2, 3)
        >>> chain_space.index(2)
        4

    """

    sequences: Incomplete
    def __init__(self, sequences: Any) -> None: ...
    @caching.CachedProperty
    def accumulated_lengths(self) -> Generator[Incomplete]:
        """
        A sequence of the accumulated length as every sequence is added.

        For example, if this chain space has sequences with lengths of 10, 100
        and 1000, this would be `[0, 10, 110, 1110]`.
        """
    length: Incomplete
    def __getitem__(self, i: Any) -> Any: ...
    def __iter__(self) -> Any: ...
    _reduced: Incomplete
    __eq__: Incomplete
    def __contains__(self, item: Any) -> bool: ...
    def index(self, item: Any) -> Any:
        """Get the index number of `item` in this space."""
    def __bool__(self) -> bool: ...



