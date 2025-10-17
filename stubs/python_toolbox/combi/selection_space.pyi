from _typeshed import Incomplete
from python_toolbox import sequence_tools as sequence_tools
from typing import Any
import abc
import collections
import collections.abc

class SelectionSpace(sequence_tools.CuteSequenceMixin, collections.abc.Sequence[Any], metaclass=abc.ABCMeta):
    """
    Space of possible selections of any number of items from `sequence`.

    For example:

        >>> tuple(SelectionSpace(range(2)))
        (set(), {1}, {0}, {0, 1})

    The selections (which are sets) can be for any number of items, from zero
    to the length of the sequence.

    Of course, this is a smart object that doesn't really create all these sets
    in advance, but rather on demand. So you can create a `SelectionSpace` like
    this:

        >>> selection_space = SelectionSpace(range(10**4))

    And take a random selection from it:

        >>> selection_space.take_random()
        {0, 3, 4, ..., 9996, 9997}

    Even though the length of this space is around 10 ** 3010, which is much
    bigger than the number of particles in the universe.
    """

    sequence: Incomplete
    sequence_length: Incomplete
    _sequence_set: Incomplete
    length: Incomplete
    def __init__(self, sequence: Any) -> None: ...
    def __getitem__(self, i: Any) -> Any: ...
    _reduced: Incomplete
    __hash__: Incomplete
    __bool__: Incomplete
    __eq__: Incomplete
    def index(self, selection: Any) -> Any:
        """Find the index number of `selection` in this `SelectionSpace`."""



