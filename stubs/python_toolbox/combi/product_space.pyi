from _typeshed import Incomplete
from python_toolbox import math_tools as math_tools, sequence_tools as sequence_tools
from typing import Any
import abc
import collections
import collections.abc

class ProductSpace(sequence_tools.CuteSequenceMixin, collections.abc.Sequence[Any], metaclass=abc.ABCMeta):
    """
    A product space between sequences.

    This is similar to Python's builtin `itertools.product`, except that it
    behaves like a sequence rather than an iterable. (Though it's also
    iterable.) You can access any item by its index number.

    Example:

        >>> product_space = ProductSpace(('abc', range(4)))
        >>> product_space
        <ProductSpace: 3 * 4>
        >>> product_space.length
        12
        >>> product_space[10]
        ('c', 2)
        >>> tuple(product_space)
        (('a', 0), ('a', 1), ('a', 2), ('a', 3), ('b', 0), ('b', 1), ('b', 2),
         ('b', 3), ('c', 0), ('c', 1), ('c', 2), ('c', 3))

    """

    sequences: Incomplete
    sequence_lengths: Incomplete
    length: Incomplete
    def __init__(self, sequences: Any) -> None: ...
    def __getitem__(self, i: Any) -> Any: ...
    _reduced: Incomplete
    __hash__: Incomplete
    __eq__: Incomplete
    def index(self, given_sequence: Any) -> Any:
        """Get the index number of `given_sequence` in this product space."""
    __bool__: Incomplete



