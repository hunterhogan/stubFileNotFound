from .perm import Perm as Perm, UnrecurrentedPerm as UnrecurrentedPerm
from typing import Any
import abc

class Comb(Perm, metaclass=abc.ABCMeta):
    """
    A combination of items from a `CombSpace`.

    In combinatorics, a combination is like a permutation except with no order.
    In the `combi` package, we implement that by making the items in `Comb` be
    in canonical order. (This has the same effect as having no order because
    each combination of items can only appear once, in the canonical order,
    rather than many different times in many different orders like with
    `Perm`.)

    Example:

        >>> comb_space = CombSpace('abcde', 3)
        >>> comb = Comb('bcd', comb_space)
        >>> comb
        <Comb, n_elements=3: ('a', 'b', 'c')>
        >>> comb_space.index(comb)
        6

    """

    def __init__(self, perm_sequence: Any, perm_space: Any=None) -> None: ...

class UnrecurrentedComb(UnrecurrentedPerm, Comb, metaclass=abc.ABCMeta):
    """A combination in a space that's been unrecurrented."""



