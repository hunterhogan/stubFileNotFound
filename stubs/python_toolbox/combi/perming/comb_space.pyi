from .perm_space import PermSpace as PermSpace
from typing import Any
import abc

class CombSpace(PermSpace, metaclass=abc.ABCMeta):
    """
    A space of combinations.

    This is a subclass of `PermSpace`; see its documentation for more details.

    Each item in a `CombSpace` is a `Comb`, i.e. a combination. This is similar
    to `itertools.combinations`, except it offers far, far more functionality.
    The combinations may be accessed by index number, the combinations can be
    of a custom type, the space may be sliced, etc.

    Here is the simplest possible `CombSpace`:

        >>> comb_space = CombSpace(4, 2)
        <CombSpace: 0..3, n_elements=2>
        >>> comb_space[2]
        <Comb, n_elements=2: (0, 3)>
        >>> tuple(comb_space)
        (<Comb, n_elements=2: (0, 1)>, <Comb, n_elements=2: (0, 2)>,
         <Comb, n_elements=2: (0, 3)>, <Comb, n_elements=2: (1, 2)>,
         <Comb, n_elements=2: (1, 3)>, <Comb, n_elements=2: (2, 3)>)

    The members are `Comb` objects, which are sequence-like objects that have
    extra functionality. (See documentation of `Comb` and `Perm` for more
    info.)
    """

    def __init__(self, iterable_or_length: Any, n_elements: Any, *, slice_: Any=None, perm_type: Any=None, _domain_for_checking: Any=None, _degrees_for_checking: Any=None) -> None: ...



