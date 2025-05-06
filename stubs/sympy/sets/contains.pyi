from _typeshed import Incomplete
from sympy.core.relational import Eq as Eq, Ne as Ne
from sympy.logic.boolalg import Boolean as Boolean

class Contains(Boolean):
    """
    Asserts that x is an element of the set S.

    Examples
    ========

    >>> from sympy import Symbol, Integer, S, Contains
    >>> Contains(Integer(2), S.Integers)
    True
    >>> Contains(Integer(-2), S.Naturals)
    False
    >>> i = Symbol('i', integer=True)
    >>> Contains(i, S.Naturals)
    Contains(i, Naturals)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Element_%28mathematics%29
    """
    def __new__(cls, x, s, evaluate: Incomplete | None = None): ...
    @property
    def binary_symbols(self): ...
    def as_set(self): ...
