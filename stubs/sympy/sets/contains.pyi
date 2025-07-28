from .sets import Set as Set
from sympy.core import S as S
from sympy.core.parameters import global_parameters as global_parameters
from sympy.core.relational import Eq as Eq, Ne as Ne
from sympy.core.sympify import sympify as sympify
from sympy.logic.boolalg import Boolean as Boolean
from sympy.utilities.misc import func_name as func_name

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
    def __new__(cls, x, s, evaluate=None): ...
    @property
    def binary_symbols(self): ...
    def as_set(self): ...
