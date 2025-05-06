from .function import Function as Function
from .kind import NumberKind as NumberKind
from .logic import fuzzy_and as fuzzy_and, fuzzy_not as fuzzy_not
from _typeshed import Incomplete

class Mod(Function):
    """Represents a modulo operation on symbolic expressions.

    Parameters
    ==========

    p : Expr
        Dividend.

    q : Expr
        Divisor.

    Notes
    =====

    The convention used is the same as Python's: the remainder always has the
    same sign as the divisor.

    Many objects can be evaluated modulo ``n`` much faster than they can be
    evaluated directly (or at all).  For this, ``evaluate=False`` is
    necessary to prevent eager evaluation:

    >>> from sympy import binomial, factorial, Mod, Pow
    >>> Mod(Pow(2, 10**16, evaluate=False), 97)
    61
    >>> Mod(factorial(10**9, evaluate=False), 10**9 + 9)
    712524808
    >>> Mod(binomial(10**18, 10**12, evaluate=False), (10**5 + 3)**2)
    3744312326

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> x**2 % y
    Mod(x**2, y)
    >>> _.subs({x: 5, y: 6})
    1

    """
    kind = NumberKind
    @classmethod
    def eval(cls, p, q): ...
    def _eval_is_integer(self): ...
    def _eval_is_nonnegative(self): ...
    def _eval_is_nonpositive(self): ...
    def _eval_rewrite_as_floor(self, a, b, **kwargs): ...
    def _eval_as_leading_term(self, x, logx: Incomplete | None = None, cdir: int = 0): ...
    def _eval_nseries(self, x, n, logx, cdir: int = 0): ...
