from _typeshed import Incomplete
from sympy.assumptions import Q as Q, ask as ask
from sympy.core.basic import Basic as Basic
from sympy.core.sympify import _sympify as _sympify

def make_eval_method(fact): ...

class AssumptionsWrapper(Basic):
    """
    Wrapper over ``Basic`` instances to call predicate query by
    ``.is_[...]`` property

    Parameters
    ==========

    expr : Basic

    assumptions : Boolean, optional

    Examples
    ========

    >>> from sympy import Q, Symbol
    >>> from sympy.assumptions.wrapper import AssumptionsWrapper
    >>> x = Symbol('x', even=True)
    >>> AssumptionsWrapper(x).is_integer
    True
    >>> y = Symbol('y')
    >>> AssumptionsWrapper(y, Q.even(y)).is_integer
    True

    With ``AssumptionsWrapper``, both evaluation and refinement can be supported
    by single implementation.

    >>> from sympy import Function
    >>> class MyAbs(Function):
    ...     @classmethod
    ...     def eval(cls, x, assumptions=True):
    ...         _x = AssumptionsWrapper(x, assumptions)
    ...         if _x.is_nonnegative:
    ...             return x
    ...         if _x.is_negative:
    ...             return -x
    ...     def _eval_refine(self, assumptions):
    ...         return MyAbs.eval(self.args[0], assumptions)
    >>> MyAbs(x)
    MyAbs(x)
    >>> MyAbs(x).refine(Q.positive(x))
    x
    >>> MyAbs(Symbol('y', negative=True))
    -y

    """
    def __new__(cls, expr, assumptions: Incomplete | None = None): ...
    _eval_is_algebraic: Incomplete
    _eval_is_antihermitian: Incomplete
    _eval_is_commutative: Incomplete
    _eval_is_complex: Incomplete
    _eval_is_composite: Incomplete
    _eval_is_even: Incomplete
    _eval_is_extended_negative: Incomplete
    _eval_is_extended_nonnegative: Incomplete
    _eval_is_extended_nonpositive: Incomplete
    _eval_is_extended_nonzero: Incomplete
    _eval_is_extended_positive: Incomplete
    _eval_is_extended_real: Incomplete
    _eval_is_finite: Incomplete
    _eval_is_hermitian: Incomplete
    _eval_is_imaginary: Incomplete
    _eval_is_infinite: Incomplete
    _eval_is_integer: Incomplete
    _eval_is_irrational: Incomplete
    _eval_is_negative: Incomplete
    _eval_is_noninteger: Incomplete
    _eval_is_nonnegative: Incomplete
    _eval_is_nonpositive: Incomplete
    _eval_is_nonzero: Incomplete
    _eval_is_odd: Incomplete
    _eval_is_polar: Incomplete
    _eval_is_positive: Incomplete
    _eval_is_prime: Incomplete
    _eval_is_rational: Incomplete
    _eval_is_real: Incomplete
    _eval_is_transcendental: Incomplete
    _eval_is_zero: Incomplete

def is_infinite(obj, assumptions: Incomplete | None = None): ...
def is_extended_real(obj, assumptions: Incomplete | None = None): ...
def is_extended_nonnegative(obj, assumptions: Incomplete | None = None): ...
