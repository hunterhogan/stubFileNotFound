from _typeshed import Incomplete
from sympy.core import Add as Add, Eq as Eq, Expr as Expr, Mul as Mul, Pow as Pow, S as S, expand_mul as expand_mul, expand_multinomial as expand_multinomial
from sympy.core.exprtools import decompose_power as decompose_power, decompose_power_rat as decompose_power_rat
from sympy.core.numbers import _illegal as _illegal
from sympy.external.gmpy import GROUND_TYPES as GROUND_TYPES
from sympy.polys.domains.modularinteger import ModularInteger as ModularInteger
from sympy.polys.polyerrors import GeneratorsError as GeneratorsError, PolynomialError as PolynomialError
from sympy.polys.polyoptions import build_options as build_options

_gens_order: Incomplete
_max_order: int
_re_gen: Incomplete

def _nsort(roots, separated: bool = False):
    """Sort the numerical roots putting the real roots first, then sorting
    according to real and imaginary parts. If ``separated`` is True, then
    the real and imaginary roots will be returned in two lists, respectively.

    This routine tries to avoid issue 6137 by separating the roots into real
    and imaginary parts before evaluation. In addition, the sorting will raise
    an error if any computation cannot be done with precision.
    """
def _sort_gens(gens, **args):
    """Sort generators in a reasonably intelligent way. """
def _unify_gens(f_gens, g_gens):
    """Unify generators in a reasonably intelligent way. """
def _analyze_gens(gens):
    """Support for passing generators as `*gens` and `[gens]`. """
def _sort_factors(factors, **args):
    """Sort low-level factors in increasing 'complexity' order. """

illegal_types: Incomplete
finf: Incomplete

def _not_a_coeff(expr):
    """Do not treat NaN and infinities as valid polynomial coefficients. """
def _parallel_dict_from_expr_if_gens(exprs, opt):
    """Transform expressions into a multinomial form given generators. """
def _parallel_dict_from_expr_no_gens(exprs, opt):
    """Transform expressions into a multinomial form and figure out generators. """
def _dict_from_expr_if_gens(expr, opt):
    """Transform an expression into a multinomial form given generators. """
def _dict_from_expr_no_gens(expr, opt):
    """Transform an expression into a multinomial form and figure out generators. """
def parallel_dict_from_expr(exprs, **args):
    """Transform expressions into a multinomial form. """
def _parallel_dict_from_expr(exprs, opt):
    """Transform expressions into a multinomial form. """
def dict_from_expr(expr, **args):
    """Transform an expression into a multinomial form. """
def _dict_from_expr(expr, opt):
    """Transform an expression into a multinomial form. """
def expr_from_dict(rep, *gens):
    """Convert a multinomial form into an expression. """
parallel_dict_from_basic = parallel_dict_from_expr
dict_from_basic = dict_from_expr
basic_from_dict = expr_from_dict

def _dict_reorder(rep, gens, new_gens):
    """Reorder levels using dict representation. """

class PicklableWithSlots:
    """
    Mixin class that allows to pickle objects with ``__slots__``.

    Examples
    ========

    First define a class that mixes :class:`PicklableWithSlots` in::

        >>> from sympy.polys.polyutils import PicklableWithSlots
        >>> class Some(PicklableWithSlots):
        ...     __slots__ = ('foo', 'bar')
        ...
        ...     def __init__(self, foo, bar):
        ...         self.foo = foo
        ...         self.bar = bar

    To make :mod:`pickle` happy in doctest we have to use these hacks::

        >>> import builtins
        >>> builtins.Some = Some
        >>> from sympy.polys import polyutils
        >>> polyutils.Some = Some

    Next lets see if we can create an instance, pickle it and unpickle::

        >>> some = Some('abc', 10)
        >>> some.foo, some.bar
        ('abc', 10)

        >>> from pickle import dumps, loads
        >>> some2 = loads(dumps(some))

        >>> some2.foo, some2.bar
        ('abc', 10)

    """
    __slots__: Incomplete
    def __getstate__(self, cls: Incomplete | None = None): ...
    def __setstate__(self, d) -> None: ...

class IntegerPowerable:
    """
    Mixin class for classes that define a `__mul__` method, and want to be
    raised to integer powers in the natural way that follows. Implements
    powering via binary expansion, for efficiency.

    By default, only integer powers $\\geq 2$ are supported. To support the
    first, zeroth, or negative powers, override the corresponding methods,
    `_first_power`, `_zeroth_power`, `_negative_power`, below.
    """
    def __pow__(self, e, modulo: Incomplete | None = None): ...
    def _negative_power(self, e, modulo: Incomplete | None = None) -> None:
        """
        Compute inverse of self, then raise that to the abs(e) power.
        For example, if the class has an `inv()` method,
            return self.inv() ** abs(e) % modulo
        """
    def _zeroth_power(self) -> None:
        """Return unity element of algebraic struct to which self belongs."""
    def _first_power(self) -> None:
        """Return a copy of self."""

_GF_types: tuple[type, ...]
