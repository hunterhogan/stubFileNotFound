from .basic import Basic as Basic
from .parameters import global_parameters as global_parameters
from _typeshed import Incomplete
from sympy.core.expr import Expr as Expr
from sympy.core.numbers import Float as Float, Integer as Integer
from sympy.core.random import choice as choice
from sympy.utilities.iterables import iterable as iterable
from typing import Any, Callable, TypeVar, overload

Tbasic = TypeVar('Tbasic', bound=Basic)

class SympifyError(ValueError):
    expr: Incomplete
    base_exc: Incomplete
    def __init__(self, expr, base_exc=None) -> None: ...
    def __str__(self) -> str: ...

converter: dict[type[Any], Callable[[Any], Basic]]
_sympy_converter: dict[type[Any], Callable[[Any], Basic]]
_external_converter = converter

class CantSympify:
    """
    Mix in this trait to a class to disallow sympification of its instances.

    Examples
    ========

    >>> from sympy import sympify
    >>> from sympy.core.sympify import CantSympify

    >>> class Something(dict):
    ...     pass
    ...
    >>> sympify(Something())
    {}

    >>> class Something(dict, CantSympify):
    ...     pass
    ...
    >>> sympify(Something())
    Traceback (most recent call last):
    ...
    SympifyError: SympifyError: {}

    """
    __slots__: Incomplete

def _is_numpy_instance(a):
    """
    Checks if an object is an instance of a type from the numpy module.
    """
def _convert_numpy_types(a, **sympify_args):
    """
    Converts a numpy datatype input to an appropriate SymPy type.
    """
@overload
def sympify(a: int, *, strict: bool = False) -> Integer: ...
@overload
def sympify(a: float, *, strict: bool = False) -> Float: ...
@overload
def sympify(a: Expr | complex, *, strict: bool = False) -> Expr: ...
@overload
def sympify(a: Tbasic, *, strict: bool = False) -> Tbasic: ...
@overload
def sympify(a: Any, *, strict: bool = False) -> Basic: ...
def _sympify(a):
    """
    Short version of :func:`~.sympify` for internal usage for ``__add__`` and
    ``__eq__`` methods where it is ok to allow some things (like Python
    integers and floats) in the expression. This excludes things (like strings)
    that are unwise to allow into such an expression.

    >>> from sympy import Integer
    >>> Integer(1) == 1
    True

    >>> Integer(1) == '1'
    False

    >>> from sympy.abc import x
    >>> x + 1
    x + 1

    >>> x + '1'
    Traceback (most recent call last):
    ...
    TypeError: unsupported operand type(s) for +: 'Symbol' and 'str'

    see: sympify

    """
def kernS(s):
    """Use a hack to try keep autosimplification from distributing a
    a number into an Add; this modification does not
    prevent the 2-arg Mul from becoming an Add, however.

    Examples
    ========

    >>> from sympy.core.sympify import kernS
    >>> from sympy.abc import x, y

    The 2-arg Mul distributes a number (or minus sign) across the terms
    of an expression, but kernS will prevent that:

    >>> 2*(x + y), -(x + 1)
    (2*x + 2*y, -x - 1)
    >>> kernS('2*(x + y)')
    2*(x + y)
    >>> kernS('-(x + 1)')
    -(x + 1)

    If use of the hack fails, the un-hacked string will be passed to sympify...
    and you get what you get.

    XXX This hack should not be necessary once issue 4596 has been resolved.
    """
