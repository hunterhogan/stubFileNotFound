from sympy.core.add import Add as Add
from sympy.core.basic import Basic as Basic
from sympy.core.function import (
	AppliedUndef as AppliedUndef, Derivative as Derivative, diff as diff, expand as expand, Function as Function)
from sympy.core.intfunc import igcd as igcd
from sympy.core.mul import Mul as Mul
from sympy.core.numbers import E as E, I as I, Integer as Integer, pi as pi, Rational as Rational
from sympy.core.singleton import S as S
from sympy.core.symbol import Symbol as Symbol, symbols as symbols, var as var
from sympy.core.sympify import sympify as sympify, SympifyError as SympifyError
from sympy.functions.elementary.exponential import exp as exp, log as log
from sympy.functions.elementary.hyperbolic import (
	acosh as acosh, acoth as acoth, asinh as asinh, atanh as atanh, cosh as cosh, coth as coth, sinh as sinh, tanh as tanh)
from sympy.functions.elementary.miscellaneous import sqrt as sqrt
from sympy.functions.elementary.trigonometric import (
	acos as acos, acot as acot, acsc as acsc, asec as asec, asin as asin, atan as atan, cos as cos, cot as cot, csc as csc,
	sec as sec, sin as sin, tan as tan)
from sympy.functions.special.gamma_functions import gamma as gamma
from sympy.matrices.dense import (
	diag as diag, eye as eye, Matrix as Matrix, ones as ones, symarray as symarray, zeros as zeros)
from sympy.matrices.immutable import ImmutableMatrix as ImmutableMatrix
from sympy.matrices.matrixbase import MatrixBase as MatrixBase
from sympy.utilities.lambdify import lambdify as lambdify

__all__ = ['Add', 'AppliedUndef', 'Basic', 'Derivative', 'E', 'Function', 'I', 'ImmutableMatrix', 'Integer', 'Matrix', 'MatrixBase', 'Mul', 'Rational', 'S', 'Symbol', 'SympifyError', 'acos', 'acosh', 'acot', 'acoth', 'acsc', 'asec', 'asin', 'asinh', 'atan', 'atanh', 'cos', 'cosh', 'cot', 'coth', 'csc', 'diag', 'diff', 'exp', 'expand', 'eye', 'gamma', 'igcd', 'lambdify', 'log', 'ones', 'pi', 'sec', 'sin', 'sinh', 'sqrt', 'symarray', 'symbols', 'sympify', 'tan', 'tanh', 'var', 'zeros']

def sympify(a, *, strict: bool = False):
    """
    Notes
    -----
    SymEngine's ``sympify`` does not accept keyword arguments and is
    therefore not compatible with SymPy's ``sympify`` with ``strict=True``
    (which ensures that only the types for which an explicit conversion has
    been defined are converted). This wrapper adds an additional parameter
    ``strict`` (with default ``False``) that will raise a ``SympifyError``
    if ``strict=True`` and the argument passed to the parameter ``a`` is a
    string.

    See Also
    --------
    sympify: Converts an arbitrary expression to a type that can be used
        inside SymPy.

    """
