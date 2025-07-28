from .codeprinter import CodePrinter as CodePrinter
from .precedence import precedence as precedence
from _typeshed import Incomplete
from collections.abc import Generator
from sympy.core import S as S
from sympy.core.mod import Mod as Mod
from sympy.printing.pycode import PythonCodePrinter as PythonCodePrinter

_kw: Incomplete
_known_functions: Incomplete
_known_functions_math: Incomplete
_known_constants_math: Incomplete

def _print_known_func(self, expr): ...
def _print_known_const(self, expr): ...

class AbstractPythonCodePrinter(CodePrinter):
    printmethod: str
    language: str
    reserved_words = _kw
    modules: Incomplete
    tab: str
    _kf: Incomplete
    _kc: Incomplete
    _operators: Incomplete
    _default_settings: Incomplete
    standard: Incomplete
    module_imports: Incomplete
    known_functions: Incomplete
    known_constants: Incomplete
    def __init__(self, settings=None) -> None: ...
    def _declare_number_const(self, name, value): ...
    def _module_format(self, fqn, register: bool = True): ...
    def _format_code(self, lines): ...
    def _get_statement(self, codestring): ...
    def _get_comment(self, text): ...
    def _expand_fold_binary_op(self, op, args):
        """
        This method expands a fold on binary operations.

        ``functools.reduce`` is an example of a folded operation.

        For example, the expression

        `A + B + C + D`

        is folded into

        `((A + B) + C) + D`
        """
    def _expand_reduce_binary_op(self, op, args):
        """
        This method expands a reduction on binary operations.

        Notice: this is NOT the same as ``functools.reduce``.

        For example, the expression

        `A + B + C + D`

        is reduced into:

        `(A + B) + (C + D)`
        """
    def _print_NaN(self, expr): ...
    def _print_Infinity(self, expr): ...
    def _print_NegativeInfinity(self, expr): ...
    def _print_ComplexInfinity(self, expr): ...
    def _print_Mod(self, expr): ...
    def _print_Piecewise(self, expr): ...
    def _print_Relational(self, expr):
        """Relational printer for Equality and Unequality"""
    def _print_ITE(self, expr): ...
    def _print_Sum(self, expr): ...
    def _print_ImaginaryUnit(self, expr): ...
    def _print_KroneckerDelta(self, expr): ...
    def _print_MatrixBase(self, expr): ...
    _print_SparseRepMatrix: Incomplete
    _print_MutableSparseMatrix: Incomplete
    _print_ImmutableSparseMatrix: Incomplete
    _print_Matrix: Incomplete
    _print_DenseMatrix: Incomplete
    _print_MutableDenseMatrix: Incomplete
    _print_ImmutableMatrix: Incomplete
    _print_ImmutableDenseMatrix: Incomplete
    def _indent_codestring(self, codestring): ...
    def _print_FunctionDefinition(self, fd): ...
    def _print_While(self, whl): ...
    def _print_Declaration(self, decl): ...
    def _print_BreakToken(self, bt): ...
    def _print_Return(self, ret): ...
    def _print_Raise(self, rs): ...
    def _print_RuntimeError_(self, re): ...
    def _print_Print(self, prnt): ...
    def _print_Stream(self, strm): ...
    def _print_NoneToken(self, arg): ...
    def _hprint_Pow(self, expr, rational: bool = False, sqrt: str = 'math.sqrt'):
        """Printing helper function for ``Pow``

        Notes
        =====

        This preprocesses the ``sqrt`` as math formatter and prints division

        Examples
        ========

        >>> from sympy import sqrt
        >>> from sympy.printing.pycode import PythonCodePrinter
        >>> from sympy.abc import x

        Python code printer automatically looks up ``math.sqrt``.

        >>> printer = PythonCodePrinter()
        >>> printer._hprint_Pow(sqrt(x), rational=True)
        'x**(1/2)'
        >>> printer._hprint_Pow(sqrt(x), rational=False)
        'math.sqrt(x)'
        >>> printer._hprint_Pow(1/sqrt(x), rational=True)
        'x**(-1/2)'
        >>> printer._hprint_Pow(1/sqrt(x), rational=False)
        '1/math.sqrt(x)'
        >>> printer._hprint_Pow(1/x, rational=False)
        '1/x'
        >>> printer._hprint_Pow(1/x, rational=True)
        'x**(-1)'

        Using sqrt from numpy or mpmath

        >>> printer._hprint_Pow(sqrt(x), sqrt='numpy.sqrt')
        'numpy.sqrt(x)'
        >>> printer._hprint_Pow(sqrt(x), sqrt='mpmath.sqrt')
        'mpmath.sqrt(x)'

        See Also
        ========

        sympy.printing.str.StrPrinter._print_Pow
        """

class ArrayPrinter:
    def _arrayify(self, indexed): ...
    def _get_einsum_string(self, subranks, contraction_indices): ...
    def _get_letter_generator_for_einsum(self) -> Generator[Incomplete]: ...
    def _print_ArrayTensorProduct(self, expr): ...
    def _print_ArrayContraction(self, expr): ...
    def _print_ArrayDiagonal(self, expr): ...
    def _print_PermuteDims(self, expr): ...
    def _print_ArrayAdd(self, expr): ...
    def _print_OneArray(self, expr): ...
    def _print_ZeroArray(self, expr): ...
    def _print_Assignment(self, expr): ...
    def _print_IndexedBase(self, expr): ...

class PythonCodePrinter(AbstractPythonCodePrinter):
    def _print_sign(self, e): ...
    def _print_Not(self, expr): ...
    def _print_IndexedBase(self, expr): ...
    def _print_Indexed(self, expr): ...
    def _print_Pow(self, expr, rational: bool = False): ...
    def _print_Rational(self, expr): ...
    def _print_Half(self, expr): ...
    def _print_frac(self, expr): ...
    def _print_Symbol(self, expr): ...
    _print_lowergamma: Incomplete
    _print_uppergamma: Incomplete
    _print_fresnelc: Incomplete
    _print_fresnels: Incomplete

def pycode(expr, **settings):
    """ Converts an expr to a string of Python code

    Parameters
    ==========

    expr : Expr
        A SymPy expression.
    fully_qualified_modules : bool
        Whether or not to write out full module names of functions
        (``math.sin`` vs. ``sin``). default: ``True``.
    standard : str or None, optional
        Only 'python3' (default) is supported.
        This parameter may be removed in the future.

    Examples
    ========

    >>> from sympy import pycode, tan, Symbol
    >>> pycode(tan(Symbol('x')) + 1)
    'math.tan(x) + 1'

    """

_known_functions_cmath: Incomplete
_known_constants_cmath: Incomplete

class CmathPrinter(PythonCodePrinter):
    """ Printer for Python's cmath module """
    printmethod: str
    language: str
    _kf: Incomplete
    _kc: Incomplete
    def _print_Pow(self, expr, rational: bool = False): ...
    def _print_Float(self, e): ...
    def _print_known_func(self, expr): ...
    def _print_known_const(self, expr): ...
    def _print_re(self, expr):
        """Prints `re(z)` as `z.real`"""
    def _print_im(self, expr):
        """Prints `im(z)` as `z.imag`"""

_not_in_mpmath: Incomplete
_in_mpmath: Incomplete
_known_functions_mpmath: Incomplete
_known_constants_mpmath: Incomplete

def _unpack_integral_limits(integral_expr):
    """ helper function for _print_Integral that
        - accepts an Integral expression
        - returns a tuple of
           - a list variables of integration
           - a list of tuples of the upper and lower limits of integration
    """

class MpmathPrinter(PythonCodePrinter):
    """
    Lambda printer for mpmath which maintains precision for floats
    """
    printmethod: str
    language: str
    _kf: Incomplete
    _kc: Incomplete
    def _print_Float(self, e): ...
    def _print_Rational(self, e): ...
    def _print_Half(self, e): ...
    def _print_uppergamma(self, e): ...
    def _print_lowergamma(self, e): ...
    def _print_log2(self, e): ...
    def _print_log1p(self, e): ...
    def _print_Pow(self, expr, rational: bool = False): ...
    def _print_Integral(self, e): ...
    def _print_Derivative_zeta(self, args, seq_orders): ...

class SymPyPrinter(AbstractPythonCodePrinter):
    language: str
    _default_settings: Incomplete
    def _print_Function(self, expr): ...
    def _print_Pow(self, expr, rational: bool = False): ...
