from _typeshed import Incomplete
from sympy.core import S as S
from sympy.core.numbers import Integer as Integer, IntegerConstant as IntegerConstant, equal_valued as equal_valued
from sympy.printing.codeprinter import CodePrinter as CodePrinter
from sympy.printing.precedence import PRECEDENCE as PRECEDENCE, precedence as precedence

_known_func_same_name: Incomplete
known_functions: Incomplete
number_symbols: Incomplete
spec_relational_ops: Incomplete
not_supported_symbol: Incomplete

class MapleCodePrinter(CodePrinter):
    """
    Printer which converts a SymPy expression into a maple code.
    """
    printmethod: str
    language: str
    _operators: Incomplete
    _default_settings: Incomplete
    known_functions: Incomplete
    def __init__(self, settings=None) -> None: ...
    def _get_statement(self, codestring): ...
    def _get_comment(self, text): ...
    def _declare_number_const(self, name, value): ...
    def _format_code(self, lines): ...
    def _print_tuple(self, expr): ...
    def _print_Tuple(self, expr): ...
    def _print_Assignment(self, expr): ...
    def _print_Pow(self, expr, **kwargs): ...
    def _print_Piecewise(self, expr): ...
    def _print_Rational(self, expr): ...
    def _print_Relational(self, expr): ...
    def _print_NumberSymbol(self, expr): ...
    def _print_NegativeInfinity(self, expr): ...
    def _print_Infinity(self, expr): ...
    def _print_BooleanTrue(self, expr): ...
    def _print_BooleanFalse(self, expr): ...
    def _print_bool(self, expr): ...
    def _print_NaN(self, expr): ...
    def _get_matrix(self, expr, sparse: bool = False): ...
    def _print_MatrixElement(self, expr): ...
    def _print_MatrixBase(self, expr): ...
    def _print_SparseRepMatrix(self, expr): ...
    def _print_Identity(self, expr): ...
    def _print_MatMul(self, expr): ...
    def _print_MatPow(self, expr): ...
    def _print_HadamardProduct(self, expr): ...
    def _print_Derivative(self, expr): ...

def maple_code(expr, assign_to=None, **settings):
    """Converts ``expr`` to a string of Maple code.

    Parameters
    ==========

    expr : Expr
        A SymPy expression to be converted.
    assign_to : optional
        When given, the argument is used as the name of the variable to which
        the expression is assigned.  Can be a string, ``Symbol``,
        ``MatrixSymbol``, or ``Indexed`` type.  This can be helpful for
        expressions that generate multi-line statements.
    precision : integer, optional
        The precision for numbers such as pi  [default=16].
    user_functions : dict, optional
        A dictionary where keys are ``FunctionClass`` instances and values are
        their string representations.  Alternatively, the dictionary value can
        be a list of tuples i.e. [(argument_test, cfunction_string)].  See
        below for examples.
    human : bool, optional
        If True, the result is a single string that may contain some constant
        declarations for the number symbols.  If False, the same information is
        returned in a tuple of (symbols_to_declare, not_supported_functions,
        code_text).  [default=True].
    contract: bool, optional
        If True, ``Indexed`` instances are assumed to obey tensor contraction
        rules and the corresponding nested loops over indices are generated.
        Setting contract=False will not generate loops, instead the user is
        responsible to provide values for the indices in the code.
        [default=True].
    inline: bool, optional
        If True, we try to create single-statement code instead of multiple
        statements.  [default=True].

    """
def print_maple_code(expr, **settings) -> None:
    """Prints the Maple representation of the given expression.

    See :func:`maple_code` for the meaning of the optional arguments.

    Examples
    ========

    >>> from sympy import print_maple_code, symbols
    >>> x, y = symbols('x y')
    >>> print_maple_code(x, assign_to=y)
    y := x
    """
