from _typeshed import Incomplete
from sympy.core import S as S
from sympy.core.numbers import equal_valued as equal_valued
from sympy.printing.codeprinter import CodePrinter as CodePrinter
from sympy.printing.precedence import PRECEDENCE as PRECEDENCE, precedence as precedence
from typing import Any

known_functions: Incomplete

class JavascriptCodePrinter(CodePrinter):
    '''"A Printer to convert Python expressions to strings of JavaScript code
    '''
    printmethod: str
    language: str
    _default_settings: dict[str, Any]
    known_functions: Incomplete
    def __init__(self, settings={}) -> None: ...
    def _rate_index_position(self, p): ...
    def _get_statement(self, codestring): ...
    def _get_comment(self, text): ...
    def _declare_number_const(self, name, value): ...
    def _format_code(self, lines): ...
    def _traverse_matrix_indices(self, mat): ...
    def _get_loop_opening_ending(self, indices): ...
    def _print_Pow(self, expr): ...
    def _print_Rational(self, expr): ...
    def _print_Mod(self, expr): ...
    def _print_Relational(self, expr): ...
    def _print_Indexed(self, expr): ...
    def _print_Exp1(self, expr): ...
    def _print_Pi(self, expr): ...
    def _print_Infinity(self, expr): ...
    def _print_NegativeInfinity(self, expr): ...
    def _print_Piecewise(self, expr): ...
    def _print_MatrixElement(self, expr): ...
    def indent_code(self, code):
        """Accepts a string of code or a list of code lines"""

def jscode(expr, assign_to=None, **settings):
    '''Converts an expr to a string of javascript code

    Parameters
    ==========

    expr : Expr
        A SymPy expression to be converted.
    assign_to : optional
        When given, the argument is used as the name of the variable to which
        the expression is assigned. Can be a string, ``Symbol``,
        ``MatrixSymbol``, or ``Indexed`` type. This is helpful in case of
        line-wrapping, or for expressions that generate multi-line statements.
    precision : integer, optional
        The precision for numbers such as pi [default=15].
    user_functions : dict, optional
        A dictionary where keys are ``FunctionClass`` instances and values are
        their string representations. Alternatively, the dictionary value can
        be a list of tuples i.e. [(argument_test, js_function_string)]. See
        below for examples.
    human : bool, optional
        If True, the result is a single string that may contain some constant
        declarations for the number symbols. If False, the same information is
        returned in a tuple of (symbols_to_declare, not_supported_functions,
        code_text). [default=True].
    contract: bool, optional
        If True, ``Indexed`` instances are assumed to obey tensor contraction
        rules and the corresponding nested loops over indices are generated.
        Setting contract=False will not generate loops, instead the user is
        responsible to provide values for the indices in the code.
        [default=True].

    Examples
    ========

    >>> from sympy import jscode, symbols, Rational, sin, ceiling, Abs
    >>> x, tau = symbols("x, tau")
    >>> jscode((2*tau)**Rational(7, 2))
    \'8*Math.sqrt(2)*Math.pow(tau, 7/2)\'
    >>> jscode(sin(x), assign_to="s")
    \'s = Math.sin(x);\'

    Custom printing can be defined for certain types by passing a dictionary of
    "type" : "function" to the ``user_functions`` kwarg. Alternatively, the
    dictionary value can be a list of tuples i.e. [(argument_test,
    js_function_string)].

    >>> custom_functions = {
    ...   "ceiling": "CEIL",
    ...   "Abs": [(lambda x: not x.is_integer, "fabs"),
    ...           (lambda x: x.is_integer, "ABS")]
    ... }
    >>> jscode(Abs(x) + ceiling(x), user_functions=custom_functions)
    \'fabs(x) + CEIL(x)\'

    ``Piecewise`` expressions are converted into conditionals. If an
    ``assign_to`` variable is provided an if statement is created, otherwise
    the ternary operator is used. Note that if the ``Piecewise`` lacks a
    default term, represented by ``(expr, True)`` then an error will be thrown.
    This is to prevent generating an expression that may not evaluate to
    anything.

    >>> from sympy import Piecewise
    >>> expr = Piecewise((x + 1, x > 0), (x, True))
    >>> print(jscode(expr, tau))
    if (x > 0) {
       tau = x + 1;
    }
    else {
       tau = x;
    }

    Support for loops is provided through ``Indexed`` types. With
    ``contract=True`` these expressions will be turned into loops, whereas
    ``contract=False`` will just print the assignment expression that should be
    looped over:

    >>> from sympy import Eq, IndexedBase, Idx
    >>> len_y = 5
    >>> y = IndexedBase(\'y\', shape=(len_y,))
    >>> t = IndexedBase(\'t\', shape=(len_y,))
    >>> Dy = IndexedBase(\'Dy\', shape=(len_y-1,))
    >>> i = Idx(\'i\', len_y-1)
    >>> e=Eq(Dy[i], (y[i+1]-y[i])/(t[i+1]-t[i]))
    >>> jscode(e.rhs, assign_to=e.lhs, contract=False)
    \'Dy[i] = (y[i + 1] - y[i])/(t[i + 1] - t[i]);\'

    Matrices are also supported, but a ``MatrixSymbol`` of the same dimensions
    must be provided to ``assign_to``. Note that any expression that can be
    generated normally can also exist inside a Matrix:

    >>> from sympy import Matrix, MatrixSymbol
    >>> mat = Matrix([x**2, Piecewise((x + 1, x > 0), (x, True)), sin(x)])
    >>> A = MatrixSymbol(\'A\', 3, 1)
    >>> print(jscode(mat, A))
    A[0] = Math.pow(x, 2);
    if (x > 0) {
       A[1] = x + 1;
    }
    else {
       A[1] = x;
    }
    A[2] = Math.sin(x);
    '''
def print_jscode(expr, **settings) -> None:
    """Prints the Javascript representation of the given expression.

       See jscode for the meaning of the optional arguments.
    """
