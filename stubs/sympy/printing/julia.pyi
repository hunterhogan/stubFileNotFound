from _typeshed import Incomplete
from sympy.core import Mul as Mul, Pow as Pow, Rational as Rational, S as S
from sympy.printing.codeprinter import CodePrinter as CodePrinter
from sympy.printing.precedence import PRECEDENCE as PRECEDENCE, precedence as precedence
from typing import Any

known_fcns_src1: Incomplete
known_fcns_src2: Incomplete

class JuliaCodePrinter(CodePrinter):
    """
    A printer to convert expressions to strings of Julia code.
    """
    printmethod: str
    language: str
    _operators: Incomplete
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
    def _print_Mul(self, expr): ...
    def _print_Relational(self, expr): ...
    def _print_Pow(self, expr): ...
    def _print_MatPow(self, expr): ...
    def _print_Pi(self, expr): ...
    def _print_ImaginaryUnit(self, expr): ...
    def _print_Exp1(self, expr): ...
    def _print_EulerGamma(self, expr): ...
    def _print_Catalan(self, expr): ...
    def _print_GoldenRatio(self, expr): ...
    def _print_Assignment(self, expr): ...
    def _print_Infinity(self, expr): ...
    def _print_NegativeInfinity(self, expr): ...
    def _print_NaN(self, expr): ...
    def _print_list(self, expr): ...
    def _print_tuple(self, expr): ...
    _print_Tuple = _print_tuple
    def _print_BooleanTrue(self, expr): ...
    def _print_BooleanFalse(self, expr): ...
    def _print_bool(self, expr): ...
    def _print_MatrixBase(self, A): ...
    def _print_SparseRepMatrix(self, A): ...
    def _print_MatrixElement(self, expr): ...
    def _print_MatrixSlice(self, expr): ...
    def _print_Indexed(self, expr): ...
    def _print_Idx(self, expr): ...
    def _print_Identity(self, expr): ...
    def _print_HadamardProduct(self, expr): ...
    def _print_HadamardPower(self, expr): ...
    def _print_Rational(self, expr): ...
    def _print_jn(self, expr): ...
    def _print_yn(self, expr): ...
    def _print_Piecewise(self, expr): ...
    def _print_MatMul(self, expr): ...
    def indent_code(self, code):
        """Accepts a string of code or a list of code lines"""

def julia_code(expr, assign_to: Incomplete | None = None, **settings):
    '''Converts `expr` to a string of Julia code.

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

    Examples
    ========

    >>> from sympy import julia_code, symbols, sin, pi
    >>> x = symbols(\'x\')
    >>> julia_code(sin(x).series(x).removeO())
    \'x .^ 5 / 120 - x .^ 3 / 6 + x\'

    >>> from sympy import Rational, ceiling
    >>> x, y, tau = symbols("x, y, tau")
    >>> julia_code((2*tau)**Rational(7, 2))
    \'8 * sqrt(2) * tau .^ (7 // 2)\'

    Note that element-wise (Hadamard) operations are used by default between
    symbols.  This is because its possible in Julia to write "vectorized"
    code.  It is harmless if the values are scalars.

    >>> julia_code(sin(pi*x*y), assign_to="s")
    \'s = sin(pi * x .* y)\'

    If you need a matrix product "*" or matrix power "^", you can specify the
    symbol as a ``MatrixSymbol``.

    >>> from sympy import Symbol, MatrixSymbol
    >>> n = Symbol(\'n\', integer=True, positive=True)
    >>> A = MatrixSymbol(\'A\', n, n)
    >>> julia_code(3*pi*A**3)
    \'(3 * pi) * A ^ 3\'

    This class uses several rules to decide which symbol to use a product.
    Pure numbers use "*", Symbols use ".*" and MatrixSymbols use "*".
    A HadamardProduct can be used to specify componentwise multiplication ".*"
    of two MatrixSymbols.  There is currently there is no easy way to specify
    scalar symbols, so sometimes the code might have some minor cosmetic
    issues.  For example, suppose x and y are scalars and A is a Matrix, then
    while a human programmer might write "(x^2*y)*A^3", we generate:

    >>> julia_code(x**2*y*A**3)
    \'(x .^ 2 .* y) * A ^ 3\'

    Matrices are supported using Julia inline notation.  When using
    ``assign_to`` with matrices, the name can be specified either as a string
    or as a ``MatrixSymbol``.  The dimensions must align in the latter case.

    >>> from sympy import Matrix, MatrixSymbol
    >>> mat = Matrix([[x**2, sin(x), ceiling(x)]])
    >>> julia_code(mat, assign_to=\'A\')
    \'A = [x .^ 2 sin(x) ceil(x)]\'

    ``Piecewise`` expressions are implemented with logical masking by default.
    Alternatively, you can pass "inline=False" to use if-else conditionals.
    Note that if the ``Piecewise`` lacks a default term, represented by
    ``(expr, True)`` then an error will be thrown.  This is to prevent
    generating an expression that may not evaluate to anything.

    >>> from sympy import Piecewise
    >>> pw = Piecewise((x + 1, x > 0), (x, True))
    >>> julia_code(pw, assign_to=tau)
    \'tau = ((x > 0) ? (x + 1) : (x))\'

    Note that any expression that can be generated normally can also exist
    inside a Matrix:

    >>> mat = Matrix([[x**2, pw, sin(x)]])
    >>> julia_code(mat, assign_to=\'A\')
    \'A = [x .^ 2 ((x > 0) ? (x + 1) : (x)) sin(x)]\'

    Custom printing can be defined for certain types by passing a dictionary of
    "type" : "function" to the ``user_functions`` kwarg.  Alternatively, the
    dictionary value can be a list of tuples i.e., [(argument_test,
    cfunction_string)].  This can be used to call a custom Julia function.

    >>> from sympy import Function
    >>> f = Function(\'f\')
    >>> g = Function(\'g\')
    >>> custom_functions = {
    ...   "f": "existing_julia_fcn",
    ...   "g": [(lambda x: x.is_Matrix, "my_mat_fcn"),
    ...         (lambda x: not x.is_Matrix, "my_fcn")]
    ... }
    >>> mat = Matrix([[1, x]])
    >>> julia_code(f(x) + g(x) + g(mat), user_functions=custom_functions)
    \'existing_julia_fcn(x) + my_fcn(x) + my_mat_fcn([1 x])\'

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
    >>> e = Eq(Dy[i], (y[i+1]-y[i])/(t[i+1]-t[i]))
    >>> julia_code(e.rhs, assign_to=e.lhs, contract=False)
    \'Dy[i] = (y[i + 1] - y[i]) ./ (t[i + 1] - t[i])\'
    '''
def print_julia_code(expr, **settings) -> None:
    """Prints the Julia representation of the given expression.

    See `julia_code` for the meaning of the optional arguments.
    """
