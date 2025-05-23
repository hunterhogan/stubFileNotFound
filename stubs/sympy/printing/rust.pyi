from _typeshed import Incomplete
from sympy.core import Float as Float, Lambda as Lambda, Rational as Rational, S as S
from sympy.core.numbers import equal_valued as equal_valued
from sympy.printing.codeprinter import CodePrinter as CodePrinter
from typing import Any

known_functions: Incomplete
reserved_words: Incomplete

class RustCodePrinter(CodePrinter):
    """A printer to convert SymPy expressions to strings of Rust code"""
    printmethod: str
    language: str
    _default_settings: dict[str, Any]
    known_functions: Incomplete
    _dereference: Incomplete
    reserved_words: Incomplete
    def __init__(self, settings={}) -> None: ...
    def _rate_index_position(self, p): ...
    def _get_statement(self, codestring): ...
    def _get_comment(self, text): ...
    def _declare_number_const(self, name, value): ...
    def _format_code(self, lines): ...
    def _traverse_matrix_indices(self, mat): ...
    def _get_loop_opening_ending(self, indices): ...
    def _print_caller_var(self, expr): ...
    def _print_Function(self, expr):
        """
        basic function for printing `Function`

        Function Style :

        1. args[0].func(args[1:]), method with arguments
        2. args[0].func(), method without arguments
        3. args[1].func(), method without arguments (e.g. (e, x) => x.exp())
        4. func(args), function with arguments
        """
    def _print_Pow(self, expr): ...
    def _print_Float(self, expr, _type: bool = False): ...
    def _print_Integer(self, expr, _type: bool = False): ...
    def _print_Rational(self, expr): ...
    def _print_Relational(self, expr): ...
    def _print_Indexed(self, expr): ...
    def _print_Idx(self, expr): ...
    def _print_Dummy(self, expr): ...
    def _print_Exp1(self, expr, _type: bool = False): ...
    def _print_Pi(self, expr, _type: bool = False): ...
    def _print_Infinity(self, expr, _type: bool = False): ...
    def _print_NegativeInfinity(self, expr, _type: bool = False): ...
    def _print_BooleanTrue(self, expr, _type: bool = False): ...
    def _print_BooleanFalse(self, expr, _type: bool = False): ...
    def _print_bool(self, expr, _type: bool = False): ...
    def _print_NaN(self, expr, _type: bool = False): ...
    def _print_Piecewise(self, expr): ...
    def _print_ITE(self, expr): ...
    def _print_MatrixBase(self, A): ...
    def _print_SparseRepMatrix(self, mat): ...
    def _print_MatrixElement(self, expr): ...
    def _print_Symbol(self, expr): ...
    def _print_Assignment(self, expr): ...
    def indent_code(self, code):
        """Accepts a string of code or a list of code lines"""

def rust_code(expr, assign_to: Incomplete | None = None, **settings):
    '''Converts an expr to a string of Rust code

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
        A dictionary where the keys are string representations of either
        ``FunctionClass`` or ``UndefinedFunction`` instances and the values
        are their desired C string representations. Alternatively, the
        dictionary value can be a list of tuples i.e. [(argument_test,
        cfunction_string)].  See below for examples.
    dereference : iterable, optional
        An iterable of symbols that should be dereferenced in the printed code
        expression. These would be values passed by address to the function.
        For example, if ``dereference=[a]``, the resulting code would print
        ``(*a)`` instead of ``a``.
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

    >>> from sympy import rust_code, symbols, Rational, sin, ceiling, Abs, Function
    >>> x, tau = symbols("x, tau")
    >>> rust_code((2*tau)**Rational(7, 2))
    \'8*1.4142135623731*tau.powf(7_f64/2.0)\'
    >>> rust_code(sin(x), assign_to="s")
    \'s = x.sin();\'

    Simple custom printing can be defined for certain types by passing a
    dictionary of {"type" : "function"} to the ``user_functions`` kwarg.
    Alternatively, the dictionary value can be a list of tuples i.e.
    [(argument_test, cfunction_string)].

    >>> custom_functions = {
    ...   "ceiling": "CEIL",
    ...   "Abs": [(lambda x: not x.is_integer, "fabs", 4),
    ...           (lambda x: x.is_integer, "ABS", 4)],
    ...   "func": "f"
    ... }
    >>> func = Function(\'func\')
    >>> rust_code(func(Abs(x) + ceiling(x)), user_functions=custom_functions)
    \'(fabs(x) + x.CEIL()).f()\'

    ``Piecewise`` expressions are converted into conditionals. If an
    ``assign_to`` variable is provided an if statement is created, otherwise
    the ternary operator is used. Note that if the ``Piecewise`` lacks a
    default term, represented by ``(expr, True)`` then an error will be thrown.
    This is to prevent generating an expression that may not evaluate to
    anything.

    >>> from sympy import Piecewise
    >>> expr = Piecewise((x + 1, x > 0), (x, True))
    >>> print(rust_code(expr, tau))
    tau = if (x > 0) {
        x + 1
    } else {
        x
    };

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
    >>> rust_code(e.rhs, assign_to=e.lhs, contract=False)
    \'Dy[i] = (y[i + 1] - y[i])/(t[i + 1] - t[i]);\'

    Matrices are also supported, but a ``MatrixSymbol`` of the same dimensions
    must be provided to ``assign_to``. Note that any expression that can be
    generated normally can also exist inside a Matrix:

    >>> from sympy import Matrix, MatrixSymbol
    >>> mat = Matrix([x**2, Piecewise((x + 1, x > 0), (x, True)), sin(x)])
    >>> A = MatrixSymbol(\'A\', 3, 1)
    >>> print(rust_code(mat, A))
    A = [x.powi(2), if (x > 0) {
        x + 1
    } else {
        x
    }, x.sin()];
    '''
def print_rust_code(expr, **settings) -> None:
    """Prints Rust representation of the given expression."""
