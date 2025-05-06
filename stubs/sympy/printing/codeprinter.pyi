from _typeshed import Incomplete
from sympy.core import Add as Add, Float as Float, Mul as Mul, Pow as Pow, S as S, sympify as sympify
from sympy.printing.precedence import PRECEDENCE as PRECEDENCE, precedence as precedence
from sympy.printing.str import StrPrinter as StrPrinter
from typing import Any

class requires:
    """ Decorator for registering requirements on print methods. """
    _req: Incomplete
    def __init__(self, **kwargs) -> None: ...
    def __call__(self, method): ...

class AssignmentError(Exception):
    """
    Raised if an assignment variable for a loop is missing.
    """
class PrintMethodNotImplementedError(NotImplementedError):
    """
    Raised if a _print_* method is missing in the Printer.
    """

def _convert_python_lists(arg): ...

class CodePrinter(StrPrinter):
    """
    The base class for code-printing subclasses.
    """
    _operators: Incomplete
    _default_settings: dict[str, Any]
    _rewriteable_functions: Incomplete
    reserved_words: Incomplete
    def __init__(self, settings: Incomplete | None = None) -> None: ...
    def _handle_UnevaluatedExpr(self, expr): ...
    _not_supported: Incomplete
    _number_symbols: Incomplete
    def doprint(self, expr, assign_to: Incomplete | None = None):
        """
        Print the expression as code.

        Parameters
        ----------
        expr : Expression
            The expression to be printed.

        assign_to : Symbol, string, MatrixSymbol, list of strings or Symbols (optional)
            If provided, the printed code will set the expression to a variable or multiple variables
            with the name or names given in ``assign_to``.
        """
    def _doprint_loops(self, expr, assign_to: Incomplete | None = None): ...
    def _get_expression_indices(self, expr, assign_to): ...
    def _sort_optimized(self, indices, expr): ...
    def _rate_index_position(self, p) -> None:
        """function to calculate score based on position among indices

        This method is used to sort loops in an optimized order, see
        CodePrinter._sort_optimized()
        """
    def _get_statement(self, codestring) -> None:
        """Formats a codestring with the proper line ending."""
    def _get_comment(self, text) -> None:
        """Formats a text string as a comment."""
    def _declare_number_const(self, name, value) -> None:
        """Declare a numeric constant at the top of a function"""
    def _format_code(self, lines) -> None:
        """Take in a list of lines of code, and format them accordingly.

        This may include indenting, wrapping long lines, etc..."""
    def _get_loop_opening_ending(self, indices) -> None:
        """Returns a tuple (open_lines, close_lines) containing lists
        of codelines"""
    def _print_Dummy(self, expr): ...
    def _print_CodeBlock(self, expr): ...
    def _print_String(self, string): ...
    def _print_QuotedString(self, arg): ...
    def _print_Comment(self, string): ...
    def _print_Assignment(self, expr): ...
    def _print_AugmentedAssignment(self, expr): ...
    def _print_FunctionCall(self, expr): ...
    def _print_Variable(self, expr): ...
    def _print_Symbol(self, expr): ...
    def _can_print(self, name):
        """ Check if function ``name`` is either a known function or has its own
            printing method. Used to check if rewriting is possible."""
    def _print_Function(self, expr): ...
    _print_Expr = _print_Function
    _print_Heaviside: Incomplete
    def _print_NumberSymbol(self, expr): ...
    def _print_Catalan(self, expr): ...
    def _print_EulerGamma(self, expr): ...
    def _print_GoldenRatio(self, expr): ...
    def _print_TribonacciConstant(self, expr): ...
    def _print_Exp1(self, expr): ...
    def _print_Pi(self, expr): ...
    def _print_And(self, expr): ...
    def _print_Or(self, expr): ...
    def _print_Xor(self, expr): ...
    def _print_Equivalent(self, expr): ...
    def _print_Not(self, expr): ...
    def _print_BooleanFunction(self, expr): ...
    def _print_Mul(self, expr): ...
    def _print_not_supported(self, expr): ...
    _print_Basic = _print_not_supported
    _print_ComplexInfinity = _print_not_supported
    _print_Derivative = _print_not_supported
    _print_ExprCondPair = _print_not_supported
    _print_GeometryEntity = _print_not_supported
    _print_Infinity = _print_not_supported
    _print_Integral = _print_not_supported
    _print_Interval = _print_not_supported
    _print_AccumulationBounds = _print_not_supported
    _print_Limit = _print_not_supported
    _print_MatrixBase = _print_not_supported
    _print_DeferredVector = _print_not_supported
    _print_NaN = _print_not_supported
    _print_NegativeInfinity = _print_not_supported
    _print_Order = _print_not_supported
    _print_RootOf = _print_not_supported
    _print_RootsOf = _print_not_supported
    _print_RootSum = _print_not_supported
    _print_Uniform = _print_not_supported
    _print_Unit = _print_not_supported
    _print_Wild = _print_not_supported
    _print_WildFunction = _print_not_supported
    _print_Relational = _print_not_supported

def ccode(expr, assign_to: Incomplete | None = None, standard: str = 'c99', **settings):
    '''Converts an expr to a string of c code

    Parameters
    ==========

    expr : Expr
        A SymPy expression to be converted.
    assign_to : optional
        When given, the argument is used as the name of the variable to which
        the expression is assigned. Can be a string, ``Symbol``,
        ``MatrixSymbol``, or ``Indexed`` type. This is helpful in case of
        line-wrapping, or for expressions that generate multi-line statements.
    standard : str, optional
        String specifying the standard. If your compiler supports a more modern
        standard you may set this to \'c99\' to allow the printer to use more math
        functions. [default=\'c89\'].
    precision : integer, optional
        The precision for numbers such as pi [default=17].
    user_functions : dict, optional
        A dictionary where the keys are string representations of either
        ``FunctionClass`` or ``UndefinedFunction`` instances and the values
        are their desired C string representations. Alternatively, the
        dictionary value can be a list of tuples i.e. [(argument_test,
        cfunction_string)] or [(argument_test, cfunction_formater)]. See below
        for examples.
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

    >>> from sympy import ccode, symbols, Rational, sin, ceiling, Abs, Function
    >>> x, tau = symbols("x, tau")
    >>> expr = (2*tau)**Rational(7, 2)
    >>> ccode(expr)
    \'8*M_SQRT2*pow(tau, 7.0/2.0)\'
    >>> ccode(expr, math_macros={})
    \'8*sqrt(2)*pow(tau, 7.0/2.0)\'
    >>> ccode(sin(x), assign_to="s")
    \'s = sin(x);\'
    >>> from sympy.codegen.ast import real, float80
    >>> ccode(expr, type_aliases={real: float80})
    \'8*M_SQRT2l*powl(tau, 7.0L/2.0L)\'

    Simple custom printing can be defined for certain types by passing a
    dictionary of {"type" : "function"} to the ``user_functions`` kwarg.
    Alternatively, the dictionary value can be a list of tuples i.e.
    [(argument_test, cfunction_string)].

    >>> custom_functions = {
    ...   "ceiling": "CEIL",
    ...   "Abs": [(lambda x: not x.is_integer, "fabs"),
    ...           (lambda x: x.is_integer, "ABS")],
    ...   "func": "f"
    ... }
    >>> func = Function(\'func\')
    >>> ccode(func(Abs(x) + ceiling(x)), standard=\'C89\', user_functions=custom_functions)
    \'f(fabs(x) + CEIL(x))\'

    or if the C-function takes a subset of the original arguments:

    >>> ccode(2**x + 3**x, standard=\'C99\', user_functions={\'Pow\': [
    ...   (lambda b, e: b == 2, lambda b, e: \'exp2(%s)\' % e),
    ...   (lambda b, e: b != 2, \'pow\')]})
    \'exp2(x) + pow(3, x)\'

    ``Piecewise`` expressions are converted into conditionals. If an
    ``assign_to`` variable is provided an if statement is created, otherwise
    the ternary operator is used. Note that if the ``Piecewise`` lacks a
    default term, represented by ``(expr, True)`` then an error will be thrown.
    This is to prevent generating an expression that may not evaluate to
    anything.

    >>> from sympy import Piecewise
    >>> expr = Piecewise((x + 1, x > 0), (x, True))
    >>> print(ccode(expr, tau, standard=\'C89\'))
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
    >>> ccode(e.rhs, assign_to=e.lhs, contract=False, standard=\'C89\')
    \'Dy[i] = (y[i + 1] - y[i])/(t[i + 1] - t[i]);\'

    Matrices are also supported, but a ``MatrixSymbol`` of the same dimensions
    must be provided to ``assign_to``. Note that any expression that can be
    generated normally can also exist inside a Matrix:

    >>> from sympy import Matrix, MatrixSymbol
    >>> mat = Matrix([x**2, Piecewise((x + 1, x > 0), (x, True)), sin(x)])
    >>> A = MatrixSymbol(\'A\', 3, 1)
    >>> print(ccode(mat, A, standard=\'C89\'))
    A[0] = pow(x, 2);
    if (x > 0) {
       A[1] = x + 1;
    }
    else {
       A[1] = x;
    }
    A[2] = sin(x);
    '''
def print_ccode(expr, **settings) -> None:
    """Prints C representation of the given expression."""
def fcode(expr, assign_to: Incomplete | None = None, **settings):
    '''Converts an expr to a string of fortran code

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
        DEPRECATED. Use type_mappings instead. The precision for numbers such
        as pi [default=17].
    user_functions : dict, optional
        A dictionary where keys are ``FunctionClass`` instances and values are
        their string representations. Alternatively, the dictionary value can
        be a list of tuples i.e. [(argument_test, cfunction_string)]. See below
        for examples.
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
    source_format : optional
        The source format can be either \'fixed\' or \'free\'. [default=\'fixed\']
    standard : integer, optional
        The Fortran standard to be followed. This is specified as an integer.
        Acceptable standards are 66, 77, 90, 95, 2003, and 2008. Default is 77.
        Note that currently the only distinction internally is between
        standards before 95, and those 95 and after. This may change later as
        more features are added.
    name_mangling : bool, optional
        If True, then the variables that would become identical in
        case-insensitive Fortran are mangled by appending different number
        of ``_`` at the end. If False, SymPy Will not interfere with naming of
        variables. [default=True]

    Examples
    ========

    >>> from sympy import fcode, symbols, Rational, sin, ceiling, floor
    >>> x, tau = symbols("x, tau")
    >>> fcode((2*tau)**Rational(7, 2))
    \'      8*sqrt(2.0d0)*tau**(7.0d0/2.0d0)\'
    >>> fcode(sin(x), assign_to="s")
    \'      s = sin(x)\'

    Custom printing can be defined for certain types by passing a dictionary of
    "type" : "function" to the ``user_functions`` kwarg. Alternatively, the
    dictionary value can be a list of tuples i.e. [(argument_test,
    cfunction_string)].

    >>> custom_functions = {
    ...   "ceiling": "CEIL",
    ...   "floor": [(lambda x: not x.is_integer, "FLOOR1"),
    ...             (lambda x: x.is_integer, "FLOOR2")]
    ... }
    >>> fcode(floor(x) + ceiling(x), user_functions=custom_functions)
    \'      CEIL(x) + FLOOR1(x)\'

    ``Piecewise`` expressions are converted into conditionals. If an
    ``assign_to`` variable is provided an if statement is created, otherwise
    the ternary operator is used. Note that if the ``Piecewise`` lacks a
    default term, represented by ``(expr, True)`` then an error will be thrown.
    This is to prevent generating an expression that may not evaluate to
    anything.

    >>> from sympy import Piecewise
    >>> expr = Piecewise((x + 1, x > 0), (x, True))
    >>> print(fcode(expr, tau))
          if (x > 0) then
             tau = x + 1
          else
             tau = x
          end if

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
    >>> fcode(e.rhs, assign_to=e.lhs, contract=False)
    \'      Dy(i) = (y(i + 1) - y(i))/(t(i + 1) - t(i))\'

    Matrices are also supported, but a ``MatrixSymbol`` of the same dimensions
    must be provided to ``assign_to``. Note that any expression that can be
    generated normally can also exist inside a Matrix:

    >>> from sympy import Matrix, MatrixSymbol
    >>> mat = Matrix([x**2, Piecewise((x + 1, x > 0), (x, True)), sin(x)])
    >>> A = MatrixSymbol(\'A\', 3, 1)
    >>> print(fcode(mat, A))
          A(1, 1) = x**2
             if (x > 0) then
          A(2, 1) = x + 1
             else
          A(2, 1) = x
             end if
          A(3, 1) = sin(x)
    '''
def print_fcode(expr, **settings) -> None:
    """Prints the Fortran representation of the given expression.

       See fcode for the meaning of the optional arguments.
    """
def cxxcode(expr, assign_to: Incomplete | None = None, standard: str = 'c++11', **settings):
    """ C++ equivalent of :func:`~.ccode`. """
