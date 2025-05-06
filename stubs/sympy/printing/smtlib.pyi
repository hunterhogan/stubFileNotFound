import typing
from _typeshed import Incomplete
from sympy.assumptions.ask import Q as Q
from sympy.assumptions.assume import AppliedPredicate as AppliedPredicate
from sympy.assumptions.relation.binrel import AppliedBinaryRelation as AppliedBinaryRelation
from sympy.assumptions.relation.equality import EqualityPredicate as EqualityPredicate, GreaterThanPredicate as GreaterThanPredicate, LessThanPredicate as LessThanPredicate, StrictGreaterThanPredicate as StrictGreaterThanPredicate, StrictLessThanPredicate as StrictLessThanPredicate
from sympy.core import Add as Add, Basic as Basic, Expr as Expr, Float as Float, Integer as Integer, Mul as Mul, Rational as Rational, Symbol as Symbol
from sympy.core.function import Function as Function, UndefinedFunction as UndefinedFunction
from sympy.core.relational import Equality as Equality, GreaterThan as GreaterThan, LessThan as LessThan, Relational as Relational, StrictGreaterThan as StrictGreaterThan, StrictLessThan as StrictLessThan, Unequality as Unequality
from sympy.functions.elementary.complexes import Abs as Abs
from sympy.functions.elementary.exponential import Pow as Pow, exp as exp, log as log
from sympy.functions.elementary.hyperbolic import cosh as cosh, sinh as sinh, tanh as tanh
from sympy.functions.elementary.miscellaneous import Max as Max, Min as Min
from sympy.functions.elementary.piecewise import Piecewise as Piecewise
from sympy.functions.elementary.trigonometric import acos as acos, asin as asin, atan as atan, atan2 as atan2, cos as cos, sin as sin, tan as tan
from sympy.logic.boolalg import And as And, Boolean as Boolean, BooleanFalse as BooleanFalse, BooleanFunction as BooleanFunction, BooleanTrue as BooleanTrue, ITE as ITE, Implies as Implies, Not as Not, Or as Or, Xor as Xor
from sympy.printing.printer import Printer as Printer
from sympy.sets import Interval as Interval

class SMTLibPrinter(Printer):
    printmethod: str
    _default_settings: dict
    symbol_table: dict
    _precision: Incomplete
    _known_types: Incomplete
    _known_constants: Incomplete
    _known_functions: Incomplete
    def __init__(self, settings: dict | None = None, symbol_table: Incomplete | None = None) -> None: ...
    def _is_legal_name(self, s: str): ...
    def _s_expr(self, op: str, args: list | tuple) -> str: ...
    def _print_Function(self, e): ...
    def _print_Relational(self, e: Relational): ...
    def _print_BooleanFunction(self, e: BooleanFunction): ...
    def _print_Expr(self, e: Expr): ...
    def _print_Unequality(self, e: Unequality): ...
    def _print_Piecewise(self, e: Piecewise): ...
    def _print_Interval(self, e: Interval): ...
    def _print_AppliedPredicate(self, e: AppliedPredicate): ...
    def _print_AppliedBinaryRelation(self, e: AppliedPredicate): ...
    def _print_BooleanTrue(self, x: BooleanTrue): ...
    def _print_BooleanFalse(self, x: BooleanFalse): ...
    def _print_Float(self, x: Float): ...
    def _print_float(self, x: float): ...
    def _print_Rational(self, x: Rational): ...
    def _print_Integer(self, x: Integer): ...
    def _print_int(self, x: int): ...
    def _print_Symbol(self, x: Symbol): ...
    def _print_NumberSymbol(self, x): ...
    def _print_UndefinedFunction(self, x): ...
    def _print_Exp1(self, x): ...
    def emptyPrinter(self, expr) -> None: ...

def smtlib_code(expr, auto_assert: bool = True, auto_declare: bool = True, precision: Incomplete | None = None, symbol_table: Incomplete | None = None, known_types: Incomplete | None = None, known_constants: Incomplete | None = None, known_functions: Incomplete | None = None, prefix_expressions: Incomplete | None = None, suffix_expressions: Incomplete | None = None, log_warn: Incomplete | None = None):
    '''Converts ``expr`` to a string of smtlib code.

    Parameters
    ==========

    expr : Expr | List[Expr]
        A SymPy expression or system to be converted.
    auto_assert : bool, optional
        If false, do not modify expr and produce only the S-Expression equivalent of expr.
        If true, assume expr is a system and assert each boolean element.
    auto_declare : bool, optional
        If false, do not produce declarations for the symbols used in expr.
        If true, prepend all necessary declarations for variables used in expr based on symbol_table.
    precision : integer, optional
        The ``evalf(..)`` precision for numbers such as pi.
    symbol_table : dict, optional
        A dictionary where keys are ``Symbol`` or ``Function`` instances and values are their Python type i.e. ``bool``, ``int``, ``float``, or ``Callable[...]``.
        If incomplete, an attempt will be made to infer types from ``expr``.
    known_types: dict, optional
        A dictionary where keys are ``bool``, ``int``, ``float`` etc. and values are their corresponding SMT type names.
        If not given, a partial listing compatible with several solvers will be used.
    known_functions : dict, optional
        A dictionary where keys are ``Function``, ``Relational``, ``BooleanFunction``, or ``Expr`` instances and values are their SMT string representations.
        If not given, a partial listing optimized for dReal solver (but compatible with others) will be used.
    known_constants: dict, optional
        A dictionary where keys are ``NumberSymbol`` instances and values are their SMT variable names.
        When using this feature, extra caution must be taken to avoid naming collisions between user symbols and listed constants.
        If not given, constants will be expanded inline i.e. ``3.14159`` instead of ``MY_SMT_VARIABLE_FOR_PI``.
    prefix_expressions: list, optional
        A list of lists of ``str`` and/or expressions to convert into SMTLib and prefix to the output.
    suffix_expressions: list, optional
        A list of lists of ``str`` and/or expressions to convert into SMTLib and postfix to the output.
    log_warn: lambda function, optional
        A function to record all warnings during potentially risky operations.
        Soundness is a core value in SMT solving, so it is good to log all assumptions made.

    Examples
    ========
    >>> from sympy import smtlib_code, symbols, sin, Eq
    >>> x = symbols(\'x\')
    >>> smtlib_code(sin(x).series(x).removeO(), log_warn=print)
    Could not infer type of `x`. Defaulting to float.
    Non-Boolean expression `x**5/120 - x**3/6 + x` will not be asserted. Converting to SMTLib verbatim.
    \'(declare-const x Real)\\n(+ x (* (/ -1 6) (pow x 3)) (* (/ 1 120) (pow x 5)))\'

    >>> from sympy import Rational
    >>> x, y, tau = symbols("x, y, tau")
    >>> smtlib_code((2*tau)**Rational(7, 2), log_warn=print)
    Could not infer type of `tau`. Defaulting to float.
    Non-Boolean expression `8*sqrt(2)*tau**(7/2)` will not be asserted. Converting to SMTLib verbatim.
    \'(declare-const tau Real)\\n(* 8 (pow 2 (/ 1 2)) (pow tau (/ 7 2)))\'

    ``Piecewise`` expressions are implemented with ``ite`` expressions by default.
    Note that if the ``Piecewise`` lacks a default term, represented by
    ``(expr, True)`` then an error will be thrown.  This is to prevent
    generating an expression that may not evaluate to anything.

    >>> from sympy import Piecewise
    >>> pw = Piecewise((x + 1, x > 0), (x, True))
    >>> smtlib_code(Eq(pw, 3), symbol_table={x: float}, log_warn=print)
    \'(declare-const x Real)\\n(assert (= (ite (> x 0) (+ 1 x) x) 3))\'

    Custom printing can be defined for certain types by passing a dictionary of
    PythonType : "SMT Name" to the ``known_types``, ``known_constants``, and ``known_functions`` kwargs.

    >>> from typing import Callable
    >>> from sympy import Function, Add
    >>> f = Function(\'f\')
    >>> g = Function(\'g\')
    >>> smt_builtin_funcs = {  # functions our SMT solver will understand
    ...   f: "existing_smtlib_fcn",
    ...   Add: "sum",
    ... }
    >>> user_def_funcs = {  # functions defined by the user must have their types specified explicitly
    ...   g: Callable[[int], float],
    ... }
    >>> smtlib_code(f(x) + g(x), symbol_table=user_def_funcs, known_functions=smt_builtin_funcs, log_warn=print)
    Non-Boolean expression `f(x) + g(x)` will not be asserted. Converting to SMTLib verbatim.
    \'(declare-const x Int)\\n(declare-fun g (Int) Real)\\n(sum (existing_smtlib_fcn x) (g x))\'
    '''
def _auto_declare_smtlib(sym: Symbol | Function, p: SMTLibPrinter, log_warn: typing.Callable[[str], None]): ...
def _auto_assert_smtlib(e: Expr, p: SMTLibPrinter, log_warn: typing.Callable[[str], None]): ...
def _auto_infer_smtlib_types(*exprs: Basic, symbol_table: dict | None = None) -> dict: ...
