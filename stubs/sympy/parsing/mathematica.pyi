from _typeshed import Incomplete
from sympy import Add as Add, And as And, Ci as Ci, Dummy as Dummy, Ei as Ei, Equality as Equality, Function as Function, GreaterThan as GreaterThan, I as I, Integer as Integer, Lambda as Lambda, LessThan as LessThan, Max as Max, Min as Min, Mod as Mod, Mul as Mul, Or as Or, Pow as Pow, Rational as Rational, S as S, Si as Si, StrictGreaterThan as StrictGreaterThan, StrictLessThan as StrictLessThan, Tuple as Tuple, UnevaluatedExpr as UnevaluatedExpr, acos as acos, acosh as acosh, acot as acot, acoth as acoth, acsc as acsc, acsch as acsch, airyai as airyai, airyaiprime as airyaiprime, airybi as airybi, asec as asec, asech as asech, asin as asin, asinh as asinh, atan as atan, atan2 as atan2, atanh as atanh, cancel as cancel, cos as cos, cosh as cosh, cot as cot, coth as coth, csc as csc, csch as csch, exp as exp, expand as expand, expand_trig as expand_trig, flatten as flatten, im as im, isprime as isprime, log as log, pi as pi, polylog as polylog, prime as prime, primepi as primepi, rf as rf, sec as sec, sech as sech, sign as sign, simplify as simplify, sin as sin, sinh as sinh, sqrt as sqrt, symbols as symbols, tan as tan, tanh as tanh
from sympy.core.sympify import _sympify as _sympify, sympify as sympify
from sympy.functions.special.bessel import airybiprime as airybiprime
from sympy.functions.special.error_functions import li as li
from sympy.utilities.exceptions import sympy_deprecation_warning as sympy_deprecation_warning
from typing import Any

from collections.abc import Callable

def mathematica(s, additional_translations: Incomplete | None = None): ...
def parse_mathematica(s):
    '''
    Translate a string containing a Wolfram Mathematica expression to a SymPy
    expression.

    If the translator is unable to find a suitable SymPy expression, the
    ``FullForm`` of the Mathematica expression will be output, using SymPy
    ``Function`` objects as nodes of the syntax tree.

    Examples
    ========

    >>> from sympy.parsing.mathematica import parse_mathematica
    >>> parse_mathematica("Sin[x]^2 Tan[y]")
    sin(x)**2*tan(y)
    >>> e = parse_mathematica("F[7,5,3]")
    >>> e
    F(7, 5, 3)
    >>> from sympy import Function, Max, Min
    >>> e.replace(Function("F"), lambda *x: Max(*x)*Min(*x))
    21

    Both standard input form and Mathematica full form are supported:

    >>> parse_mathematica("x*(a + b)")
    x*(a + b)
    >>> parse_mathematica("Times[x, Plus[a, b]]")
    x*(a + b)

    To get a matrix from Wolfram\'s code:

    >>> m = parse_mathematica("{{a, b}, {c, d}}")
    >>> m
    ((a, b), (c, d))
    >>> from sympy import Matrix
    >>> Matrix(m)
    Matrix([
    [a, b],
    [c, d]])

    If the translation into equivalent SymPy expressions fails, an SymPy
    expression equivalent to Wolfram Mathematica\'s "FullForm" will be created:

    >>> parse_mathematica("x_.")
    Optional(Pattern(x, Blank()))
    >>> parse_mathematica("Plus @@ {x, y, z}")
    Apply(Plus, (x, y, z))
    >>> parse_mathematica("f[x_, 3] := x^3 /; x > 0")
    SetDelayed(f(Pattern(x, Blank()), 3), Condition(x**3, x > 0))
    '''
def _parse_Function(*args): ...
def _deco(cls): ...

class MathematicaParser:
    """
    An instance of this class converts a string of a Wolfram Mathematica
    expression to a SymPy expression.

    The main parser acts internally in three stages:

    1. tokenizer: tokenizes the Mathematica expression and adds the missing *
        operators. Handled by ``_from_mathematica_to_tokens(...)``
    2. full form list: sort the list of strings output by the tokenizer into a
        syntax tree of nested lists and strings, equivalent to Mathematica's
        ``FullForm`` expression output. This is handled by the function
        ``_from_tokens_to_fullformlist(...)``.
    3. SymPy expression: the syntax tree expressed as full form list is visited
        and the nodes with equivalent classes in SymPy are replaced. Unknown
        syntax tree nodes are cast to SymPy ``Function`` objects. This is
        handled by ``_from_fullformlist_to_sympy(...)``.

    """
    CORRESPONDENCES: Incomplete
    fm: Incomplete
    fs: Incomplete
    REPLACEMENTS: Incomplete
    RULES: Incomplete
    FM_PATTERN: Incomplete
    ARG_MTRX_PATTERN: Incomplete
    ARGS_PATTERN_TEMPLATE: str
    TRANSLATIONS: dict[tuple[str, int], dict[str, Any]]
    cache_original: dict[tuple[str, int], dict[str, Any]]
    cache_compiled: dict[tuple[str, int], dict[str, Any]]
    @classmethod
    def _initialize_class(cls) -> None: ...
    translations: Incomplete
    def __init__(self, additional_translations: Incomplete | None = None) -> None: ...
    @classmethod
    def _compile_dictionary(cls, dic): ...
    def _convert_function(self, s):
        """Parse Mathematica function to SymPy one"""
    def _convert_one_function(self, s, fm, args, bgn, end): ...
    @classmethod
    def _get_args(cls, m):
        """Get arguments of a Mathematica function"""
    @classmethod
    def _replace(cls, s, bef): ...
    @classmethod
    def _apply_rules(cls, s, bef): ...
    @classmethod
    def _check_input(cls, s) -> None: ...
    def _parse_old(self, s): ...
    def parse(self, s): ...
    INFIX: str
    PREFIX: str
    POSTFIX: str
    FLAT: str
    RIGHT: str
    LEFT: str
    _mathematica_op_precedence: list[tuple[str, str | None, dict[str, str | Callable]]]
    _missing_arguments_default: Incomplete
    _literal: str
    _number: str
    _enclosure_open: Incomplete
    _enclosure_close: Incomplete
    @classmethod
    def _get_neg(cls, x): ...
    @classmethod
    def _get_inv(cls, x): ...
    _regex_tokenizer: Incomplete
    def _get_tokenizer(self): ...
    def _from_mathematica_to_tokens(self, code: str): ...
    def _is_op(self, token: str | list) -> bool: ...
    def _is_valid_star1(self, token: str | list) -> bool: ...
    def _is_valid_star2(self, token: str | list) -> bool: ...
    def _from_tokens_to_fullformlist(self, tokens: list): ...
    def _util_remove_newlines(self, lines: list, tokens: list, inside_enclosure: bool): ...
    def _util_add_missing_asterisks(self, tokens: list): ...
    def _parse_after_braces(self, tokens: list, inside_enclosure: bool = False): ...
    def _check_op_compatible(self, op1: str, op2: str): ...
    def _from_fullform_to_fullformlist(self, wmexpr: str):
        """
        Parses FullForm[Downvalues[]] generated by Mathematica
        """
    def _from_fullformlist_to_fullformsympy(self, pylist: list): ...
    _node_conversions: Incomplete
    _atom_conversions: Incomplete
    def _from_fullformlist_to_sympy(self, full_form_list): ...
    def _from_fullformsympy_to_sympy(self, mform): ...
