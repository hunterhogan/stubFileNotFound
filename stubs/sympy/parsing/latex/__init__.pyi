from .errors import LaTeXParsingError as LaTeXParsingError
from _typeshed import Incomplete
from sympy.external import import_module as import_module
from sympy.parsing.latex.lark import LarkLaTeXParser as LarkLaTeXParser, TransformToSymPyExpr as TransformToSymPyExpr, parse_latex_lark as parse_latex_lark
from sympy.utilities.decorator import doctest_depends_on as doctest_depends_on

IGNORE_L: str
IGNORE_R: str
NO_LEFT: str
BEGIN_AMS_MAT: str
END_AMS_MAT: str
BEGIN_ARR: str
END_ARR: str
MATRIX_DELIMS: Incomplete
MATRIX_DELIMS_INV: Incomplete
BEGIN_DELIM_REPR: Incomplete
END_DELIM_REPR: Incomplete

def check_matrix_delimiters(latex_str):
    """Report mismatched, excess, or missing matrix delimiters."""

__doctest_requires__: Incomplete

def parse_latex(s, strict: bool = False, backend: str = 'antlr'):
    '''Converts the input LaTeX string ``s`` to a SymPy ``Expr``.

    Parameters
    ==========

    s : str
        The LaTeX string to parse. In Python source containing LaTeX,
        *raw strings* (denoted with ``r"``, like this one) are preferred,
        as LaTeX makes liberal use of the ``\\`` character, which would
        trigger escaping in normal Python strings.
    backend : str, optional
        Currently, there are two backends supported: ANTLR, and Lark.
        The default setting is to use the ANTLR backend, which can be
        changed to Lark if preferred.

        Use ``backend="antlr"`` for the ANTLR-based parser, and
        ``backend="lark"`` for the Lark-based parser.

        The ``backend`` option is case-sensitive, and must be in
        all lowercase.
    strict : bool, optional
        This option is only available with the ANTLR backend.

        If True, raise an exception if the string cannot be parsed as
        valid LaTeX. If False, try to recover gracefully from common
        mistakes.

    Examples
    ========

    >>> from sympy.parsing.latex import parse_latex
    >>> expr = parse_latex(r"\\frac {1 + \\sqrt {\\a}} {\\b}")
    >>> expr
    (sqrt(a) + 1)/b
    >>> expr.evalf(4, subs=dict(a=5, b=2))
    1.618
    >>> func = parse_latex(r"\\int_1^\\alpha \\dfrac{\\mathrm{d}t}{t}", backend="lark")
    >>> func.evalf(subs={"alpha": 2})
    0.693147180559945
    '''
