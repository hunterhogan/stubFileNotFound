from _typeshed import Incomplete
from sympy.core import AtomicExpr as AtomicExpr, S as S, Symbol as Symbol
from sympy.core.sympify import _sympify as _sympify
from sympy.printing.precedence import PRECEDENCE as PRECEDENCE
from sympy.printing.pretty.stringpict import prettyForm as prettyForm

class BaseScalar(AtomicExpr):
    """
    A coordinate symbol/base scalar.

    Ideally, users should not instantiate this class.

    """
    def __new__(cls, index, system, pretty_str: Incomplete | None = None, latex_str: Incomplete | None = None): ...
    is_commutative: bool
    is_symbol: bool
    @property
    def free_symbols(self): ...
    _diff_wrt: bool
    def _eval_derivative(self, s): ...
    def _latex(self, printer: Incomplete | None = None): ...
    def _pretty(self, printer: Incomplete | None = None): ...
    precedence: Incomplete
    @property
    def system(self): ...
    def _sympystr(self, printer): ...
