from .matexpr import MatrixExpr as MatrixExpr
from .special import Identity as Identity
from sympy.core import S as S
from sympy.core.cache import cacheit as cacheit
from sympy.core.expr import ExprBuilder as ExprBuilder
from sympy.core.power import Pow as Pow
from sympy.core.sympify import _sympify as _sympify
from sympy.matrices import MatrixBase as MatrixBase
from sympy.matrices.exceptions import NonSquareMatrixError as NonSquareMatrixError

class MatPow(MatrixExpr):
    def __new__(cls, base, exp, evaluate: bool = False, **options): ...
    @property
    def base(self): ...
    @property
    def exp(self): ...
    @property
    def shape(self): ...
    def _get_explicit_matrix(self): ...
    def _entry(self, i, j, **kwargs): ...
    def doit(self, **hints): ...
    def _eval_transpose(self): ...
    def _eval_adjoint(self): ...
    def _eval_conjugate(self): ...
    def _eval_derivative(self, x): ...
    def _eval_derivative_matrix_lines(self, x): ...
    def _eval_inverse(self): ...
