from sympy.core.expr import Expr as Expr, ExprBuilder as ExprBuilder

class Trace(Expr):
    """Matrix Trace

    Represents the trace of a matrix expression.

    Examples
    ========

    >>> from sympy import MatrixSymbol, Trace, eye
    >>> A = MatrixSymbol('A', 3, 3)
    >>> Trace(A)
    Trace(A)
    >>> Trace(eye(3))
    Trace(Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]]))
    >>> Trace(eye(3)).simplify()
    3
    """
    is_Trace: bool
    is_commutative: bool
    def __new__(cls, mat): ...
    def _eval_transpose(self): ...
    def _eval_derivative(self, v): ...
    def _eval_derivative_matrix_lines(self, x): ...
    @property
    def arg(self): ...
    def doit(self, **hints): ...
    def as_explicit(self): ...
    def _normalize(self): ...
    def _eval_rewrite_as_Sum(self, expr, **kwargs): ...

def trace(expr):
    """Trace of a Matrix.  Sum of the diagonal elements.

    Examples
    ========

    >>> from sympy import trace, Symbol, MatrixSymbol, eye
    >>> n = Symbol('n')
    >>> X = MatrixSymbol('X', n, n)  # A square matrix
    >>> trace(2*X)
    2*Trace(X)
    >>> trace(eye(3))
    3
    """
