from .matexpr import MatrixExpr as MatrixExpr
from sympy.core.assumptions import check_assumptions as check_assumptions
from sympy.core.kind import NumberKind as NumberKind
from sympy.core.logic import fuzzy_and as fuzzy_and
from sympy.core.sympify import _sympify as _sympify
from sympy.matrices.kind import MatrixKind as MatrixKind
from sympy.sets.sets import Set as Set, SetKind as SetKind

class MatrixSet(Set):
    """
    MatrixSet represents the set of matrices with ``shape = (n, m)`` over the
    given set.

    Examples
    ========

    >>> from sympy.matrices import MatrixSet
    >>> from sympy import S, I, Matrix
    >>> M = MatrixSet(2, 2, set=S.Reals)
    >>> X = Matrix([[1, 2], [3, 4]])
    >>> X in M
    True
    >>> X = Matrix([[1, 2], [I, 4]])
    >>> X in M
    False

    """
    is_empty: bool
    def __new__(cls, n, m, set): ...
    @property
    def shape(self): ...
    @property
    def set(self): ...
    def _contains(self, other): ...
    @classmethod
    def _check_dim(cls, dim) -> None:
        """Helper function to check invalid matrix dimensions"""
    def _kind(self): ...
