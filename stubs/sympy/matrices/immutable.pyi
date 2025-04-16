from _typeshed import Incomplete
from sympy.core import Basic as Basic, Dict as Dict, Tuple as Tuple
from sympy.core.cache import cacheit as cacheit
from sympy.core.numbers import Integer as Integer
from sympy.core.sympify import _sympify as _sympify
from sympy.matrices.dense import DenseMatrix as DenseMatrix
from sympy.matrices.expressions import MatrixExpr as MatrixExpr
from sympy.matrices.matrixbase import MatrixBase as MatrixBase
from sympy.matrices.repmatrix import RepMatrix as RepMatrix
from sympy.matrices.sparse import SparseRepMatrix as SparseRepMatrix
from sympy.multipledispatch import dispatch as dispatch

def sympify_matrix(arg): ...
def sympify_mpmath_matrix(arg): ...

class ImmutableRepMatrix(RepMatrix, MatrixExpr):
    """Immutable matrix based on RepMatrix

    Uses DomainMAtrix as the internal representation.
    """
    def __new__(cls, *args, **kwargs): ...
    __hash__: Incomplete
    def copy(self): ...
    @property
    def cols(self): ...
    @property
    def rows(self): ...
    @property
    def shape(self): ...
    def as_immutable(self): ...
    def _entry(self, i, j, **kwargs): ...
    def __setitem__(self, *args) -> None: ...
    def is_diagonalizable(self, reals_only: bool = False, **kwargs): ...
    is_diagonalizable: Incomplete
    def analytic_func(self, f, x): ...

class ImmutableDenseMatrix(DenseMatrix, ImmutableRepMatrix):
    """Create an immutable version of a matrix.

    Examples
    ========

    >>> from sympy import eye, ImmutableMatrix
    >>> ImmutableMatrix(eye(3))
    Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])
    >>> _[0, 0] = 42
    Traceback (most recent call last):
    ...
    TypeError: Cannot set values of ImmutableDenseMatrix
    """
    _iterable: bool
    _class_priority: int
    _op_priority: float
    @classmethod
    def _new(cls, *args, **kwargs): ...
    @classmethod
    def _fromrep(cls, rep): ...
ImmutableMatrix = ImmutableDenseMatrix

class ImmutableSparseMatrix(SparseRepMatrix, ImmutableRepMatrix):
    """Create an immutable version of a sparse matrix.

    Examples
    ========

    >>> from sympy import eye, ImmutableSparseMatrix
    >>> ImmutableSparseMatrix(1, 1, {})
    Matrix([[0]])
    >>> ImmutableSparseMatrix(eye(3))
    Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])
    >>> _[0, 0] = 42
    Traceback (most recent call last):
    ...
    TypeError: Cannot set values of ImmutableSparseMatrix
    >>> _.shape
    (3, 3)
    """
    is_Matrix: bool
    _class_priority: int
    @classmethod
    def _new(cls, *args, **kwargs): ...
    @classmethod
    def _fromrep(cls, rep): ...

def _eval_is_eq(lhs, rhs):
    """Helper method for Equality with matrices.sympy.

    Relational automatically converts matrices to ImmutableDenseMatrix
    instances, so this method only applies here.  Returns True if the
    matrices are definitively the same, False if they are definitively
    different, and None if undetermined (e.g. if they contain Symbols).
    Returning None triggers default handling of Equalities.

    """
