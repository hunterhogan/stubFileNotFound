from .determinant import Determinant as Determinant
from .inverse import Inverse as Inverse
from .matadd import MatAdd as MatAdd
from .matmul import MatMul as MatMul
from .matpow import MatPow as MatPow
from .special import Identity as Identity, ZeroMatrix as ZeroMatrix
from .transpose import Transpose as Transpose
from _typeshed import Incomplete
from sympy.core import Add as Add, Basic as Basic, Integer as Integer, Mul as Mul, S as S
from sympy.core.assumptions import check_assumptions as check_assumptions
from sympy.core.decorators import call_highest_priority as call_highest_priority
from sympy.core.expr import Expr as Expr, ExprBuilder as ExprBuilder
from sympy.core.logic import FuzzyBool as FuzzyBool
from sympy.core.symbol import Dummy as Dummy, Str as Str, Symbol as Symbol, symbols as symbols
from sympy.core.sympify import SympifyError as SympifyError, _sympify as _sympify
from sympy.external.gmpy import SYMPY_INTS as SYMPY_INTS
from sympy.functions import adjoint as adjoint, conjugate as conjugate
from sympy.functions.special.tensor_functions import KroneckerDelta as KroneckerDelta
from sympy.matrices.exceptions import NonSquareMatrixError as NonSquareMatrixError
from sympy.matrices.kind import MatrixKind as MatrixKind
from sympy.matrices.matrixbase import MatrixBase as MatrixBase
from sympy.multipledispatch import dispatch as dispatch
from sympy.utilities.misc import filldedent as filldedent

def _sympifyit(arg, retval: Incomplete | None = None): ...

class MatrixExpr(Expr):
    """Superclass for Matrix Expressions

    MatrixExprs represent abstract matrices, linear transformations represented
    within a particular basis.

    Examples
    ========

    >>> from sympy import MatrixSymbol
    >>> A = MatrixSymbol('A', 3, 3)
    >>> y = MatrixSymbol('y', 3, 1)
    >>> x = (A.T*A).I * A * y

    See Also
    ========

    MatrixSymbol, MatAdd, MatMul, Transpose, Inverse
    """
    __slots__: tuple[str, ...]
    _iterable: bool
    _op_priority: float
    is_Matrix: bool
    is_MatrixExpr: bool
    is_Identity: FuzzyBool
    is_Inverse: bool
    is_Transpose: bool
    is_ZeroMatrix: bool
    is_MatAdd: bool
    is_MatMul: bool
    is_commutative: bool
    is_number: bool
    is_symbol: bool
    is_scalar: bool
    kind: MatrixKind
    def __new__(cls, *args, **kwargs): ...
    @property
    def shape(self) -> tuple[Expr, Expr]: ...
    @property
    def _add_handler(self): ...
    @property
    def _mul_handler(self): ...
    def __neg__(self): ...
    def __abs__(self) -> None: ...
    def __add__(self, other): ...
    def __radd__(self, other): ...
    def __sub__(self, other): ...
    def __rsub__(self, other): ...
    def __mul__(self, other): ...
    def __matmul__(self, other): ...
    def __rmul__(self, other): ...
    def __rmatmul__(self, other): ...
    def __pow__(self, other): ...
    def __rpow__(self, other) -> None: ...
    def __truediv__(self, other): ...
    def __rtruediv__(self, other) -> None: ...
    @property
    def rows(self): ...
    @property
    def cols(self): ...
    @property
    def is_square(self) -> bool | None: ...
    def _eval_conjugate(self): ...
    def as_real_imag(self, deep: bool = True, **hints): ...
    def _eval_as_real_imag(self): ...
    def _eval_inverse(self): ...
    def _eval_determinant(self): ...
    def _eval_transpose(self): ...
    def _eval_trace(self) -> None: ...
    def _eval_power(self, exp):
        """
        Override this in sub-classes to implement simplification of powers.  The cases where the exponent
        is -1, 0, 1 are already covered in MatPow.doit(), so implementations can exclude these cases.
        """
    def _eval_simplify(self, **kwargs): ...
    def _eval_adjoint(self): ...
    def _eval_derivative_n_times(self, x, n): ...
    def _eval_derivative(self, x): ...
    @classmethod
    def _check_dim(cls, dim) -> None:
        """Helper function to check invalid matrix dimensions"""
    def _entry(self, i, j, **kwargs) -> None: ...
    def adjoint(self): ...
    def as_coeff_Mul(self, rational: bool = False):
        """Efficiently extract the coefficient of a product."""
    def conjugate(self): ...
    def transpose(self): ...
    @property
    def T(self):
        """Matrix transposition"""
    def inverse(self): ...
    def inv(self): ...
    def det(self): ...
    @property
    def I(self): ...
    def valid_index(self, i, j): ...
    def __getitem__(self, key): ...
    def _is_shape_symbolic(self) -> bool: ...
    def as_explicit(self):
        """
        Returns a dense Matrix with elements represented explicitly

        Returns an object of type ImmutableDenseMatrix.

        Examples
        ========

        >>> from sympy import Identity
        >>> I = Identity(3)
        >>> I
        I
        >>> I.as_explicit()
        Matrix([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]])

        See Also
        ========
        as_mutable: returns mutable Matrix type

        """
    def as_mutable(self):
        """
        Returns a dense, mutable matrix with elements represented explicitly

        Examples
        ========

        >>> from sympy import Identity
        >>> I = Identity(3)
        >>> I
        I
        >>> I.shape
        (3, 3)
        >>> I.as_mutable()
        Matrix([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]])

        See Also
        ========
        as_explicit: returns ImmutableDenseMatrix
        """
    def __array__(self, dtype=..., copy: Incomplete | None = None): ...
    def equals(self, other):
        """
        Test elementwise equality between matrices, potentially of different
        types

        >>> from sympy import Identity, eye
        >>> Identity(3).equals(eye(3))
        True
        """
    def canonicalize(self): ...
    def as_coeff_mmul(self): ...
    @staticmethod
    def from_index_summation(expr, first_index: Incomplete | None = None, last_index: Incomplete | None = None, dimensions: Incomplete | None = None):
        '''
        Parse expression of matrices with explicitly summed indices into a
        matrix expression without indices, if possible.

        This transformation expressed in mathematical notation:

        `\\sum_{j=0}^{N-1} A_{i,j} B_{j,k} \\Longrightarrow \\mathbf{A}\\cdot \\mathbf{B}`

        Optional parameter ``first_index``: specify which free index to use as
        the index starting the expression.

        Examples
        ========

        >>> from sympy import MatrixSymbol, MatrixExpr, Sum
        >>> from sympy.abc import i, j, k, l, N
        >>> A = MatrixSymbol("A", N, N)
        >>> B = MatrixSymbol("B", N, N)
        >>> expr = Sum(A[i, j]*B[j, k], (j, 0, N-1))
        >>> MatrixExpr.from_index_summation(expr)
        A*B

        Transposition is detected:

        >>> expr = Sum(A[j, i]*B[j, k], (j, 0, N-1))
        >>> MatrixExpr.from_index_summation(expr)
        A.T*B

        Detect the trace:

        >>> expr = Sum(A[i, i], (i, 0, N-1))
        >>> MatrixExpr.from_index_summation(expr)
        Trace(A)

        More complicated expressions:

        >>> expr = Sum(A[i, j]*B[k, j]*A[l, k], (j, 0, N-1), (k, 0, N-1))
        >>> MatrixExpr.from_index_summation(expr)
        A*B.T*A.T
        '''
    def applyfunc(self, func): ...

def get_postprocessor(cls): ...
def _matrix_derivative(expr, x, old_algorithm: bool = False): ...
def _matrix_derivative_old_algorithm(expr, x): ...

class MatrixElement(Expr):
    parent: Incomplete
    i: Incomplete
    j: Incomplete
    _diff_wrt: bool
    is_symbol: bool
    is_commutative: bool
    def __new__(cls, name, n, m): ...
    @property
    def symbol(self): ...
    def doit(self, **hints): ...
    @property
    def indices(self): ...
    def _eval_derivative(self, v): ...

class MatrixSymbol(MatrixExpr):
    """Symbolic representation of a Matrix object

    Creates a SymPy Symbol to represent a Matrix. This matrix has a shape and
    can be included in Matrix Expressions

    Examples
    ========

    >>> from sympy import MatrixSymbol, Identity
    >>> A = MatrixSymbol('A', 3, 4) # A 3 by 4 Matrix
    >>> B = MatrixSymbol('B', 4, 3) # A 4 by 3 Matrix
    >>> A.shape
    (3, 4)
    >>> 2*A*B + Identity(3)
    I + 2*A*B
    """
    is_commutative: bool
    is_symbol: bool
    _diff_wrt: bool
    def __new__(cls, name, n, m): ...
    @property
    def shape(self): ...
    @property
    def name(self): ...
    def _entry(self, i, j, **kwargs): ...
    @property
    def free_symbols(self): ...
    def _eval_simplify(self, **kwargs): ...
    def _eval_derivative(self, x): ...
    def _eval_derivative_matrix_lines(self, x): ...

def matrix_symbols(expr): ...

class _LeftRightArgs:
    """
    Helper class to compute matrix derivatives.

    The logic: when an expression is derived by a matrix `X_{mn}`, two lines of
    matrix multiplications are created: the one contracted to `m` (first line),
    and the one contracted to `n` (second line).

    Transposition flips the side by which new matrices are connected to the
    lines.

    The trace connects the end of the two lines.
    """
    _lines: Incomplete
    _first_pointer_parent: Incomplete
    _first_pointer_index: int
    _first_line_index: int
    _second_pointer_parent: Incomplete
    _second_pointer_index: int
    _second_line_index: int
    higher: Incomplete
    def __init__(self, lines, higher=...) -> None: ...
    @property
    def first_pointer(self): ...
    @first_pointer.setter
    def first_pointer(self, value) -> None: ...
    @property
    def second_pointer(self): ...
    @second_pointer.setter
    def second_pointer(self, value) -> None: ...
    def __repr__(self) -> str: ...
    def transpose(self): ...
    @staticmethod
    def _build(expr): ...
    def build(self): ...
    def matrix_form(self): ...
    def rank(self):
        """
        Number of dimensions different from trivial (warning: not related to
        matrix rank).
        """
    def _multiply_pointer(self, pointer, other): ...
    def append_first(self, other) -> None: ...
    def append_second(self, other) -> None: ...

def _make_matrix(x): ...
