from .common import MatrixCommon as MatrixCommon
from .determinant import _adjugate as _adjugate, _charpoly as _charpoly, _cofactor as _cofactor, _cofactor_matrix as _cofactor_matrix, _det as _det, _det_LU as _det_LU, _det_bareiss as _det_bareiss, _det_berkowitz as _det_berkowitz, _det_bird as _det_bird, _det_laplace as _det_laplace, _find_reasonable_pivot as _find_reasonable_pivot, _find_reasonable_pivot_naive as _find_reasonable_pivot_naive, _minor as _minor, _minor_submatrix as _minor_submatrix, _per as _per
from .eigen import _bidiagonal_decomposition as _bidiagonal_decomposition, _bidiagonalize as _bidiagonalize, _diagonalize as _diagonalize, _eigenvals as _eigenvals, _eigenvects as _eigenvects, _is_diagonalizable as _is_diagonalizable, _is_indefinite as _is_indefinite, _is_negative_definite as _is_negative_definite, _is_negative_semidefinite as _is_negative_semidefinite, _is_positive_definite as _is_positive_definite, _is_positive_semidefinite as _is_positive_semidefinite, _jordan_form as _jordan_form, _left_eigenvects as _left_eigenvects, _singular_values as _singular_values
from .reductions import _echelon_form as _echelon_form, _is_echelon as _is_echelon, _rank as _rank, _rref as _rref
from .subspaces import _columnspace as _columnspace, _nullspace as _nullspace, _orthogonalize as _orthogonalize, _rowspace as _rowspace
from .utilities import _is_zero_after_expand_mul as _is_zero_after_expand_mul, _iszero as _iszero, _simplify as _simplify
from _typeshed import Incomplete

__doctest_requires__: Incomplete

class MatrixDeterminant(MatrixCommon):
    """Provides basic matrix determinant operations. Should not be instantiated
    directly. See ``determinant.py`` for their implementations."""
    def _eval_det_bareiss(self, iszerofunc=...): ...
    def _eval_det_berkowitz(self): ...
    def _eval_det_lu(self, iszerofunc=..., simpfunc: Incomplete | None = None): ...
    def _eval_det_bird(self): ...
    def _eval_det_laplace(self): ...
    def _eval_determinant(self): ...
    def adjugate(self, method: str = 'berkowitz'): ...
    def charpoly(self, x: str = 'lambda', simplify=...): ...
    def cofactor(self, i, j, method: str = 'berkowitz'): ...
    def cofactor_matrix(self, method: str = 'berkowitz'): ...
    def det(self, method: str = 'bareiss', iszerofunc: Incomplete | None = None): ...
    def per(self): ...
    def minor(self, i, j, method: str = 'berkowitz'): ...
    def minor_submatrix(self, i, j): ...

class MatrixReductions(MatrixDeterminant):
    """Provides basic matrix row/column operations. Should not be instantiated
    directly. See ``reductions.py`` for some of their implementations."""
    def echelon_form(self, iszerofunc=..., simplify: bool = False, with_pivots: bool = False): ...
    @property
    def is_echelon(self): ...
    def rank(self, iszerofunc=..., simplify: bool = False): ...
    def rref_rhs(self, rhs):
        """Return reduced row-echelon form of matrix, matrix showing
        rhs after reduction steps. ``rhs`` must have the same number
        of rows as ``self``.

        Examples
        ========

        >>> from sympy import Matrix, symbols
        >>> r1, r2 = symbols('r1 r2')
        >>> Matrix([[1, 1], [2, 1]]).rref_rhs(Matrix([r1, r2]))
        (Matrix([
        [1, 0],
        [0, 1]]), Matrix([
        [ -r1 + r2],
        [2*r1 - r2]]))
        """
    def rref(self, iszerofunc=..., simplify: bool = False, pivots: bool = True, normalize_last: bool = True): ...
    def _normalize_op_args(self, op, col, k, col1, col2, error_str: str = 'col'):
        '''Validate the arguments for a row/column operation.  ``error_str``
        can be one of "row" or "col" depending on the arguments being parsed.'''
    def _eval_col_op_multiply_col_by_const(self, col, k): ...
    def _eval_col_op_swap(self, col1, col2): ...
    def _eval_col_op_add_multiple_to_other_col(self, col, k, col2): ...
    def _eval_row_op_swap(self, row1, row2): ...
    def _eval_row_op_multiply_row_by_const(self, row, k): ...
    def _eval_row_op_add_multiple_to_other_row(self, row, k, row2): ...
    def elementary_col_op(self, op: str = 'n->kn', col: Incomplete | None = None, k: Incomplete | None = None, col1: Incomplete | None = None, col2: Incomplete | None = None):
        '''Performs the elementary column operation `op`.

        `op` may be one of

            * ``"n->kn"`` (column n goes to k*n)
            * ``"n<->m"`` (swap column n and column m)
            * ``"n->n+km"`` (column n goes to column n + k*column m)

        Parameters
        ==========

        op : string; the elementary row operation
        col : the column to apply the column operation
        k : the multiple to apply in the column operation
        col1 : one column of a column swap
        col2 : second column of a column swap or column "m" in the column operation
               "n->n+km"
        '''
    def elementary_row_op(self, op: str = 'n->kn', row: Incomplete | None = None, k: Incomplete | None = None, row1: Incomplete | None = None, row2: Incomplete | None = None):
        '''Performs the elementary row operation `op`.

        `op` may be one of

            * ``"n->kn"`` (row n goes to k*n)
            * ``"n<->m"`` (swap row n and row m)
            * ``"n->n+km"`` (row n goes to row n + k*row m)

        Parameters
        ==========

        op : string; the elementary row operation
        row : the row to apply the row operation
        k : the multiple to apply in the row operation
        row1 : one row of a row swap
        row2 : second row of a row swap or row "m" in the row operation
               "n->n+km"
        '''

class MatrixSubspaces(MatrixReductions):
    """Provides methods relating to the fundamental subspaces of a matrix.
    Should not be instantiated directly. See ``subspaces.py`` for their
    implementations."""
    def columnspace(self, simplify: bool = False): ...
    def nullspace(self, simplify: bool = False, iszerofunc=...): ...
    def rowspace(self, simplify: bool = False): ...
    def orthogonalize(cls, *vecs, **kwargs): ...
    orthogonalize: Incomplete

class MatrixEigen(MatrixSubspaces):
    """Provides basic matrix eigenvalue/vector operations.
    Should not be instantiated directly. See ``eigen.py`` for their
    implementations."""
    def eigenvals(self, error_when_incomplete: bool = True, **flags): ...
    def eigenvects(self, error_when_incomplete: bool = True, iszerofunc=..., **flags): ...
    def is_diagonalizable(self, reals_only: bool = False, **kwargs): ...
    def diagonalize(self, reals_only: bool = False, sort: bool = False, normalize: bool = False): ...
    def bidiagonalize(self, upper: bool = True): ...
    def bidiagonal_decomposition(self, upper: bool = True): ...
    @property
    def is_positive_definite(self): ...
    @property
    def is_positive_semidefinite(self): ...
    @property
    def is_negative_definite(self): ...
    @property
    def is_negative_semidefinite(self): ...
    @property
    def is_indefinite(self): ...
    def jordan_form(self, calc_transform: bool = True, **kwargs): ...
    def left_eigenvects(self, **flags): ...
    def singular_values(self): ...

class MatrixCalculus(MatrixCommon):
    """Provides calculus-related matrix operations."""
    def diff(self, *args, evaluate: bool = True, **kwargs):
        """Calculate the derivative of each element in the matrix.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.abc import x, y
        >>> M = Matrix([[x, y], [1, 0]])
        >>> M.diff(x)
        Matrix([
        [1, 0],
        [0, 0]])

        See Also
        ========

        integrate
        limit
        """
    def _eval_derivative(self, arg): ...
    def integrate(self, *args, **kwargs):
        """Integrate each element of the matrix.  ``args`` will
        be passed to the ``integrate`` function.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.abc import x, y
        >>> M = Matrix([[x, y], [1, 0]])
        >>> M.integrate((x, ))
        Matrix([
        [x**2/2, x*y],
        [     x,   0]])
        >>> M.integrate((x, 0, 2))
        Matrix([
        [2, 2*y],
        [2,   0]])

        See Also
        ========

        limit
        diff
        """
    def jacobian(self, X):
        """Calculates the Jacobian matrix (derivative of a vector-valued function).

        Parameters
        ==========

        ``self`` : vector of expressions representing functions f_i(x_1, ..., x_n).
        X : set of x_i's in order, it can be a list or a Matrix

        Both ``self`` and X can be a row or a column matrix in any order
        (i.e., jacobian() should always work).

        Examples
        ========

        >>> from sympy import sin, cos, Matrix
        >>> from sympy.abc import rho, phi
        >>> X = Matrix([rho*cos(phi), rho*sin(phi), rho**2])
        >>> Y = Matrix([rho, phi])
        >>> X.jacobian(Y)
        Matrix([
        [cos(phi), -rho*sin(phi)],
        [sin(phi),  rho*cos(phi)],
        [   2*rho,             0]])
        >>> X = Matrix([rho*cos(phi), rho*sin(phi)])
        >>> X.jacobian(Y)
        Matrix([
        [cos(phi), -rho*sin(phi)],
        [sin(phi),  rho*cos(phi)]])

        See Also
        ========

        hessian
        wronskian
        """
    def limit(self, *args):
        """Calculate the limit of each element in the matrix.
        ``args`` will be passed to the ``limit`` function.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.abc import x, y
        >>> M = Matrix([[x, y], [1, 0]])
        >>> M.limit(x, 2)
        Matrix([
        [2, y],
        [1, 0]])

        See Also
        ========

        integrate
        diff
        """

class MatrixDeprecated(MatrixCommon):
    """A class to house deprecated matrix methods."""
    def berkowitz_charpoly(self, x=..., simplify=...): ...
    def berkowitz_det(self):
        """Computes determinant using Berkowitz method.

        See Also
        ========

        det
        berkowitz
        """
    def berkowitz_eigenvals(self, **flags):
        """Computes eigenvalues of a Matrix using Berkowitz method.

        See Also
        ========

        berkowitz
        """
    def berkowitz_minors(self):
        """Computes principal minors using Berkowitz method.

        See Also
        ========

        berkowitz
        """
    def berkowitz(self): ...
    def cofactorMatrix(self, method: str = 'berkowitz'): ...
    def det_bareis(self): ...
    def det_LU_decomposition(self):
        """Compute matrix determinant using LU decomposition.


        Note that this method fails if the LU decomposition itself
        fails. In particular, if the matrix has no inverse this method
        will fail.

        TODO: Implement algorithm for sparse matrices (SFF),
        https://www.eecis.udel.edu/~saunders/papers/sffge/it5.ps

        See Also
        ========


        det
        det_bareiss
        berkowitz_det
        """
    def jordan_cell(self, eigenval, n): ...
    def jordan_cells(self, calc_transformation: bool = True): ...
    def minorEntry(self, i, j, method: str = 'berkowitz'): ...
    def minorMatrix(self, i, j): ...
    def permuteBkwd(self, perm):
        """Permute the rows of the matrix with the given permutation in reverse."""
    def permuteFwd(self, perm):
        """Permute the rows of the matrix with the given permutation."""
