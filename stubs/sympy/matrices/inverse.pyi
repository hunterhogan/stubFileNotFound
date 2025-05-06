from .exceptions import MatrixError as MatrixError, NonInvertibleMatrixError as NonInvertibleMatrixError, NonSquareMatrixError as NonSquareMatrixError
from .utilities import _iszero as _iszero
from _typeshed import Incomplete
from sympy.polys.domains import EX as EX
from sympy.polys.matrices.exceptions import DMNonInvertibleMatrixError as DMNonInvertibleMatrixError

def _pinv_full_rank(M):
    """Subroutine for full row or column rank matrices.

    For full row rank matrices, inverse of ``A * A.H`` Exists.
    For full column rank matrices, inverse of ``A.H * A`` Exists.

    This routine can apply for both cases by checking the shape
    and have small decision.
    """
def _pinv_rank_decomposition(M):
    """Subroutine for rank decomposition

    With rank decompositions, `A` can be decomposed into two full-
    rank matrices, and each matrix can take pseudoinverse
    individually.
    """
def _pinv_diagonalization(M):
    """Subroutine using diagonalization

    This routine can sometimes fail if SymPy's eigenvalue
    computation is not reliable.
    """
def _pinv(M, method: str = 'RD'):
    """Calculate the Moore-Penrose pseudoinverse of the matrix.

    The Moore-Penrose pseudoinverse exists and is unique for any matrix.
    If the matrix is invertible, the pseudoinverse is the same as the
    inverse.

    Parameters
    ==========

    method : String, optional
        Specifies the method for computing the pseudoinverse.

        If ``'RD'``, Rank-Decomposition will be used.

        If ``'ED'``, Diagonalization will be used.

    Examples
    ========

    Computing pseudoinverse by rank decomposition :

    >>> from sympy import Matrix
    >>> A = Matrix([[1, 2, 3], [4, 5, 6]])
    >>> A.pinv()
    Matrix([
    [-17/18,  4/9],
    [  -1/9,  1/9],
    [ 13/18, -2/9]])

    Computing pseudoinverse by diagonalization :

    >>> B = A.pinv(method='ED')
    >>> B.simplify()
    >>> B
    Matrix([
    [-17/18,  4/9],
    [  -1/9,  1/9],
    [ 13/18, -2/9]])

    See Also
    ========

    inv
    pinv_solve

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Moore-Penrose_pseudoinverse

    """
def _verify_invertible(M, iszerofunc=...):
    """Initial check to see if a matrix is invertible. Raises or returns
    determinant for use in _inv_ADJ."""
def _inv_ADJ(M, iszerofunc=...):
    """Calculates the inverse using the adjugate matrix and a determinant.

    See Also
    ========

    inv
    inverse_GE
    inverse_LU
    inverse_CH
    inverse_LDL
    """
def _inv_GE(M, iszerofunc=...):
    """Calculates the inverse using Gaussian elimination.

    See Also
    ========

    inv
    inverse_ADJ
    inverse_LU
    inverse_CH
    inverse_LDL
    """
def _inv_LU(M, iszerofunc=...):
    """Calculates the inverse using LU decomposition.

    See Also
    ========

    inv
    inverse_ADJ
    inverse_GE
    inverse_CH
    inverse_LDL
    """
def _inv_CH(M, iszerofunc=...):
    """Calculates the inverse using cholesky decomposition.

    See Also
    ========

    inv
    inverse_ADJ
    inverse_GE
    inverse_LU
    inverse_LDL
    """
def _inv_LDL(M, iszerofunc=...):
    """Calculates the inverse using LDL decomposition.

    See Also
    ========

    inv
    inverse_ADJ
    inverse_GE
    inverse_LU
    inverse_CH
    """
def _inv_QR(M, iszerofunc=...):
    """Calculates the inverse using QR decomposition.

    See Also
    ========

    inv
    inverse_ADJ
    inverse_GE
    inverse_CH
    inverse_LDL
    """
def _try_DM(M, use_EX: bool = False):
    """Try to convert a matrix to a ``DomainMatrix``."""
def _use_exact_domain(dom):
    """Check whether to convert to an exact domain."""
def _inv_DM(dM, cancel: bool = True):
    """Calculates the inverse using ``DomainMatrix``.

    See Also
    ========

    inv
    inverse_ADJ
    inverse_GE
    inverse_CH
    inverse_LDL
    sympy.polys.matrices.domainmatrix.DomainMatrix.inv
    """
def _inv_block(M, iszerofunc=...):
    """Calculates the inverse using BLOCKWISE inversion.

    See Also
    ========

    inv
    inverse_ADJ
    inverse_GE
    inverse_CH
    inverse_LDL
    """
def _inv(M, method: Incomplete | None = None, iszerofunc=..., try_block_diag: bool = False):
    """
    Return the inverse of a matrix using the method indicated. The default
    is DM if a suitable domain is found or otherwise GE for dense matrices
    LDL for sparse matrices.

    Parameters
    ==========

    method : ('DM', 'DMNC', 'GE', 'LU', 'ADJ', 'CH', 'LDL', 'QR')

    iszerofunc : function, optional
        Zero-testing function to use.

    try_block_diag : bool, optional
        If True then will try to form block diagonal matrices using the
        method get_diag_blocks(), invert these individually, and then
        reconstruct the full inverse matrix.

    Examples
    ========

    >>> from sympy import SparseMatrix, Matrix
    >>> A = SparseMatrix([
    ... [ 2, -1,  0],
    ... [-1,  2, -1],
    ... [ 0,  0,  2]])
    >>> A.inv('CH')
    Matrix([
    [2/3, 1/3, 1/6],
    [1/3, 2/3, 1/3],
    [  0,   0, 1/2]])
    >>> A.inv(method='LDL') # use of 'method=' is optional
    Matrix([
    [2/3, 1/3, 1/6],
    [1/3, 2/3, 1/3],
    [  0,   0, 1/2]])
    >>> A * _
    Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])
    >>> A = Matrix(A)
    >>> A.inv('CH')
    Matrix([
    [2/3, 1/3, 1/6],
    [1/3, 2/3, 1/3],
    [  0,   0, 1/2]])
    >>> A.inv('ADJ') == A.inv('GE') == A.inv('LU') == A.inv('CH') == A.inv('LDL') == A.inv('QR')
    True

    Notes
    =====

    According to the ``method`` keyword, it calls the appropriate method:

        DM .... Use DomainMatrix ``inv_den`` method
        DMNC .... Use DomainMatrix ``inv_den`` method without cancellation
        GE .... inverse_GE(); default for dense matrices
        LU .... inverse_LU()
        ADJ ... inverse_ADJ()
        CH ... inverse_CH()
        LDL ... inverse_LDL(); default for sparse matrices
        QR ... inverse_QR()

    Note, the GE and LU methods may require the matrix to be simplified
    before it is inverted in order to properly detect zeros during
    pivoting. In difficult cases a custom zero detection function can
    be provided by setting the ``iszerofunc`` argument to a function that
    should return True if its argument is zero. The ADJ routine computes
    the determinant and uses that to detect singular matrices in addition
    to testing for zeros on the diagonal.

    See Also
    ========

    inverse_ADJ
    inverse_GE
    inverse_LU
    inverse_CH
    inverse_LDL

    Raises
    ======

    ValueError
        If the determinant of the matrix is zero.
    """
