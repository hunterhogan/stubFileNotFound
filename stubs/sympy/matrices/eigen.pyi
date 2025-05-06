from .exceptions import MatrixError as MatrixError, NonSquareMatrixError as NonSquareMatrixError
from .utilities import _iszero as _iszero, _simplify as _simplify
from _typeshed import Incomplete
from sympy.core.evalf import DEFAULT_MAXPREC as DEFAULT_MAXPREC, PrecisionExhausted as PrecisionExhausted
from sympy.core.logic import fuzzy_and as fuzzy_and, fuzzy_or as fuzzy_or
from sympy.polys import CRootOf as CRootOf, EX as EX, QQ as QQ, ZZ as ZZ, roots as roots
from sympy.polys.matrices.eigen import dom_eigenvects as dom_eigenvects, dom_eigenvects_to_sympy as dom_eigenvects_to_sympy

__doctest_requires__: Incomplete

def _eigenvals_eigenvects_mpmath(M): ...
def _eigenvals_mpmath(M, multiple: bool = False):
    """Compute eigenvalues using mpmath"""
def _eigenvects_mpmath(M): ...
def _eigenvals(M, error_when_incomplete: bool = True, *, simplify: bool = False, multiple: bool = False, rational: bool = False, **flags):
    """Compute eigenvalues of the matrix.

    Parameters
    ==========

    error_when_incomplete : bool, optional
        If it is set to ``True``, it will raise an error if not all
        eigenvalues are computed. This is caused by ``roots`` not returning
        a full list of eigenvalues.

    simplify : bool or function, optional
        If it is set to ``True``, it attempts to return the most
        simplified form of expressions returned by applying default
        simplification method in every routine.

        If it is set to ``False``, it will skip simplification in this
        particular routine to save computation resources.

        If a function is passed to, it will attempt to apply
        the particular function as simplification method.

    rational : bool, optional
        If it is set to ``True``, every floating point numbers would be
        replaced with rationals before computation. It can solve some
        issues of ``roots`` routine not working well with floats.

    multiple : bool, optional
        If it is set to ``True``, the result will be in the form of a
        list.

        If it is set to ``False``, the result will be in the form of a
        dictionary.

    Returns
    =======

    eigs : list or dict
        Eigenvalues of a matrix. The return format would be specified by
        the key ``multiple``.

    Raises
    ======

    MatrixError
        If not enough roots had got computed.

    NonSquareMatrixError
        If attempted to compute eigenvalues from a non-square matrix.

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix(3, 3, [0, 1, 1, 1, 0, 0, 1, 1, 1])
    >>> M.eigenvals()
    {-1: 1, 0: 1, 2: 1}

    See Also
    ========

    MatrixBase.charpoly
    eigenvects

    Notes
    =====

    Eigenvalues of a matrix $A$ can be computed by solving a matrix
    equation $\\det(A - \\lambda I) = 0$

    It's not always possible to return radical solutions for
    eigenvalues for matrices larger than $4, 4$ shape due to
    Abel-Ruffini theorem.

    If there is no radical solution is found for the eigenvalue,
    it may return eigenvalues in the form of
    :class:`sympy.polys.rootoftools.ComplexRootOf`.
    """

eigenvals_error_message: Incomplete

def _eigenvals_list(M, error_when_incomplete: bool = True, simplify: bool = False, **flags): ...
def _eigenvals_dict(M, error_when_incomplete: bool = True, simplify: bool = False, **flags): ...
def _eigenspace(M, eigenval, iszerofunc=..., simplify: bool = False):
    """Get a basis for the eigenspace for a particular eigenvalue"""
def _eigenvects_DOM(M, **kwargs): ...
def _eigenvects_sympy(M, iszerofunc, simplify: bool = True, **flags): ...
def _eigenvects(M, error_when_incomplete: bool = True, iszerofunc=..., *, chop: bool = False, **flags):
    """Compute eigenvectors of the matrix.

    Parameters
    ==========

    error_when_incomplete : bool, optional
        Raise an error when not all eigenvalues are computed. This is
        caused by ``roots`` not returning a full list of eigenvalues.

    iszerofunc : function, optional
        Specifies a zero testing function to be used in ``rref``.

        Default value is ``_iszero``, which uses SymPy's naive and fast
        default assumption handler.

        It can also accept any user-specified zero testing function, if it
        is formatted as a function which accepts a single symbolic argument
        and returns ``True`` if it is tested as zero and ``False`` if it
        is tested as non-zero, and ``None`` if it is undecidable.

    simplify : bool or function, optional
        If ``True``, ``as_content_primitive()`` will be used to tidy up
        normalization artifacts.

        It will also be used by the ``nullspace`` routine.

    chop : bool or positive number, optional
        If the matrix contains any Floats, they will be changed to Rationals
        for computation purposes, but the answers will be returned after
        being evaluated with evalf. The ``chop`` flag is passed to ``evalf``.
        When ``chop=True`` a default precision will be used; a number will
        be interpreted as the desired level of precision.

    Returns
    =======

    ret : [(eigenval, multiplicity, eigenspace), ...]
        A ragged list containing tuples of data obtained by ``eigenvals``
        and ``nullspace``.

        ``eigenspace`` is a list containing the ``eigenvector`` for each
        eigenvalue.

        ``eigenvector`` is a vector in the form of a ``Matrix``. e.g.
        a vector of length 3 is returned as ``Matrix([a_1, a_2, a_3])``.

    Raises
    ======

    NotImplementedError
        If failed to compute nullspace.

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix(3, 3, [0, 1, 1, 1, 0, 0, 1, 1, 1])
    >>> M.eigenvects()
    [(-1, 1, [Matrix([
    [-1],
    [ 1],
    [ 0]])]), (0, 1, [Matrix([
    [ 0],
    [-1],
    [ 1]])]), (2, 1, [Matrix([
    [2/3],
    [1/3],
    [  1]])])]

    See Also
    ========

    eigenvals
    MatrixBase.nullspace
    """
def _is_diagonalizable_with_eigen(M, reals_only: bool = False):
    """See _is_diagonalizable. This function returns the bool along with the
    eigenvectors to avoid calculating them again in functions like
    ``diagonalize``."""
def _is_diagonalizable(M, reals_only: bool = False, **kwargs):
    """Returns ``True`` if a matrix is diagonalizable.

    Parameters
    ==========

    reals_only : bool, optional
        If ``True``, it tests whether the matrix can be diagonalized
        to contain only real numbers on the diagonal.


        If ``False``, it tests whether the matrix can be diagonalized
        at all, even with numbers that may not be real.

    Examples
    ========

    Example of a diagonalizable matrix:

    >>> from sympy import Matrix
    >>> M = Matrix([[1, 2, 0], [0, 3, 0], [2, -4, 2]])
    >>> M.is_diagonalizable()
    True

    Example of a non-diagonalizable matrix:

    >>> M = Matrix([[0, 1], [0, 0]])
    >>> M.is_diagonalizable()
    False

    Example of a matrix that is diagonalized in terms of non-real entries:

    >>> M = Matrix([[0, 1], [-1, 0]])
    >>> M.is_diagonalizable(reals_only=False)
    True
    >>> M.is_diagonalizable(reals_only=True)
    False

    See Also
    ========

    sympy.matrices.matrixbase.MatrixBase.is_diagonal
    diagonalize
    """
def _householder_vector(x): ...
def _bidiagonal_decmp_hholder(M): ...
def _eval_bidiag_hholder(M): ...
def _bidiagonal_decomposition(M, upper: bool = True):
    """
    Returns $(U,B,V.H)$ for

    $$A = UBV^{H}$$

    where $A$ is the input matrix, and $B$ is its Bidiagonalized form

    Note: Bidiagonal Computation can hang for symbolic matrices.

    Parameters
    ==========

    upper : bool. Whether to do upper bidiagnalization or lower.
                True for upper and False for lower.

    References
    ==========

    .. [1] Algorithm 5.4.2, Matrix computations by Golub and Van Loan, 4th edition
    .. [2] Complex Matrix Bidiagonalization, https://github.com/vslobody/Householder-Bidiagonalization

    """
def _bidiagonalize(M, upper: bool = True):
    """
    Returns $B$, the Bidiagonalized form of the input matrix.

    Note: Bidiagonal Computation can hang for symbolic matrices.

    Parameters
    ==========

    upper : bool. Whether to do upper bidiagnalization or lower.
                True for upper and False for lower.

    References
    ==========

    .. [1] Algorithm 5.4.2, Matrix computations by Golub and Van Loan, 4th edition
    .. [2] Complex Matrix Bidiagonalization : https://github.com/vslobody/Householder-Bidiagonalization

    """
def _diagonalize(M, reals_only: bool = False, sort: bool = False, normalize: bool = False):
    """
    Return (P, D), where D is diagonal and

        D = P^-1 * M * P

    where M is current matrix.

    Parameters
    ==========

    reals_only : bool. Whether to throw an error if complex numbers are need
                    to diagonalize. (Default: False)

    sort : bool. Sort the eigenvalues along the diagonal. (Default: False)

    normalize : bool. If True, normalize the columns of P. (Default: False)

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix(3, 3, [1, 2, 0, 0, 3, 0, 2, -4, 2])
    >>> M
    Matrix([
    [1,  2, 0],
    [0,  3, 0],
    [2, -4, 2]])
    >>> (P, D) = M.diagonalize()
    >>> D
    Matrix([
    [1, 0, 0],
    [0, 2, 0],
    [0, 0, 3]])
    >>> P
    Matrix([
    [-1, 0, -1],
    [ 0, 0, -1],
    [ 2, 1,  2]])
    >>> P.inv() * M * P
    Matrix([
    [1, 0, 0],
    [0, 2, 0],
    [0, 0, 3]])

    See Also
    ========

    sympy.matrices.matrixbase.MatrixBase.is_diagonal
    is_diagonalizable
    """
def _fuzzy_positive_definite(M): ...
def _fuzzy_positive_semidefinite(M): ...
def _is_positive_definite(M): ...
def _is_positive_semidefinite(M): ...
def _is_negative_definite(M): ...
def _is_negative_semidefinite(M): ...
def _is_indefinite(M): ...
def _is_positive_definite_GE(M):
    """A division-free gaussian elimination method for testing
    positive-definiteness."""
def _is_positive_semidefinite_cholesky(M):
    """Uses Cholesky factorization with complete pivoting

    References
    ==========

    .. [1] http://eprints.ma.man.ac.uk/1199/1/covered/MIMS_ep2008_116.pdf

    .. [2] https://www.value-at-risk.net/cholesky-factorization/
    """

_doc_positive_definite: str

def _jordan_form(M, calc_transform: bool = True, *, chop: bool = False):
    """Return $(P, J)$ where $J$ is a Jordan block
    matrix and $P$ is a matrix such that $M = P J P^{-1}$

    Parameters
    ==========

    calc_transform : bool
        If ``False``, then only $J$ is returned.

    chop : bool
        All matrices are converted to exact types when computing
        eigenvalues and eigenvectors.  As a result, there may be
        approximation errors.  If ``chop==True``, these errors
        will be truncated.

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix([[ 6,  5, -2, -3], [-3, -1,  3,  3], [ 2,  1, -2, -3], [-1,  1,  5,  5]])
    >>> P, J = M.jordan_form()
    >>> J
    Matrix([
    [2, 1, 0, 0],
    [0, 2, 0, 0],
    [0, 0, 2, 1],
    [0, 0, 0, 2]])

    See Also
    ========

    jordan_block
    """
def _left_eigenvects(M, **flags):
    """Returns left eigenvectors and eigenvalues.

    This function returns the list of triples (eigenval, multiplicity,
    basis) for the left eigenvectors. Options are the same as for
    eigenvects(), i.e. the ``**flags`` arguments gets passed directly to
    eigenvects().

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix([[0, 1, 1], [1, 0, 0], [1, 1, 1]])
    >>> M.eigenvects()
    [(-1, 1, [Matrix([
    [-1],
    [ 1],
    [ 0]])]), (0, 1, [Matrix([
    [ 0],
    [-1],
    [ 1]])]), (2, 1, [Matrix([
    [2/3],
    [1/3],
    [  1]])])]
    >>> M.left_eigenvects()
    [(-1, 1, [Matrix([[-2, 1, 1]])]), (0, 1, [Matrix([[-1, -1, 1]])]), (2,
    1, [Matrix([[1, 1, 1]])])]

    """
def _singular_values(M):
    """Compute the singular values of a Matrix

    Examples
    ========

    >>> from sympy import Matrix, Symbol
    >>> x = Symbol('x', real=True)
    >>> M = Matrix([[0, 1, 0], [0, x, 0], [-1, 0, 0]])
    >>> M.singular_values()
    [sqrt(x**2 + 1), 1, 0]

    See Also
    ========

    condition_number
    """
