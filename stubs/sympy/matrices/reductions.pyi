from .utilities import _dotprodsimp as _dotprodsimp, _get_intermediate_simp as _get_intermediate_simp, _iszero as _iszero, _simplify as _simplify
from sympy.polys.domains import QQ as QQ, ZZ as ZZ

def _row_reduce_list(mat, rows, cols, one, iszerofunc, simpfunc, normalize_last: bool = True, normalize: bool = True, zero_above: bool = True):
    """Row reduce a flat list representation of a matrix and return a tuple
    (rref_matrix, pivot_cols, swaps) where ``rref_matrix`` is a flat list,
    ``pivot_cols`` are the pivot columns and ``swaps`` are any row swaps that
    were used in the process of row reduction.

    Parameters
    ==========

    mat : list
        list of matrix elements, must be ``rows`` * ``cols`` in length

    rows, cols : integer
        number of rows and columns in flat list representation

    one : SymPy object
        represents the value one, from ``Matrix.one``

    iszerofunc : determines if an entry can be used as a pivot

    simpfunc : used to simplify elements and test if they are
        zero if ``iszerofunc`` returns `None`

    normalize_last : indicates where all row reduction should
        happen in a fraction-free manner and then the rows are
        normalized (so that the pivots are 1), or whether
        rows should be normalized along the way (like the naive
        row reduction algorithm)

    normalize : whether pivot rows should be normalized so that
        the pivot value is 1

    zero_above : whether entries above the pivot should be zeroed.
        If ``zero_above=False``, an echelon matrix will be returned.
    """
def _row_reduce(M, iszerofunc, simpfunc, normalize_last: bool = True, normalize: bool = True, zero_above: bool = True): ...
def _is_echelon(M, iszerofunc=...):
    """Returns `True` if the matrix is in echelon form. That is, all rows of
    zeros are at the bottom, and below each leading non-zero in a row are
    exclusively zeros."""
def _echelon_form(M, iszerofunc=..., simplify: bool = False, with_pivots: bool = False):
    """Returns a matrix row-equivalent to ``M`` that is in echelon form. Note
    that echelon form of a matrix is *not* unique, however, properties like the
    row space and the null space are preserved.

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix([[1, 2], [3, 4]])
    >>> M.echelon_form()
    Matrix([
    [1,  2],
    [0, -2]])
    """
def _rank(M, iszerofunc=..., simplify: bool = False):
    """Returns the rank of a matrix.

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.abc import x
    >>> m = Matrix([[1, 2], [x, 1 - 1/x]])
    >>> m.rank()
    2
    >>> n = Matrix(3, 3, range(1, 10))
    >>> n.rank()
    2
    """
def _to_DM_ZZ_QQ(M): ...
def _rref_dm(dM):
    """Compute the reduced row echelon form of a DomainMatrix."""
def _rref(M, iszerofunc=..., simplify: bool = False, pivots: bool = True, normalize_last: bool = True):
    """Return reduced row-echelon form of matrix and indices
    of pivot vars.

    Parameters
    ==========

    iszerofunc : Function
        A function used for detecting whether an element can
        act as a pivot.  ``lambda x: x.is_zero`` is used by default.

    simplify : Function
        A function used to simplify elements when looking for a pivot.
        By default SymPy's ``simplify`` is used.

    pivots : True or False
        If ``True``, a tuple containing the row-reduced matrix and a tuple
        of pivot columns is returned.  If ``False`` just the row-reduced
        matrix is returned.

    normalize_last : True or False
        If ``True``, no pivots are normalized to `1` until after all
        entries above and below each pivot are zeroed.  This means the row
        reduction algorithm is fraction free until the very last step.
        If ``False``, the naive row reduction procedure is used where
        each pivot is normalized to be `1` before row operations are
        used to zero above and below the pivot.

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.abc import x
    >>> m = Matrix([[1, 2], [x, 1 - 1/x]])
    >>> m.rref()
    (Matrix([
    [1, 0],
    [0, 1]]), (0, 1))
    >>> rref_matrix, rref_pivots = m.rref()
    >>> rref_matrix
    Matrix([
    [1, 0],
    [0, 1]])
    >>> rref_pivots
    (0, 1)

    ``iszerofunc`` can correct rounding errors in matrices with float
    values. In the following example, calling ``rref()`` leads to
    floating point errors, incorrectly row reducing the matrix.
    ``iszerofunc= lambda x: abs(x) < 1e-9`` sets sufficiently small numbers
    to zero, avoiding this error.

    >>> m = Matrix([[0.9, -0.1, -0.2, 0], [-0.8, 0.9, -0.4, 0], [-0.1, -0.8, 0.6, 0]])
    >>> m.rref()
    (Matrix([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0]]), (0, 1, 2))
    >>> m.rref(iszerofunc=lambda x:abs(x)<1e-9)
    (Matrix([
    [1, 0, -0.301369863013699, 0],
    [0, 1, -0.712328767123288, 0],
    [0, 0,         0,          0]]), (0, 1))

    Notes
    =====

    The default value of ``normalize_last=True`` can provide significant
    speedup to row reduction, especially on matrices with symbols.  However,
    if you depend on the form row reduction algorithm leaves entries
    of the matrix, set ``normalize_last=False``
    """
