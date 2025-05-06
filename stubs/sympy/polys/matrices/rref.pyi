from sympy.polys.domains import ZZ as ZZ
from sympy.polys.matrices.ddm import DDM as DDM
from sympy.polys.matrices.dense import ddm_irref as ddm_irref, ddm_irref_den as ddm_irref_den
from sympy.polys.matrices.sdm import SDM as SDM, sdm_irref as sdm_irref, sdm_rref_den as sdm_rref_den

def _dm_rref(M, *, method: str = 'auto'):
    """
    Compute the reduced row echelon form of a ``DomainMatrix``.

    This function is the implementation of :meth:`DomainMatrix.rref`.

    Chooses the best algorithm depending on the domain, shape, and sparsity of
    the matrix as well as things like the bit count in the case of :ref:`ZZ` or
    :ref:`QQ`. The result is returned over the field associated with the domain
    of the Matrix.

    See Also
    ========

    sympy.polys.matrices.domainmatrix.DomainMatrix.rref
        The ``DomainMatrix`` method that calls this function.
    sympy.polys.matrices.rref._dm_rref_den
        Alternative function for computing RREF with denominator.
    """
def _dm_rref_den(M, *, keep_domain: bool = True, method: str = 'auto'):
    """
    Compute the reduced row echelon form of a ``DomainMatrix`` with denominator.

    This function is the implementation of :meth:`DomainMatrix.rref_den`.

    Chooses the best algorithm depending on the domain, shape, and sparsity of
    the matrix as well as things like the bit count in the case of :ref:`ZZ` or
    :ref:`QQ`. The result is returned over the same domain as the input matrix
    unless ``keep_domain=False`` in which case the result might be over an
    associated ring or field domain.

    See Also
    ========

    sympy.polys.matrices.domainmatrix.DomainMatrix.rref_den
        The ``DomainMatrix`` method that calls this function.
    sympy.polys.matrices.rref._dm_rref
        Alternative function for computing RREF without denominator.
    """
def _dm_to_fmt(M, fmt):
    """Convert a matrix to the given format and return the old format."""
def _dm_rref_GJ(M):
    """Compute RREF using Gauss-Jordan elimination with division."""
def _dm_rref_den_FF(M):
    """Compute RREF using fraction-free Gauss-Jordan elimination."""
def _dm_rref_GJ_sparse(M):
    """Compute RREF using sparse Gauss-Jordan elimination with division."""
def _dm_rref_GJ_dense(M):
    """Compute RREF using dense Gauss-Jordan elimination with division."""
def _dm_rref_den_FF_sparse(M):
    """Compute RREF using sparse fraction-free Gauss-Jordan elimination."""
def _dm_rref_den_FF_dense(M):
    """Compute RREF using sparse fraction-free Gauss-Jordan elimination."""
def _dm_rref_choose_method(M, method, *, denominator: bool = False):
    """Choose the fastest method for computing RREF for M."""
def _dm_rref_choose_method_QQ(M, *, denominator: bool = False):
    """Choose the fastest method for computing RREF over QQ."""
def _dm_rref_choose_method_ZZ(M, *, denominator: bool = False):
    """Choose the fastest method for computing RREF over ZZ."""
def _dm_row_density(M):
    '''Density measure for sparse matrices.

    Defines the "density", ``d`` as the average number of non-zero entries per
    row except ignoring rows that are fully zero. RREF can ignore fully zero
    rows so they are excluded. By definition ``d >= 1`` except that we define
    ``d = 0`` for the zero matrix.

    Returns ``(density, nrows_nz, ncols)`` where ``nrows_nz`` counts the number
    of nonzero rows and ``ncols`` is the number of columns.
    '''
def _dm_elements(M):
    """Return nonzero elements of a DomainMatrix."""
def _dm_QQ_numers_denoms(Mq):
    """Returns the numerators and denominators of a DomainMatrix over QQ."""
def _to_field(M):
    """Convert a DomainMatrix to a field if possible."""
