from .dense import ddm_berk as ddm_berk, ddm_iadd as ddm_iadd, ddm_idet as ddm_idet, ddm_iinv as ddm_iinv, ddm_ilu_solve as ddm_ilu_solve, ddm_ilu_split as ddm_ilu_split, ddm_imatmul as ddm_imatmul, ddm_imul as ddm_imul, ddm_ineg as ddm_ineg, ddm_irmul as ddm_irmul, ddm_irref as ddm_irref, ddm_irref_den as ddm_irref_den, ddm_isub as ddm_isub, ddm_transpose as ddm_transpose
from .exceptions import DMBadInputError as DMBadInputError, DMDomainError as DMDomainError, DMNonSquareMatrixError as DMNonSquareMatrixError, DMShapeError as DMShapeError
from .lll import ddm_lll as ddm_lll, ddm_lll_transform as ddm_lll_transform
from _typeshed import Incomplete
from collections.abc import Generator

__doctest_skip__: Incomplete

class DDM(list):
    """Dense matrix based on polys domain elements

    This is a list subclass and is a wrapper for a list of lists that supports
    basic matrix arithmetic +, -, *, **.
    """
    fmt: str
    is_DFM: bool
    is_DDM: bool
    shape: Incomplete
    rows: Incomplete
    cols: Incomplete
    domain: Incomplete
    def __init__(self, rowslist, shape, domain) -> None: ...
    def getitem(self, i, j): ...
    def setitem(self, i, j, value) -> None: ...
    def extract_slice(self, slice1, slice2): ...
    def extract(self, rows, cols): ...
    @classmethod
    def from_list(cls, rowslist, shape, domain):
        """
        Create a :class:`DDM` from a list of lists.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.ddm import DDM
        >>> A = DDM.from_list([[ZZ(0), ZZ(1)], [ZZ(-1), ZZ(0)]], (2, 2), ZZ)
        >>> A
        [[0, 1], [-1, 0]]
        >>> A == DDM([[ZZ(0), ZZ(1)], [ZZ(-1), ZZ(0)]], (2, 2), ZZ)
        True

        See Also
        ========

        from_list_flat
        """
    @classmethod
    def from_ddm(cls, other): ...
    def to_list(self):
        """
        Convert to a list of lists.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.ddm import DDM
        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)
        >>> A.to_list()
        [[1, 2], [3, 4]]

        See Also
        ========

        to_list_flat
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_list
        """
    def to_list_flat(self):
        """
        Convert to a flat list of elements.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.ddm import DDM
        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)
        >>> A.to_list_flat()
        [1, 2, 3, 4]
        >>> A == DDM.from_list_flat(A.to_list_flat(), A.shape, A.domain)
        True

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.to_list_flat
        """
    @classmethod
    def from_list_flat(cls, flat, shape, domain):
        """
        Create a :class:`DDM` from a flat list of elements.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.ddm import DDM
        >>> A = DDM.from_list_flat([1, 2, 3, 4], (2, 2), QQ)
        >>> A
        [[1, 2], [3, 4]]
        >>> A == DDM.from_list_flat(A.to_list_flat(), A.shape, A.domain)
        True

        See Also
        ========

        to_list_flat
        sympy.polys.matrices.domainmatrix.DomainMatrix.from_list_flat
        """
    def flatiter(self): ...
    def flat(self): ...
    def to_flat_nz(self):
        """
        Convert to a flat list of nonzero elements and data.

        Explanation
        ===========

        This is used to operate on a list of the elements of a matrix and then
        reconstruct a matrix using :meth:`from_flat_nz`. Zero elements are
        included in the list but that may change in the future.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)
        >>> elements, data = A.to_flat_nz()
        >>> elements
        [1, 2, 3, 4]
        >>> A == DDM.from_flat_nz(elements, data, A.domain)
        True

        See Also
        ========

        from_flat_nz
        sympy.polys.matrices.sdm.SDM.to_flat_nz
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_flat_nz
        """
    @classmethod
    def from_flat_nz(cls, elements, data, domain):
        """
        Reconstruct a :class:`DDM` after calling :meth:`to_flat_nz`.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)
        >>> elements, data = A.to_flat_nz()
        >>> elements
        [1, 2, 3, 4]
        >>> A == DDM.from_flat_nz(elements, data, A.domain)
        True

        See Also
        ========

        to_flat_nz
        sympy.polys.matrices.sdm.SDM.from_flat_nz
        sympy.polys.matrices.domainmatrix.DomainMatrix.from_flat_nz
        """
    def to_dod(self):
        """
        Convert to a dictionary of dictionaries (dod) format.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)
        >>> A.to_dod()
        {0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}

        See Also
        ========

        from_dod
        sympy.polys.matrices.sdm.SDM.to_dod
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_dod
        """
    @classmethod
    def from_dod(cls, dod, shape, domain):
        """
        Create a :class:`DDM` from a dictionary of dictionaries (dod) format.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> dod = {0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}
        >>> A = DDM.from_dod(dod, (2, 2), QQ)
        >>> A
        [[1, 2], [3, 4]]

        See Also
        ========

        to_dod
        sympy.polys.matrices.sdm.SDM.from_dod
        sympy.polys.matrices.domainmatrix.DomainMatrix.from_dod
        """
    def to_dok(self):
        """
        Convert :class:`DDM` to dictionary of keys (dok) format.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)
        >>> A.to_dok()
        {(0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4}

        See Also
        ========

        from_dok
        sympy.polys.matrices.sdm.SDM.to_dok
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_dok
        """
    @classmethod
    def from_dok(cls, dok, shape, domain):
        """
        Create a :class:`DDM` from a dictionary of keys (dok) format.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> dok = {(0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4}
        >>> A = DDM.from_dok(dok, (2, 2), QQ)
        >>> A
        [[1, 2], [3, 4]]

        See Also
        ========

        to_dok
        sympy.polys.matrices.sdm.SDM.from_dok
        sympy.polys.matrices.domainmatrix.DomainMatrix.from_dok
        """
    def iter_values(self) -> Generator[Incomplete, Incomplete]:
        """
        Iterater over the non-zero values of the matrix.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> A = DDM([[QQ(1), QQ(0)], [QQ(3), QQ(4)]], (2, 2), QQ)
        >>> list(A.iter_values())
        [1, 3, 4]

        See Also
        ========

        iter_items
        to_list_flat
        sympy.polys.matrices.domainmatrix.DomainMatrix.iter_values
        """
    def iter_items(self) -> Generator[Incomplete]:
        """
        Iterate over indices and values of nonzero elements of the matrix.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> A = DDM([[QQ(1), QQ(0)], [QQ(3), QQ(4)]], (2, 2), QQ)
        >>> list(A.iter_items())
        [((0, 0), 1), ((1, 0), 3), ((1, 1), 4)]

        See Also
        ========

        iter_values
        to_dok
        sympy.polys.matrices.domainmatrix.DomainMatrix.iter_items
        """
    def to_ddm(self):
        """
        Convert to a :class:`DDM`.

        This just returns ``self`` but exists to parallel the corresponding
        method in other matrix types like :class:`~.SDM`.

        See Also
        ========

        to_sdm
        to_dfm
        to_dfm_or_ddm
        sympy.polys.matrices.sdm.SDM.to_ddm
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_ddm
        """
    def to_sdm(self):
        """
        Convert to a :class:`~.SDM`.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)
        >>> A.to_sdm()
        {0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}
        >>> type(A.to_sdm())
        <class 'sympy.polys.matrices.sdm.SDM'>

        See Also
        ========

        SDM
        sympy.polys.matrices.sdm.SDM.to_ddm
        """
    def to_dfm(self):
        """
        Convert to :class:`~.DDM` to :class:`~.DFM`.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)
        >>> A.to_dfm()
        [[1, 2], [3, 4]]
        >>> type(A.to_dfm())
        <class 'sympy.polys.matrices._dfm.DFM'>

        See Also
        ========

        DFM
        sympy.polys.matrices._dfm.DFM.to_ddm
        """
    def to_dfm_or_ddm(self):
        """
        Convert to :class:`~.DFM` if possible or otherwise return self.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy import QQ
        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)
        >>> A.to_dfm_or_ddm()
        [[1, 2], [3, 4]]
        >>> type(A.to_dfm_or_ddm())
        <class 'sympy.polys.matrices._dfm.DFM'>

        See Also
        ========

        to_dfm
        to_ddm
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_dfm_or_ddm
        """
    def convert_to(self, K): ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other): ...
    def __ne__(self, other): ...
    @classmethod
    def zeros(cls, shape, domain): ...
    @classmethod
    def ones(cls, shape, domain): ...
    @classmethod
    def eye(cls, size, domain): ...
    def copy(self): ...
    def transpose(self): ...
    def __add__(a, b): ...
    def __sub__(a, b): ...
    def __neg__(a): ...
    def __mul__(a, b): ...
    def __rmul__(a, b): ...
    def __matmul__(a, b): ...
    @classmethod
    def _check(cls, a, op, b, ashape, bshape) -> None: ...
    def add(a, b):
        """a + b"""
    def sub(a, b):
        """a - b"""
    def neg(a):
        """-a"""
    def mul(a, b): ...
    def rmul(a, b): ...
    def matmul(a, b):
        """a @ b (matrix product)"""
    def mul_elementwise(a, b): ...
    def hstack(A, *B):
        """Horizontally stacks :py:class:`~.DDM` matrices.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import DDM

        >>> A = DDM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> B = DDM([[ZZ(5), ZZ(6)], [ZZ(7), ZZ(8)]], (2, 2), ZZ)
        >>> A.hstack(B)
        [[1, 2, 5, 6], [3, 4, 7, 8]]

        >>> C = DDM([[ZZ(9), ZZ(10)], [ZZ(11), ZZ(12)]], (2, 2), ZZ)
        >>> A.hstack(B, C)
        [[1, 2, 5, 6, 9, 10], [3, 4, 7, 8, 11, 12]]
        """
    def vstack(A, *B):
        """Vertically stacks :py:class:`~.DDM` matrices.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import DDM

        >>> A = DDM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> B = DDM([[ZZ(5), ZZ(6)], [ZZ(7), ZZ(8)]], (2, 2), ZZ)
        >>> A.vstack(B)
        [[1, 2], [3, 4], [5, 6], [7, 8]]

        >>> C = DDM([[ZZ(9), ZZ(10)], [ZZ(11), ZZ(12)]], (2, 2), ZZ)
        >>> A.vstack(B, C)
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]
        """
    def applyfunc(self, func, domain): ...
    def nnz(a):
        """Number of non-zero entries in :py:class:`~.DDM` matrix.

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.nnz
        """
    def scc(a):
        """Strongly connected components of a square matrix *a*.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import DDM
        >>> A = DDM([[ZZ(1), ZZ(0)], [ZZ(0), ZZ(1)]], (2, 2), ZZ)
        >>> A.scc()
        [[0], [1]]

        See also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.scc

        """
    @classmethod
    def diag(cls, values, domain):
        """Returns a square diagonal matrix with *values* on the diagonal.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import DDM
        >>> DDM.diag([ZZ(1), ZZ(2), ZZ(3)], ZZ)
        [[1, 0, 0], [0, 2, 0], [0, 0, 3]]

        See also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.diag
        """
    def rref(a):
        """Reduced-row echelon form of a and list of pivots.

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.rref
            Higher level interface to this function.
        sympy.polys.matrices.dense.ddm_irref
            The underlying algorithm.
        """
    def rref_den(a):
        """Reduced-row echelon form of a with denominator and list of pivots

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.rref_den
            Higher level interface to this function.
        sympy.polys.matrices.dense.ddm_irref_den
            The underlying algorithm.
        """
    def nullspace(a):
        """Returns a basis for the nullspace of a.

        The domain of the matrix must be a field.

        See Also
        ========

        rref
        sympy.polys.matrices.domainmatrix.DomainMatrix.nullspace
        """
    def nullspace_from_rref(a, pivots: Incomplete | None = None):
        """Compute the nullspace of a matrix from its rref.

        The domain of the matrix can be any domain.

        Returns a tuple (basis, nonpivots).

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.nullspace
            The higher level interface to this function.
        """
    def particular(a): ...
    def det(a):
        """Determinant of a"""
    def inv(a):
        """Inverse of a"""
    def lu(a):
        """L, U decomposition of a"""
    def lu_solve(a, b):
        """x where a*x = b"""
    def charpoly(a):
        """Coefficients of characteristic polynomial of a"""
    def is_zero_matrix(self):
        """
        Says whether this matrix has all zero entries.
        """
    def is_upper(self):
        """
        Says whether this matrix is upper-triangular. True can be returned
        even if the matrix is not square.
        """
    def is_lower(self):
        """
        Says whether this matrix is lower-triangular. True can be returned
        even if the matrix is not square.
        """
    def is_diagonal(self):
        """
        Says whether this matrix is diagonal. True can be returned even if
        the matrix is not square.
        """
    def diagonal(self):
        """
        Returns a list of the elements from the diagonal of the matrix.
        """
    def lll(A, delta=...): ...
    def lll_transform(A, delta=...): ...
