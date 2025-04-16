from .ddm import DDM as DDM
from .exceptions import DMBadInputError as DMBadInputError, DMDomainError as DMDomainError, DMShapeError as DMShapeError
from _typeshed import Incomplete
from collections.abc import Generator
from sympy.external.gmpy import GROUND_TYPES as GROUND_TYPES
from sympy.polys.domains import QQ as QQ
from sympy.utilities.decorator import doctest_depends_on as doctest_depends_on
from sympy.utilities.iterables import _strongly_connected_components as _strongly_connected_components

__doctest_skip__: Incomplete

class SDM(dict):
    """Sparse matrix based on polys domain elements

    This is a dict subclass and is a wrapper for a dict of dicts that supports
    basic matrix arithmetic +, -, *, **.


    In order to create a new :py:class:`~.SDM`, a dict
    of dicts mapping non-zero elements to their
    corresponding row and column in the matrix is needed.

    We also need to specify the shape and :py:class:`~.Domain`
    of our :py:class:`~.SDM` object.

    We declare a 2x2 :py:class:`~.SDM` matrix belonging
    to QQ domain as shown below.
    The 2x2 Matrix in the example is

    .. math::
           A = \\left[\\begin{array}{ccc}
                0 & \\frac{1}{2} \\\\\n                0 & 0 \\end{array} \\right]


    >>> from sympy.polys.matrices.sdm import SDM
    >>> from sympy import QQ
    >>> elemsdict = {0:{1:QQ(1, 2)}}
    >>> A = SDM(elemsdict, (2, 2), QQ)
    >>> A
    {0: {1: 1/2}}

    We can manipulate :py:class:`~.SDM` the same way
    as a Matrix class

    >>> from sympy import ZZ
    >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)
    >>> B  = SDM({0:{0: ZZ(3)}, 1:{1:ZZ(4)}}, (2, 2), ZZ)
    >>> A + B
    {0: {0: 3, 1: 2}, 1: {0: 1, 1: 4}}

    Multiplication

    >>> A*B
    {0: {1: 8}, 1: {0: 3}}
    >>> A*ZZ(2)
    {0: {1: 4}, 1: {0: 2}}

    """
    fmt: str
    is_DFM: bool
    is_DDM: bool
    shape: Incomplete
    domain: Incomplete
    def __init__(self, elemsdict, shape, domain) -> None: ...
    def getitem(self, i, j): ...
    def setitem(self, i, j, value) -> None: ...
    def extract_slice(self, slice1, slice2): ...
    def extract(self, rows, cols): ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    @classmethod
    def new(cls, sdm, shape, domain):
        """

        Parameters
        ==========

        sdm: A dict of dicts for non-zero elements in SDM
        shape: tuple representing dimension of SDM
        domain: Represents :py:class:`~.Domain` of SDM

        Returns
        =======

        An :py:class:`~.SDM` object

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> elemsdict = {0:{1: QQ(2)}}
        >>> A = SDM.new(elemsdict, (2, 2), QQ)
        >>> A
        {0: {1: 2}}

        """
    def copy(A):
        """
        Returns the copy of a :py:class:`~.SDM` object

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> elemsdict = {0:{1:QQ(2)}, 1:{}}
        >>> A = SDM(elemsdict, (2, 2), QQ)
        >>> B = A.copy()
        >>> B
        {0: {1: 2}, 1: {}}

        """
    @classmethod
    def from_list(cls, ddm, shape, domain):
        """
        Create :py:class:`~.SDM` object from a list of lists.

        Parameters
        ==========

        ddm:
            list of lists containing domain elements
        shape:
            Dimensions of :py:class:`~.SDM` matrix
        domain:
            Represents :py:class:`~.Domain` of :py:class:`~.SDM` object

        Returns
        =======

        :py:class:`~.SDM` containing elements of ddm

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> ddm = [[QQ(1, 2), QQ(0)], [QQ(0), QQ(3, 4)]]
        >>> A = SDM.from_list(ddm, (2, 2), QQ)
        >>> A
        {0: {0: 1/2}, 1: {1: 3/4}}

        See Also
        ========

        to_list
        from_list_flat
        from_dok
        from_ddm
        """
    @classmethod
    def from_ddm(cls, ddm):
        """
        Create :py:class:`~.SDM` from a :py:class:`~.DDM`.

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> ddm = DDM( [[QQ(1, 2), 0], [0, QQ(3, 4)]], (2, 2), QQ)
        >>> A = SDM.from_ddm(ddm)
        >>> A
        {0: {0: 1/2}, 1: {1: 3/4}}
        >>> SDM.from_ddm(ddm).to_ddm() == ddm
        True

        See Also
        ========

        to_ddm
        from_list
        from_list_flat
        from_dok
        """
    def to_list(M):
        """
        Convert a :py:class:`~.SDM` object to a list of lists.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> elemsdict = {0:{1:QQ(2)}, 1:{}}
        >>> A = SDM(elemsdict, (2, 2), QQ)
        >>> A.to_list()
        [[0, 2], [0, 0]]


        """
    def to_list_flat(M):
        """
        Convert :py:class:`~.SDM` to a flat list.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0:{1:QQ(2)}, 1:{0: QQ(3)}}, (2, 2), QQ)
        >>> A.to_list_flat()
        [0, 2, 3, 0]
        >>> A == A.from_list_flat(A.to_list_flat(), A.shape, A.domain)
        True

        See Also
        ========

        from_list_flat
        to_list
        to_dok
        to_ddm
        """
    @classmethod
    def from_list_flat(cls, elements, shape, domain):
        """
        Create :py:class:`~.SDM` from a flat list of elements.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM.from_list_flat([QQ(0), QQ(2), QQ(0), QQ(0)], (2, 2), QQ)
        >>> A
        {0: {1: 2}}
        >>> A == A.from_list_flat(A.to_list_flat(), A.shape, A.domain)
        True

        See Also
        ========

        to_list_flat
        from_list
        from_dok
        from_ddm
        """
    def to_flat_nz(M):
        """
        Convert :class:`SDM` to a flat list of nonzero elements and data.

        Explanation
        ===========

        This is used to operate on a list of the elements of a matrix and then
        reconstruct a modified matrix with elements in the same positions using
        :meth:`from_flat_nz`. Zero elements are omitted from the list.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0:{1:QQ(2)}, 1:{0: QQ(3)}}, (2, 2), QQ)
        >>> elements, data = A.to_flat_nz()
        >>> elements
        [2, 3]
        >>> A == A.from_flat_nz(elements, data, A.domain)
        True

        See Also
        ========

        from_flat_nz
        to_list_flat
        sympy.polys.matrices.ddm.DDM.to_flat_nz
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_flat_nz
        """
    @classmethod
    def from_flat_nz(cls, elements, data, domain):
        """
        Reconstruct a :class:`~.SDM` after calling :meth:`to_flat_nz`.

        See :meth:`to_flat_nz` for explanation.

        See Also
        ========

        to_flat_nz
        from_list_flat
        sympy.polys.matrices.ddm.DDM.from_flat_nz
        sympy.polys.matrices.domainmatrix.DomainMatrix.from_flat_nz
        """
    def to_dod(M):
        """
        Convert to dictionary of dictionaries (dod) format.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0: {1: QQ(2)}, 1: {0: QQ(3)}}, (2, 2), QQ)
        >>> A.to_dod()
        {0: {1: 2}, 1: {0: 3}}

        See Also
        ========

        from_dod
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_dod
        """
    @classmethod
    def from_dod(cls, dod, shape, domain):
        """
        Create :py:class:`~.SDM` from dictionary of dictionaries (dod) format.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> dod = {0: {1: QQ(2)}, 1: {0: QQ(3)}}
        >>> A = SDM.from_dod(dod, (2, 2), QQ)
        >>> A
        {0: {1: 2}, 1: {0: 3}}
        >>> A == SDM.from_dod(A.to_dod(), A.shape, A.domain)
        True

        See Also
        ========

        to_dod
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_dod
        """
    def to_dok(M):
        """
        Convert to dictionary of keys (dok) format.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0: {1: QQ(2)}, 1: {0: QQ(3)}}, (2, 2), QQ)
        >>> A.to_dok()
        {(0, 1): 2, (1, 0): 3}

        See Also
        ========

        from_dok
        to_list
        to_list_flat
        to_ddm
        """
    @classmethod
    def from_dok(cls, dok, shape, domain):
        """
        Create :py:class:`~.SDM` from dictionary of keys (dok) format.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> dok = {(0, 1): QQ(2), (1, 0): QQ(3)}
        >>> A = SDM.from_dok(dok, (2, 2), QQ)
        >>> A
        {0: {1: 2}, 1: {0: 3}}
        >>> A == SDM.from_dok(A.to_dok(), A.shape, A.domain)
        True

        See Also
        ========

        to_dok
        from_list
        from_list_flat
        from_ddm
        """
    def iter_values(M) -> Generator[Incomplete, Incomplete]:
        """
        Iterate over the nonzero values of a :py:class:`~.SDM` matrix.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0: {1: QQ(2)}, 1: {0: QQ(3)}}, (2, 2), QQ)
        >>> list(A.iter_values())
        [2, 3]

        """
    def iter_items(M) -> Generator[Incomplete]:
        """
        Iterate over indices and values of the nonzero elements.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0: {1: QQ(2)}, 1: {0: QQ(3)}}, (2, 2), QQ)
        >>> list(A.iter_items())
        [((0, 1), 2), ((1, 0), 3)]

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.iter_items
        """
    def to_ddm(M):
        """
        Convert a :py:class:`~.SDM` object to a :py:class:`~.DDM` object

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0:{1:QQ(2)}, 1:{}}, (2, 2), QQ)
        >>> A.to_ddm()
        [[0, 2], [0, 0]]

        """
    def to_sdm(M):
        """
        Convert to :py:class:`~.SDM` format (returns self).
        """
    def to_dfm(M):
        """
        Convert a :py:class:`~.SDM` object to a :py:class:`~.DFM` object

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0:{1:QQ(2)}, 1:{}}, (2, 2), QQ)
        >>> A.to_dfm()
        [[0, 2], [0, 0]]

        See Also
        ========

        to_ddm
        to_dfm_or_ddm
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_dfm
        """
    def to_dfm_or_ddm(M):
        """
        Convert to :py:class:`~.DFM` if possible, else :py:class:`~.DDM`.

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0:{1:QQ(2)}, 1:{}}, (2, 2), QQ)
        >>> A.to_dfm_or_ddm()
        [[0, 2], [0, 0]]
        >>> type(A.to_dfm_or_ddm())  # depends on the ground types
        <class 'sympy.polys.matrices._dfm.DFM'>

        See Also
        ========

        to_ddm
        to_dfm
        sympy.polys.matrices.domainmatrix.DomainMatrix.to_dfm_or_ddm
        """
    @classmethod
    def zeros(cls, shape, domain):
        """

        Returns a :py:class:`~.SDM` of size shape,
        belonging to the specified domain

        In the example below we declare a matrix A where,

        .. math::
            A := \\left[\\begin{array}{ccc}
            0 & 0 & 0 \\\\\n            0 & 0 & 0 \\end{array} \\right]

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM.zeros((2, 3), QQ)
        >>> A
        {}

        """
    @classmethod
    def ones(cls, shape, domain): ...
    @classmethod
    def eye(cls, shape, domain):
        """

        Returns a identity :py:class:`~.SDM` matrix of dimensions
        size x size, belonging to the specified domain

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> I = SDM.eye((2, 2), QQ)
        >>> I
        {0: {0: 1}, 1: {1: 1}}

        """
    @classmethod
    def diag(cls, diagonal, domain, shape: Incomplete | None = None): ...
    def transpose(M):
        """

        Returns the transpose of a :py:class:`~.SDM` matrix

        Examples
        ========

        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> A = SDM({0:{1:QQ(2)}, 1:{}}, (2, 2), QQ)
        >>> A.transpose()
        {1: {0: 2}}

        """
    def __add__(A, B): ...
    def __sub__(A, B): ...
    def __neg__(A): ...
    def __mul__(A, B):
        """A * B"""
    def __rmul__(a, b): ...
    def matmul(A, B):
        """
        Performs matrix multiplication of two SDM matrices

        Parameters
        ==========

        A, B: SDM to multiply

        Returns
        =======

        SDM
            SDM after multiplication

        Raises
        ======

        DomainError
            If domain of A does not match
            with that of B

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)
        >>> B = SDM({0:{0:ZZ(2), 1:ZZ(3)}, 1:{0:ZZ(4)}}, (2, 2), ZZ)
        >>> A.matmul(B)
        {0: {0: 8}, 1: {0: 2, 1: 3}}

        """
    def mul(A, b):
        """
        Multiplies each element of A with a scalar b

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)
        >>> A.mul(ZZ(3))
        {0: {1: 6}, 1: {0: 3}}

        """
    def rmul(A, b): ...
    def mul_elementwise(A, B): ...
    def add(A, B):
        """

        Adds two :py:class:`~.SDM` matrices

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)
        >>> B = SDM({0:{0: ZZ(3)}, 1:{1:ZZ(4)}}, (2, 2), ZZ)
        >>> A.add(B)
        {0: {0: 3, 1: 2}, 1: {0: 1, 1: 4}}

        """
    def sub(A, B):
        """

        Subtracts two :py:class:`~.SDM` matrices

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)
        >>> B  = SDM({0:{0: ZZ(3)}, 1:{1:ZZ(4)}}, (2, 2), ZZ)
        >>> A.sub(B)
        {0: {0: -3, 1: 2}, 1: {0: 1, 1: -4}}

        """
    def neg(A):
        """

        Returns the negative of a :py:class:`~.SDM` matrix

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)
        >>> A.neg()
        {0: {1: -2}, 1: {0: -1}}

        """
    def convert_to(A, K):
        """
        Converts the :py:class:`~.Domain` of a :py:class:`~.SDM` matrix to K

        Examples
        ========

        >>> from sympy import ZZ, QQ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)
        >>> A.convert_to(QQ)
        {0: {1: 2}, 1: {0: 1}}

        """
    def nnz(A):
        """Number of non-zero elements in the :py:class:`~.SDM` matrix.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)
        >>> A.nnz()
        2

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.nnz
        """
    def scc(A):
        """Strongly connected components of a square matrix *A*.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{0: ZZ(2)}, 1:{1:ZZ(1)}}, (2, 2), ZZ)
        >>> A.scc()
        [[0], [1]]

        See also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.scc
        """
    def rref(A):
        """

        Returns reduced-row echelon form and list of pivots for the :py:class:`~.SDM`

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(2), 1:QQ(4)}}, (2, 2), QQ)
        >>> A.rref()
        ({0: {0: 1, 1: 2}}, [0])

        """
    def rref_den(A):
        """

        Returns reduced-row echelon form (RREF) with denominator and pivots.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(2), 1:QQ(4)}}, (2, 2), QQ)
        >>> A.rref_den()
        ({0: {0: 1, 1: 2}}, 1, [0])

        """
    def inv(A):
        """

        Returns inverse of a matrix A

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(3), 1:QQ(4)}}, (2, 2), QQ)
        >>> A.inv()
        {0: {0: -2, 1: 1}, 1: {0: 3/2, 1: -1/2}}

        """
    def det(A):
        """
        Returns determinant of A

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(3), 1:QQ(4)}}, (2, 2), QQ)
        >>> A.det()
        -2

        """
    def lu(A):
        """

        Returns LU decomposition for a matrix A

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(3), 1:QQ(4)}}, (2, 2), QQ)
        >>> A.lu()
        ({0: {0: 1}, 1: {0: 3, 1: 1}}, {0: {0: 1, 1: 2}, 1: {1: -2}}, [])

        """
    def lu_solve(A, b):
        """

        Uses LU decomposition to solve Ax = b,

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(3), 1:QQ(4)}}, (2, 2), QQ)
        >>> b = SDM({0:{0:QQ(1)}, 1:{0:QQ(2)}}, (2, 1), QQ)
        >>> A.lu_solve(b)
        {1: {0: 1/2}}

        """
    def nullspace(A):
        """
        Nullspace of a :py:class:`~.SDM` matrix A.

        The domain of the matrix must be a field.

        It is better to use the :meth:`~.DomainMatrix.nullspace` method rather
        than this method which is otherwise no longer used.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0: QQ(2), 1: QQ(4)}}, (2, 2), QQ)
        >>> A.nullspace()
        ({0: {0: -2, 1: 1}}, [1])


        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.nullspace
            The preferred way to get the nullspace of a matrix.

        """
    def nullspace_from_rref(A, pivots: Incomplete | None = None):
        """
        Returns nullspace for a :py:class:`~.SDM` matrix ``A`` in RREF.

        The domain of the matrix can be any domain.

        The matrix must already be in reduced row echelon form (RREF).

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices.sdm import SDM
        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0: QQ(2), 1: QQ(4)}}, (2, 2), QQ)
        >>> A_rref, pivots = A.rref()
        >>> A_null, nonpivots = A_rref.nullspace_from_rref(pivots)
        >>> A_null
        {0: {0: -2, 1: 1}}
        >>> pivots
        [0]
        >>> nonpivots
        [1]

        See Also
        ========

        sympy.polys.matrices.domainmatrix.DomainMatrix.nullspace
            The higher-level function that would usually be called instead of
            calling this one directly.

        sympy.polys.matrices.domainmatrix.DomainMatrix.nullspace_from_rref
            The higher-level direct equivalent of this function.

        sympy.polys.matrices.ddm.DDM.nullspace_from_rref
            The equivalent function for dense :py:class:`~.DDM` matrices.

        """
    def particular(A): ...
    def hstack(A, *B):
        """Horizontally stacks :py:class:`~.SDM` matrices.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import SDM

        >>> A = SDM({0: {0: ZZ(1), 1: ZZ(2)}, 1: {0: ZZ(3), 1: ZZ(4)}}, (2, 2), ZZ)
        >>> B = SDM({0: {0: ZZ(5), 1: ZZ(6)}, 1: {0: ZZ(7), 1: ZZ(8)}}, (2, 2), ZZ)
        >>> A.hstack(B)
        {0: {0: 1, 1: 2, 2: 5, 3: 6}, 1: {0: 3, 1: 4, 2: 7, 3: 8}}

        >>> C = SDM({0: {0: ZZ(9), 1: ZZ(10)}, 1: {0: ZZ(11), 1: ZZ(12)}}, (2, 2), ZZ)
        >>> A.hstack(B, C)
        {0: {0: 1, 1: 2, 2: 5, 3: 6, 4: 9, 5: 10}, 1: {0: 3, 1: 4, 2: 7, 3: 8, 4: 11, 5: 12}}
        """
    def vstack(A, *B):
        """Vertically stacks :py:class:`~.SDM` matrices.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices.sdm import SDM

        >>> A = SDM({0: {0: ZZ(1), 1: ZZ(2)}, 1: {0: ZZ(3), 1: ZZ(4)}}, (2, 2), ZZ)
        >>> B = SDM({0: {0: ZZ(5), 1: ZZ(6)}, 1: {0: ZZ(7), 1: ZZ(8)}}, (2, 2), ZZ)
        >>> A.vstack(B)
        {0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}, 2: {0: 5, 1: 6}, 3: {0: 7, 1: 8}}

        >>> C = SDM({0: {0: ZZ(9), 1: ZZ(10)}, 1: {0: ZZ(11), 1: ZZ(12)}}, (2, 2), ZZ)
        >>> A.vstack(B, C)
        {0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}, 2: {0: 5, 1: 6}, 3: {0: 7, 1: 8}, 4: {0: 9, 1: 10}, 5: {0: 11, 1: 12}}
        """
    def applyfunc(self, func, domain): ...
    def charpoly(A):
        """
        Returns the coefficients of the characteristic polynomial
        of the :py:class:`~.SDM` matrix. These elements will be domain elements.
        The domain of the elements will be same as domain of the :py:class:`~.SDM`.

        Examples
        ========

        >>> from sympy import QQ, Symbol
        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy.polys import Poly
        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(3), 1:QQ(4)}}, (2, 2), QQ)
        >>> A.charpoly()
        [1, -5, -2]

        We can create a polynomial using the
        coefficients using :py:class:`~.Poly`

        >>> x = Symbol('x')
        >>> p = Poly(A.charpoly(), x, domain=A.domain)
        >>> p
        Poly(x**2 - 5*x - 2, x, domain='QQ')

        """
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
        Says whether this matrix is diagonal. True can be returned
        even if the matrix is not square.
        """
    def diagonal(self):
        """
        Returns the diagonal of the matrix as a list.
        """
    def lll(A, delta=...):
        """
        Returns the LLL-reduced basis for the :py:class:`~.SDM` matrix.
        """
    def lll_transform(A, delta=...):
        """
        Returns the LLL-reduced basis and transformation matrix.
        """

def binop_dict(A, B, fab, fa, fb): ...
def unop_dict(A, f): ...
def sdm_transpose(M): ...
def sdm_dotvec(A, B, K): ...
def sdm_matvecmul(A, B, K): ...
def sdm_matmul(A, B, K, m, o): ...
def sdm_matmul_exraw(A, B, K, m, o): ...
def sdm_irref(A):
    """RREF and pivots of a sparse matrix *A*.

    Compute the reduced row echelon form (RREF) of the matrix *A* and return a
    list of the pivot columns. This routine does not work in place and leaves
    the original matrix *A* unmodified.

    The domain of the matrix must be a field.

    Examples
    ========

    This routine works with a dict of dicts sparse representation of a matrix:

    >>> from sympy import QQ
    >>> from sympy.polys.matrices.sdm import sdm_irref
    >>> A = {0: {0: QQ(1), 1: QQ(2)}, 1: {0: QQ(3), 1: QQ(4)}}
    >>> Arref, pivots, _ = sdm_irref(A)
    >>> Arref
    {0: {0: 1}, 1: {1: 1}}
    >>> pivots
    [0, 1]

    The analogous calculation with :py:class:`~.MutableDenseMatrix` would be

    >>> from sympy import Matrix
    >>> M = Matrix([[1, 2], [3, 4]])
    >>> Mrref, pivots = M.rref()
    >>> Mrref
    Matrix([
    [1, 0],
    [0, 1]])
    >>> pivots
    (0, 1)

    Notes
    =====

    The cost of this algorithm is determined purely by the nonzero elements of
    the matrix. No part of the cost of any step in this algorithm depends on
    the number of rows or columns in the matrix. No step depends even on the
    number of nonzero rows apart from the primary loop over those rows. The
    implementation is much faster than ddm_rref for sparse matrices. In fact
    at the time of writing it is also (slightly) faster than the dense
    implementation even if the input is a fully dense matrix so it seems to be
    faster in all cases.

    The elements of the matrix should support exact division with ``/``. For
    example elements of any domain that is a field (e.g. ``QQ``) should be
    fine. No attempt is made to handle inexact arithmetic.

    See Also
    ========

    sympy.polys.matrices.domainmatrix.DomainMatrix.rref
        The higher-level function that would normally be used to call this
        routine.
    sympy.polys.matrices.dense.ddm_irref
        The dense equivalent of this routine.
    sdm_rref_den
        Fraction-free version of this routine.
    """
def sdm_rref_den(A, K):
    """
    Return the reduced row echelon form (RREF) of A with denominator.

    The RREF is computed using fraction-free Gauss-Jordan elimination.

    Explanation
    ===========

    The algorithm used is the fraction-free version of Gauss-Jordan elimination
    described as FFGJ in [1]_. Here it is modified to handle zero or missing
    pivots and to avoid redundant arithmetic. This implementation is also
    optimized for sparse matrices.

    The domain $K$ must support exact division (``K.exquo``) but does not need
    to be a field. This method is suitable for most exact rings and fields like
    :ref:`ZZ`, :ref:`QQ` and :ref:`QQ(a)`. In the case of :ref:`QQ` or
    :ref:`K(x)` it might be more efficient to clear denominators and use
    :ref:`ZZ` or :ref:`K[x]` instead.

    For inexact domains like :ref:`RR` and :ref:`CC` use ``ddm_irref`` instead.

    Examples
    ========

    >>> from sympy.polys.matrices.sdm import sdm_rref_den
    >>> from sympy.polys.domains import ZZ
    >>> A = {0: {0: ZZ(1), 1: ZZ(2)}, 1: {0: ZZ(3), 1: ZZ(4)}}
    >>> A_rref, den, pivots = sdm_rref_den(A, ZZ)
    >>> A_rref
    {0: {0: -2}, 1: {1: -2}}
    >>> den
    -2
    >>> pivots
    [0, 1]

    See Also
    ========

    sympy.polys.matrices.domainmatrix.DomainMatrix.rref_den
        Higher-level interface to ``sdm_rref_den`` that would usually be used
        instead of calling this function directly.
    sympy.polys.matrices.sdm.sdm_rref_den
        The ``SDM`` method that uses this function.
    sdm_irref
        Computes RREF using field division.
    ddm_irref_den
        The dense version of this algorithm.

    References
    ==========

    .. [1] Fraction-free algorithms for linear and polynomial equations.
        George C. Nakos , Peter R. Turner , Robert M. Williams.
        https://dl.acm.org/doi/10.1145/271130.271133
    """
def sdm_nullspace_from_rref(A, one, ncols, pivots, nonzero_cols):
    """Get nullspace from A which is in RREF"""
def sdm_particular_from_rref(A, ncols, pivots):
    """Get a particular solution from A which is in RREF"""
def sdm_berk(M, n, K):
    """
    Berkowitz algorithm for computing the characteristic polynomial.

    Explanation
    ===========

    The Berkowitz algorithm is a division-free algorithm for computing the
    characteristic polynomial of a matrix over any commutative ring using only
    arithmetic in the coefficient ring. This implementation is for sparse
    matrices represented in a dict-of-dicts format (like :class:`SDM`).

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.polys.matrices.sdm import sdm_berk
    >>> from sympy.polys.domains import ZZ
    >>> M = {0: {0: ZZ(1), 1:ZZ(2)}, 1: {0:ZZ(3), 1:ZZ(4)}}
    >>> sdm_berk(M, 2, ZZ)
    {0: 1, 1: -5, 2: -2}
    >>> Matrix([[1, 2], [3, 4]]).charpoly()
    PurePoly(lambda**2 - 5*lambda - 2, lambda, domain='ZZ')

    See Also
    ========

    sympy.polys.matrices.domainmatrix.DomainMatrix.charpoly
        The high-level interface to this function.
    sympy.polys.matrices.dense.ddm_berk
        The dense version of this function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Samuelson%E2%80%93Berkowitz_algorithm
    """
