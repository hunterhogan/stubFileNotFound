import typing
from _typeshed import Incomplete
from sympy.combinatorics import Permutation as Permutation
from sympy.combinatorics.permutations import _af_invert as _af_invert
from sympy.core.basic import Basic as Basic
from sympy.core.containers import Tuple as Tuple
from sympy.core.expr import Expr as Expr
from sympy.core.function import Function as Function, Lambda as Lambda
from sympy.core.mul import Mul as Mul
from sympy.core.numbers import Integer as Integer
from sympy.core.relational import Equality as Equality
from sympy.core.singleton import S as S
from sympy.core.sorting import default_sort_key as default_sort_key
from sympy.core.symbol import Dummy as Dummy, Symbol as Symbol
from sympy.core.sympify import _sympify as _sympify
from sympy.functions.special.tensor_functions import KroneckerDelta as KroneckerDelta
from sympy.matrices.expressions.diagonal import diagonalize_vector as diagonalize_vector
from sympy.matrices.expressions.matexpr import MatrixElement as MatrixElement, MatrixExpr as MatrixExpr
from sympy.matrices.expressions.special import ZeroMatrix as ZeroMatrix
from sympy.matrices.matrixbase import MatrixBase as MatrixBase
from sympy.tensor.array.arrayop import permutedims as permutedims, tensorcontraction as tensorcontraction, tensordiagonal as tensordiagonal, tensorproduct as tensorproduct
from sympy.tensor.array.dense_ndim_array import ImmutableDenseNDimArray as ImmutableDenseNDimArray
from sympy.tensor.array.expressions.utils import _apply_recursively_over_nested_lists as _apply_recursively_over_nested_lists, _build_push_indices_down_func_transformation as _build_push_indices_down_func_transformation, _build_push_indices_up_func_transformation as _build_push_indices_up_func_transformation, _get_contraction_links as _get_contraction_links, _get_mapping_from_subranks as _get_mapping_from_subranks, _sort_contraction_indices as _sort_contraction_indices
from sympy.tensor.array.ndim_array import NDimArray as NDimArray
from sympy.tensor.indexed import Indexed as Indexed, IndexedBase as IndexedBase

class _ArrayExpr(Expr):
    shape: tuple[Expr, ...]
    def __getitem__(self, item): ...
    def _get(self, item): ...

class ArraySymbol(_ArrayExpr):
    """
    Symbol representing an array expression
    """
    def __new__(cls, symbol, shape: typing.Iterable) -> ArraySymbol: ...
    @property
    def name(self): ...
    @property
    def shape(self): ...
    def as_explicit(self): ...

class ArrayElement(Expr):
    """
    An element of an array.
    """
    _diff_wrt: bool
    is_symbol: bool
    is_commutative: bool
    def __new__(cls, name, indices): ...
    @classmethod
    def _check_shape(cls, name, indices) -> None: ...
    @property
    def name(self): ...
    @property
    def indices(self): ...
    def _eval_derivative(self, s): ...

class ZeroArray(_ArrayExpr):
    """
    Symbolic array of zeros. Equivalent to ``ZeroMatrix`` for matrices.
    """
    def __new__(cls, *shape): ...
    @property
    def shape(self): ...
    def as_explicit(self): ...
    def _get(self, item): ...

class OneArray(_ArrayExpr):
    """
    Symbolic array of ones.
    """
    def __new__(cls, *shape): ...
    @property
    def shape(self): ...
    def as_explicit(self): ...
    def _get(self, item): ...

class _CodegenArrayAbstract(Basic):
    @property
    def subranks(self):
        '''
        Returns the ranks of the objects in the uppermost tensor product inside
        the current object.  In case no tensor products are contained, return
        the atomic ranks.

        Examples
        ========

        >>> from sympy.tensor.array import tensorproduct, tensorcontraction
        >>> from sympy import MatrixSymbol
        >>> M = MatrixSymbol("M", 3, 3)
        >>> N = MatrixSymbol("N", 3, 3)
        >>> P = MatrixSymbol("P", 3, 3)

        Important: do not confuse the rank of the matrix with the rank of an array.

        >>> tp = tensorproduct(M, N, P)
        >>> tp.subranks
        [2, 2, 2]

        >>> co = tensorcontraction(tp, (1, 2), (3, 4))
        >>> co.subranks
        [2, 2, 2]
        '''
    def subrank(self):
        """
        The sum of ``subranks``.
        """
    @property
    def shape(self): ...
    def doit(self, **hints): ...

class ArrayTensorProduct(_CodegenArrayAbstract):
    """
    Class to represent the tensor product of array-like objects.
    """
    def __new__(cls, *args, **kwargs): ...
    def _canonicalize(self): ...
    @classmethod
    def _flatten(cls, args): ...
    def as_explicit(self): ...

class ArrayAdd(_CodegenArrayAbstract):
    """
    Class for elementwise array additions.
    """
    def __new__(cls, *args, **kwargs): ...
    def _canonicalize(self): ...
    @classmethod
    def _flatten_args(cls, args): ...
    def as_explicit(self): ...

class PermuteDims(_CodegenArrayAbstract):
    '''
    Class to represent permutation of axes of arrays.

    Examples
    ========

    >>> from sympy.tensor.array import permutedims
    >>> from sympy import MatrixSymbol
    >>> M = MatrixSymbol("M", 3, 3)
    >>> cg = permutedims(M, [1, 0])

    The object ``cg`` represents the transposition of ``M``, as the permutation
    ``[1, 0]`` will act on its indices by switching them:

    `M_{ij} \\Rightarrow M_{ji}`

    This is evident when transforming back to matrix form:

    >>> from sympy.tensor.array.expressions.from_array_to_matrix import convert_array_to_matrix
    >>> convert_array_to_matrix(cg)
    M.T

    >>> N = MatrixSymbol("N", 3, 2)
    >>> cg = permutedims(N, [1, 0])
    >>> cg.shape
    (2, 3)

    There are optional parameters that can be used as alternative to the permutation:

    >>> from sympy.tensor.array.expressions import ArraySymbol, PermuteDims
    >>> M = ArraySymbol("M", (1, 2, 3, 4, 5))
    >>> expr = PermuteDims(M, index_order_old="ijklm", index_order_new="kijml")
    >>> expr
    PermuteDims(M, (0 2 1)(3 4))
    >>> expr.shape
    (3, 1, 2, 5, 4)

    Permutations of tensor products are simplified in order to achieve a
    standard form:

    >>> from sympy.tensor.array import tensorproduct
    >>> M = MatrixSymbol("M", 4, 5)
    >>> tp = tensorproduct(M, N)
    >>> tp.shape
    (4, 5, 3, 2)
    >>> perm1 = permutedims(tp, [2, 3, 1, 0])

    The args ``(M, N)`` have been sorted and the permutation has been
    simplified, the expression is equivalent:

    >>> perm1.expr.args
    (N, M)
    >>> perm1.shape
    (3, 2, 5, 4)
    >>> perm1.permutation
    (2 3)

    The permutation in its array form has been simplified from
    ``[2, 3, 1, 0]`` to ``[0, 1, 3, 2]``, as the arguments of the tensor
    product `M` and `N` have been switched:

    >>> perm1.permutation.array_form
    [0, 1, 3, 2]

    We can nest a second permutation:

    >>> perm2 = permutedims(perm1, [1, 0, 2, 3])
    >>> perm2.shape
    (2, 3, 5, 4)
    >>> perm2.permutation.array_form
    [1, 0, 3, 2]
    '''
    def __new__(cls, expr, permutation: Incomplete | None = None, index_order_old: Incomplete | None = None, index_order_new: Incomplete | None = None, **kwargs): ...
    def _canonicalize(self): ...
    @property
    def expr(self): ...
    @property
    def permutation(self): ...
    @classmethod
    def _PermuteDims_denestarg_ArrayTensorProduct(cls, expr, permutation): ...
    @classmethod
    def _PermuteDims_denestarg_ArrayContraction(cls, expr, permutation): ...
    @classmethod
    def _check_permutation_mapping(cls, expr, permutation): ...
    @classmethod
    def _check_if_there_are_closed_cycles(cls, expr, permutation): ...
    def nest_permutation(self):
        """
        DEPRECATED.
        """
    @classmethod
    def _nest_permutation(cls, expr, permutation): ...
    def as_explicit(self): ...
    @classmethod
    def _get_permutation_from_arguments(cls, permutation, index_order_old, index_order_new, dim): ...
    @classmethod
    def _get_permutation_from_index_orders(cls, index_order_old, index_order_new, dim): ...

class ArrayDiagonal(_CodegenArrayAbstract):
    """
    Class to represent the diagonal operator.

    Explanation
    ===========

    In a 2-dimensional array it returns the diagonal, this looks like the
    operation:

    `A_{ij} \\rightarrow A_{ii}`

    The diagonal over axes 1 and 2 (the second and third) of the tensor product
    of two 2-dimensional arrays `A \\otimes B` is

    `\\Big[ A_{ab} B_{cd} \\Big]_{abcd} \\rightarrow \\Big[ A_{ai} B_{id} \\Big]_{adi}`

    In this last example the array expression has been reduced from
    4-dimensional to 3-dimensional. Notice that no contraction has occurred,
    rather there is a new index `i` for the diagonal, contraction would have
    reduced the array to 2 dimensions.

    Notice that the diagonalized out dimensions are added as new dimensions at
    the end of the indices.
    """
    def __new__(cls, expr, *diagonal_indices, **kwargs): ...
    def _canonicalize(self): ...
    @staticmethod
    def _validate(expr, *diagonal_indices, **kwargs) -> None: ...
    @staticmethod
    def _remove_trivial_dimensions(shape, *diagonal_indices): ...
    @property
    def expr(self): ...
    @property
    def diagonal_indices(self): ...
    @staticmethod
    def _flatten(expr, *outer_diagonal_indices): ...
    @classmethod
    def _ArrayDiagonal_denest_ArrayAdd(cls, expr, *diagonal_indices): ...
    @classmethod
    def _ArrayDiagonal_denest_ArrayDiagonal(cls, expr, *diagonal_indices): ...
    @classmethod
    def _ArrayDiagonal_denest_PermuteDims(cls, expr: PermuteDims, *diagonal_indices): ...
    def _push_indices_down_nonstatic(self, indices): ...
    def _push_indices_up_nonstatic(self, indices): ...
    @classmethod
    def _push_indices_down(cls, diagonal_indices, indices, rank): ...
    @classmethod
    def _push_indices_up(cls, diagonal_indices, indices, rank): ...
    @classmethod
    def _get_positions_shape(cls, shape, diagonal_indices): ...
    def as_explicit(self): ...

class ArrayElementwiseApplyFunc(_CodegenArrayAbstract):
    def __new__(cls, function, element): ...
    @property
    def function(self): ...
    @property
    def expr(self): ...
    @property
    def shape(self): ...
    def _get_function_fdiff(self): ...
    def as_explicit(self): ...

class ArrayContraction(_CodegenArrayAbstract):
    """
    This class is meant to represent contractions of arrays in a form easily
    processable by the code printers.
    """
    def __new__(cls, expr, *contraction_indices, **kwargs): ...
    def _canonicalize(self): ...
    def __mul__(self, other): ...
    def __rmul__(self, other): ...
    @staticmethod
    def _validate(expr, *contraction_indices) -> None: ...
    @classmethod
    def _push_indices_down(cls, contraction_indices, indices): ...
    @classmethod
    def _push_indices_up(cls, contraction_indices, indices): ...
    @classmethod
    def _lower_contraction_to_addends(cls, expr, contraction_indices): ...
    def split_multiple_contractions(self):
        """
        Recognize multiple contractions and attempt at rewriting them as paired-contractions.

        This allows some contractions involving more than two indices to be
        rewritten as multiple contractions involving two indices, thus allowing
        the expression to be rewritten as a matrix multiplication line.

        Examples:

        * `A_ij b_j0 C_jk` ===> `A*DiagMatrix(b)*C`

        Care for:
        - matrix being diagonalized (i.e. `A_ii`)
        - vectors being diagonalized (i.e. `a_i0`)

        Multiple contractions can be split into matrix multiplications if
        not more than two arguments are non-diagonals or non-vectors.
        Vectors get diagonalized while diagonal matrices remain diagonal.
        The non-diagonal matrices can be at the beginning or at the end
        of the final matrix multiplication line.
        """
    def flatten_contraction_of_diagonal(self): ...
    @staticmethod
    def _get_free_indices_to_position_map(free_indices, contraction_indices): ...
    @staticmethod
    def _get_index_shifts(expr):
        '''
        Get the mapping of indices at the positions before the contraction
        occurs.

        Examples
        ========

        >>> from sympy.tensor.array import tensorproduct, tensorcontraction
        >>> from sympy import MatrixSymbol
        >>> M = MatrixSymbol("M", 3, 3)
        >>> N = MatrixSymbol("N", 3, 3)
        >>> cg = tensorcontraction(tensorproduct(M, N), [1, 2])
        >>> cg._get_index_shifts(cg)
        [0, 2]

        Indeed, ``cg`` after the contraction has two dimensions, 0 and 1. They
        need to be shifted by 0 and 2 to get the corresponding positions before
        the contraction (that is, 0 and 3).
        '''
    @staticmethod
    def _convert_outer_indices_to_inner_indices(expr, *outer_contraction_indices): ...
    @staticmethod
    def _flatten(expr, *outer_contraction_indices): ...
    @classmethod
    def _ArrayContraction_denest_ArrayContraction(cls, expr, *contraction_indices): ...
    @classmethod
    def _ArrayContraction_denest_ZeroArray(cls, expr, *contraction_indices): ...
    @classmethod
    def _ArrayContraction_denest_ArrayAdd(cls, expr, *contraction_indices): ...
    @classmethod
    def _ArrayContraction_denest_PermuteDims(cls, expr, *contraction_indices): ...
    @classmethod
    def _ArrayContraction_denest_ArrayDiagonal(cls, expr: ArrayDiagonal, *contraction_indices): ...
    @classmethod
    def _sort_fully_contracted_args(cls, expr, contraction_indices): ...
    def _get_contraction_tuples(self):
        '''
        Return tuples containing the argument index and position within the
        argument of the index position.

        Examples
        ========

        >>> from sympy import MatrixSymbol
        >>> from sympy.abc import N
        >>> from sympy.tensor.array import tensorproduct, tensorcontraction
        >>> A = MatrixSymbol("A", N, N)
        >>> B = MatrixSymbol("B", N, N)

        >>> cg = tensorcontraction(tensorproduct(A, B), (1, 2))
        >>> cg._get_contraction_tuples()
        [[(0, 1), (1, 0)]]

        Notes
        =====

        Here the contraction pair `(1, 2)` meaning that the 2nd and 3rd indices
        of the tensor product `A\\otimes B` are contracted, has been transformed
        into `(0, 1)` and `(1, 0)`, identifying the same indices in a different
        notation. `(0, 1)` is the second index (1) of the first argument (i.e.
                0 or `A`). `(1, 0)` is the first index (i.e. 0) of the second
        argument (i.e. 1 or `B`).
        '''
    @staticmethod
    def _contraction_tuples_to_contraction_indices(expr, contraction_tuples): ...
    @property
    def free_indices(self): ...
    @property
    def free_indices_to_position(self): ...
    @property
    def expr(self): ...
    @property
    def contraction_indices(self): ...
    def _contraction_indices_to_components(self): ...
    def sort_args_by_name(self):
        '''
        Sort arguments in the tensor product so that their order is lexicographical.

        Examples
        ========

        >>> from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array
        >>> from sympy import MatrixSymbol
        >>> from sympy.abc import N
        >>> A = MatrixSymbol("A", N, N)
        >>> B = MatrixSymbol("B", N, N)
        >>> C = MatrixSymbol("C", N, N)
        >>> D = MatrixSymbol("D", N, N)

        >>> cg = convert_matrix_to_array(C*D*A*B)
        >>> cg
        ArrayContraction(ArrayTensorProduct(A, D, C, B), (0, 3), (1, 6), (2, 5))
        >>> cg.sort_args_by_name()
        ArrayContraction(ArrayTensorProduct(A, D, B, C), (0, 3), (1, 4), (2, 7))
        '''
    def _get_contraction_links(self):
        '''
        Returns a dictionary of links between arguments in the tensor product
        being contracted.

        See the example for an explanation of the values.

        Examples
        ========

        >>> from sympy import MatrixSymbol
        >>> from sympy.abc import N
        >>> from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array
        >>> A = MatrixSymbol("A", N, N)
        >>> B = MatrixSymbol("B", N, N)
        >>> C = MatrixSymbol("C", N, N)
        >>> D = MatrixSymbol("D", N, N)

        Matrix multiplications are pairwise contractions between neighboring
        matrices:

        `A_{ij} B_{jk} C_{kl} D_{lm}`

        >>> cg = convert_matrix_to_array(A*B*C*D)
        >>> cg
        ArrayContraction(ArrayTensorProduct(B, C, A, D), (0, 5), (1, 2), (3, 6))

        >>> cg._get_contraction_links()
        {0: {0: (2, 1), 1: (1, 0)}, 1: {0: (0, 1), 1: (3, 0)}, 2: {1: (0, 0)}, 3: {0: (1, 1)}}

        This dictionary is interpreted as follows: argument in position 0 (i.e.
        matrix `A`) has its second index (i.e. 1) contracted to `(1, 0)`, that
        is argument in position 1 (matrix `B`) on the first index slot of `B`,
        this is the contraction provided by the index `j` from `A`.

        The argument in position 1 (that is, matrix `B`) has two contractions,
        the ones provided by the indices `j` and `k`, respectively the first
        and second indices (0 and 1 in the sub-dict).  The link `(0, 1)` and
        `(2, 0)` respectively. `(0, 1)` is the index slot 1 (the 2nd) of
        argument in position 0 (that is, `A_{\\ldot j}`), and so on.
        '''
    def as_explicit(self): ...

class Reshape(_CodegenArrayAbstract):
    '''
    Reshape the dimensions of an array expression.

    Examples
    ========

    >>> from sympy.tensor.array.expressions import ArraySymbol, Reshape
    >>> A = ArraySymbol("A", (6,))
    >>> A.shape
    (6,)
    >>> Reshape(A, (3, 2)).shape
    (3, 2)

    Check the component-explicit forms:

    >>> A.as_explicit()
    [A[0], A[1], A[2], A[3], A[4], A[5]]
    >>> Reshape(A, (3, 2)).as_explicit()
    [[A[0], A[1]], [A[2], A[3]], [A[4], A[5]]]

    '''
    def __new__(cls, expr, shape): ...
    @property
    def shape(self): ...
    @property
    def expr(self): ...
    def doit(self, *args, **kwargs): ...
    def as_explicit(self): ...

class _ArgE:
    """
    The ``_ArgE`` object contains references to the array expression
    (``.element``) and a list containing the information about index
    contractions (``.indices``).

    Index contractions are numbered and contracted indices show the number of
    the contraction. Uncontracted indices have ``None`` value.

    For example:
    ``_ArgE(M, [None, 3])``
    This object means that expression ``M`` is part of an array contraction
    and has two indices, the first is not contracted (value ``None``),
    the second index is contracted to the 4th (i.e. number ``3``) group of the
    array contraction object.
    """
    indices: list[int | None]
    element: Incomplete
    def __init__(self, element, indices: list[int | None] | None = None) -> None: ...
    def __str__(self) -> str: ...
    __repr__ = __str__

class _IndPos:
    """
    Index position, requiring two integers in the constructor:

    - arg: the position of the argument in the tensor product,
    - rel: the relative position of the index inside the argument.
    """
    arg: Incomplete
    rel: Incomplete
    def __init__(self, arg: int, rel: int) -> None: ...
    def __str__(self) -> str: ...
    __repr__ = __str__
    def __iter__(self): ...

class _EditArrayContraction:
    """
    Utility class to help manipulate array contraction objects.

    This class takes as input an ``ArrayContraction`` object and turns it into
    an editable object.

    The field ``args_with_ind`` of this class is a list of ``_ArgE`` objects
    which can be used to easily edit the contraction structure of the
    expression.

    Once editing is finished, the ``ArrayContraction`` object may be recreated
    by calling the ``.to_array_contraction()`` method.
    """
    args_with_ind: list[_ArgE]
    number_of_contraction_indices: int
    _track_permutation: list[list[int]] | None
    def __init__(self, base_array: ArrayContraction | ArrayDiagonal | ArrayTensorProduct) -> None: ...
    def insert_after(self, arg: _ArgE, new_arg: _ArgE): ...
    def get_new_contraction_index(self): ...
    def refresh_indices(self) -> None: ...
    def merge_scalars(self) -> None: ...
    def to_array_contraction(self): ...
    def get_contraction_indices(self) -> list[list[int]]: ...
    def get_mapping_for_index(self, ind) -> list[_IndPos]: ...
    def get_contraction_indices_to_ind_rel_pos(self) -> list[list[_IndPos]]: ...
    def count_args_with_index(self, index: int) -> int:
        """
        Count the number of arguments that have the given index.
        """
    def get_args_with_index(self, index: int) -> list[_ArgE]:
        """
        Get a list of arguments having the given index.
        """
    @property
    def number_of_diagonal_indices(self): ...
    def track_permutation_start(self) -> None: ...
    def track_permutation_merge(self, destination: _ArgE, from_element: _ArgE): ...
    def get_absolute_free_range(self, arg: _ArgE) -> tuple[int, int]:
        """
        Return the range of the free indices of the arg as absolute positions
        among all free indices.
        """
    def get_absolute_range(self, arg: _ArgE) -> tuple[int, int]:
        """
        Return the absolute range of indices for arg, disregarding dummy
        indices.
        """

def get_rank(expr): ...
def _get_subrank(expr): ...
def _get_subranks(expr): ...
def get_shape(expr): ...
def nest_permutation(expr): ...
def _array_tensor_product(*args, **kwargs): ...
def _array_contraction(expr, *contraction_indices, **kwargs): ...
def _array_diagonal(expr, *diagonal_indices, **kwargs): ...
def _permute_dims(expr, permutation, **kwargs): ...
def _array_add(*args, **kwargs): ...
def _get_array_element_or_slice(expr, indices): ...
