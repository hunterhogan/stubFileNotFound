from functools import singledispatch
from sympy import Basic as Basic, KroneckerProduct as KroneckerProduct, MatMul as MatMul, Wild as Wild
from sympy.assumptions.ask import Q as Q, ask as ask
from sympy.combinatorics.permutations import Permutation as Permutation, _af_invert as _af_invert
from sympy.core.mul import Mul as Mul
from sympy.core.singleton import S as S
from sympy.matrices.expressions.applyfunc import ElementwiseApplyFunction as ElementwiseApplyFunction
from sympy.matrices.expressions.diagonal import DiagMatrix as DiagMatrix
from sympy.matrices.expressions.hadamard import HadamardPower as HadamardPower, hadamard_product as hadamard_product
from sympy.matrices.expressions.matexpr import MatrixElement as MatrixElement, MatrixExpr as MatrixExpr
from sympy.matrices.expressions.special import Identity as Identity, OneMatrix as OneMatrix, ZeroMatrix as ZeroMatrix
from sympy.matrices.expressions.trace import Trace as Trace
from sympy.matrices.expressions.transpose import Transpose as Transpose
from sympy.matrices.matrixbase import MatrixBase as MatrixBase
from sympy.tensor.array.expressions.array_expressions import ArrayAdd as ArrayAdd, ArrayContraction as ArrayContraction, ArrayDiagonal as ArrayDiagonal, ArrayElement as ArrayElement, ArrayElementwiseApplyFunc as ArrayElementwiseApplyFunc, ArrayTensorProduct as ArrayTensorProduct, OneArray as OneArray, PermuteDims as PermuteDims, ZeroArray as ZeroArray, _ArgE as _ArgE, _ArrayExpr as _ArrayExpr, _CodegenArrayAbstract as _CodegenArrayAbstract, _EditArrayContraction as _EditArrayContraction, _array_add as _array_add, _array_contraction as _array_contraction, _array_diagonal as _array_diagonal, _array_tensor_product as _array_tensor_product, _get_subrank as _get_subrank, _permute_dims as _permute_dims, get_rank as get_rank, get_shape as get_shape
from sympy.tensor.array.expressions.utils import _get_mapping_from_subranks as _get_mapping_from_subranks

def _get_candidate_for_matmul_from_contraction(scan_indices: list[int | None], remaining_args: list[_ArgE]) -> tuple[_ArgE | None, bool, int]: ...
def _insert_candidate_into_editor(editor: _EditArrayContraction, arg_with_ind: _ArgE, candidate: _ArgE, transpose1: bool, transpose2: bool): ...
def _support_function_tp1_recognize(contraction_indices, args): ...
def _find_trivial_matrices_rewrite(expr: ArrayTensorProduct): ...
def _find_trivial_kronecker_products_broadcast(expr: ArrayTensorProduct): ...
@singledispatch
def _array2matrix(expr): ...
def _(expr: ZeroArray): ...
@singledispatch
def _remove_trivial_dims(expr): ...
def _remove_diagonalized_identity_matrices(expr: ArrayDiagonal): ...
def convert_array_to_matrix(expr):
    '''
    Recognize matrix expressions in codegen objects.

    If more than one matrix multiplication line have been detected, return a
    list with the matrix expressions.

    Examples
    ========

    >>> from sympy.tensor.array.expressions.from_indexed_to_array import convert_indexed_to_array
    >>> from sympy.tensor.array import tensorcontraction, tensorproduct
    >>> from sympy import MatrixSymbol, Sum
    >>> from sympy.abc import i, j, k, l, N
    >>> from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array
    >>> from sympy.tensor.array.expressions.from_array_to_matrix import convert_array_to_matrix
    >>> A = MatrixSymbol("A", N, N)
    >>> B = MatrixSymbol("B", N, N)
    >>> C = MatrixSymbol("C", N, N)
    >>> D = MatrixSymbol("D", N, N)

    >>> expr = Sum(A[i, j]*B[j, k], (j, 0, N-1))
    >>> cg = convert_indexed_to_array(expr)
    >>> convert_array_to_matrix(cg)
    A*B
    >>> cg = convert_indexed_to_array(expr, first_indices=[k])
    >>> convert_array_to_matrix(cg)
    B.T*A.T

    Transposition is detected:

    >>> expr = Sum(A[j, i]*B[j, k], (j, 0, N-1))
    >>> cg = convert_indexed_to_array(expr)
    >>> convert_array_to_matrix(cg)
    A.T*B
    >>> cg = convert_indexed_to_array(expr, first_indices=[k])
    >>> convert_array_to_matrix(cg)
    B.T*A

    Detect the trace:

    >>> expr = Sum(A[i, i], (i, 0, N-1))
    >>> cg = convert_indexed_to_array(expr)
    >>> convert_array_to_matrix(cg)
    Trace(A)

    Recognize some more complex traces:

    >>> expr = Sum(A[i, j]*B[j, i], (i, 0, N-1), (j, 0, N-1))
    >>> cg = convert_indexed_to_array(expr)
    >>> convert_array_to_matrix(cg)
    Trace(A*B)

    More complicated expressions:

    >>> expr = Sum(A[i, j]*B[k, j]*A[l, k], (j, 0, N-1), (k, 0, N-1))
    >>> cg = convert_indexed_to_array(expr)
    >>> convert_array_to_matrix(cg)
    A*B.T*A.T

    Expressions constructed from matrix expressions do not contain literal
    indices, the positions of free indices are returned instead:

    >>> expr = A*B
    >>> cg = convert_matrix_to_array(expr)
    >>> convert_array_to_matrix(cg)
    A*B

    If more than one line of matrix multiplications is detected, return
    separate matrix multiplication factors embedded in a tensor product object:

    >>> cg = tensorcontraction(tensorproduct(A, B, C, D), (1, 2), (5, 6))
    >>> convert_array_to_matrix(cg)
    ArrayTensorProduct(A*B, C*D)

    The two lines have free indices at axes 0, 3 and 4, 7, respectively.
    '''
def _array_diag2contr_diagmatrix(expr: ArrayDiagonal): ...
def _a2m_mul(*args): ...
def _a2m_tensor_product(*args): ...
def _a2m_add(*args): ...
def _a2m_trace(arg): ...
def _a2m_transpose(arg): ...
def identify_hadamard_products(expr: ArrayContraction | ArrayDiagonal): ...
def identify_removable_identity_matrices(expr): ...
def remove_identity_matrices(expr: ArrayContraction): ...
def _combine_removed(dim: int, removed1: list[int], removed2: list[int]) -> list[int]: ...
def _array_contraction_to_diagonal_multiple_identity(expr: ArrayContraction): ...
