from sympy.combinatorics.permutations import _af_invert as _af_invert
from sympy.core.expr import Expr as Expr
from sympy.core.singleton import S as S
from sympy.matrices.expressions.applyfunc import ElementwiseApplyFunction as ElementwiseApplyFunction
from sympy.matrices.expressions.hadamard import HadamardProduct as HadamardProduct
from sympy.matrices.expressions.inverse import Inverse as Inverse
from sympy.matrices.expressions.matexpr import MatrixExpr as MatrixExpr, MatrixSymbol as MatrixSymbol
from sympy.matrices.expressions.special import Identity as Identity, OneMatrix as OneMatrix
from sympy.matrices.expressions.transpose import Transpose as Transpose
from sympy.tensor.array.expressions.array_expressions import ArrayAdd as ArrayAdd, ArrayContraction as ArrayContraction, ArrayDiagonal as ArrayDiagonal, ArrayElementwiseApplyFunc as ArrayElementwiseApplyFunc, ArraySymbol as ArraySymbol, ArrayTensorProduct as ArrayTensorProduct, PermuteDims as PermuteDims, Reshape as Reshape, ZeroArray as ZeroArray, _ArrayExpr as _ArrayExpr, _array_add as _array_add, _array_contraction as _array_contraction, _array_diagonal as _array_diagonal, _array_tensor_product as _array_tensor_product, _permute_dims as _permute_dims, get_rank as get_rank, get_shape as get_shape
from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array as convert_matrix_to_array

def array_derive(expr, x) -> None:
    """
    Derivatives (gradients) for array expressions.
    """
def _(expr: Expr, x: _ArrayExpr): ...
def matrix_derive(expr, x): ...
