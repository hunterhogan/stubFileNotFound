from sympy.core.expr import Expr as Expr
from sympy.matrices.expressions.matexpr import MatrixExpr as MatrixExpr, MatrixSymbol as MatrixSymbol
from sympy.matrices.expressions.special import Identity as Identity, OneMatrix as OneMatrix
from sympy.tensor.array.expressions.array_expressions import ArrayAdd as ArrayAdd, ArrayContraction as ArrayContraction, ArrayDiagonal as ArrayDiagonal, ArrayElementwiseApplyFunc as ArrayElementwiseApplyFunc, ArraySymbol as ArraySymbol, ArrayTensorProduct as ArrayTensorProduct, PermuteDims as PermuteDims, Reshape as Reshape, ZeroArray as ZeroArray, _ArrayExpr as _ArrayExpr, _array_add as _array_add, _array_contraction as _array_contraction, _array_diagonal as _array_diagonal, _array_tensor_product as _array_tensor_product, _permute_dims as _permute_dims, get_rank as get_rank, get_shape as get_shape

def array_derive(expr, x) -> None:
    """
    Derivatives (gradients) for array expressions.
    """
def _(expr: Expr, x: _ArrayExpr): ...
def matrix_derive(expr, x): ...
