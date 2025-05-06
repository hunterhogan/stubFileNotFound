from sympy.core.basic import Basic as Basic
from sympy.core.symbol import Dummy as Dummy, symbols as symbols
from sympy.matrices.expressions.hadamard import HadamardPower as HadamardPower, HadamardProduct as HadamardProduct
from sympy.tensor.array.expressions.array_expressions import ArrayElementwiseApplyFunc as ArrayElementwiseApplyFunc, Reshape as Reshape, _array_add as _array_add, _array_contraction as _array_contraction, _array_diagonal as _array_diagonal, _array_tensor_product as _array_tensor_product, _permute_dims as _permute_dims

def convert_matrix_to_array(expr: Basic) -> Basic: ...
