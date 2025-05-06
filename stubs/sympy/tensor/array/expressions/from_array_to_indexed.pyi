from sympy import Add as Add, Dummy as Dummy, Mul as Mul, Sum as Sum
from sympy.tensor.array.expressions import ArrayAdd as ArrayAdd, ArrayElementwiseApplyFunc as ArrayElementwiseApplyFunc, PermuteDims as PermuteDims, Reshape as Reshape
from sympy.tensor.array.expressions.array_expressions import ArrayContraction as ArrayContraction, ArrayDiagonal as ArrayDiagonal, ArrayTensorProduct as ArrayTensorProduct, _ArrayExpr as _ArrayExpr, _get_array_element_or_slice as _get_array_element_or_slice, get_rank as get_rank, get_shape as get_shape
from sympy.tensor.array.expressions.utils import _apply_permutation_to_list as _apply_permutation_to_list

def convert_array_to_indexed(expr, indices): ...

class _ConvertArrayToIndexed:
    count_dummies: int
    def __init__(self) -> None: ...
    def do_convert(self, expr, indices): ...
