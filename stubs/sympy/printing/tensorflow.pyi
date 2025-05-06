from _typeshed import Incomplete
from sympy.printing.pycode import AbstractPythonCodePrinter as AbstractPythonCodePrinter, ArrayPrinter as ArrayPrinter

tensorflow: Incomplete

class TensorflowPrinter(ArrayPrinter, AbstractPythonCodePrinter):
    """
    Tensorflow printer which handles vectorized piecewise functions,
    logical operators, max/min, and relational operators.
    """
    printmethod: str
    mapping: Incomplete
    _default_settings: Incomplete
    tensorflow_version: Incomplete
    def __init__(self, settings: Incomplete | None = None) -> None: ...
    def _print_Function(self, expr): ...
    _print_Expr = _print_Function
    _print_Application = _print_Function
    _print_MatrixExpr = _print_Function
    _print_Relational = _print_Function
    _print_Not = _print_Function
    _print_And = _print_Function
    _print_Or = _print_Function
    _print_HadamardProduct = _print_Function
    _print_Trace = _print_Function
    _print_Determinant = _print_Function
    def _print_Inverse(self, expr): ...
    def _print_Transpose(self, expr): ...
    def _print_Derivative(self, expr): ...
    def _print_Piecewise(self, expr): ...
    def _print_Pow(self, expr): ...
    def _print_MatrixBase(self, expr): ...
    def _print_MatMul(self, expr): ...
    def _print_MatPow(self, expr): ...
    def _print_CodeBlock(self, expr): ...
    _module: str
    _einsum: str
    _add: str
    _transpose: str
    _ones: str
    _zeros: str

def tensorflow_code(expr, **settings): ...
