from .codeprinter import CodePrinter as CodePrinter
from .pycode import ArrayPrinter as ArrayPrinter, PythonCodePrinter as PythonCodePrinter, _known_functions_math as _known_functions_math, _print_known_const as _print_known_const, _print_known_func as _print_known_func, _unpack_integral_limits as _unpack_integral_limits
from _typeshed import Incomplete
from sympy.core import S as S
from sympy.core.function import Lambda as Lambda
from sympy.core.power import Pow as Pow

_not_in_numpy: Incomplete
_in_numpy: Incomplete
_known_functions_numpy: Incomplete
_known_constants_numpy: Incomplete
_numpy_known_functions: Incomplete
_numpy_known_constants: Incomplete

class NumPyPrinter(ArrayPrinter, PythonCodePrinter):
    """
    Numpy printer which handles vectorized piecewise functions,
    logical operators, etc.
    """
    _module: str
    _kf = _numpy_known_functions
    _kc = _numpy_known_constants
    language: Incomplete
    printmethod: Incomplete
    def __init__(self, settings: Incomplete | None = None) -> None:
        """
        `settings` is passed to CodePrinter.__init__()
        `module` specifies the array module to use, currently 'NumPy', 'CuPy'
        or 'JAX'.
        """
    def _print_seq(self, seq):
        """General sequence printer: converts to tuple"""
    def _print_NegativeInfinity(self, expr): ...
    def _print_MatMul(self, expr):
        """Matrix multiplication printer"""
    def _print_MatPow(self, expr):
        """Matrix power printer"""
    def _print_Inverse(self, expr):
        """Matrix inverse printer"""
    def _print_DotProduct(self, expr): ...
    def _print_MatrixSolve(self, expr): ...
    def _print_ZeroMatrix(self, expr): ...
    def _print_OneMatrix(self, expr): ...
    def _print_FunctionMatrix(self, expr): ...
    def _print_HadamardProduct(self, expr): ...
    def _print_KroneckerProduct(self, expr): ...
    def _print_Adjoint(self, expr): ...
    def _print_DiagonalOf(self, expr): ...
    def _print_DiagMatrix(self, expr): ...
    def _print_DiagonalMatrix(self, expr): ...
    def _print_Piecewise(self, expr):
        """Piecewise function printer"""
    def _print_Relational(self, expr):
        """Relational printer for Equality and Unequality"""
    def _print_And(self, expr):
        """Logical And printer"""
    def _print_Or(self, expr):
        """Logical Or printer"""
    def _print_Not(self, expr):
        """Logical Not printer"""
    def _print_Pow(self, expr, rational: bool = False): ...
    def _print_Min(self, expr): ...
    def _print_Max(self, expr): ...
    def _print_arg(self, expr): ...
    def _print_im(self, expr): ...
    def _print_Mod(self, expr): ...
    def _print_re(self, expr): ...
    def _print_sinc(self, expr): ...
    def _print_MatrixBase(self, expr): ...
    def _print_Identity(self, expr): ...
    def _print_BlockMatrix(self, expr): ...
    def _print_NDimArray(self, expr): ...
    _add: str
    _einsum: str
    _transpose: str
    _ones: str
    _zeros: str
    _print_lowergamma: Incomplete
    _print_uppergamma: Incomplete
    _print_fresnelc: Incomplete
    _print_fresnels: Incomplete

_known_functions_scipy_special: Incomplete
_known_constants_scipy_constants: Incomplete
_scipy_known_functions: Incomplete
_scipy_known_constants: Incomplete

class SciPyPrinter(NumPyPrinter):
    _kf: Incomplete
    _kc: Incomplete
    language: str
    def __init__(self, settings: Incomplete | None = None) -> None: ...
    def _print_SparseRepMatrix(self, expr): ...
    _print_ImmutableSparseMatrix = _print_SparseRepMatrix
    def _print_assoc_legendre(self, expr): ...
    def _print_lowergamma(self, expr): ...
    def _print_uppergamma(self, expr): ...
    def _print_betainc(self, expr): ...
    def _print_betainc_regularized(self, expr): ...
    def _print_fresnels(self, expr): ...
    def _print_fresnelc(self, expr): ...
    def _print_airyai(self, expr): ...
    def _print_airyaiprime(self, expr): ...
    def _print_airybi(self, expr): ...
    def _print_airybiprime(self, expr): ...
    def _print_bernoulli(self, expr): ...
    def _print_harmonic(self, expr): ...
    def _print_Integral(self, e): ...
    def _print_Si(self, expr): ...
    def _print_Ci(self, expr): ...

_cupy_known_functions: Incomplete
_cupy_known_constants: Incomplete

class CuPyPrinter(NumPyPrinter):
    """
    CuPy printer which handles vectorized piecewise functions,
    logical operators, etc.
    """
    _module: str
    _kf = _cupy_known_functions
    _kc = _cupy_known_constants
    def __init__(self, settings: Incomplete | None = None) -> None: ...

_jax_known_functions: Incomplete
_jax_known_constants: Incomplete

class JaxPrinter(NumPyPrinter):
    """
    JAX printer which handles vectorized piecewise functions,
    logical operators, etc.
    """
    _module: str
    _kf = _jax_known_functions
    _kc = _jax_known_constants
    printmethod: str
    def __init__(self, settings: Incomplete | None = None) -> None: ...
    def _print_And(self, expr):
        """Logical And printer"""
    def _print_Or(self, expr):
        """Logical Or printer"""
