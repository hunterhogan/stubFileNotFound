from .pythonrational import PythonRational as PythonRational
from gmpy2 import (
	denom as gmpy_denom, gcd as gmpy_gcd, gcdext as gmpy_gcdex, lcm as gmpy_lcm, mpq as GMPYRational, mpz as GMPYInteger,
	numer as gmpy_numer, qdiv as gmpy_qdiv)
from sympy.core.intfunc import igcd2 as python_gcd, igcdex as python_gcdex, ilcm as python_lcm
from sympy.core.numbers import Float as SymPyReal, Integer as SymPyInteger, Rational as SymPyRational
from sympy.external.gmpy import factorial as factorial, is_square as is_square, sqrt as sqrt, sqrtrem as sqrtrem
import builtins

__all__ = ['GMPYInteger', 'GMPYInteger', 'GMPYRational', 'GMPYRational', 'PythonComplex', 'PythonInteger', 'PythonRational', 'PythonReal', 'SymPyInteger', 'SymPyRational', 'SymPyReal', 'factorial', 'gmpy_denom', 'gmpy_gcd', 'gmpy_gcdex', 'gmpy_lcm', 'gmpy_numer', 'gmpy_qdiv', 'is_square', 'python_gcd', 'python_gcdex', 'python_lcm', 'sqrt', 'sqrtrem']

PythonInteger = builtins.int
PythonReal = builtins.float
PythonComplex = builtins.complex

class _GMPYInteger:
    def __init__(self, obj) -> None: ...

class _GMPYRational:
    def __init__(self, obj) -> None: ...
gcdex = gmpy_gcdex
gcd = gmpy_gcd
lcm = gmpy_lcm
GMPYInteger = _GMPYInteger
GMPYRational = _GMPYRational
GMPYInteger = _GMPYInteger
GMPYRational = _GMPYRational
gcdex = python_gcdex
gcd = python_gcd
lcm = python_lcm
