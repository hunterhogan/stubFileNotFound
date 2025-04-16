from _typeshed import Incomplete
from sympy.polys.domains.groundtypes import PythonRational
from sympy.polys.domains.rationalfield import RationalField

__all__ = ['PythonRationalField']

class PythonRationalField(RationalField):
    """Rational field based on :ref:`MPQ`.

    This will be used as :ref:`QQ` if ``gmpy`` and ``gmpy2`` are not
    installed. Elements are instances of :ref:`MPQ`.
    """
    dtype = PythonRational
    zero: Incomplete
    one: Incomplete
    alias: str
    def __init__(self) -> None: ...
    def get_ring(self):
        """Returns ring associated with ``self``. """
    def to_sympy(self, a):
        """Convert `a` to a SymPy object. """
    def from_sympy(self, a):
        """Convert SymPy's Rational to `dtype`. """
    def from_ZZ_python(K1, a, K0):
        """Convert a Python `int` object to `dtype`. """
    def from_QQ_python(K1, a, K0):
        """Convert a Python `Fraction` object to `dtype`. """
    def from_ZZ_gmpy(K1, a, K0):
        """Convert a GMPY `mpz` object to `dtype`. """
    def from_QQ_gmpy(K1, a, K0):
        """Convert a GMPY `mpq` object to `dtype`. """
    def from_RealField(K1, a, K0):
        """Convert a mpmath `mpf` object to `dtype`. """
    def numer(self, a):
        """Returns numerator of `a`. """
    def denom(self, a):
        """Returns denominator of `a`. """
