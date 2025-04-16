from _typeshed import Incomplete

__all__ = ['PythonMPQ']

class PythonMPQ:
    """Rational number implementation that is intended to be compatible with
    gmpy2's mpq.

    Also slightly faster than fractions.Fraction.

    PythonMPQ should be treated as immutable although no effort is made to
    prevent mutation (since that might slow down calculations).
    """
    __slots__: Incomplete
    def __new__(cls, numerator, denominator: Incomplete | None = None):
        """Construct PythonMPQ with gcd computation and checks"""
    @classmethod
    def _new_check(cls, numerator, denominator):
        """Construct PythonMPQ, check divide by zero and canonicalize signs"""
    @classmethod
    def _new(cls, numerator, denominator):
        """Construct PythonMPQ efficiently (no checks)"""
    def __int__(self) -> int:
        """Convert to int (truncates towards zero)"""
    def __float__(self) -> float:
        """Convert to float (approximately)"""
    def __bool__(self) -> bool:
        """True/False if nonzero/zero"""
    def __eq__(self, other):
        """Compare equal with PythonMPQ, int, float, Decimal or Fraction"""
    def __hash__(self):
        """hash - same as mpq/Fraction"""
    def __reduce__(self):
        """Deconstruct for pickling"""
    def __str__(self) -> str:
        """Convert to string"""
    def __repr__(self) -> str:
        """Convert to string"""
    def _cmp(self, other, op):
        """Helper for lt/le/gt/ge"""
    def __lt__(self, other):
        """self < other"""
    def __le__(self, other):
        """self <= other"""
    def __gt__(self, other):
        """self > other"""
    def __ge__(self, other):
        """self >= other"""
    def __abs__(self):
        """abs(q)"""
    def __pos__(self):
        """+q"""
    def __neg__(self):
        """-q"""
    def __add__(self, other):
        """q1 + q2"""
    def __radd__(self, other):
        """z1 + q2"""
    def __sub__(self, other):
        """q1 - q2"""
    def __rsub__(self, other):
        """z1 - q2"""
    def __mul__(self, other):
        """q1 * q2"""
    def __rmul__(self, other):
        """z1 * q2"""
    def __pow__(self, exp):
        """q ** z"""
    def __truediv__(self, other):
        """q1 / q2"""
    def __rtruediv__(self, other):
        """z / q"""
    _compatible_types: tuple[type, ...]
