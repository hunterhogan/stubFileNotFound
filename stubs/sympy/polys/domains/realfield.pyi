from _typeshed import Incomplete
from sympy.polys.domains.characteristiczero import CharacteristicZero
from sympy.polys.domains.field import Field
from sympy.polys.domains.simpledomain import SimpleDomain

__all__ = ['RealField']

class RealField(Field, CharacteristicZero, SimpleDomain):
    """Real numbers up to the given precision. """
    rep: str
    is_RealField: bool
    is_RR: bool
    is_Exact: bool
    is_Numerical: bool
    is_PID: bool
    has_assoc_Ring: bool
    has_assoc_Field: bool
    _default_precision: int
    @property
    def has_default_precision(self): ...
    @property
    def precision(self): ...
    @property
    def dps(self): ...
    @property
    def tolerance(self): ...
    _context: Incomplete
    _dtype: Incomplete
    zero: Incomplete
    one: Incomplete
    def __init__(self, prec=..., dps: Incomplete | None = None, tol: Incomplete | None = None) -> None: ...
    @property
    def tp(self): ...
    def dtype(self, arg): ...
    def __eq__(self, other): ...
    def __hash__(self): ...
    def to_sympy(self, element):
        """Convert ``element`` to SymPy number. """
    def from_sympy(self, expr):
        """Convert SymPy's number to ``dtype``. """
    def from_ZZ(self, element, base): ...
    def from_ZZ_python(self, element, base): ...
    def from_ZZ_gmpy(self, element, base): ...
    def from_QQ(self, element, base): ...
    def from_QQ_python(self, element, base): ...
    def from_QQ_gmpy(self, element, base): ...
    def from_AlgebraicField(self, element, base): ...
    def from_RealField(self, element, base): ...
    def from_ComplexField(self, element, base): ...
    def to_rational(self, element, limit: bool = True):
        """Convert a real number to rational number. """
    def get_ring(self):
        """Returns a ring associated with ``self``. """
    def get_exact(self):
        """Returns an exact domain associated with ``self``. """
    def gcd(self, a, b):
        """Returns GCD of ``a`` and ``b``. """
    def lcm(self, a, b):
        """Returns LCM of ``a`` and ``b``. """
    def almosteq(self, a, b, tolerance: Incomplete | None = None):
        """Check if ``a`` and ``b`` are almost equal. """
    def is_square(self, a):
        """Returns ``True`` if ``a >= 0`` and ``False`` otherwise. """
    def exsqrt(self, a):
        """Non-negative square root for ``a >= 0`` and ``None`` otherwise.

        Explanation
        ===========
        The square root may be slightly inaccurate due to floating point
        rounding error.
        """
