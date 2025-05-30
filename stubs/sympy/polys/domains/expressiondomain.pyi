from _typeshed import Incomplete
from sympy.polys.domains.characteristiczero import CharacteristicZero
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.domains.field import Field
from sympy.polys.domains.simpledomain import SimpleDomain
from sympy.polys.polyutils import PicklableWithSlots

__all__ = ['ExpressionDomain']

class ExpressionDomain(Field, CharacteristicZero, SimpleDomain):
    """A class for arbitrary expressions. """
    is_SymbolicDomain: bool
    is_EX: bool
    class Expression(DomainElement, PicklableWithSlots):
        """An arbitrary expression. """
        __slots__: Incomplete
        ex: Incomplete
        def __init__(self, ex) -> None: ...
        def __repr__(f) -> str: ...
        def __str__(f) -> str: ...
        def __hash__(self): ...
        def parent(self): ...
        def as_expr(f): ...
        def numer(f): ...
        def denom(f): ...
        def simplify(f, ex): ...
        def __abs__(f): ...
        def __neg__(f): ...
        def _to_ex(f, g): ...
        def __lt__(f, g): ...
        def __add__(f, g): ...
        def __radd__(f, g): ...
        def __sub__(f, g): ...
        def __rsub__(f, g): ...
        def __mul__(f, g): ...
        def __rmul__(f, g): ...
        def __pow__(f, n): ...
        def __truediv__(f, g): ...
        def __rtruediv__(f, g): ...
        def __eq__(f, g): ...
        def __ne__(f, g): ...
        def __bool__(f) -> bool: ...
        def gcd(f, g): ...
        def lcm(f, g): ...
    dtype = Expression
    zero: Incomplete
    one: Incomplete
    rep: str
    has_assoc_Ring: bool
    has_assoc_Field: bool
    def __init__(self) -> None: ...
    def __eq__(self, other): ...
    def __hash__(self): ...
    def to_sympy(self, a):
        """Convert ``a`` to a SymPy object. """
    def from_sympy(self, a):
        """Convert SymPy's expression to ``dtype``. """
    def from_ZZ(K1, a, K0):
        """Convert a Python ``int`` object to ``dtype``. """
    def from_ZZ_python(K1, a, K0):
        """Convert a Python ``int`` object to ``dtype``. """
    def from_QQ(K1, a, K0):
        """Convert a Python ``Fraction`` object to ``dtype``. """
    def from_QQ_python(K1, a, K0):
        """Convert a Python ``Fraction`` object to ``dtype``. """
    def from_ZZ_gmpy(K1, a, K0):
        """Convert a GMPY ``mpz`` object to ``dtype``. """
    def from_QQ_gmpy(K1, a, K0):
        """Convert a GMPY ``mpq`` object to ``dtype``. """
    def from_GaussianIntegerRing(K1, a, K0):
        """Convert a ``GaussianRational`` object to ``dtype``. """
    def from_GaussianRationalField(K1, a, K0):
        """Convert a ``GaussianRational`` object to ``dtype``. """
    def from_AlgebraicField(K1, a, K0):
        """Convert an ``ANP`` object to ``dtype``. """
    def from_RealField(K1, a, K0):
        """Convert a mpmath ``mpf`` object to ``dtype``. """
    def from_ComplexField(K1, a, K0):
        """Convert a mpmath ``mpc`` object to ``dtype``. """
    def from_PolynomialRing(K1, a, K0):
        """Convert a ``DMP`` object to ``dtype``. """
    def from_FractionField(K1, a, K0):
        """Convert a ``DMF`` object to ``dtype``. """
    def from_ExpressionDomain(K1, a, K0):
        """Convert a ``EX`` object to ``dtype``. """
    def get_ring(self):
        """Returns a ring associated with ``self``. """
    def get_field(self):
        """Returns a field associated with ``self``. """
    def is_positive(self, a):
        """Returns True if ``a`` is positive. """
    def is_negative(self, a):
        """Returns True if ``a`` is negative. """
    def is_nonpositive(self, a):
        """Returns True if ``a`` is non-positive. """
    def is_nonnegative(self, a):
        """Returns True if ``a`` is non-negative. """
    def numer(self, a):
        """Returns numerator of ``a``. """
    def denom(self, a):
        """Returns denominator of ``a``. """
    def gcd(self, a, b): ...
    def lcm(self, a, b): ...
