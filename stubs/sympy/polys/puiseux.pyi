from _typeshed import Incomplete
from collections.abc import Iterable, Iterator
from sympy.core.add import Add as Add
from sympy.core.expr import Expr as Expr
from sympy.core.mul import Mul as Mul
from sympy.external.gmpy import gcd as gcd, lcm as lcm
from sympy.polys.domains import Domain as Domain, QQ as QQ
from sympy.polys.rings import PolyElement as PolyElement, PolyRing as PolyRing
from typing import Any, Unpack

def puiseux_ring(symbols: str | list[Expr], domain: Domain) -> tuple[PuiseuxRing, Unpack[tuple[PuiseuxPoly, ...]]]:
    """Construct a Puiseux ring.

    This function constructs a Puiseux ring with the given symbols and domain.

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.puiseux import puiseux_ring
    >>> R, x, y = puiseux_ring('x y', QQ)
    >>> R
    PuiseuxRing((x, y), QQ)
    >>> p = 5*x**QQ(1,2) + 7/y
    >>> p
    7*y**(-1) + 5*x**(1/2)
    """

class PuiseuxRing:
    """Ring of Puiseux polynomials.

    A Puiseux polynomial is a truncated Puiseux series. The exponents of the
    monomials can be negative or rational numbers. This ring is used by the
    ring_series module:

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.puiseux import puiseux_ring
    >>> from sympy.polys.ring_series import rs_exp, rs_nth_root
    >>> ring, x, y = puiseux_ring('x y', QQ)
    >>> f = x**2 + y**3
    >>> f
    y**3 + x**2
    >>> f.diff(x)
    2*x
    >>> rs_exp(x, x, 5)
    1 + x + 1/2*x**2 + 1/6*x**3 + 1/24*x**4

    Importantly the Puiseux ring can represent truncated series with negative
    and fractional exponents:

    >>> f = 1/x + 1/y**2
    >>> f
    x**(-1) + y**(-2)
    >>> f.diff(x)
    -1*x**(-2)

    >>> rs_nth_root(8*x + x**2 + x**3, 3, x, 5)
    2*x**(1/3) + 1/12*x**(4/3) + 23/288*x**(7/3) + -139/20736*x**(10/3)

    See Also
    ========

    sympy.polys.ring_series.rs_series
    PuiseuxPoly
    """
    poly_ring: Incomplete
    domain: Incomplete
    symbols: Incomplete
    gens: Incomplete
    ngens: Incomplete
    zero: Incomplete
    one: Incomplete
    zero_monom: Incomplete
    monomial_mul: Incomplete
    def __init__(self, symbols: str | list[Expr], domain: Domain) -> None: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: Any) -> bool: ...
    def from_poly(self, poly: PolyElement) -> PuiseuxPoly:
        """Create a Puiseux polynomial from a polynomial.

        >>> from sympy.polys.domains import QQ
        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.puiseux import puiseux_ring
        >>> R1, x1 = ring('x', QQ)
        >>> R2, x2 = puiseux_ring('x', QQ)
        >>> R2.from_poly(x1**2)
        x**2
        """
    def from_dict(self, terms: dict[tuple[int, ...], Any]) -> PuiseuxPoly:
        """Create a Puiseux polynomial from a dictionary of terms.

        >>> from sympy.polys.domains import QQ
        >>> from sympy.polys.puiseux import puiseux_ring
        >>> R, x = puiseux_ring('x', QQ)
        >>> R.from_dict({(QQ(1,2),): QQ(3)})
        3*x**(1/2)
        """
    def from_int(self, n: int) -> PuiseuxPoly:
        """Create a Puiseux polynomial from an integer.

        >>> from sympy.polys.domains import QQ
        >>> from sympy.polys.puiseux import puiseux_ring
        >>> R, x = puiseux_ring('x', QQ)
        >>> R.from_int(3)
        3
        """
    def domain_new(self, arg: Any) -> Any:
        """Create a new element of the domain.

        >>> from sympy.polys.domains import QQ
        >>> from sympy.polys.puiseux import puiseux_ring
        >>> R, x = puiseux_ring('x', QQ)
        >>> R.domain_new(3)
        3
        >>> QQ.of_type(_)
        True
        """
    def ground_new(self, arg: Any) -> PuiseuxPoly:
        """Create a new element from a ground element.

        >>> from sympy.polys.domains import QQ
        >>> from sympy.polys.puiseux import puiseux_ring, PuiseuxPoly
        >>> R, x = puiseux_ring('x', QQ)
        >>> R.ground_new(3)
        3
        >>> isinstance(_, PuiseuxPoly)
        True
        """
    def __call__(self, arg: Any) -> PuiseuxPoly:
        """Coerce an element into the ring.

        >>> from sympy.polys.domains import QQ
        >>> from sympy.polys.puiseux import puiseux_ring
        >>> R, x = puiseux_ring('x', QQ)
        >>> R(3)
        3
        >>> R({(QQ(1,2),): QQ(3)})
        3*x**(1/2)
        """
    def index(self, x: PuiseuxPoly) -> int:
        """Return the index of a generator.

        >>> from sympy.polys.domains import QQ
        >>> from sympy.polys.puiseux import puiseux_ring
        >>> R, x, y = puiseux_ring('x y', QQ)
        >>> R.index(x)
        0
        >>> R.index(y)
        1
        """

def _div_poly_monom(poly: PolyElement, monom: Iterable[int]) -> PolyElement: ...
def _mul_poly_monom(poly: PolyElement, monom: Iterable[int]) -> PolyElement: ...
def _div_monom(monom: Iterable[int], div: Iterable[int]) -> tuple[int, ...]: ...

class PuiseuxPoly:
    """Puiseux polynomial. Represents a truncated Puiseux series.

    See the :class:`PuiseuxRing` class for more information.

    >>> from sympy import QQ
    >>> from sympy.polys.puiseux import puiseux_ring
    >>> R, x, y = puiseux_ring('x, y', QQ)
    >>> p = 5*x**2 + 7*y**3
    >>> p
    7*y**3 + 5*x**2

    The internal representation of a Puiseux polynomial wraps a normal
    polynomial. To support negative powers the polynomial is considered to be
    divided by a monomial.

    >>> p2 = 1/x + 1/y**2
    >>> p2.monom # x*y**2
    (1, 2)
    >>> p2.poly
    x + y**2
    >>> (y**2 + x) / (x*y**2) == p2
    True

    To support fractional powers the polynomial is considered to be a function
    of ``x**(1/nx), y**(1/ny), ...``. The representation keeps track of a
    monomial and a list of exponent denominators so that the polynomial can be
    used to represent both negative and fractional powers.

    >>> p3 = x**QQ(1,2) + y**QQ(2,3)
    >>> p3.ns
    (2, 3)
    >>> p3.poly
    x + y**2

    See Also
    ========

    sympy.polys.puiseux.PuiseuxRing
    sympy.polys.rings.PolyElement
    """
    ring: PuiseuxRing
    poly: PolyElement
    monom: tuple[int, ...] | None
    ns: tuple[int, ...] | None
    def __new__(cls, poly: PolyElement, ring: PuiseuxRing) -> PuiseuxPoly: ...
    @classmethod
    def _new(cls, ring: PuiseuxRing, poly: PolyElement, monom: tuple[int, ...] | None, ns: tuple[int, ...] | None) -> PuiseuxPoly: ...
    @classmethod
    def _new_raw(cls, ring: PuiseuxRing, poly: PolyElement, monom: tuple[int, ...] | None, ns: tuple[int, ...] | None) -> PuiseuxPoly: ...
    def __eq__(self, other: Any) -> bool: ...
    @classmethod
    def _normalize(cls, poly: PolyElement, monom: tuple[int, ...] | None, ns: tuple[int, ...] | None) -> tuple[PolyElement, tuple[int, ...] | None, tuple[int, ...] | None]: ...
    @classmethod
    def _monom_fromint(cls, monom: tuple[int, ...], dmonom: tuple[int, ...] | None, ns: tuple[int, ...] | None) -> tuple[Any, ...]: ...
    @classmethod
    def _monom_toint(cls, monom: tuple[Any, ...], dmonom: tuple[int, ...] | None, ns: tuple[int, ...] | None) -> tuple[int, ...]: ...
    def itermonoms(self) -> Iterator[tuple[Any, ...]]:
        """Iterate over the monomials of a Puiseux polynomial.

        >>> from sympy import QQ
        >>> from sympy.polys.puiseux import puiseux_ring
        >>> R, x, y = puiseux_ring('x, y', QQ)
        >>> p = 5*x**2 + 7*y**3
        >>> list(p.itermonoms())
        [(2, 0), (0, 3)]
        >>> p[(2, 0)]
        5
        """
    def monoms(self) -> list[tuple[Any, ...]]:
        """Return a list of the monomials of a Puiseux polynomial."""
    def __iter__(self) -> Iterator[tuple[tuple[Any, ...], Any]]: ...
    def __getitem__(self, monom: tuple[int, ...]) -> Any: ...
    def __len__(self) -> int: ...
    def iterterms(self) -> Iterator[tuple[tuple[Any, ...], Any]]:
        """Iterate over the terms of a Puiseux polynomial.

        >>> from sympy import QQ
        >>> from sympy.polys.puiseux import puiseux_ring
        >>> R, x, y = puiseux_ring('x, y', QQ)
        >>> p = 5*x**2 + 7*y**3
        >>> list(p.iterterms())
        [((2, 0), 5), ((0, 3), 7)]
        """
    def terms(self) -> list[tuple[tuple[Any, ...], Any]]:
        """Return a list of the terms of a Puiseux polynomial."""
    @property
    def is_term(self) -> bool:
        """Return True if the Puiseux polynomial is a single term."""
    def to_dict(self) -> dict[tuple[int, ...], Any]:
        """Return a dictionary representation of a Puiseux polynomial."""
    @classmethod
    def from_dict(cls, terms: dict[tuple[Any, ...], Any], ring: PuiseuxRing) -> PuiseuxPoly:
        """Create a Puiseux polynomial from a dictionary of terms.

        >>> from sympy import QQ
        >>> from sympy.polys.puiseux import puiseux_ring, PuiseuxPoly
        >>> R, x = puiseux_ring('x', QQ)
        >>> PuiseuxPoly.from_dict({(QQ(1,2),): QQ(3)}, R)
        3*x**(1/2)
        >>> R.from_dict({(QQ(1,2),): QQ(3)})
        3*x**(1/2)
        """
    def as_expr(self) -> Expr:
        """Convert a Puiseux polynomial to :class:`~sympy.core.expr.Expr`.

        >>> from sympy import QQ, Expr
        >>> from sympy.polys.puiseux import puiseux_ring
        >>> R, x = puiseux_ring('x', QQ)
        >>> p = 5*x**2 + 7*x**3
        >>> p.as_expr()
        7*x**3 + 5*x**2
        >>> isinstance(_, Expr)
        True
        """
    def __repr__(self) -> str: ...
    def _unify(self, other: PuiseuxPoly) -> tuple[PolyElement, PolyElement, tuple[int, ...] | None, tuple[int, ...] | None]:
        """Bring two Puiseux polynomials to a common monom and ns."""
    def __pos__(self) -> PuiseuxPoly: ...
    def __neg__(self) -> PuiseuxPoly: ...
    def __add__(self, other: Any) -> PuiseuxPoly: ...
    def __radd__(self, other: Any) -> PuiseuxPoly: ...
    def __sub__(self, other: Any) -> PuiseuxPoly: ...
    def __rsub__(self, other: Any) -> PuiseuxPoly: ...
    def __mul__(self, other: Any) -> PuiseuxPoly: ...
    def __rmul__(self, other: Any) -> PuiseuxPoly: ...
    def __pow__(self, other: Any) -> PuiseuxPoly: ...
    def __truediv__(self, other: Any) -> PuiseuxPoly: ...
    def __rtruediv__(self, other: Any) -> PuiseuxPoly: ...
    def _add(self, other: PuiseuxPoly) -> PuiseuxPoly: ...
    def _add_ground(self, ground: Any) -> PuiseuxPoly: ...
    def _sub(self, other: PuiseuxPoly) -> PuiseuxPoly: ...
    def _sub_ground(self, ground: Any) -> PuiseuxPoly: ...
    def _rsub_ground(self, ground: Any) -> PuiseuxPoly: ...
    def _mul(self, other: PuiseuxPoly) -> PuiseuxPoly: ...
    def _mul_ground(self, ground: Any) -> PuiseuxPoly: ...
    def _div_ground(self, ground: Any) -> PuiseuxPoly: ...
    def _pow_pint(self, n: int) -> PuiseuxPoly: ...
    def _pow_nint(self, n: int) -> PuiseuxPoly: ...
    def _pow_rational(self, n: Any) -> PuiseuxPoly: ...
    def _inv(self) -> PuiseuxPoly: ...
    def diff(self, x: PuiseuxPoly) -> PuiseuxPoly:
        """Differentiate a Puiseux polynomial with respect to a variable.

        >>> from sympy import QQ
        >>> from sympy.polys.puiseux import puiseux_ring
        >>> R, x, y = puiseux_ring('x, y', QQ)
        >>> p = 5*x**2 + 7*y**3
        >>> p.diff(x)
        10*x
        >>> p.diff(y)
        21*y**2
        """
