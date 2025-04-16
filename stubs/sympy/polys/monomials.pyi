from _typeshed import Incomplete
from collections.abc import Generator
from sympy.polys.polyutils import PicklableWithSlots

__all__ = ['itermonomials', 'Monomial']

def itermonomials(variables, max_degrees, min_degrees: Incomplete | None = None) -> Generator[Incomplete, Incomplete]:
    """
    ``max_degrees`` and ``min_degrees`` are either both integers or both lists.
    Unless otherwise specified, ``min_degrees`` is either ``0`` or
    ``[0, ..., 0]``.

    A generator of all monomials ``monom`` is returned, such that
    either
    ``min_degree <= total_degree(monom) <= max_degree``,
    or
    ``min_degrees[i] <= degree_list(monom)[i] <= max_degrees[i]``,
    for all ``i``.

    Case I. ``max_degrees`` and ``min_degrees`` are both integers
    =============================================================

    Given a set of variables $V$ and a min_degree $N$ and a max_degree $M$
    generate a set of monomials of degree less than or equal to $N$ and greater
    than or equal to $M$. The total number of monomials in commutative
    variables is huge and is given by the following formula if $M = 0$:

        .. math::
            \\frac{(\\#V + N)!}{\\#V! N!}

    For example if we would like to generate a dense polynomial of
    a total degree $N = 50$ and $M = 0$, which is the worst case, in 5
    variables, assuming that exponents and all of coefficients are 32-bit long
    and stored in an array we would need almost 80 GiB of memory! Fortunately
    most polynomials, that we will encounter, are sparse.

    Consider monomials in commutative variables $x$ and $y$
    and non-commutative variables $a$ and $b$::

        >>> from sympy import symbols
        >>> from sympy.polys.monomials import itermonomials
        >>> from sympy.polys.orderings import monomial_key
        >>> from sympy.abc import x, y

        >>> sorted(itermonomials([x, y], 2), key=monomial_key('grlex', [y, x]))
        [1, x, y, x**2, x*y, y**2]

        >>> sorted(itermonomials([x, y], 3), key=monomial_key('grlex', [y, x]))
        [1, x, y, x**2, x*y, y**2, x**3, x**2*y, x*y**2, y**3]

        >>> a, b = symbols('a, b', commutative=False)
        >>> set(itermonomials([a, b, x], 2))
        {1, a, a**2, b, b**2, x, x**2, a*b, b*a, x*a, x*b}

        >>> sorted(itermonomials([x, y], 2, 1), key=monomial_key('grlex', [y, x]))
        [x, y, x**2, x*y, y**2]

    Case II. ``max_degrees`` and ``min_degrees`` are both lists
    ===========================================================

    If ``max_degrees = [d_1, ..., d_n]`` and
    ``min_degrees = [e_1, ..., e_n]``, the number of monomials generated
    is:

    .. math::
        (d_1 - e_1 + 1) (d_2 - e_2 + 1) \\cdots (d_n - e_n + 1)

    Let us generate all monomials ``monom`` in variables $x$ and $y$
    such that ``[1, 2][i] <= degree_list(monom)[i] <= [2, 4][i]``,
    ``i = 0, 1`` ::

        >>> from sympy import symbols
        >>> from sympy.polys.monomials import itermonomials
        >>> from sympy.polys.orderings import monomial_key
        >>> from sympy.abc import x, y

        >>> sorted(itermonomials([x, y], [2, 4], [1, 2]), reverse=True, key=monomial_key('lex', [x, y]))
        [x**2*y**4, x**2*y**3, x**2*y**2, x*y**4, x*y**3, x*y**2]
    """

class MonomialOps:
    """Code generator of fast monomial arithmetic functions. """
    ngens: Incomplete
    def __init__(self, ngens) -> None: ...
    def _build(self, code, name): ...
    def _vars(self, name): ...
    def mul(self): ...
    def pow(self): ...
    def mulpow(self): ...
    def ldiv(self): ...
    def div(self): ...
    def lcm(self): ...
    def gcd(self): ...

class Monomial(PicklableWithSlots):
    """Class representing a monomial, i.e. a product of powers. """
    __slots__: Incomplete
    exponents: Incomplete
    gens: Incomplete
    def __init__(self, monom, gens: Incomplete | None = None) -> None: ...
    def rebuild(self, exponents, gens: Incomplete | None = None): ...
    def __len__(self) -> int: ...
    def __iter__(self): ...
    def __getitem__(self, item): ...
    def __hash__(self): ...
    def __str__(self) -> str: ...
    def as_expr(self, *gens):
        """Convert a monomial instance to a SymPy expression. """
    def __eq__(self, other): ...
    def __ne__(self, other): ...
    def __mul__(self, other): ...
    def __truediv__(self, other): ...
    __floordiv__ = __truediv__
    def __pow__(self, other): ...
    def gcd(self, other):
        """Greatest common divisor of monomials. """
    def lcm(self, other):
        """Least common multiple of monomials. """
