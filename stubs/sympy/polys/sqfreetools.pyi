from _typeshed import Incomplete
from collections.abc import Generator
from sympy.polys.densearith import dmp_mul as dmp_mul, dmp_mul_ground as dmp_mul_ground, dmp_neg as dmp_neg, dmp_quo as dmp_quo, dmp_sub as dmp_sub, dup_mul as dup_mul, dup_mul_ground as dup_mul_ground, dup_neg as dup_neg, dup_quo as dup_quo, dup_sub as dup_sub
from sympy.polys.densebasic import dmp_degree as dmp_degree, dmp_degree_in as dmp_degree_in, dmp_degree_list as dmp_degree_list, dmp_ground as dmp_ground, dmp_ground_LC as dmp_ground_LC, dmp_inject as dmp_inject, dmp_raise as dmp_raise, dmp_zero_p as dmp_zero_p, dup_LC as dup_LC, dup_convert as dup_convert, dup_degree as dup_degree, dup_strip as dup_strip
from sympy.polys.densetools import dmp_diff as dmp_diff, dmp_diff_in as dmp_diff_in, dmp_ground_monic as dmp_ground_monic, dmp_ground_primitive as dmp_ground_primitive, dmp_shift as dmp_shift, dup_diff as dup_diff, dup_monic as dup_monic, dup_primitive as dup_primitive, dup_shift as dup_shift
from sympy.polys.euclidtools import dmp_gcd as dmp_gcd, dmp_inner_gcd as dmp_inner_gcd, dmp_primitive as dmp_primitive, dmp_resultant as dmp_resultant, dup_gcd as dup_gcd, dup_inner_gcd as dup_inner_gcd
from sympy.polys.galoistools import gf_sqf_list as gf_sqf_list, gf_sqf_part as gf_sqf_part
from sympy.polys.polyerrors import DomainError as DomainError, MultivariatePolynomialError as MultivariatePolynomialError

def _dup_check_degrees(f, result) -> None:
    """Sanity check the degrees of a computed factorization in K[x]."""
def _dmp_check_degrees(f, u, result) -> None:
    """Sanity check the degrees of a computed factorization in K[X]."""
def dup_sqf_p(f, K):
    '''
    Return ``True`` if ``f`` is a square-free polynomial in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_sqf_p(x**2 - 2*x + 1)
    False
    >>> R.dup_sqf_p(x**2 - 1)
    True

    '''
def dmp_sqf_p(f, u, K):
    '''
    Return ``True`` if ``f`` is a square-free polynomial in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_sqf_p(x**2 + 2*x*y + y**2)
    False
    >>> R.dmp_sqf_p(x**2 + y**2)
    True

    '''
def dup_sqf_norm(f, K):
    '''
    Find a shift of `f` in `K[x]` that has square-free norm.

    The domain `K` must be an algebraic number field `k(a)` (see :ref:`QQ(a)`).

    Returns `(s,g,r)`, such that `g(x)=f(x-sa)`, `r(x)=\\text{Norm}(g(x))` and
    `r` is a square-free polynomial over `k`.

    Examples
    ========

    We first create the algebraic number field `K=k(a)=\\mathbb{Q}(\\sqrt{3})`
    and rings `K[x]` and `k[x]`:

    >>> from sympy.polys import ring, QQ
    >>> from sympy import sqrt

    >>> K = QQ.algebraic_field(sqrt(3))
    >>> R, x = ring("x", K)
    >>> _, X = ring("x", QQ)

    We can now find a square free norm for a shift of `f`:

    >>> f = x**2 - 1
    >>> s, g, r = R.dup_sqf_norm(f)

    The choice of shift `s` is arbitrary and the particular values returned for
    `g` and `r` are determined by `s`.

    >>> s == 1
    True
    >>> g == x**2 - 2*sqrt(3)*x + 2
    True
    >>> r == X**4 - 8*X**2 + 4
    True

    The invariants are:

    >>> g == f.shift(-s*K.unit)
    True
    >>> g.norm() == r
    True
    >>> r.is_squarefree
    True

    Explanation
    ===========

    This is part of Trager\'s algorithm for factorizing polynomials over
    algebraic number fields. In particular this function is algorithm
    ``sqfr_norm`` from [Trager76]_.

    See Also
    ========

    dmp_sqf_norm:
        Analogous function for multivariate polynomials over ``k(a)``.
    dmp_norm:
        Computes the norm of `f` directly without any shift.
    dup_ext_factor:
        Function implementing Trager\'s algorithm that uses this.
    sympy.polys.polytools.sqf_norm:
        High-level interface for using this function.
    '''
def _dmp_sqf_norm_shifts(f, u, K) -> Generator[Incomplete]:
    """Generate a sequence of candidate shifts for dmp_sqf_norm."""
def dmp_sqf_norm(f, u, K):
    '''
    Find a shift of ``f`` in ``K[X]`` that has square-free norm.

    The domain `K` must be an algebraic number field `k(a)` (see :ref:`QQ(a)`).

    Returns `(s,g,r)`, such that `g(x_1,x_2,\\cdots)=f(x_1-s_1 a, x_2 - s_2 a,
    \\cdots)`, `r(x)=\\text{Norm}(g(x))` and `r` is a square-free polynomial over
    `k`.

    Examples
    ========

    We first create the algebraic number field `K=k(a)=\\mathbb{Q}(i)` and rings
    `K[x,y]` and `k[x,y]`:

    >>> from sympy.polys import ring, QQ
    >>> from sympy import I

    >>> K = QQ.algebraic_field(I)
    >>> R, x, y = ring("x,y", K)
    >>> _, X, Y = ring("x,y", QQ)

    We can now find a square free norm for a shift of `f`:

    >>> f = x*y + y**2
    >>> s, g, r = R.dmp_sqf_norm(f)

    The choice of shifts ``s`` is arbitrary and the particular values returned
    for ``g`` and ``r`` are determined by ``s``.

    >>> s
    [0, 1]
    >>> g == x*y - I*x + y**2 - 2*I*y - 1
    True
    >>> r == X**2*Y**2 + X**2 + 2*X*Y**3 + 2*X*Y + Y**4 + 2*Y**2 + 1
    True

    The required invariants are:

    >>> g == f.shift_list([-si*K.unit for si in s])
    True
    >>> g.norm() == r
    True
    >>> r.is_squarefree
    True

    Explanation
    ===========

    This is part of Trager\'s algorithm for factorizing polynomials over
    algebraic number fields. In particular this function is a multivariate
    generalization of algorithm ``sqfr_norm`` from [Trager76]_.

    See Also
    ========

    dup_sqf_norm:
        Analogous function for univariate polynomials over ``k(a)``.
    dmp_norm:
        Computes the norm of `f` directly without any shift.
    dmp_ext_factor:
        Function implementing Trager\'s algorithm that uses this.
    sympy.polys.polytools.sqf_norm:
        High-level interface for using this function.
    '''
def dmp_norm(f, u, K):
    """
    Norm of ``f`` in ``K[X]``, often not square-free.

    The domain `K` must be an algebraic number field `k(a)` (see :ref:`QQ(a)`).

    Examples
    ========

    We first define the algebraic number field `K = k(a) = \\mathbb{Q}(\\sqrt{2})`:

    >>> from sympy import QQ, sqrt
    >>> from sympy.polys.sqfreetools import dmp_norm
    >>> k = QQ
    >>> K = k.algebraic_field(sqrt(2))

    We can now compute the norm of a polynomial `p` in `K[x,y]`:

    >>> p = [[K(1)], [K(1),K.unit]]                  # x + y + sqrt(2)
    >>> N = [[k(1)], [k(2),k(0)], [k(1),k(0),k(-2)]] # x**2 + 2*x*y + y**2 - 2
    >>> dmp_norm(p, 1, K) == N
    True

    In higher level functions that is:

    >>> from sympy import expand, roots, minpoly
    >>> from sympy.abc import x, y
    >>> from math import prod
    >>> a = sqrt(2)
    >>> e = (x + y + a)
    >>> e.as_poly([x, y], extension=a).norm()
    Poly(x**2 + 2*x*y + y**2 - 2, x, y, domain='QQ')

    This is equal to the product of the expressions `x + y + a_i` where the
    `a_i` are the conjugates of `a`:

    >>> pa = minpoly(a)
    >>> pa
    _x**2 - 2
    >>> rs = roots(pa, multiple=True)
    >>> rs
    [sqrt(2), -sqrt(2)]
    >>> n = prod(e.subs(a, r) for r in rs)
    >>> n
    (x + y - sqrt(2))*(x + y + sqrt(2))
    >>> expand(n)
    x**2 + 2*x*y + y**2 - 2

    Explanation
    ===========

    Given an algebraic number field `K = k(a)` any element `b` of `K` can be
    represented as polynomial function `b=g(a)` where `g` is in `k[x]`. If the
    minimal polynomial of `a` over `k` is `p_a` then the roots `a_1`, `a_2`,
    `\\cdots` of `p_a(x)` are the conjugates of `a`. The norm of `b` is the
    product `g(a1) \\times g(a2) \\times \\cdots` and is an element of `k`.

    As in [Trager76]_ we extend this norm to multivariate polynomials over `K`.
    If `b(x)` is a polynomial in `k(a)[X]` then we can think of `b` as being
    alternately a function `g_X(a)` where `g_X` is an element of `k[X][y]` i.e.
    a polynomial function with coefficients that are elements of `k[X]`. Then
    the norm of `b` is the product `g_X(a1) \\times g_X(a2) \\times \\cdots` and
    will be an element of `k[X]`.

    See Also
    ========

    dmp_sqf_norm:
        Compute a shift of `f` so that the `\\text{Norm}(f)` is square-free.
    sympy.polys.polytools.Poly.norm:
        Higher-level function that calls this.
    """
def dup_gf_sqf_part(f, K):
    """Compute square-free part of ``f`` in ``GF(p)[x]``. """
def dmp_gf_sqf_part(f, u, K) -> None:
    """Compute square-free part of ``f`` in ``GF(p)[X]``. """
def dup_sqf_part(f, K):
    '''
    Returns square-free part of a polynomial in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_sqf_part(x**3 - 3*x - 2)
    x**2 - x - 2

    See Also
    ========

    sympy.polys.polytools.Poly.sqf_part
    '''
def dmp_sqf_part(f, u, K):
    '''
    Returns square-free part of a polynomial in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_sqf_part(x**3 + 2*x**2*y + x*y**2)
    x**2 + x*y

    '''
def dup_gf_sqf_list(f, K, all: bool = False):
    """Compute square-free decomposition of ``f`` in ``GF(p)[x]``. """
def dmp_gf_sqf_list(f, u, K, all: bool = False) -> None:
    """Compute square-free decomposition of ``f`` in ``GF(p)[X]``. """
def dup_sqf_list(f, K, all: bool = False):
    '''
    Return square-free decomposition of a polynomial in ``K[x]``.

    Uses Yun\'s algorithm from [Yun76]_.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> f = 2*x**5 + 16*x**4 + 50*x**3 + 76*x**2 + 56*x + 16

    >>> R.dup_sqf_list(f)
    (2, [(x + 1, 2), (x + 2, 3)])
    >>> R.dup_sqf_list(f, all=True)
    (2, [(1, 1), (x + 1, 2), (x + 2, 3)])

    See Also
    ========

    dmp_sqf_list:
        Corresponding function for multivariate polynomials.
    sympy.polys.polytools.sqf_list:
        High-level function for square-free factorization of expressions.
    sympy.polys.polytools.Poly.sqf_list:
        Analogous method on :class:`~.Poly`.

    References
    ==========

    [Yun76]_
    '''
def dup_sqf_list_include(f, K, all: bool = False):
    '''
    Return square-free decomposition of a polynomial in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> f = 2*x**5 + 16*x**4 + 50*x**3 + 76*x**2 + 56*x + 16

    >>> R.dup_sqf_list_include(f)
    [(2, 1), (x + 1, 2), (x + 2, 3)]
    >>> R.dup_sqf_list_include(f, all=True)
    [(2, 1), (x + 1, 2), (x + 2, 3)]

    '''
def dmp_sqf_list(f, u, K, all: bool = False):
    '''
    Return square-free decomposition of a polynomial in `K[X]`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = x**5 + 2*x**4*y + x**3*y**2

    >>> R.dmp_sqf_list(f)
    (1, [(x + y, 2), (x, 3)])
    >>> R.dmp_sqf_list(f, all=True)
    (1, [(1, 1), (x + y, 2), (x, 3)])

    Explanation
    ===========

    Uses Yun\'s algorithm for univariate polynomials from [Yun76]_ recrusively.
    The multivariate polynomial is treated as a univariate polynomial in its
    leading variable. Then Yun\'s algorithm computes the square-free
    factorization of the primitive and the content is factored recursively.

    It would be better to use a dedicated algorithm for multivariate
    polynomials instead.

    See Also
    ========

    dup_sqf_list:
        Corresponding function for univariate polynomials.
    sympy.polys.polytools.sqf_list:
        High-level function for square-free factorization of expressions.
    sympy.polys.polytools.Poly.sqf_list:
        Analogous method on :class:`~.Poly`.
    '''
def dmp_sqf_list_include(f, u, K, all: bool = False):
    '''
    Return square-free decomposition of a polynomial in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = x**5 + 2*x**4*y + x**3*y**2

    >>> R.dmp_sqf_list_include(f)
    [(1, 1), (x + y, 2), (x, 3)]
    >>> R.dmp_sqf_list_include(f, all=True)
    [(1, 1), (x + y, 2), (x, 3)]

    '''
def dup_gff_list(f, K):
    '''
    Compute greatest factorial factorization of ``f`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_gff_list(x**5 + 2*x**4 - x**3 - 2*x**2)
    [(x, 1), (x + 2, 4)]

    '''
def dmp_gff_list(f, u, K):
    '''
    Compute greatest factorial factorization of ``f`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    '''
