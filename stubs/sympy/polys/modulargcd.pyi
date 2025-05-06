from _typeshed import Incomplete
from sympy.core.symbol import Dummy as Dummy
from sympy.ntheory import nextprime as nextprime
from sympy.ntheory.modular import crt as crt
from sympy.polys.domains import PolynomialRing as PolynomialRing
from sympy.polys.galoistools import gf_div as gf_div, gf_from_dict as gf_from_dict, gf_gcd as gf_gcd, gf_gcdex as gf_gcdex, gf_lcm as gf_lcm
from sympy.polys.polyerrors import ModularGCDFailed as ModularGCDFailed

def _trivial_gcd(f, g):
    """
    Compute the GCD of two polynomials in trivial cases, i.e. when one
    or both polynomials are zero.
    """
def _gf_gcd(fp, gp, p):
    """
    Compute the GCD of two univariate polynomials in `\\mathbb{Z}_p[x]`.
    """
def _degree_bound_univariate(f, g):
    """
    Compute an upper bound for the degree of the GCD of two univariate
    integer polynomials `f` and `g`.

    The function chooses a suitable prime `p` and computes the GCD of
    `f` and `g` in `\\mathbb{Z}_p[x]`. The choice of `p` guarantees that
    the degree in `\\mathbb{Z}_p[x]` is greater than or equal to the degree
    in `\\mathbb{Z}[x]`.

    Parameters
    ==========

    f : PolyElement
        univariate integer polynomial
    g : PolyElement
        univariate integer polynomial

    """
def _chinese_remainder_reconstruction_univariate(hp, hq, p, q):
    '''
    Construct a polynomial `h_{pq}` in `\\mathbb{Z}_{p q}[x]` such that

    .. math ::

        h_{pq} = h_p \\; \\mathrm{mod} \\, p

        h_{pq} = h_q \\; \\mathrm{mod} \\, q

    for relatively prime integers `p` and `q` and polynomials
    `h_p` and `h_q` in `\\mathbb{Z}_p[x]` and `\\mathbb{Z}_q[x]`
    respectively.

    The coefficients of the polynomial `h_{pq}` are computed with the
    Chinese Remainder Theorem. The symmetric representation in
    `\\mathbb{Z}_p[x]`, `\\mathbb{Z}_q[x]` and `\\mathbb{Z}_{p q}[x]` is used.
    It is assumed that `h_p` and `h_q` have the same degree.

    Parameters
    ==========

    hp : PolyElement
        univariate integer polynomial with coefficients in `\\mathbb{Z}_p`
    hq : PolyElement
        univariate integer polynomial with coefficients in `\\mathbb{Z}_q`
    p : Integer
        modulus of `h_p`, relatively prime to `q`
    q : Integer
        modulus of `h_q`, relatively prime to `p`

    Examples
    ========

    >>> from sympy.polys.modulargcd import _chinese_remainder_reconstruction_univariate
    >>> from sympy.polys import ring, ZZ

    >>> R, x = ring("x", ZZ)
    >>> p = 3
    >>> q = 5

    >>> hp = -x**3 - 1
    >>> hq = 2*x**3 - 2*x**2 + x

    >>> hpq = _chinese_remainder_reconstruction_univariate(hp, hq, p, q)
    >>> hpq
    2*x**3 + 3*x**2 + 6*x + 5

    >>> hpq.trunc_ground(p) == hp
    True
    >>> hpq.trunc_ground(q) == hq
    True

    '''
def modgcd_univariate(f, g):
    '''
    Computes the GCD of two polynomials in `\\mathbb{Z}[x]` using a modular
    algorithm.

    The algorithm computes the GCD of two univariate integer polynomials
    `f` and `g` by computing the GCD in `\\mathbb{Z}_p[x]` for suitable
    primes `p` and then reconstructing the coefficients with the Chinese
    Remainder Theorem. Trial division is only made for candidates which
    are very likely the desired GCD.

    Parameters
    ==========

    f : PolyElement
        univariate integer polynomial
    g : PolyElement
        univariate integer polynomial

    Returns
    =======

    h : PolyElement
        GCD of the polynomials `f` and `g`
    cff : PolyElement
        cofactor of `f`, i.e. `\\frac{f}{h}`
    cfg : PolyElement
        cofactor of `g`, i.e. `\\frac{g}{h}`

    Examples
    ========

    >>> from sympy.polys.modulargcd import modgcd_univariate
    >>> from sympy.polys import ring, ZZ

    >>> R, x = ring("x", ZZ)

    >>> f = x**5 - 1
    >>> g = x - 1

    >>> h, cff, cfg = modgcd_univariate(f, g)
    >>> h, cff, cfg
    (x - 1, x**4 + x**3 + x**2 + x + 1, 1)

    >>> cff * h == f
    True
    >>> cfg * h == g
    True

    >>> f = 6*x**2 - 6
    >>> g = 2*x**2 + 4*x + 2

    >>> h, cff, cfg = modgcd_univariate(f, g)
    >>> h, cff, cfg
    (2*x + 2, 3*x - 3, x + 1)

    >>> cff * h == f
    True
    >>> cfg * h == g
    True

    References
    ==========

    1. [Monagan00]_

    '''
def _primitive(f, p):
    '''
    Compute the content and the primitive part of a polynomial in
    `\\mathbb{Z}_p[x_0, \\ldots, x_{k-2}, y] \\cong \\mathbb{Z}_p[y][x_0, \\ldots, x_{k-2}]`.

    Parameters
    ==========

    f : PolyElement
        integer polynomial in `\\mathbb{Z}_p[x0, \\ldots, x{k-2}, y]`
    p : Integer
        modulus of `f`

    Returns
    =======

    contf : PolyElement
        integer polynomial in `\\mathbb{Z}_p[y]`, content of `f`
    ppf : PolyElement
        primitive part of `f`, i.e. `\\frac{f}{contf}`

    Examples
    ========

    >>> from sympy.polys.modulargcd import _primitive
    >>> from sympy.polys import ring, ZZ

    >>> R, x, y = ring("x, y", ZZ)
    >>> p = 3

    >>> f = x**2*y**2 + x**2*y - y**2 - y
    >>> _primitive(f, p)
    (y**2 + y, x**2 - 1)

    >>> R, x, y, z = ring("x, y, z", ZZ)

    >>> f = x*y*z - y**2*z**2
    >>> _primitive(f, p)
    (z, x*y - y**2*z)

    '''
def _deg(f):
    '''
    Compute the degree of a multivariate polynomial
    `f \\in K[x_0, \\ldots, x_{k-2}, y] \\cong K[y][x_0, \\ldots, x_{k-2}]`.

    Parameters
    ==========

    f : PolyElement
        polynomial in `K[x_0, \\ldots, x_{k-2}, y]`

    Returns
    =======

    degf : Integer tuple
        degree of `f` in `x_0, \\ldots, x_{k-2}`

    Examples
    ========

    >>> from sympy.polys.modulargcd import _deg
    >>> from sympy.polys import ring, ZZ

    >>> R, x, y = ring("x, y", ZZ)

    >>> f = x**2*y**2 + x**2*y - 1
    >>> _deg(f)
    (2,)

    >>> R, x, y, z = ring("x, y, z", ZZ)

    >>> f = x**2*y**2 + x**2*y - 1
    >>> _deg(f)
    (2, 2)

    >>> f = x*y*z - y**2*z**2
    >>> _deg(f)
    (1, 1)

    '''
def _LC(f):
    '''
    Compute the leading coefficient of a multivariate polynomial
    `f \\in K[x_0, \\ldots, x_{k-2}, y] \\cong K[y][x_0, \\ldots, x_{k-2}]`.

    Parameters
    ==========

    f : PolyElement
        polynomial in `K[x_0, \\ldots, x_{k-2}, y]`

    Returns
    =======

    lcf : PolyElement
        polynomial in `K[y]`, leading coefficient of `f`

    Examples
    ========

    >>> from sympy.polys.modulargcd import _LC
    >>> from sympy.polys import ring, ZZ

    >>> R, x, y = ring("x, y", ZZ)

    >>> f = x**2*y**2 + x**2*y - 1
    >>> _LC(f)
    y**2 + y

    >>> R, x, y, z = ring("x, y, z", ZZ)

    >>> f = x**2*y**2 + x**2*y - 1
    >>> _LC(f)
    1

    >>> f = x*y*z - y**2*z**2
    >>> _LC(f)
    z

    '''
def _swap(f, i):
    """
    Make the variable `x_i` the leading one in a multivariate polynomial `f`.
    """
def _degree_bound_bivariate(f, g):
    """
    Compute upper degree bounds for the GCD of two bivariate
    integer polynomials `f` and `g`.

    The GCD is viewed as a polynomial in `\\mathbb{Z}[y][x]` and the
    function returns an upper bound for its degree and one for the degree
    of its content. This is done by choosing a suitable prime `p` and
    computing the GCD of the contents of `f \\; \\mathrm{mod} \\, p` and
    `g \\; \\mathrm{mod} \\, p`. The choice of `p` guarantees that the degree
    of the content in `\\mathbb{Z}_p[y]` is greater than or equal to the
    degree in `\\mathbb{Z}[y]`. To obtain the degree bound in the variable
    `x`, the polynomials are evaluated at `y = a` for a suitable
    `a \\in \\mathbb{Z}_p` and then their GCD in `\\mathbb{Z}_p[x]` is
    computed. If no such `a` exists, i.e. the degree in `\\mathbb{Z}_p[x]`
    is always smaller than the one in `\\mathbb{Z}[y][x]`, then the bound is
    set to the minimum of the degrees of `f` and `g` in `x`.

    Parameters
    ==========

    f : PolyElement
        bivariate integer polynomial
    g : PolyElement
        bivariate integer polynomial

    Returns
    =======

    xbound : Integer
        upper bound for the degree of the GCD of the polynomials `f` and
        `g` in the variable `x`
    ycontbound : Integer
        upper bound for the degree of the content of the GCD of the
        polynomials `f` and `g` in the variable `y`

    References
    ==========

    1. [Monagan00]_

    """
def _chinese_remainder_reconstruction_multivariate(hp, hq, p, q):
    '''
    Construct a polynomial `h_{pq}` in
    `\\mathbb{Z}_{p q}[x_0, \\ldots, x_{k-1}]` such that

    .. math ::

        h_{pq} = h_p \\; \\mathrm{mod} \\, p

        h_{pq} = h_q \\; \\mathrm{mod} \\, q

    for relatively prime integers `p` and `q` and polynomials
    `h_p` and `h_q` in `\\mathbb{Z}_p[x_0, \\ldots, x_{k-1}]` and
    `\\mathbb{Z}_q[x_0, \\ldots, x_{k-1}]` respectively.

    The coefficients of the polynomial `h_{pq}` are computed with the
    Chinese Remainder Theorem. The symmetric representation in
    `\\mathbb{Z}_p[x_0, \\ldots, x_{k-1}]`,
    `\\mathbb{Z}_q[x_0, \\ldots, x_{k-1}]` and
    `\\mathbb{Z}_{p q}[x_0, \\ldots, x_{k-1}]` is used.

    Parameters
    ==========

    hp : PolyElement
        multivariate integer polynomial with coefficients in `\\mathbb{Z}_p`
    hq : PolyElement
        multivariate integer polynomial with coefficients in `\\mathbb{Z}_q`
    p : Integer
        modulus of `h_p`, relatively prime to `q`
    q : Integer
        modulus of `h_q`, relatively prime to `p`

    Examples
    ========

    >>> from sympy.polys.modulargcd import _chinese_remainder_reconstruction_multivariate
    >>> from sympy.polys import ring, ZZ

    >>> R, x, y = ring("x, y", ZZ)
    >>> p = 3
    >>> q = 5

    >>> hp = x**3*y - x**2 - 1
    >>> hq = -x**3*y - 2*x*y**2 + 2

    >>> hpq = _chinese_remainder_reconstruction_multivariate(hp, hq, p, q)
    >>> hpq
    4*x**3*y + 5*x**2 + 3*x*y**2 + 2

    >>> hpq.trunc_ground(p) == hp
    True
    >>> hpq.trunc_ground(q) == hq
    True

    >>> R, x, y, z = ring("x, y, z", ZZ)
    >>> p = 6
    >>> q = 5

    >>> hp = 3*x**4 - y**3*z + z
    >>> hq = -2*x**4 + z

    >>> hpq = _chinese_remainder_reconstruction_multivariate(hp, hq, p, q)
    >>> hpq
    3*x**4 + 5*y**3*z + z

    >>> hpq.trunc_ground(p) == hp
    True
    >>> hpq.trunc_ground(q) == hq
    True

    '''
def _interpolate_multivariate(evalpoints, hpeval, ring, i, p, ground: bool = False):
    """
    Reconstruct a polynomial `h_p` in `\\mathbb{Z}_p[x_0, \\ldots, x_{k-1}]`
    from a list of evaluation points in `\\mathbb{Z}_p` and a list of
    polynomials in
    `\\mathbb{Z}_p[x_0, \\ldots, x_{i-1}, x_{i+1}, \\ldots, x_{k-1}]`, which
    are the images of `h_p` evaluated in the variable `x_i`.

    It is also possible to reconstruct a parameter of the ground domain,
    i.e. if `h_p` is a polynomial over `\\mathbb{Z}_p[x_0, \\ldots, x_{k-1}]`.
    In this case, one has to set ``ground=True``.

    Parameters
    ==========

    evalpoints : list of Integer objects
        list of evaluation points in `\\mathbb{Z}_p`
    hpeval : list of PolyElement objects
        list of polynomials in (resp. over)
        `\\mathbb{Z}_p[x_0, \\ldots, x_{i-1}, x_{i+1}, \\ldots, x_{k-1}]`,
        images of `h_p` evaluated in the variable `x_i`
    ring : PolyRing
        `h_p` will be an element of this ring
    i : Integer
        index of the variable which has to be reconstructed
    p : Integer
        prime number, modulus of `h_p`
    ground : Boolean
        indicates whether `x_i` is in the ground domain, default is
        ``False``

    Returns
    =======

    hp : PolyElement
        interpolated polynomial in (resp. over)
        `\\mathbb{Z}_p[x_0, \\ldots, x_{k-1}]`

    """
def modgcd_bivariate(f, g):
    '''
    Computes the GCD of two polynomials in `\\mathbb{Z}[x, y]` using a
    modular algorithm.

    The algorithm computes the GCD of two bivariate integer polynomials
    `f` and `g` by calculating the GCD in `\\mathbb{Z}_p[x, y]` for
    suitable primes `p` and then reconstructing the coefficients with the
    Chinese Remainder Theorem. To compute the bivariate GCD over
    `\\mathbb{Z}_p`, the polynomials `f \\; \\mathrm{mod} \\, p` and
    `g \\; \\mathrm{mod} \\, p` are evaluated at `y = a` for certain
    `a \\in \\mathbb{Z}_p` and then their univariate GCD in `\\mathbb{Z}_p[x]`
    is computed. Interpolating those yields the bivariate GCD in
    `\\mathbb{Z}_p[x, y]`. To verify the result in `\\mathbb{Z}[x, y]`, trial
    division is done, but only for candidates which are very likely the
    desired GCD.

    Parameters
    ==========

    f : PolyElement
        bivariate integer polynomial
    g : PolyElement
        bivariate integer polynomial

    Returns
    =======

    h : PolyElement
        GCD of the polynomials `f` and `g`
    cff : PolyElement
        cofactor of `f`, i.e. `\\frac{f}{h}`
    cfg : PolyElement
        cofactor of `g`, i.e. `\\frac{g}{h}`

    Examples
    ========

    >>> from sympy.polys.modulargcd import modgcd_bivariate
    >>> from sympy.polys import ring, ZZ

    >>> R, x, y = ring("x, y", ZZ)

    >>> f = x**2 - y**2
    >>> g = x**2 + 2*x*y + y**2

    >>> h, cff, cfg = modgcd_bivariate(f, g)
    >>> h, cff, cfg
    (x + y, x - y, x + y)

    >>> cff * h == f
    True
    >>> cfg * h == g
    True

    >>> f = x**2*y - x**2 - 4*y + 4
    >>> g = x + 2

    >>> h, cff, cfg = modgcd_bivariate(f, g)
    >>> h, cff, cfg
    (x + 2, x*y - x - 2*y + 2, 1)

    >>> cff * h == f
    True
    >>> cfg * h == g
    True

    References
    ==========

    1. [Monagan00]_

    '''
def _modgcd_multivariate_p(f, g, p, degbound, contbound):
    """
    Compute the GCD of two polynomials in
    `\\mathbb{Z}_p[x_0, \\ldots, x_{k-1}]`.

    The algorithm reduces the problem step by step by evaluating the
    polynomials `f` and `g` at `x_{k-1} = a` for suitable
    `a \\in \\mathbb{Z}_p` and then calls itself recursively to compute the GCD
    in `\\mathbb{Z}_p[x_0, \\ldots, x_{k-2}]`. If these recursive calls are
    successful for enough evaluation points, the GCD in `k` variables is
    interpolated, otherwise the algorithm returns ``None``. Every time a GCD
    or a content is computed, their degrees are compared with the bounds. If
    a degree greater then the bound is encountered, then the current call
    returns ``None`` and a new evaluation point has to be chosen. If at some
    point the degree is smaller, the correspondent bound is updated and the
    algorithm fails.

    Parameters
    ==========

    f : PolyElement
        multivariate integer polynomial with coefficients in `\\mathbb{Z}_p`
    g : PolyElement
        multivariate integer polynomial with coefficients in `\\mathbb{Z}_p`
    p : Integer
        prime number, modulus of `f` and `g`
    degbound : list of Integer objects
        ``degbound[i]`` is an upper bound for the degree of the GCD of `f`
        and `g` in the variable `x_i`
    contbound : list of Integer objects
        ``contbound[i]`` is an upper bound for the degree of the content of
        the GCD in `\\mathbb{Z}_p[x_i][x_0, \\ldots, x_{i-1}]`,
        ``contbound[0]`` is not used can therefore be chosen
        arbitrarily.

    Returns
    =======

    h : PolyElement
        GCD of the polynomials `f` and `g` or ``None``

    References
    ==========

    1. [Monagan00]_
    2. [Brown71]_

    """
def modgcd_multivariate(f, g):
    '''
    Compute the GCD of two polynomials in `\\mathbb{Z}[x_0, \\ldots, x_{k-1}]`
    using a modular algorithm.

    The algorithm computes the GCD of two multivariate integer polynomials
    `f` and `g` by calculating the GCD in
    `\\mathbb{Z}_p[x_0, \\ldots, x_{k-1}]` for suitable primes `p` and then
    reconstructing the coefficients with the Chinese Remainder Theorem. To
    compute the multivariate GCD over `\\mathbb{Z}_p` the recursive
    subroutine :func:`_modgcd_multivariate_p` is used. To verify the result in
    `\\mathbb{Z}[x_0, \\ldots, x_{k-1}]`, trial division is done, but only for
    candidates which are very likely the desired GCD.

    Parameters
    ==========

    f : PolyElement
        multivariate integer polynomial
    g : PolyElement
        multivariate integer polynomial

    Returns
    =======

    h : PolyElement
        GCD of the polynomials `f` and `g`
    cff : PolyElement
        cofactor of `f`, i.e. `\\frac{f}{h}`
    cfg : PolyElement
        cofactor of `g`, i.e. `\\frac{g}{h}`

    Examples
    ========

    >>> from sympy.polys.modulargcd import modgcd_multivariate
    >>> from sympy.polys import ring, ZZ

    >>> R, x, y = ring("x, y", ZZ)

    >>> f = x**2 - y**2
    >>> g = x**2 + 2*x*y + y**2

    >>> h, cff, cfg = modgcd_multivariate(f, g)
    >>> h, cff, cfg
    (x + y, x - y, x + y)

    >>> cff * h == f
    True
    >>> cfg * h == g
    True

    >>> R, x, y, z = ring("x, y, z", ZZ)

    >>> f = x*z**2 - y*z**2
    >>> g = x**2*z + z

    >>> h, cff, cfg = modgcd_multivariate(f, g)
    >>> h, cff, cfg
    (z, x*z - y*z, x**2 + 1)

    >>> cff * h == f
    True
    >>> cfg * h == g
    True

    References
    ==========

    1. [Monagan00]_
    2. [Brown71]_

    See also
    ========

    _modgcd_multivariate_p

    '''
def _gf_div(f, g, p):
    """
    Compute `\\frac f g` modulo `p` for two univariate polynomials over
    `\\mathbb Z_p`.
    """
def _rational_function_reconstruction(c, p, m):
    """
    Reconstruct a rational function `\\frac a b` in `\\mathbb Z_p(t)` from

    .. math::

        c = \\frac a b \\; \\mathrm{mod} \\, m,

    where `c` and `m` are polynomials in `\\mathbb Z_p[t]` and `m` has
    positive degree.

    The algorithm is based on the Euclidean Algorithm. In general, `m` is
    not irreducible, so it is possible that `b` is not invertible modulo
    `m`. In that case ``None`` is returned.

    Parameters
    ==========

    c : PolyElement
        univariate polynomial in `\\mathbb Z[t]`
    p : Integer
        prime number
    m : PolyElement
        modulus, not necessarily irreducible

    Returns
    =======

    frac : FracElement
        either `\\frac a b` in `\\mathbb Z(t)` or ``None``

    References
    ==========

    1. [Hoeij04]_

    """
def _rational_reconstruction_func_coeffs(hm, p, m, ring, k):
    """
    Reconstruct every coefficient `c_h` of a polynomial `h` in
    `\\mathbb Z_p(t_k)[t_1, \\ldots, t_{k-1}][x, z]` from the corresponding
    coefficient `c_{h_m}` of a polynomial `h_m` in
    `\\mathbb Z_p[t_1, \\ldots, t_k][x, z] \\cong \\mathbb Z_p[t_k][t_1, \\ldots, t_{k-1}][x, z]`
    such that

    .. math::

        c_{h_m} = c_h \\; \\mathrm{mod} \\, m,

    where `m \\in \\mathbb Z_p[t]`.

    The reconstruction is based on the Euclidean Algorithm. In general, `m`
    is not irreducible, so it is possible that this fails for some
    coefficient. In that case ``None`` is returned.

    Parameters
    ==========

    hm : PolyElement
        polynomial in `\\mathbb Z[t_1, \\ldots, t_k][x, z]`
    p : Integer
        prime number, modulus of `\\mathbb Z_p`
    m : PolyElement
        modulus, polynomial in `\\mathbb Z[t]`, not necessarily irreducible
    ring : PolyRing
        `\\mathbb Z(t_k)[t_1, \\ldots, t_{k-1}][x, z]`, `h` will be an
        element of this ring
    k : Integer
        index of the parameter `t_k` which will be reconstructed

    Returns
    =======

    h : PolyElement
        reconstructed polynomial in
        `\\mathbb Z(t_k)[t_1, \\ldots, t_{k-1}][x, z]` or ``None``

    See also
    ========

    _rational_function_reconstruction

    """
def _gf_gcdex(f, g, p):
    """
    Extended Euclidean Algorithm for two univariate polynomials over
    `\\mathbb Z_p`.

    Returns polynomials `s, t` and `h`, such that `h` is the GCD of `f` and
    `g` and `sf + tg = h \\; \\mathrm{mod} \\, p`.

    """
def _trunc(f, minpoly, p):
    """
    Compute the reduced representation of a polynomial `f` in
    `\\mathbb Z_p[z] / (\\check m_{\\alpha}(z))[x]`

    Parameters
    ==========

    f : PolyElement
        polynomial in `\\mathbb Z[x, z]`
    minpoly : PolyElement
        polynomial `\\check m_{\\alpha} \\in \\mathbb Z[z]`, not necessarily
        irreducible
    p : Integer
        prime number, modulus of `\\mathbb Z_p`

    Returns
    =======

    ftrunc : PolyElement
        polynomial in `\\mathbb Z[x, z]`, reduced modulo
        `\\check m_{\\alpha}(z)` and `p`

    """
def _euclidean_algorithm(f, g, minpoly, p):
    """
    Compute the monic GCD of two univariate polynomials in
    `\\mathbb{Z}_p[z]/(\\check m_{\\alpha}(z))[x]` with the Euclidean
    Algorithm.

    In general, `\\check m_{\\alpha}(z)` is not irreducible, so it is possible
    that some leading coefficient is not invertible modulo
    `\\check m_{\\alpha}(z)`. In that case ``None`` is returned.

    Parameters
    ==========

    f, g : PolyElement
        polynomials in `\\mathbb Z[x, z]`
    minpoly : PolyElement
        polynomial in `\\mathbb Z[z]`, not necessarily irreducible
    p : Integer
        prime number, modulus of `\\mathbb Z_p`

    Returns
    =======

    h : PolyElement
        GCD of `f` and `g` in `\\mathbb Z[z, x]` or ``None``, coefficients
        are in `\\left[ -\\frac{p-1} 2, \\frac{p-1} 2 \\right]`

    """
def _trial_division(f, h, minpoly, p: Incomplete | None = None):
    """
    Check if `h` divides `f` in
    `\\mathbb K[t_1, \\ldots, t_k][z]/(m_{\\alpha}(z))`, where `\\mathbb K` is
    either `\\mathbb Q` or `\\mathbb Z_p`.

    This algorithm is based on pseudo division and does not use any
    fractions. By default `\\mathbb K` is `\\mathbb Q`, if a prime number `p`
    is given, `\\mathbb Z_p` is chosen instead.

    Parameters
    ==========

    f, h : PolyElement
        polynomials in `\\mathbb Z[t_1, \\ldots, t_k][x, z]`
    minpoly : PolyElement
        polynomial `m_{\\alpha}(z)` in `\\mathbb Z[t_1, \\ldots, t_k][z]`
    p : Integer or None
        if `p` is given, `\\mathbb K` is set to `\\mathbb Z_p` instead of
        `\\mathbb Q`, default is ``None``

    Returns
    =======

    rem : PolyElement
        remainder of `\\frac f h`

    References
    ==========

    .. [1] [Hoeij02]_

    """
def _evaluate_ground(f, i, a):
    """
    Evaluate a polynomial `f` at `a` in the `i`-th variable of the ground
    domain.
    """
def _func_field_modgcd_p(f, g, minpoly, p):
    """
    Compute the GCD of two polynomials `f` and `g` in
    `\\mathbb Z_p(t_1, \\ldots, t_k)[z]/(\\check m_\\alpha(z))[x]`.

    The algorithm reduces the problem step by step by evaluating the
    polynomials `f` and `g` at `t_k = a` for suitable `a \\in \\mathbb Z_p`
    and then calls itself recursively to compute the GCD in
    `\\mathbb Z_p(t_1, \\ldots, t_{k-1})[z]/(\\check m_\\alpha(z))[x]`. If these
    recursive calls are successful, the GCD over `k` variables is
    interpolated, otherwise the algorithm returns ``None``. After
    interpolation, Rational Function Reconstruction is used to obtain the
    correct coefficients. If this fails, a new evaluation point has to be
    chosen, otherwise the desired polynomial is obtained by clearing
    denominators. The result is verified with a fraction free trial
    division.

    Parameters
    ==========

    f, g : PolyElement
        polynomials in `\\mathbb Z[t_1, \\ldots, t_k][x, z]`
    minpoly : PolyElement
        polynomial in `\\mathbb Z[t_1, \\ldots, t_k][z]`, not necessarily
        irreducible
    p : Integer
        prime number, modulus of `\\mathbb Z_p`

    Returns
    =======

    h : PolyElement
        primitive associate in `\\mathbb Z[t_1, \\ldots, t_k][x, z]` of the
        GCD of the polynomials `f` and `g`  or ``None``, coefficients are
        in `\\left[ -\\frac{p-1} 2, \\frac{p-1} 2 \\right]`

    References
    ==========

    1. [Hoeij04]_

    """
def _integer_rational_reconstruction(c, m, domain):
    """
    Reconstruct a rational number `\\frac a b` from

    .. math::

        c = \\frac a b \\; \\mathrm{mod} \\, m,

    where `c` and `m` are integers.

    The algorithm is based on the Euclidean Algorithm. In general, `m` is
    not a prime number, so it is possible that `b` is not invertible modulo
    `m`. In that case ``None`` is returned.

    Parameters
    ==========

    c : Integer
        `c = \\frac a b \\; \\mathrm{mod} \\, m`
    m : Integer
        modulus, not necessarily prime
    domain : IntegerRing
        `a, b, c` are elements of ``domain``

    Returns
    =======

    frac : Rational
        either `\\frac a b` in `\\mathbb Q` or ``None``

    References
    ==========

    1. [Wang81]_

    """
def _rational_reconstruction_int_coeffs(hm, m, ring):
    """
    Reconstruct every rational coefficient `c_h` of a polynomial `h` in
    `\\mathbb Q[t_1, \\ldots, t_k][x, z]` from the corresponding integer
    coefficient `c_{h_m}` of a polynomial `h_m` in
    `\\mathbb Z[t_1, \\ldots, t_k][x, z]` such that

    .. math::

        c_{h_m} = c_h \\; \\mathrm{mod} \\, m,

    where `m \\in \\mathbb Z`.

    The reconstruction is based on the Euclidean Algorithm. In general,
    `m` is not a prime number, so it is possible that this fails for some
    coefficient. In that case ``None`` is returned.

    Parameters
    ==========

    hm : PolyElement
        polynomial in `\\mathbb Z[t_1, \\ldots, t_k][x, z]`
    m : Integer
        modulus, not necessarily prime
    ring : PolyRing
        `\\mathbb Q[t_1, \\ldots, t_k][x, z]`, `h` will be an element of this
        ring

    Returns
    =======

    h : PolyElement
        reconstructed polynomial in `\\mathbb Q[t_1, \\ldots, t_k][x, z]` or
        ``None``

    See also
    ========

    _integer_rational_reconstruction

    """
def _func_field_modgcd_m(f, g, minpoly):
    """
    Compute the GCD of two polynomials in
    `\\mathbb Q(t_1, \\ldots, t_k)[z]/(m_{\\alpha}(z))[x]` using a modular
    algorithm.

    The algorithm computes the GCD of two polynomials `f` and `g` by
    calculating the GCD in
    `\\mathbb Z_p(t_1, \\ldots, t_k)[z] / (\\check m_{\\alpha}(z))[x]` for
    suitable primes `p` and the primitive associate `\\check m_{\\alpha}(z)`
    of `m_{\\alpha}(z)`. Then the coefficients are reconstructed with the
    Chinese Remainder Theorem and Rational Reconstruction. To compute the
    GCD over `\\mathbb Z_p(t_1, \\ldots, t_k)[z] / (\\check m_{\\alpha})[x]`,
    the recursive subroutine ``_func_field_modgcd_p`` is used. To verify the
    result in `\\mathbb Q(t_1, \\ldots, t_k)[z] / (m_{\\alpha}(z))[x]`, a
    fraction free trial division is used.

    Parameters
    ==========

    f, g : PolyElement
        polynomials in `\\mathbb Z[t_1, \\ldots, t_k][x, z]`
    minpoly : PolyElement
        irreducible polynomial in `\\mathbb Z[t_1, \\ldots, t_k][z]`

    Returns
    =======

    h : PolyElement
        the primitive associate in `\\mathbb Z[t_1, \\ldots, t_k][x, z]` of
        the GCD of `f` and `g`

    Examples
    ========

    >>> from sympy.polys.modulargcd import _func_field_modgcd_m
    >>> from sympy.polys import ring, ZZ

    >>> R, x, z = ring('x, z', ZZ)
    >>> minpoly = (z**2 - 2).drop(0)

    >>> f = x**2 + 2*x*z + 2
    >>> g = x + z
    >>> _func_field_modgcd_m(f, g, minpoly)
    x + z

    >>> D, t = ring('t', ZZ)
    >>> R, x, z = ring('x, z', D)
    >>> minpoly = (z**2-3).drop(0)

    >>> f = x**2 + (t + 1)*x*z + 3*t
    >>> g = x*z + 3*t
    >>> _func_field_modgcd_m(f, g, minpoly)
    x + t*z

    References
    ==========

    1. [Hoeij04]_

    See also
    ========

    _func_field_modgcd_p

    """
def _to_ZZ_poly(f, ring):
    """
    Compute an associate of a polynomial
    `f \\in \\mathbb Q(\\alpha)[x_0, \\ldots, x_{n-1}]` in
    `\\mathbb Z[x_1, \\ldots, x_{n-1}][z] / (\\check m_{\\alpha}(z))[x_0]`,
    where `\\check m_{\\alpha}(z) \\in \\mathbb Z[z]` is the primitive associate
    of the minimal polynomial `m_{\\alpha}(z)` of `\\alpha` over
    `\\mathbb Q`.

    Parameters
    ==========

    f : PolyElement
        polynomial in `\\mathbb Q(\\alpha)[x_0, \\ldots, x_{n-1}]`
    ring : PolyRing
        `\\mathbb Z[x_1, \\ldots, x_{n-1}][x_0, z]`

    Returns
    =======

    f_ : PolyElement
        associate of `f` in
        `\\mathbb Z[x_1, \\ldots, x_{n-1}][x_0, z]`

    """
def _to_ANP_poly(f, ring):
    """
    Convert a polynomial
    `f \\in \\mathbb Z[x_1, \\ldots, x_{n-1}][z]/(\\check m_{\\alpha}(z))[x_0]`
    to a polynomial in `\\mathbb Q(\\alpha)[x_0, \\ldots, x_{n-1}]`,
    where `\\check m_{\\alpha}(z) \\in \\mathbb Z[z]` is the primitive associate
    of the minimal polynomial `m_{\\alpha}(z)` of `\\alpha` over
    `\\mathbb Q`.

    Parameters
    ==========

    f : PolyElement
        polynomial in `\\mathbb Z[x_1, \\ldots, x_{n-1}][x_0, z]`
    ring : PolyRing
        `\\mathbb Q(\\alpha)[x_0, \\ldots, x_{n-1}]`

    Returns
    =======

    f_ : PolyElement
        polynomial in `\\mathbb Q(\\alpha)[x_0, \\ldots, x_{n-1}]`

    """
def _minpoly_from_dense(minpoly, ring):
    """
    Change representation of the minimal polynomial from ``DMP`` to
    ``PolyElement`` for a given ring.
    """
def _primitive_in_x0(f):
    """
    Compute the content in `x_0` and the primitive part of a polynomial `f`
    in
    `\\mathbb Q(\\alpha)[x_0, x_1, \\ldots, x_{n-1}] \\cong \\mathbb Q(\\alpha)[x_1, \\ldots, x_{n-1}][x_0]`.
    """
def func_field_modgcd(f, g):
    """
    Compute the GCD of two polynomials `f` and `g` in
    `\\mathbb Q(\\alpha)[x_0, \\ldots, x_{n-1}]` using a modular algorithm.

    The algorithm first computes the primitive associate
    `\\check m_{\\alpha}(z)` of the minimal polynomial `m_{\\alpha}` in
    `\\mathbb{Z}[z]` and the primitive associates of `f` and `g` in
    `\\mathbb{Z}[x_1, \\ldots, x_{n-1}][z]/(\\check m_{\\alpha})[x_0]`. Then it
    computes the GCD in
    `\\mathbb Q(x_1, \\ldots, x_{n-1})[z]/(m_{\\alpha}(z))[x_0]`.
    This is done by calculating the GCD in
    `\\mathbb{Z}_p(x_1, \\ldots, x_{n-1})[z]/(\\check m_{\\alpha}(z))[x_0]` for
    suitable primes `p` and then reconstructing the coefficients with the
    Chinese Remainder Theorem and Rational Reconstuction. The GCD over
    `\\mathbb{Z}_p(x_1, \\ldots, x_{n-1})[z]/(\\check m_{\\alpha}(z))[x_0]` is
    computed with a recursive subroutine, which evaluates the polynomials at
    `x_{n-1} = a` for suitable evaluation points `a \\in \\mathbb Z_p` and
    then calls itself recursively until the ground domain does no longer
    contain any parameters. For
    `\\mathbb{Z}_p[z]/(\\check m_{\\alpha}(z))[x_0]` the Euclidean Algorithm is
    used. The results of those recursive calls are then interpolated and
    Rational Function Reconstruction is used to obtain the correct
    coefficients. The results, both in
    `\\mathbb Q(x_1, \\ldots, x_{n-1})[z]/(m_{\\alpha}(z))[x_0]` and
    `\\mathbb{Z}_p(x_1, \\ldots, x_{n-1})[z]/(\\check m_{\\alpha}(z))[x_0]`, are
    verified by a fraction free trial division.

    Apart from the above GCD computation some GCDs in
    `\\mathbb Q(\\alpha)[x_1, \\ldots, x_{n-1}]` have to be calculated,
    because treating the polynomials as univariate ones can result in
    a spurious content of the GCD. For this ``func_field_modgcd`` is
    called recursively.

    Parameters
    ==========

    f, g : PolyElement
        polynomials in `\\mathbb Q(\\alpha)[x_0, \\ldots, x_{n-1}]`

    Returns
    =======

    h : PolyElement
        monic GCD of the polynomials `f` and `g`
    cff : PolyElement
        cofactor of `f`, i.e. `\\frac f h`
    cfg : PolyElement
        cofactor of `g`, i.e. `\\frac g h`

    Examples
    ========

    >>> from sympy.polys.modulargcd import func_field_modgcd
    >>> from sympy.polys import AlgebraicField, QQ, ring
    >>> from sympy import sqrt

    >>> A = AlgebraicField(QQ, sqrt(2))
    >>> R, x = ring('x', A)

    >>> f = x**2 - 2
    >>> g = x + sqrt(2)

    >>> h, cff, cfg = func_field_modgcd(f, g)

    >>> h == x + sqrt(2)
    True
    >>> cff * h == f
    True
    >>> cfg * h == g
    True

    >>> R, x, y = ring('x, y', A)

    >>> f = x**2 + 2*sqrt(2)*x*y + 2*y**2
    >>> g = x + sqrt(2)*y

    >>> h, cff, cfg = func_field_modgcd(f, g)

    >>> h == x + sqrt(2)*y
    True
    >>> cff * h == f
    True
    >>> cfg * h == g
    True

    >>> f = x + sqrt(2)*y
    >>> g = x + y

    >>> h, cff, cfg = func_field_modgcd(f, g)

    >>> h == R.one
    True
    >>> cff * h == f
    True
    >>> cfg * h == g
    True

    References
    ==========

    1. [Hoeij04]_

    """
