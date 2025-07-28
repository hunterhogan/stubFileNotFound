from sympy.utilities import public

__all__ = ['roots']

@public
def roots(f, *gens, auto: bool = True, cubics: bool = True, trig: bool = False, quartics: bool = True, quintics: bool = False, multiple: bool = False, filter=None, predicate=None, strict: bool = False, **flags):
    """
    Computes symbolic roots of a univariate polynomial.

    Given a univariate polynomial f with symbolic coefficients (or
    a list of the polynomial's coefficients), returns a dictionary
    with its roots and their multiplicities.

    Only roots expressible via radicals will be returned.  To get
    a complete set of roots use RootOf class or numerical methods
    instead. By default cubic and quartic formulas are used in
    the algorithm. To disable them because of unreadable output
    set ``cubics=False`` or ``quartics=False`` respectively. If cubic
    roots are real but are expressed in terms of complex numbers
    (casus irreducibilis [1]) the ``trig`` flag can be set to True to
    have the solutions returned in terms of cosine and inverse cosine
    functions.

    To get roots from a specific domain set the ``filter`` flag with
    one of the following specifiers: Z, Q, R, I, C. By default all
    roots are returned (this is equivalent to setting ``filter='C'``).

    By default a dictionary is returned giving a compact result in
    case of multiple roots.  However to get a list containing all
    those roots set the ``multiple`` flag to True; the list will
    have identical roots appearing next to each other in the result.
    (For a given Poly, the all_roots method will give the roots in
    sorted numerical order.)

    If the ``strict`` flag is True, ``UnsolvableFactorError`` will be
    raised if the roots found are known to be incomplete (because
    some roots are not expressible in radicals).

    Examples
    ========

    >>> from sympy import Poly, roots, degree
    >>> from sympy.abc import x, y

    >>> roots(x**2 - 1, x)
    {-1: 1, 1: 1}

    >>> p = Poly(x**2-1, x)
    >>> roots(p)
    {-1: 1, 1: 1}

    >>> p = Poly(x**2-y, x, y)

    >>> roots(Poly(p, x))
    {-sqrt(y): 1, sqrt(y): 1}

    >>> roots(x**2 - y, x)
    {-sqrt(y): 1, sqrt(y): 1}

    >>> roots([1, 0, -1])
    {-1: 1, 1: 1}

    ``roots`` will only return roots expressible in radicals. If
    the given polynomial has some or all of its roots inexpressible in
    radicals, the result of ``roots`` will be incomplete or empty
    respectively.

    Example where result is incomplete:

    >>> roots((x-1)*(x**5-x+1), x)
    {1: 1}

    In this case, the polynomial has an unsolvable quintic factor
    whose roots cannot be expressed by radicals. The polynomial has a
    rational root (due to the factor `(x-1)`), which is returned since
    ``roots`` always finds all rational roots.

    Example where result is empty:

    >>> roots(x**7-3*x**2+1, x)
    {}

    Here, the polynomial has no roots expressible in radicals, so
    ``roots`` returns an empty dictionary.

    The result produced by ``roots`` is complete if and only if the
    sum of the multiplicity of each root is equal to the degree of
    the polynomial. If strict=True, UnsolvableFactorError will be
    raised if the result is incomplete.

    The result can be be checked for completeness as follows:

    >>> f = x**3-2*x**2+1
    >>> sum(roots(f, x).values()) == degree(f, x)
    True
    >>> f = (x-1)*(x**5-x+1)
    >>> sum(roots(f, x).values()) == degree(f, x)
    False


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Cubic_equation#Trigonometric_and_hyperbolic_solutions

    """
