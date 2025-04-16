from sympy.polys.numberfields.galois_resolvents import GaloisGroupException

__all__ = ['galois_group']

class MaxTriesException(GaloisGroupException): ...

def galois_group(f, *gens, by_name: bool = False, max_tries: int = 30, randomize: bool = False, **args):
    """
    Compute the Galois group for polynomials *f* up to degree 6.

    Examples
    ========

    >>> from sympy import galois_group
    >>> from sympy.abc import x
    >>> f = x**4 + 1
    >>> G, alt = galois_group(f)
    >>> print(G)
    PermutationGroup([
    (0 1)(2 3),
    (0 2)(1 3)])

    The group is returned along with a boolean, indicating whether it is
    contained in the alternating group $A_n$, where $n$ is the degree of *T*.
    Along with other group properties, this can help determine which group it
    is:

    >>> alt
    True
    >>> G.order()
    4

    Alternatively, the group can be returned by name:

    >>> G_name, _ = galois_group(f, by_name=True)
    >>> print(G_name)
    S4TransitiveSubgroups.V

    The group itself can then be obtained by calling the name's
    ``get_perm_group()`` method:

    >>> G_name.get_perm_group()
    PermutationGroup([
    (0 1)(2 3),
    (0 2)(1 3)])

    Group names are values of the enum classes
    :py:class:`sympy.combinatorics.galois.S1TransitiveSubgroups`,
    :py:class:`sympy.combinatorics.galois.S2TransitiveSubgroups`,
    etc.

    Parameters
    ==========

    f : Expr
        Irreducible polynomial over :ref:`ZZ` or :ref:`QQ`, whose Galois group
        is to be determined.
    gens : optional list of symbols
        For converting *f* to Poly, and will be passed on to the
        :py:func:`~.poly_from_expr` function.
    by_name : bool, default False
        If ``True``, the Galois group will be returned by name.
        Otherwise it will be returned as a :py:class:`~.PermutationGroup`.
    max_tries : int, default 30
        Make at most this many attempts in those steps that involve
        generating Tschirnhausen transformations.
    randomize : bool, default False
        If ``True``, then use random coefficients when generating Tschirnhausen
        transformations. Otherwise try transformations in a fixed order. Both
        approaches start with small coefficients and degrees and work upward.
    args : optional
        For converting *f* to Poly, and will be passed on to the
        :py:func:`~.poly_from_expr` function.

    Returns
    =======

    Pair ``(G, alt)``
        The first element ``G`` indicates the Galois group. It is an instance
        of one of the :py:class:`sympy.combinatorics.galois.S1TransitiveSubgroups`
        :py:class:`sympy.combinatorics.galois.S2TransitiveSubgroups`, etc. enum
        classes if *by_name* was ``True``, and a :py:class:`~.PermutationGroup`
        if ``False``.

        The second element is a boolean, saying whether the group is contained
        in the alternating group $A_n$ ($n$ the degree of *T*).

    Raises
    ======

    ValueError
        if *f* is of an unsupported degree.

    MaxTriesException
        if could not complete before exceeding *max_tries* in those steps
        that involve generating Tschirnhausen transformations.

    See Also
    ========

    .Poly.galois_group

    """
