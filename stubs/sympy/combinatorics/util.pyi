from _typeshed import Incomplete
from sympy.combinatorics.permutations import Permutation as Permutation, _af_invert as _af_invert, _af_rmul as _af_rmul

rmul: Incomplete
_af_new: Incomplete

def _base_ordering(base, degree):
    '''
    Order `\\{0, 1, \\dots, n-1\\}` so that base points come first and in order.

    Parameters
    ==========

    base : the base
    degree : the degree of the associated permutation group

    Returns
    =======

    A list ``base_ordering`` such that ``base_ordering[point]`` is the
    number of ``point`` in the ordering.

    Examples
    ========

    >>> from sympy.combinatorics import SymmetricGroup
    >>> from sympy.combinatorics.util import _base_ordering
    >>> S = SymmetricGroup(4)
    >>> S.schreier_sims()
    >>> _base_ordering(S.base, S.degree)
    [0, 1, 2, 3]

    Notes
    =====

    This is used in backtrack searches, when we define a relation `\\ll` on
    the underlying set for a permutation group of degree `n`,
    `\\{0, 1, \\dots, n-1\\}`, so that if `(b_1, b_2, \\dots, b_k)` is a base we
    have `b_i \\ll b_j` whenever `i<j` and `b_i \\ll a` for all
    `i\\in\\{1,2, \\dots, k\\}` and `a` is not in the base. The idea is developed
    and applied to backtracking algorithms in [1], pp.108-132. The points
    that are not in the base are taken in increasing order.

    References
    ==========

    .. [1] Holt, D., Eick, B., O\'Brien, E.
           "Handbook of computational group theory"

    '''
def _check_cycles_alt_sym(perm):
    """
    Checks for cycles of prime length p with n/2 < p < n-2.

    Explanation
    ===========

    Here `n` is the degree of the permutation. This is a helper function for
    the function is_alt_sym from sympy.combinatorics.perm_groups.

    Examples
    ========

    >>> from sympy.combinatorics.util import _check_cycles_alt_sym
    >>> from sympy.combinatorics import Permutation
    >>> a = Permutation([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12]])
    >>> _check_cycles_alt_sym(a)
    False
    >>> b = Permutation([[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10]])
    >>> _check_cycles_alt_sym(b)
    True

    See Also
    ========

    sympy.combinatorics.perm_groups.PermutationGroup.is_alt_sym

    """
def _distribute_gens_by_base(base, gens):
    """
    Distribute the group elements ``gens`` by membership in basic stabilizers.

    Explanation
    ===========

    Notice that for a base `(b_1, b_2, \\dots, b_k)`, the basic stabilizers
    are defined as `G^{(i)} = G_{b_1, \\dots, b_{i-1}}` for
    `i \\in\\{1, 2, \\dots, k\\}`.

    Parameters
    ==========

    base : a sequence of points in `\\{0, 1, \\dots, n-1\\}`
    gens : a list of elements of a permutation group of degree `n`.

    Returns
    =======
    list
        List of length `k`, where `k` is the length of *base*. The `i`-th entry
        contains those elements in *gens* which fix the first `i` elements of
        *base* (so that the `0`-th entry is equal to *gens* itself). If no
        element fixes the first `i` elements of *base*, the `i`-th element is
        set to a list containing the identity element.

    Examples
    ========

    >>> from sympy.combinatorics.named_groups import DihedralGroup
    >>> from sympy.combinatorics.util import _distribute_gens_by_base
    >>> D = DihedralGroup(3)
    >>> D.schreier_sims()
    >>> D.strong_gens
    [(0 1 2), (0 2), (1 2)]
    >>> D.base
    [0, 1]
    >>> _distribute_gens_by_base(D.base, D.strong_gens)
    [[(0 1 2), (0 2), (1 2)],
     [(1 2)]]

    See Also
    ========

    _strong_gens_from_distr, _orbits_transversals_from_bsgs,
    _handle_precomputed_bsgs

    """
def _handle_precomputed_bsgs(base, strong_gens, transversals: Incomplete | None = None, basic_orbits: Incomplete | None = None, strong_gens_distr: Incomplete | None = None):
    """
    Calculate BSGS-related structures from those present.

    Explanation
    ===========

    The base and strong generating set must be provided; if any of the
    transversals, basic orbits or distributed strong generators are not
    provided, they will be calculated from the base and strong generating set.

    Parameters
    ==========

    base : the base
    strong_gens : the strong generators
    transversals : basic transversals
    basic_orbits : basic orbits
    strong_gens_distr : strong generators distributed by membership in basic stabilizers

    Returns
    =======

    (transversals, basic_orbits, strong_gens_distr)
        where *transversals* are the basic transversals, *basic_orbits* are the
        basic orbits, and *strong_gens_distr* are the strong generators distributed
        by membership in basic stabilizers.

    Examples
    ========

    >>> from sympy.combinatorics.named_groups import DihedralGroup
    >>> from sympy.combinatorics.util import _handle_precomputed_bsgs
    >>> D = DihedralGroup(3)
    >>> D.schreier_sims()
    >>> _handle_precomputed_bsgs(D.base, D.strong_gens,
    ... basic_orbits=D.basic_orbits)
    ([{0: (2), 1: (0 1 2), 2: (0 2)}, {1: (2), 2: (1 2)}], [[0, 1, 2], [1, 2]], [[(0 1 2), (0 2), (1 2)], [(1 2)]])

    See Also
    ========

    _orbits_transversals_from_bsgs, _distribute_gens_by_base

    """
def _orbits_transversals_from_bsgs(base, strong_gens_distr, transversals_only: bool = False, slp: bool = False):
    """
    Compute basic orbits and transversals from a base and strong generating set.

    Explanation
    ===========

    The generators are provided as distributed across the basic stabilizers.
    If the optional argument ``transversals_only`` is set to True, only the
    transversals are returned.

    Parameters
    ==========

    base : The base.
    strong_gens_distr : Strong generators distributed by membership in basic stabilizers.
    transversals_only : bool, default: False
        A flag switching between returning only the
        transversals and both orbits and transversals.
    slp : bool, default: False
        If ``True``, return a list of dictionaries containing the
        generator presentations of the elements of the transversals,
        i.e. the list of indices of generators from ``strong_gens_distr[i]``
        such that their product is the relevant transversal element.

    Examples
    ========

    >>> from sympy.combinatorics import SymmetricGroup
    >>> from sympy.combinatorics.util import _distribute_gens_by_base
    >>> S = SymmetricGroup(3)
    >>> S.schreier_sims()
    >>> strong_gens_distr = _distribute_gens_by_base(S.base, S.strong_gens)
    >>> (S.base, strong_gens_distr)
    ([0, 1], [[(0 1 2), (2)(0 1), (1 2)], [(1 2)]])

    See Also
    ========

    _distribute_gens_by_base, _handle_precomputed_bsgs

    """
def _remove_gens(base, strong_gens, basic_orbits: Incomplete | None = None, strong_gens_distr: Incomplete | None = None):
    '''
    Remove redundant generators from a strong generating set.

    Parameters
    ==========

    base : a base
    strong_gens : a strong generating set relative to *base*
    basic_orbits : basic orbits
    strong_gens_distr : strong generators distributed by membership in basic stabilizers

    Returns
    =======

    A strong generating set with respect to ``base`` which is a subset of
    ``strong_gens``.

    Examples
    ========

    >>> from sympy.combinatorics import SymmetricGroup
    >>> from sympy.combinatorics.util import _remove_gens
    >>> from sympy.combinatorics.testutil import _verify_bsgs
    >>> S = SymmetricGroup(15)
    >>> base, strong_gens = S.schreier_sims_incremental()
    >>> new_gens = _remove_gens(base, strong_gens)
    >>> len(new_gens)
    14
    >>> _verify_bsgs(S, base, new_gens)
    True

    Notes
    =====

    This procedure is outlined in [1],p.95.

    References
    ==========

    .. [1] Holt, D., Eick, B., O\'Brien, E.
           "Handbook of computational group theory"

    '''
def _strip(g, base, orbits, transversals):
    '''
    Attempt to decompose a permutation using a (possibly partial) BSGS
    structure.

    Explanation
    ===========

    This is done by treating the sequence ``base`` as an actual base, and
    the orbits ``orbits`` and transversals ``transversals`` as basic orbits and
    transversals relative to it.

    This process is called "sifting". A sift is unsuccessful when a certain
    orbit element is not found or when after the sift the decomposition
    does not end with the identity element.

    The argument ``transversals`` is a list of dictionaries that provides
    transversal elements for the orbits ``orbits``.

    Parameters
    ==========

    g : permutation to be decomposed
    base : sequence of points
    orbits : list
        A list in which the ``i``-th entry is an orbit of ``base[i]``
        under some subgroup of the pointwise stabilizer of `
        `base[0], base[1], ..., base[i - 1]``. The groups themselves are implicit
        in this function since the only information we need is encoded in the orbits
        and transversals
    transversals : list
        A list of orbit transversals associated with the orbits *orbits*.

    Examples
    ========

    >>> from sympy.combinatorics import Permutation, SymmetricGroup
    >>> from sympy.combinatorics.util import _strip
    >>> S = SymmetricGroup(5)
    >>> S.schreier_sims()
    >>> g = Permutation([0, 2, 3, 1, 4])
    >>> _strip(g, S.base, S.basic_orbits, S.basic_transversals)
    ((4), 5)

    Notes
    =====

    The algorithm is described in [1],pp.89-90. The reason for returning
    both the current state of the element being decomposed and the level
    at which the sifting ends is that they provide important information for
    the randomized version of the Schreier-Sims algorithm.

    References
    ==========

    .. [1] Holt, D., Eick, B., O\'Brien, E."Handbook of computational group theory"

    See Also
    ========

    sympy.combinatorics.perm_groups.PermutationGroup.schreier_sims
    sympy.combinatorics.perm_groups.PermutationGroup.schreier_sims_random

    '''
def _strip_af(h, base, orbits, transversals, j, slp=[], slps={}):
    """
    optimized _strip, with h, transversals and result in array form
    if the stripped elements is the identity, it returns False, base_len + 1

    j    h[base[i]] == base[i] for i <= j

    """
def _strong_gens_from_distr(strong_gens_distr):
    """
    Retrieve strong generating set from generators of basic stabilizers.

    This is just the union of the generators of the first and second basic
    stabilizers.

    Parameters
    ==========

    strong_gens_distr : strong generators distributed by membership in basic stabilizers

    Examples
    ========

    >>> from sympy.combinatorics import SymmetricGroup
    >>> from sympy.combinatorics.util import (_strong_gens_from_distr,
    ... _distribute_gens_by_base)
    >>> S = SymmetricGroup(3)
    >>> S.schreier_sims()
    >>> S.strong_gens
    [(0 1 2), (2)(0 1), (1 2)]
    >>> strong_gens_distr = _distribute_gens_by_base(S.base, S.strong_gens)
    >>> _strong_gens_from_distr(strong_gens_distr)
    [(0 1 2), (2)(0 1), (1 2)]

    See Also
    ========

    _distribute_gens_by_base

    """
