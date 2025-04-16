from sympy.polys.monomials import monomial_div as monomial_div, monomial_mul as monomial_mul

def matrix_fglm(F, ring, O_to):
    """
    Converts the reduced Groebner basis ``F`` of a zero-dimensional
    ideal w.r.t. ``O_from`` to a reduced Groebner basis
    w.r.t. ``O_to``.

    References
    ==========

    .. [1] J.C. Faugere, P. Gianni, D. Lazard, T. Mora (1994). Efficient
           Computation of Zero-dimensional Groebner Bases by Change of
           Ordering
    """
def _incr_k(m, k): ...
def _identity_matrix(n, domain): ...
def _matrix_mul(M, v): ...
def _update(s, _lambda, P):
    """
    Update ``P`` such that for the updated `P'` `P' v = e_{s}`.
    """
def _representing_matrices(basis, G, ring):
    """
    Compute the matrices corresponding to the linear maps `m \\mapsto
    x_i m` for all variables `x_i`.
    """
def _basis(G, ring):
    """
    Computes a list of monomials which are not divisible by the leading
    monomials wrt to ``O`` of ``G``. These monomials are a basis of
    `K[X_1, \\ldots, X_n]/(G)`.
    """
