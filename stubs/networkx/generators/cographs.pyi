from _typeshed import Incomplete

__all__ = ['random_cograph']

def random_cograph(n, seed: Incomplete | None = None):
    '''Returns a random cograph with $2 ^ n$ nodes.

    A cograph is a graph containing no path on four vertices.
    Cographs or $P_4$-free graphs can be obtained from a single vertex
    by disjoint union and complementation operations.

    This generator starts off from a single vertex and performs disjoint
    union and full join operations on itself.
    The decision on which operation will take place is random.

    Parameters
    ----------
    n : int
        The order of the cograph.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G : A random graph containing no path on four vertices.

    See Also
    --------
    full_join
    union

    References
    ----------
    .. [1] D.G. Corneil, H. Lerchs, L.Stewart Burlingham,
       "Complement reducible graphs",
       Discrete Applied Mathematics, Volume 3, Issue 3, 1981, Pages 163-174,
       ISSN 0166-218X.
    '''
