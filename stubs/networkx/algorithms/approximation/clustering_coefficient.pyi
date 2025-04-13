from _typeshed import Incomplete

__all__ = ['average_clustering']

def average_clustering(G, trials: int = 1000, seed: Incomplete | None = None):
    """Estimates the average clustering coefficient of G.

    The local clustering of each node in `G` is the fraction of triangles
    that actually exist over all possible triangles in its neighborhood.
    The average clustering coefficient of a graph `G` is the mean of
    local clusterings.

    This function finds an approximate average clustering coefficient
    for G by repeating `n` times (defined in `trials`) the following
    experiment: choose a node at random, choose two of its neighbors
    at random, and check if they are connected. The approximate
    coefficient is the fraction of triangles found over the number
    of trials [1]_.

    Parameters
    ----------
    G : NetworkX graph

    trials : integer
        Number of trials to perform (default 1000).

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    c : float
        Approximated average clustering coefficient.

    Examples
    --------
    >>> from networkx.algorithms import approximation
    >>> G = nx.erdos_renyi_graph(10, 0.2, seed=10)
    >>> approximation.average_clustering(G, trials=1000, seed=10)
    0.214

    Raises
    ------
    NetworkXNotImplemented
        If G is directed.

    References
    ----------
    .. [1] Schank, Thomas, and Dorothea Wagner. Approximating clustering
       coefficient and transitivity. Universität Karlsruhe, Fakultät für
       Informatik, 2004.
       https://doi.org/10.5445/IR/1000001239

    """
