from _typeshed import Incomplete

__all__ = ['uniform_random_intersection_graph', 'k_random_intersection_graph', 'general_random_intersection_graph']

def uniform_random_intersection_graph(n, m, p, seed: Incomplete | None = None):
    """Returns a uniform random intersection graph.

    Parameters
    ----------
    n : int
        The number of nodes in the first bipartite set (nodes)
    m : int
        The number of nodes in the second bipartite set (attributes)
    p : float
        Probability of connecting nodes between bipartite sets
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    See Also
    --------
    gnp_random_graph

    References
    ----------
    .. [1] K.B. Singer-Cohen, Random Intersection Graphs, 1995,
       PhD thesis, Johns Hopkins University
    .. [2] Fill, J. A., Scheinerman, E. R., and Singer-Cohen, K. B.,
       Random intersection graphs when m = !(n):
       An equivalence theorem relating the evolution of the g(n, m, p)
       and g(n, p) models. Random Struct. Algorithms 16, 2 (2000), 156–176.
    """
def k_random_intersection_graph(n, m, k, seed: Incomplete | None = None):
    """Returns a intersection graph with randomly chosen attribute sets for
    each node that are of equal size (k).

    Parameters
    ----------
    n : int
        The number of nodes in the first bipartite set (nodes)
    m : int
        The number of nodes in the second bipartite set (attributes)
    k : float
        Size of attribute set to assign to each node.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    See Also
    --------
    gnp_random_graph, uniform_random_intersection_graph

    References
    ----------
    .. [1] Godehardt, E., and Jaworski, J.
       Two models of random intersection graphs and their applications.
       Electronic Notes in Discrete Mathematics 10 (2001), 129--132.
    """
def general_random_intersection_graph(n, m, p, seed: Incomplete | None = None):
    """Returns a random intersection graph with independent probabilities
    for connections between node and attribute sets.

    Parameters
    ----------
    n : int
        The number of nodes in the first bipartite set (nodes)
    m : int
        The number of nodes in the second bipartite set (attributes)
    p : list of floats of length m
        Probabilities for connecting nodes to each attribute
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    See Also
    --------
    gnp_random_graph, uniform_random_intersection_graph

    References
    ----------
    .. [1] Nikoletseas, S. E., Raptopoulos, C., and Spirakis, P. G.
       The existence and efficient construction of large independent sets
       in general random intersection graphs. In ICALP (2004), J. D´ıaz,
       J. Karhum¨aki, A. Lepist¨o, and D. Sannella, Eds., vol. 3142
       of Lecture Notes in Computer Science, Springer, pp. 1029–1040.
    """
