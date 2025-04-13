from _typeshed import Incomplete

__all__ = ['min_weighted_dominating_set', 'min_edge_dominating_set']

def min_weighted_dominating_set(G, weight: Incomplete | None = None):
    """Returns a dominating set that approximates the minimum weight node
    dominating set.

    Parameters
    ----------
    G : NetworkX graph
        Undirected graph.

    weight : string
        The node attribute storing the weight of an node. If provided,
        the node attribute with this key must be a number for each
        node. If not provided, each node is assumed to have weight one.

    Returns
    -------
    min_weight_dominating_set : set
        A set of nodes, the sum of whose weights is no more than `(\\log
        w(V)) w(V^*)`, where `w(V)` denotes the sum of the weights of
        each node in the graph and `w(V^*)` denotes the sum of the
        weights of each node in the minimum weight dominating set.

    Examples
    --------
    >>> G = nx.Graph([(0, 1), (0, 4), (1, 4), (1, 2), (2, 3), (3, 4), (2, 5)])
    >>> nx.approximation.min_weighted_dominating_set(G)
    {1, 2, 4}

    Raises
    ------
    NetworkXNotImplemented
        If G is directed.

    Notes
    -----
    This algorithm computes an approximate minimum weighted dominating
    set for the graph `G`. The returned solution has weight `(\\log
    w(V)) w(V^*)`, where `w(V)` denotes the sum of the weights of each
    node in the graph and `w(V^*)` denotes the sum of the weights of
    each node in the minimum weight dominating set for the graph.

    This implementation of the algorithm runs in $O(m)$ time, where $m$
    is the number of edges in the graph.

    References
    ----------
    .. [1] Vazirani, Vijay V.
           *Approximation Algorithms*.
           Springer Science & Business Media, 2001.

    """
def min_edge_dominating_set(G):
    """Returns minimum cardinality edge dominating set.

    Parameters
    ----------
    G : NetworkX graph
      Undirected graph

    Returns
    -------
    min_edge_dominating_set : set
      Returns a set of dominating edges whose size is no more than 2 * OPT.

    Examples
    --------
    >>> G = nx.petersen_graph()
    >>> nx.approximation.min_edge_dominating_set(G)
    {(0, 1), (4, 9), (6, 8), (5, 7), (2, 3)}

    Raises
    ------
    ValueError
        If the input graph `G` is empty.

    Notes
    -----
    The algorithm computes an approximate solution to the edge dominating set
    problem. The result is no more than 2 * OPT in terms of size of the set.
    Runtime of the algorithm is $O(|E|)$.
    """
