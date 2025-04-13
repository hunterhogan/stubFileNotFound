__all__ = ['second_order_centrality']

def second_order_centrality(G, weight: str = 'weight'):
    '''Compute the second order centrality for nodes of G.

    The second order centrality of a given node is the standard deviation of
    the return times to that node of a perpetual random walk on G:

    Parameters
    ----------
    G : graph
      A NetworkX connected and undirected graph.

    weight : string or None, optional (default="weight")
        The name of an edge attribute that holds the numerical value
        used as a weight. If None then each edge has weight 1.

    Returns
    -------
    nodes : dictionary
       Dictionary keyed by node with second order centrality as the value.

    Examples
    --------
    >>> G = nx.star_graph(10)
    >>> soc = nx.second_order_centrality(G)
    >>> print(sorted(soc.items(), key=lambda x: x[1])[0][0])  # pick first id
    0

    Raises
    ------
    NetworkXException
        If the graph G is empty, non connected or has negative weights.

    See Also
    --------
    betweenness_centrality

    Notes
    -----
    Lower values of second order centrality indicate higher centrality.

    The algorithm is from Kermarrec, Le Merrer, Sericola and Trédan [1]_.

    This code implements the analytical version of the algorithm, i.e.,
    there is no simulation of a random walk process involved. The random walk
    is here unbiased (corresponding to eq 6 of the paper [1]_), thus the
    centrality values are the standard deviations for random walk return times
    on the transformed input graph G (equal in-degree at each nodes by adding
    self-loops).

    Complexity of this implementation, made to run locally on a single machine,
    is O(n^3), with n the size of G, which makes it viable only for small
    graphs.

    References
    ----------
    .. [1] Anne-Marie Kermarrec, Erwan Le Merrer, Bruno Sericola, Gilles Trédan
       "Second order centrality: Distributed assessment of nodes criticity in
       complex networks", Elsevier Computer Communications 34(5):619-628, 2011.
    '''
