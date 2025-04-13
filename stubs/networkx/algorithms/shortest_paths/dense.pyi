from _typeshed import Incomplete

__all__ = ['floyd_warshall', 'floyd_warshall_predecessor_and_distance', 'reconstruct_path', 'floyd_warshall_numpy']

def floyd_warshall_numpy(G, nodelist: Incomplete | None = None, weight: str = 'weight'):
    """Find all-pairs shortest path lengths using Floyd's algorithm.

    This algorithm for finding shortest paths takes advantage of
    matrix representations of a graph and works well for dense
    graphs where all-pairs shortest path lengths are desired.
    The results are returned as a NumPy array, distance[i, j],
    where i and j are the indexes of two nodes in nodelist.
    The entry distance[i, j] is the distance along a shortest
    path from i to j. If no path exists the distance is Inf.

    Parameters
    ----------
    G : NetworkX graph

    nodelist : list, optional (default=G.nodes)
       The rows and columns are ordered by the nodes in nodelist.
       If nodelist is None then the ordering is produced by G.nodes.
       Nodelist should include all nodes in G.

    weight: string, optional (default='weight')
       Edge data key corresponding to the edge weight.

    Returns
    -------
    distance : 2D numpy.ndarray
        A numpy array of shortest path distances between nodes.
        If there is no path between two nodes the value is Inf.

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_weighted_edges_from(
    ...     [(0, 1, 5), (1, 2, 2), (2, 3, -3), (1, 3, 10), (3, 2, 8)]
    ... )
    >>> nx.floyd_warshall_numpy(G)
    array([[ 0.,  5.,  7.,  4.],
           [inf,  0.,  2., -1.],
           [inf, inf,  0., -3.],
           [inf, inf,  8.,  0.]])

    Notes
    -----
    Floyd's algorithm is appropriate for finding shortest paths in
    dense graphs or graphs with negative weights when Dijkstra's
    algorithm fails. This algorithm can still fail if there are negative
    cycles. It has running time $O(n^3)$ with running space of $O(n^2)$.

    Raises
    ------
    NetworkXError
        If nodelist is not a list of the nodes in G.
    """
def floyd_warshall_predecessor_and_distance(G, weight: str = 'weight'):
    '''Find all-pairs shortest path lengths using Floyd\'s algorithm.

    Parameters
    ----------
    G : NetworkX graph

    weight: string, optional (default= \'weight\')
       Edge data key corresponding to the edge weight.

    Returns
    -------
    predecessor,distance : dictionaries
       Dictionaries, keyed by source and target, of predecessors and distances
       in the shortest path.

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_weighted_edges_from(
    ...     [
    ...         ("s", "u", 10),
    ...         ("s", "x", 5),
    ...         ("u", "v", 1),
    ...         ("u", "x", 2),
    ...         ("v", "y", 1),
    ...         ("x", "u", 3),
    ...         ("x", "v", 5),
    ...         ("x", "y", 2),
    ...         ("y", "s", 7),
    ...         ("y", "v", 6),
    ...     ]
    ... )
    >>> predecessors, _ = nx.floyd_warshall_predecessor_and_distance(G)
    >>> print(nx.reconstruct_path("s", "v", predecessors))
    [\'s\', \'x\', \'u\', \'v\']

    Notes
    -----
    Floyd\'s algorithm is appropriate for finding shortest paths
    in dense graphs or graphs with negative weights when Dijkstra\'s algorithm
    fails.  This algorithm can still fail if there are negative cycles.
    It has running time $O(n^3)$ with running space of $O(n^2)$.

    See Also
    --------
    floyd_warshall
    floyd_warshall_numpy
    all_pairs_shortest_path
    all_pairs_shortest_path_length
    '''
def reconstruct_path(source, target, predecessors):
    """Reconstruct a path from source to target using the predecessors
    dict as returned by floyd_warshall_predecessor_and_distance

    Parameters
    ----------
    source : node
       Starting node for path

    target : node
       Ending node for path

    predecessors: dictionary
       Dictionary, keyed by source and target, of predecessors in the
       shortest path, as returned by floyd_warshall_predecessor_and_distance

    Returns
    -------
    path : list
       A list of nodes containing the shortest path from source to target

       If source and target are the same, an empty list is returned

    Notes
    -----
    This function is meant to give more applicability to the
    floyd_warshall_predecessor_and_distance function

    See Also
    --------
    floyd_warshall_predecessor_and_distance
    """
def floyd_warshall(G, weight: str = 'weight'):
    '''Find all-pairs shortest path lengths using Floyd\'s algorithm.

    Parameters
    ----------
    G : NetworkX graph

    weight: string, optional (default= \'weight\')
       Edge data key corresponding to the edge weight.


    Returns
    -------
    distance : dict
       A dictionary,  keyed by source and target, of shortest paths distances
       between nodes.

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_weighted_edges_from(
    ...     [(0, 1, 5), (1, 2, 2), (2, 3, -3), (1, 3, 10), (3, 2, 8)]
    ... )
    >>> fw = nx.floyd_warshall(G, weight="weight")
    >>> results = {a: dict(b) for a, b in fw.items()}
    >>> print(results)
    {0: {0: 0, 1: 5, 2: 7, 3: 4}, 1: {1: 0, 2: 2, 3: -1, 0: inf}, 2: {2: 0, 3: -3, 0: inf, 1: inf}, 3: {3: 0, 2: 8, 0: inf, 1: inf}}

    Notes
    -----
    Floyd\'s algorithm is appropriate for finding shortest paths
    in dense graphs or graphs with negative weights when Dijkstra\'s algorithm
    fails.  This algorithm can still fail if there are negative cycles.
    It has running time $O(n^3)$ with running space of $O(n^2)$.

    See Also
    --------
    floyd_warshall_predecessor_and_distance
    floyd_warshall_numpy
    all_pairs_shortest_path
    all_pairs_shortest_path_length
    '''
