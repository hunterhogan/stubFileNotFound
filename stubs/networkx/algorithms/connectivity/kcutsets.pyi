from _typeshed import Incomplete
from collections.abc import Generator
from networkx.algorithms.flow import edmonds_karp

__all__ = ['all_node_cuts']

default_flow_func = edmonds_karp

def all_node_cuts(G, k: Incomplete | None = None, flow_func: Incomplete | None = None) -> Generator[Incomplete, Incomplete]:
    """Returns all minimum k cutsets of an undirected graph G.

    This implementation is based on Kanevsky's algorithm [1]_ for finding all
    minimum-size node cut-sets of an undirected graph G; ie the set (or sets)
    of nodes of cardinality equal to the node connectivity of G. Thus if
    removed, would break G into two or more connected components.

    Parameters
    ----------
    G : NetworkX graph
        Undirected graph

    k : Integer
        Node connectivity of the input graph. If k is None, then it is
        computed. Default value: None.

    flow_func : function
        Function to perform the underlying flow computations. Default value is
        :func:`~networkx.algorithms.flow.edmonds_karp`. This function performs
        better in sparse graphs with right tailed degree distributions.
        :func:`~networkx.algorithms.flow.shortest_augmenting_path` will
        perform better in denser graphs.


    Returns
    -------
    cuts : a generator of node cutsets
        Each node cutset has cardinality equal to the node connectivity of
        the input graph.

    Examples
    --------
    >>> # A two-dimensional grid graph has 4 cutsets of cardinality 2
    >>> G = nx.grid_2d_graph(5, 5)
    >>> cutsets = list(nx.all_node_cuts(G))
    >>> len(cutsets)
    4
    >>> all(2 == len(cutset) for cutset in cutsets)
    True
    >>> nx.node_connectivity(G)
    2

    Notes
    -----
    This implementation is based on the sequential algorithm for finding all
    minimum-size separating vertex sets in a graph [1]_. The main idea is to
    compute minimum cuts using local maximum flow computations among a set
    of nodes of highest degree and all other non-adjacent nodes in the Graph.
    Once we find a minimum cut, we add an edge between the high degree
    node and the target node of the local maximum flow computation to make
    sure that we will not find that minimum cut again.

    See also
    --------
    node_connectivity
    edmonds_karp
    shortest_augmenting_path

    References
    ----------
    .. [1]  Kanevsky, A. (1993). Finding all minimum-size separating vertex
            sets in a graph. Networks 23(6), 533--541.
            http://onlinelibrary.wiley.com/doi/10.1002/net.3230230604/abstract

    """
