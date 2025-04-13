from _typeshed import Incomplete

__all__ = ['hnm_harary_graph', 'hkn_harary_graph']

def hnm_harary_graph(n, m, create_using: Incomplete | None = None):
    '''Returns the Harary graph with given numbers of nodes and edges.

    The Harary graph $H_{n,m}$ is the graph that maximizes node connectivity
    with $n$ nodes and $m$ edges.

    This maximum node connectivity is known to be floor($2m/n$). [1]_

    Parameters
    ----------
    n: integer
       The number of nodes the generated graph is to contain

    m: integer
       The number of edges the generated graph is to contain

    create_using : NetworkX graph constructor, optional Graph type
     to create (default=nx.Graph). If graph instance, then cleared
     before populated.

    Returns
    -------
    NetworkX graph
        The Harary graph $H_{n,m}$.

    See Also
    --------
    hkn_harary_graph

    Notes
    -----
    This algorithm runs in $O(m)$ time.
    It is implemented by following the Reference [2]_.

    References
    ----------
    .. [1] F. T. Boesch, A. Satyanarayana, and C. L. Suffel,
       "A Survey of Some Network Reliability Analysis and Synthesis Results,"
       Networks, pp. 99-107, 2009.

    .. [2] Harary, F. "The Maximum Connectivity of a Graph."
       Proc. Nat. Acad. Sci. USA 48, 1142-1146, 1962.
    '''
def hkn_harary_graph(k, n, create_using: Incomplete | None = None):
    '''Returns the Harary graph with given node connectivity and node number.

    The Harary graph $H_{k,n}$ is the graph that minimizes the number of
    edges needed with given node connectivity $k$ and node number $n$.

    This smallest number of edges is known to be ceil($kn/2$) [1]_.

    Parameters
    ----------
    k: integer
       The node connectivity of the generated graph

    n: integer
       The number of nodes the generated graph is to contain

    create_using : NetworkX graph constructor, optional Graph type
     to create (default=nx.Graph). If graph instance, then cleared
     before populated.

    Returns
    -------
    NetworkX graph
        The Harary graph $H_{k,n}$.

    See Also
    --------
    hnm_harary_graph

    Notes
    -----
    This algorithm runs in $O(kn)$ time.
    It is implemented by following the Reference [2]_.

    References
    ----------
    .. [1] Weisstein, Eric W. "Harary Graph." From MathWorld--A Wolfram Web
     Resource. http://mathworld.wolfram.com/HararyGraph.html.

    .. [2] Harary, F. "The Maximum Connectivity of a Graph."
      Proc. Nat. Acad. Sci. USA 48, 1142-1146, 1962.
    '''
