__all__ = ['maximal_extendability']

def maximal_extendability(G):
    '''Computes the extendability of a graph.

    The extendability of a graph is defined as the maximum $k$ for which `G`
    is $k$-extendable. Graph `G` is $k$-extendable if and only if `G` has a
    perfect matching and every set of $k$ independent edges can be extended
    to a perfect matching in `G`.

    Parameters
    ----------
    G : NetworkX Graph
        A fully-connected bipartite graph without self-loops

    Returns
    -------
    extendability : int

    Raises
    ------
    NetworkXError
       If the graph `G` is disconnected.
       If the graph `G` is not bipartite.
       If the graph `G` does not contain a perfect matching.
       If the residual graph of `G` is not strongly connected.

    Notes
    -----
    Definition:
    Let `G` be a simple, connected, undirected and bipartite graph with a perfect
    matching M and bipartition (U,V). The residual graph of `G`, denoted by $G_M$,
    is the graph obtained from G by directing the edges of M from V to U and the
    edges that do not belong to M from U to V.

    Lemma [1]_ :
    Let M be a perfect matching of `G`. `G` is $k$-extendable if and only if its residual
    graph $G_M$ is strongly connected and there are $k$ vertex-disjoint directed
    paths between every vertex of U and every vertex of V.

    Assuming that input graph `G` is undirected, simple, connected, bipartite and contains
    a perfect matching M, this function constructs the residual graph $G_M$ of G and
    returns the minimum value among the maximum vertex-disjoint directed paths between
    every vertex of U and every vertex of V in $G_M$. By combining the definitions
    and the lemma, this value represents the extendability of the graph `G`.

    Time complexity O($n^3$ $m^2$)) where $n$ is the number of vertices
    and $m$ is the number of edges.

    References
    ----------
    .. [1] "A polynomial algorithm for the extendability problem in bipartite graphs",
          J. Lakhal, L. Litzler, Information Processing Letters, 1998.
    .. [2] "On n-extendible graphs", M. D. Plummer, Discrete Mathematics, 31:201–210, 1980
          https://doi.org/10.1016/0012-365X(80)90037-0

    '''
