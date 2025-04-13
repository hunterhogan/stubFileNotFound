from _typeshed import Incomplete

__all__ = ['constraint', 'local_constraint', 'effective_size']

def effective_size(G, nodes: Incomplete | None = None, weight: Incomplete | None = None):
    '''Returns the effective size of all nodes in the graph ``G``.

    The *effective size* of a node\'s ego network is based on the concept
    of redundancy. A person\'s ego network has redundancy to the extent
    that her contacts are connected to each other as well. The
    nonredundant part of a person\'s relationships is the effective
    size of her ego network [1]_.  Formally, the effective size of a
    node $u$, denoted $e(u)$, is defined by

    .. math::

       e(u) = \\sum_{v \\in N(u) \\setminus \\{u\\}}
       \\left(1 - \\sum_{w \\in N(v)} p_{uw} m_{vw}\\right)

    where $N(u)$ is the set of neighbors of $u$ and $p_{uw}$ is the
    normalized mutual weight of the (directed or undirected) edges
    joining $u$ and $v$, for each vertex $u$ and $v$ [1]_. And $m_{vw}$
    is the mutual weight of $v$ and $w$ divided by $v$ highest mutual
    weight with any of its neighbors. The *mutual weight* of $u$ and $v$
    is the sum of the weights of edges joining them (edge weights are
    assumed to be one if the graph is unweighted).

    For the case of unweighted and undirected graphs, Borgatti proposed
    a simplified formula to compute effective size [2]_

    .. math::

       e(u) = n - \\frac{2t}{n}

    where `t` is the number of ties in the ego network (not including
    ties to ego) and `n` is the number of nodes (excluding ego).

    Parameters
    ----------
    G : NetworkX graph
        The graph containing ``v``. Directed graphs are treated like
        undirected graphs when computing neighbors of ``v``.

    nodes : container, optional
        Container of nodes in the graph ``G`` to compute the effective size.
        If None, the effective size of every node is computed.

    weight : None or string, optional
      If None, all edge weights are considered equal.
      Otherwise holds the name of the edge attribute used as weight.

    Returns
    -------
    dict
        Dictionary with nodes as keys and the effective size of the node as values.

    Notes
    -----
    Burt also defined the related concept of *efficiency* of a node\'s ego
    network, which is its effective size divided by the degree of that
    node [1]_. So you can easily compute efficiency:

    >>> G = nx.DiGraph()
    >>> G.add_edges_from([(0, 1), (0, 2), (1, 0), (2, 1)])
    >>> esize = nx.effective_size(G)
    >>> efficiency = {n: v / G.degree(n) for n, v in esize.items()}

    See also
    --------
    constraint

    References
    ----------
    .. [1] Burt, Ronald S.
           *Structural Holes: The Social Structure of Competition.*
           Cambridge: Harvard University Press, 1995.

    .. [2] Borgatti, S.
           "Structural Holes: Unpacking Burt\'s Redundancy Measures"
           CONNECTIONS 20(1):35-38.
           http://www.analytictech.com/connections/v20(1)/holes.htm

    '''
def constraint(G, nodes: Incomplete | None = None, weight: Incomplete | None = None):
    '''Returns the constraint on all nodes in the graph ``G``.

    The *constraint* is a measure of the extent to which a node *v* is
    invested in those nodes that are themselves invested in the
    neighbors of *v*. Formally, the *constraint on v*, denoted `c(v)`,
    is defined by

    .. math::

       c(v) = \\sum_{w \\in N(v) \\setminus \\{v\\}} \\ell(v, w)

    where $N(v)$ is the subset of the neighbors of `v` that are either
    predecessors or successors of `v` and $\\ell(v, w)$ is the local
    constraint on `v` with respect to `w` [1]_. For the definition of local
    constraint, see :func:`local_constraint`.

    Parameters
    ----------
    G : NetworkX graph
        The graph containing ``v``. This can be either directed or undirected.

    nodes : container, optional
        Container of nodes in the graph ``G`` to compute the constraint. If
        None, the constraint of every node is computed.

    weight : None or string, optional
      If None, all edge weights are considered equal.
      Otherwise holds the name of the edge attribute used as weight.

    Returns
    -------
    dict
        Dictionary with nodes as keys and the constraint on the node as values.

    See also
    --------
    local_constraint

    References
    ----------
    .. [1] Burt, Ronald S.
           "Structural holes and good ideas".
           American Journal of Sociology (110): 349–399.

    '''
def local_constraint(G, u, v, weight: Incomplete | None = None):
    '''Returns the local constraint on the node ``u`` with respect to
    the node ``v`` in the graph ``G``.

    Formally, the *local constraint on u with respect to v*, denoted
    $\\ell(u, v)$, is defined by

    .. math::

       \\ell(u, v) = \\left(p_{uv} + \\sum_{w \\in N(v)} p_{uw} p_{wv}\\right)^2,

    where $N(v)$ is the set of neighbors of $v$ and $p_{uv}$ is the
    normalized mutual weight of the (directed or undirected) edges
    joining $u$ and $v$, for each vertex $u$ and $v$ [1]_. The *mutual
    weight* of $u$ and $v$ is the sum of the weights of edges joining
    them (edge weights are assumed to be one if the graph is
    unweighted).

    Parameters
    ----------
    G : NetworkX graph
        The graph containing ``u`` and ``v``. This can be either
        directed or undirected.

    u : node
        A node in the graph ``G``.

    v : node
        A node in the graph ``G``.

    weight : None or string, optional
      If None, all edge weights are considered equal.
      Otherwise holds the name of the edge attribute used as weight.

    Returns
    -------
    float
        The constraint of the node ``v`` in the graph ``G``.

    See also
    --------
    constraint

    References
    ----------
    .. [1] Burt, Ronald S.
           "Structural holes and good ideas".
           American Journal of Sociology (110): 349–399.

    '''
