from _typeshed import Incomplete
from collections.abc import Generator

__all__ = ['greedy_color', 'strategy_connected_sequential', 'strategy_connected_sequential_bfs', 'strategy_connected_sequential_dfs', 'strategy_independent_set', 'strategy_largest_first', 'strategy_random_sequential', 'strategy_saturation_largest_first', 'strategy_smallest_last']

def strategy_largest_first(G, colors):
    """Returns a list of the nodes of ``G`` in decreasing order by
    degree.

    ``G`` is a NetworkX graph. ``colors`` is ignored.

    """
def strategy_random_sequential(G, colors, seed: Incomplete | None = None):
    """Returns a random permutation of the nodes of ``G`` as a list.

    ``G`` is a NetworkX graph. ``colors`` is ignored.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    """
def strategy_smallest_last(G, colors):
    '''Returns a deque of the nodes of ``G``, "smallest" last.

    Specifically, the degrees of each node are tracked in a bucket queue.
    From this, the node of minimum degree is repeatedly popped from the
    graph, updating its neighbors\' degrees.

    ``G`` is a NetworkX graph. ``colors`` is ignored.

    This implementation of the strategy runs in $O(n + m)$ time
    (ignoring polylogarithmic factors), where $n$ is the number of nodes
    and $m$ is the number of edges.

    This strategy is related to :func:`strategy_independent_set`: if we
    interpret each node removed as an independent set of size one, then
    this strategy chooses an independent set of size one instead of a
    maximal independent set.

    '''
def strategy_independent_set(G, colors) -> Generator[Incomplete, Incomplete]:
    """Uses a greedy independent set removal strategy to determine the
    colors.

    This function updates ``colors`` **in-place** and return ``None``,
    unlike the other strategy functions in this module.

    This algorithm repeatedly finds and removes a maximal independent
    set, assigning each node in the set an unused color.

    ``G`` is a NetworkX graph.

    This strategy is related to :func:`strategy_smallest_last`: in that
    strategy, an independent set of size one is chosen at each step
    instead of a maximal independent set.

    """
def strategy_connected_sequential_bfs(G, colors):
    """Returns an iterable over nodes in ``G`` in the order given by a
    breadth-first traversal.

    The generated sequence has the property that for each node except
    the first, at least one neighbor appeared earlier in the sequence.

    ``G`` is a NetworkX graph. ``colors`` is ignored.

    """
def strategy_connected_sequential_dfs(G, colors):
    """Returns an iterable over nodes in ``G`` in the order given by a
    depth-first traversal.

    The generated sequence has the property that for each node except
    the first, at least one neighbor appeared earlier in the sequence.

    ``G`` is a NetworkX graph. ``colors`` is ignored.

    """
def strategy_connected_sequential(G, colors, traversal: str = 'bfs') -> Generator[Incomplete]:
    """Returns an iterable over nodes in ``G`` in the order given by a
    breadth-first or depth-first traversal.

    ``traversal`` must be one of the strings ``'dfs'`` or ``'bfs'``,
    representing depth-first traversal or breadth-first traversal,
    respectively.

    The generated sequence has the property that for each node except
    the first, at least one neighbor appeared earlier in the sequence.

    ``G`` is a NetworkX graph. ``colors`` is ignored.

    """
def strategy_saturation_largest_first(G, colors) -> Generator[Incomplete, None, Incomplete]:
    '''Iterates over all the nodes of ``G`` in "saturation order" (also
    known as "DSATUR").

    ``G`` is a NetworkX graph. ``colors`` is a dictionary mapping nodes of
    ``G`` to colors, for those nodes that have already been colored.

    '''
def greedy_color(G, strategy: str = 'largest_first', interchange: bool = False):
    '''Color a graph using various strategies of greedy graph coloring.

    Attempts to color a graph using as few colors as possible, where no
    neighbors of a node can have same color as the node itself. The
    given strategy determines the order in which nodes are colored.

    The strategies are described in [1]_, and smallest-last is based on
    [2]_.

    Parameters
    ----------
    G : NetworkX graph

    strategy : string or function(G, colors)
       A function (or a string representing a function) that provides
       the coloring strategy, by returning nodes in the ordering they
       should be colored. ``G`` is the graph, and ``colors`` is a
       dictionary of the currently assigned colors, keyed by nodes. The
       function must return an iterable over all the nodes in ``G``.

       If the strategy function is an iterator generator (that is, a
       function with ``yield`` statements), keep in mind that the
       ``colors`` dictionary will be updated after each ``yield``, since
       this function chooses colors greedily.

       If ``strategy`` is a string, it must be one of the following,
       each of which represents one of the built-in strategy functions.

       * ``\'largest_first\'``
       * ``\'random_sequential\'``
       * ``\'smallest_last\'``
       * ``\'independent_set\'``
       * ``\'connected_sequential_bfs\'``
       * ``\'connected_sequential_dfs\'``
       * ``\'connected_sequential\'`` (alias for the previous strategy)
       * ``\'saturation_largest_first\'``
       * ``\'DSATUR\'`` (alias for the previous strategy)

    interchange: bool
       Will use the color interchange algorithm described by [3]_ if set
       to ``True``.

       Note that ``saturation_largest_first`` and ``independent_set``
       do not work with interchange. Furthermore, if you use
       interchange with your own strategy function, you cannot rely
       on the values in the ``colors`` argument.

    Returns
    -------
    A dictionary with keys representing nodes and values representing
    corresponding coloring.

    Examples
    --------
    >>> G = nx.cycle_graph(4)
    >>> d = nx.coloring.greedy_color(G, strategy="largest_first")
    >>> d in [{0: 0, 1: 1, 2: 0, 3: 1}, {0: 1, 1: 0, 2: 1, 3: 0}]
    True

    Raises
    ------
    NetworkXPointlessConcept
        If ``strategy`` is ``saturation_largest_first`` or
        ``independent_set`` and ``interchange`` is ``True``.

    References
    ----------
    .. [1] Adrian Kosowski, and Krzysztof Manuszewski,
       Classical Coloring of Graphs, Graph Colorings, 2-19, 2004.
       ISBN 0-8218-3458-4.
    .. [2] David W. Matula, and Leland L. Beck, "Smallest-last
       ordering and clustering and graph coloring algorithms." *J. ACM* 30,
       3 (July 1983), 417-427. <https://doi.org/10.1145/2402.322385>
    .. [3] Maciej M. Sysło, Narsingh Deo, Janusz S. Kowalik,
       Discrete Optimization Algorithms with Pascal Programs, 415-424, 1983.
       ISBN 0-486-45353-7.

    '''

class _Node:
    __slots__: Incomplete
    node_id: Incomplete
    color: int
    adj_list: Incomplete
    adj_color: Incomplete
    def __init__(self, node_id, n) -> None: ...
    def __repr__(self) -> str: ...
    def assign_color(self, adj_entry, color) -> None: ...
    def clear_color(self, adj_entry, color) -> None: ...
    def iter_neighbors(self) -> Generator[Incomplete]: ...
    def iter_neighbors_color(self, color) -> Generator[Incomplete]: ...

class _AdjEntry:
    __slots__: Incomplete
    node_id: Incomplete
    next: Incomplete
    mate: Incomplete
    col_next: Incomplete
    col_prev: Incomplete
    def __init__(self, node_id) -> None: ...
    def __repr__(self) -> str: ...
