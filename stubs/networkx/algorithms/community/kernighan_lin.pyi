from _typeshed import Incomplete

__all__ = ['kernighan_lin_bisection']

def kernighan_lin_bisection(G, partition: Incomplete | None = None, max_iter: int = 10, weight: str = 'weight', seed: Incomplete | None = None):
    '''Partition a graph into two blocks using the Kernighan–Lin
    algorithm.

    This algorithm partitions a network into two sets by iteratively
    swapping pairs of nodes to reduce the edge cut between the two sets.  The
    pairs are chosen according to a modified form of Kernighan-Lin [1]_, which
    moves node individually, alternating between sides to keep the bisection
    balanced.

    Parameters
    ----------
    G : NetworkX graph
        Graph must be undirected.

    partition : tuple
        Pair of iterables containing an initial partition. If not
        specified, a random balanced partition is used.

    max_iter : int
        Maximum number of times to attempt swaps to find an
        improvement before giving up.

    weight : key
        Edge data key to use as weight. If None, the weights are all
        set to one.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
        Only used if partition is None

    Returns
    -------
    partition : tuple
        A pair of sets of nodes representing the bipartition.

    Raises
    ------
    NetworkXError
        If partition is not a valid partition of the nodes of the graph.

    References
    ----------
    .. [1] Kernighan, B. W.; Lin, Shen (1970).
       "An efficient heuristic procedure for partitioning graphs."
       *Bell Systems Technical Journal* 49: 291--307.
       Oxford University Press 2011.

    '''
