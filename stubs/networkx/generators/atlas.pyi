__all__ = ['graph_atlas', 'graph_atlas_g']

def graph_atlas(i):
    """Returns graph number `i` from the Graph Atlas.

    For more information, see :func:`.graph_atlas_g`.

    Parameters
    ----------
    i : int
        The index of the graph from the atlas to get. The graph at index
        0 is assumed to be the null graph.

    Returns
    -------
    list
        A list of :class:`~networkx.Graph` objects, the one at index *i*
        corresponding to the graph *i* in the Graph Atlas.

    See also
    --------
    graph_atlas_g

    Notes
    -----
    The time required by this function increases linearly with the
    argument `i`, since it reads a large file sequentially in order to
    generate the graph [1]_.

    References
    ----------
    .. [1] Ronald C. Read and Robin J. Wilson, *An Atlas of Graphs*.
           Oxford University Press, 1998.

    """
def graph_atlas_g():
    '''Returns the list of all graphs with up to seven nodes named in the
    Graph Atlas.

    The graphs are listed in increasing order by

    1. number of nodes,
    2. number of edges,
    3. degree sequence (for example 111223 < 112222),
    4. number of automorphisms,

    in that order, with three exceptions as described in the *Notes*
    section below. This causes the list to correspond with the index of
    the graphs in the Graph Atlas [atlas]_, with the first graph,
    ``G[0]``, being the null graph.

    Returns
    -------
    list
        A list of :class:`~networkx.Graph` objects, the one at index *i*
        corresponding to the graph *i* in the Graph Atlas.

    See also
    --------
    graph_atlas

    Notes
    -----
    This function may be expensive in both time and space, since it
    reads a large file sequentially in order to populate the list.

    Although the NetworkX atlas functions match the order of graphs
    given in the "Atlas of Graphs" book, there are (at least) three
    errors in the ordering described in the book. The following three
    pairs of nodes violate the lexicographically nondecreasing sorted
    degree sequence rule:

    - graphs 55 and 56 with degree sequences 001111 and 000112,
    - graphs 1007 and 1008 with degree sequences 3333444 and 3333336,
    - graphs 1012 and 1213 with degree sequences 1244555 and 1244456.

    References
    ----------
    .. [atlas] Ronald C. Read and Robin J. Wilson,
               *An Atlas of Graphs*.
               Oxford University Press, 1998.

    '''
