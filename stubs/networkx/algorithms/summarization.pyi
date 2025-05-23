from _typeshed import Incomplete

__all__ = ['dedensify', 'snap_aggregation']

def dedensify(G, threshold, prefix: Incomplete | None = None, copy: bool = True):
    '''Compresses neighborhoods around high-degree nodes

    Reduces the number of edges to high-degree nodes by adding compressor nodes
    that summarize multiple edges of the same type to high-degree nodes (nodes
    with a degree greater than a given threshold).  Dedensification also has
    the added benefit of reducing the number of edges around high-degree nodes.
    The implementation currently supports graphs with a single edge type.

    Parameters
    ----------
    G: graph
       A networkx graph
    threshold: int
       Minimum degree threshold of a node to be considered a high degree node.
       The threshold must be greater than or equal to 2.
    prefix: str or None, optional (default: None)
       An optional prefix for denoting compressor nodes
    copy: bool, optional (default: True)
       Indicates if dedensification should be done inplace

    Returns
    -------
    dedensified networkx graph : (graph, set)
        2-tuple of the dedensified graph and set of compressor nodes

    Notes
    -----
    According to the algorithm in [1]_, removes edges in a graph by
    compressing/decompressing the neighborhoods around high degree nodes by
    adding compressor nodes that summarize multiple edges of the same type
    to high-degree nodes.  Dedensification will only add a compressor node when
    doing so will reduce the total number of edges in the given graph. This
    implementation currently supports graphs with a single edge type.

    Examples
    --------
    Dedensification will only add compressor nodes when doing so would result
    in fewer edges::

        >>> original_graph = nx.DiGraph()
        >>> original_graph.add_nodes_from(
        ...     ["1", "2", "3", "4", "5", "6", "A", "B", "C"]
        ... )
        >>> original_graph.add_edges_from(
        ...     [
        ...         ("1", "C"), ("1", "B"),
        ...         ("2", "C"), ("2", "B"), ("2", "A"),
        ...         ("3", "B"), ("3", "A"), ("3", "6"),
        ...         ("4", "C"), ("4", "B"), ("4", "A"),
        ...         ("5", "B"), ("5", "A"),
        ...         ("6", "5"),
        ...         ("A", "6")
        ...     ]
        ... )
        >>> c_graph, c_nodes = nx.dedensify(original_graph, threshold=2)
        >>> original_graph.number_of_edges()
        15
        >>> c_graph.number_of_edges()
        14

    A dedensified, directed graph can be "densified" to reconstruct the
    original graph::

        >>> original_graph = nx.DiGraph()
        >>> original_graph.add_nodes_from(
        ...     ["1", "2", "3", "4", "5", "6", "A", "B", "C"]
        ... )
        >>> original_graph.add_edges_from(
        ...     [
        ...         ("1", "C"), ("1", "B"),
        ...         ("2", "C"), ("2", "B"), ("2", "A"),
        ...         ("3", "B"), ("3", "A"), ("3", "6"),
        ...         ("4", "C"), ("4", "B"), ("4", "A"),
        ...         ("5", "B"), ("5", "A"),
        ...         ("6", "5"),
        ...         ("A", "6")
        ...     ]
        ... )
        >>> c_graph, c_nodes = nx.dedensify(original_graph, threshold=2)
        >>> # re-densifies the compressed graph into the original graph
        >>> for c_node in c_nodes:
        ...     all_neighbors = set(nx.all_neighbors(c_graph, c_node))
        ...     out_neighbors = set(c_graph.neighbors(c_node))
        ...     for out_neighbor in out_neighbors:
        ...         c_graph.remove_edge(c_node, out_neighbor)
        ...     in_neighbors = all_neighbors - out_neighbors
        ...     for in_neighbor in in_neighbors:
        ...         c_graph.remove_edge(in_neighbor, c_node)
        ...         for out_neighbor in out_neighbors:
        ...             c_graph.add_edge(in_neighbor, out_neighbor)
        ...     c_graph.remove_node(c_node)
        ...
        >>> nx.is_isomorphic(original_graph, c_graph)
        True

    References
    ----------
    .. [1] Maccioni, A., & Abadi, D. J. (2016, August).
       Scalable pattern matching over compressed graphs via dedensification.
       In Proceedings of the 22nd ACM SIGKDD International Conference on
       Knowledge Discovery and Data Mining (pp. 1755-1764).
       http://www.cs.umd.edu/~abadi/papers/graph-dedense.pdf
    '''
def snap_aggregation(G, node_attributes, edge_attributes=(), prefix: str = 'Supernode-', supernode_attribute: str = 'group', superedge_attribute: str = 'types'):
    '''Creates a summary graph based on attributes and connectivity.

    This function uses the Summarization by Grouping Nodes on Attributes
    and Pairwise edges (SNAP) algorithm for summarizing a given
    graph by grouping nodes by node attributes and their edge attributes
    into supernodes in a summary graph.  This name SNAP should not be
    confused with the Stanford Network Analysis Project (SNAP).

    Here is a high-level view of how this algorithm works:

    1) Group nodes by node attribute values.

    2) Iteratively split groups until all nodes in each group have edges
    to nodes in the same groups. That is, until all the groups are homogeneous
    in their member nodes\' edges to other groups.  For example,
    if all the nodes in group A only have edge to nodes in group B, then the
    group is homogeneous and does not need to be split. If all nodes in group B
    have edges with nodes in groups {A, C}, but some also have edges with other
    nodes in B, then group B is not homogeneous and needs to be split into
    groups have edges with {A, C} and a group of nodes having
    edges with {A, B, C}.  This way, viewers of the summary graph can
    assume that all nodes in the group have the exact same node attributes and
    the exact same edges.

    3) Build the output summary graph, where the groups are represented by
    super-nodes. Edges represent the edges shared between all the nodes in each
    respective groups.

    A SNAP summary graph can be used to visualize graphs that are too large to display
    or visually analyze, or to efficiently identify sets of similar nodes with similar connectivity
    patterns to other sets of similar nodes based on specified node and/or edge attributes in a graph.

    Parameters
    ----------
    G: graph
        Networkx Graph to be summarized
    node_attributes: iterable, required
        An iterable of the node attributes used to group nodes in the summarization process. Nodes
        with the same values for these attributes will be grouped together in the summary graph.
    edge_attributes: iterable, optional
        An iterable of the edge attributes considered in the summarization process.  If provided, unique
        combinations of the attribute values found in the graph are used to
        determine the edge types in the graph.  If not provided, all edges
        are considered to be of the same type.
    prefix: str
        The prefix used to denote supernodes in the summary graph. Defaults to \'Supernode-\'.
    supernode_attribute: str
        The node attribute for recording the supernode groupings of nodes. Defaults to \'group\'.
    superedge_attribute: str
        The edge attribute for recording the edge types of multiple edges. Defaults to \'types\'.

    Returns
    -------
    networkx.Graph: summary graph

    Examples
    --------
    SNAP aggregation takes a graph and summarizes it in the context of user-provided
    node and edge attributes such that a viewer can more easily extract and
    analyze the information represented by the graph

    >>> nodes = {
    ...     "A": dict(color="Red"),
    ...     "B": dict(color="Red"),
    ...     "C": dict(color="Red"),
    ...     "D": dict(color="Red"),
    ...     "E": dict(color="Blue"),
    ...     "F": dict(color="Blue"),
    ... }
    >>> edges = [
    ...     ("A", "E", "Strong"),
    ...     ("B", "F", "Strong"),
    ...     ("C", "E", "Weak"),
    ...     ("D", "F", "Weak"),
    ... ]
    >>> G = nx.Graph()
    >>> for node in nodes:
    ...     attributes = nodes[node]
    ...     G.add_node(node, **attributes)
    >>> for source, target, type in edges:
    ...     G.add_edge(source, target, type=type)
    >>> node_attributes = ("color",)
    >>> edge_attributes = ("type",)
    >>> summary_graph = nx.snap_aggregation(
    ...     G, node_attributes=node_attributes, edge_attributes=edge_attributes
    ... )

    Notes
    -----
    The summary graph produced is called a maximum Attribute-edge
    compatible (AR-compatible) grouping.  According to [1]_, an
    AR-compatible grouping means that all nodes in each group have the same
    exact node attribute values and the same exact edges and
    edge types to one or more nodes in the same groups.  The maximal
    AR-compatible grouping is the grouping with the minimal cardinality.

    The AR-compatible grouping is the most detailed grouping provided by
    any of the SNAP algorithms.

    References
    ----------
    .. [1] Y. Tian, R. A. Hankins, and J. M. Patel. Efficient aggregation
       for graph summarization. In Proc. 2008 ACM-SIGMOD Int. Conf.
       Management of Data (SIGMOD’08), pages 567–580, Vancouver, Canada,
       June 2008.
    '''
