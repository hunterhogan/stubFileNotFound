from _typeshed import Incomplete
from abc import ABC
from collections.abc import Mapping, Set

__all__ = ['NodeView', 'NodeDataView', 'EdgeView', 'OutEdgeView', 'InEdgeView', 'EdgeDataView', 'OutEdgeDataView', 'InEdgeDataView', 'MultiEdgeView', 'OutMultiEdgeView', 'InMultiEdgeView', 'MultiEdgeDataView', 'OutMultiEdgeDataView', 'InMultiEdgeDataView', 'DegreeView', 'DiDegreeView', 'InDegreeView', 'OutDegreeView', 'MultiDegreeView', 'DiMultiDegreeView', 'InMultiDegreeView', 'OutMultiDegreeView']

class NodeView(Mapping, Set):
    '''A NodeView class to act as G.nodes for a NetworkX Graph

    Set operations act on the nodes without considering data.
    Iteration is over nodes. Node data can be looked up like a dict.
    Use NodeDataView to iterate over node data or to specify a data
    attribute for lookup. NodeDataView is created by calling the NodeView.

    Parameters
    ----------
    graph : NetworkX graph-like class

    Examples
    --------
    >>> G = nx.path_graph(3)
    >>> NV = G.nodes()
    >>> 2 in NV
    True
    >>> for n in NV:
    ...     print(n)
    0
    1
    2
    >>> assert NV & {1, 2, 3} == {1, 2}

    >>> G.add_node(2, color="blue")
    >>> NV[2]
    {\'color\': \'blue\'}
    >>> G.add_node(8, color="red")
    >>> NDV = G.nodes(data=True)
    >>> (2, NV[2]) in NDV
    True
    >>> for n, dd in NDV:
    ...     print((n, dd.get("color", "aqua")))
    (0, \'aqua\')
    (1, \'aqua\')
    (2, \'blue\')
    (8, \'red\')
    >>> NDV[2] == NV[2]
    True

    >>> NVdata = G.nodes(data="color", default="aqua")
    >>> (2, NVdata[2]) in NVdata
    True
    >>> for n, dd in NVdata:
    ...     print((n, dd))
    (0, \'aqua\')
    (1, \'aqua\')
    (2, \'blue\')
    (8, \'red\')
    >>> NVdata[2] == NV[2]  # NVdata gets \'color\', NV gets datadict
    False
    '''
    __slots__: Incomplete
    def __getstate__(self): ...
    _nodes: Incomplete
    def __setstate__(self, state) -> None: ...
    def __init__(self, graph) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self): ...
    def __getitem__(self, n): ...
    def __contains__(self, n) -> bool: ...
    @classmethod
    def _from_iterable(cls, it): ...
    def __call__(self, data: bool = False, default: Incomplete | None = None): ...
    def data(self, data: bool = True, default: Incomplete | None = None):
        '''
        Return a read-only view of node data.

        Parameters
        ----------
        data : bool or node data key, default=True
            If ``data=True`` (the default), return a `NodeDataView` object that
            maps each node to *all* of its attributes. `data` may also be an
            arbitrary key, in which case the `NodeDataView` maps each node to
            the value for the keyed attribute. In this case, if a node does
            not have the `data` attribute, the `default` value is used.
        default : object, default=None
            The value used when a node does not have a specific attribute.

        Returns
        -------
        NodeDataView
            The layout of the returned NodeDataView depends on the value of the
            `data` parameter.

        Notes
        -----
        If ``data=False``, returns a `NodeView` object without data.

        See Also
        --------
        NodeDataView

        Examples
        --------
        >>> G = nx.Graph()
        >>> G.add_nodes_from(
        ...     [
        ...         (0, {"color": "red", "weight": 10}),
        ...         (1, {"color": "blue"}),
        ...         (2, {"color": "yellow", "weight": 2}),
        ...     ]
        ... )

        Accessing node data with ``data=True`` (the default) returns a
        NodeDataView mapping each node to all of its attributes:

        >>> G.nodes.data()
        NodeDataView({0: {\'color\': \'red\', \'weight\': 10}, 1: {\'color\': \'blue\'}, 2: {\'color\': \'yellow\', \'weight\': 2}})

        If `data` represents  a key in the node attribute dict, a NodeDataView mapping
        the nodes to the value for that specific key is returned:

        >>> G.nodes.data("color")
        NodeDataView({0: \'red\', 1: \'blue\', 2: \'yellow\'}, data=\'color\')

        If a specific key is not found in an attribute dict, the value specified
        by `default` is returned:

        >>> G.nodes.data("weight", default=-999)
        NodeDataView({0: 10, 1: -999, 2: 2}, data=\'weight\')

        Note that there is no check that the `data` key is in any of the
        node attribute dictionaries:

        >>> G.nodes.data("height")
        NodeDataView({0: None, 1: None, 2: None}, data=\'height\')
        '''
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class NodeDataView(Set):
    """A DataView class for nodes of a NetworkX Graph

    The main use for this class is to iterate through node-data pairs.
    The data can be the entire data-dictionary for each node, or it
    can be a specific attribute (with default) for each node.
    Set operations are enabled with NodeDataView, but don't work in
    cases where the data is not hashable. Use with caution.
    Typically, set operations on nodes use NodeView, not NodeDataView.
    That is, they use `G.nodes` instead of `G.nodes(data='foo')`.

    Parameters
    ==========
    graph : NetworkX graph-like class
    data : bool or string (default=False)
    default : object (default=None)
    """
    __slots__: Incomplete
    def __getstate__(self): ...
    _nodes: Incomplete
    _data: Incomplete
    _default: Incomplete
    def __setstate__(self, state) -> None: ...
    def __init__(self, nodedict, data: bool = False, default: Incomplete | None = None) -> None: ...
    @classmethod
    def _from_iterable(cls, it): ...
    def __len__(self) -> int: ...
    def __iter__(self): ...
    def __contains__(self, n) -> bool: ...
    def __getitem__(self, n): ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class DiDegreeView:
    '''A View class for degree of nodes in a NetworkX Graph

    The functionality is like dict.items() with (node, degree) pairs.
    Additional functionality includes read-only lookup of node degree,
    and calling with optional features nbunch (for only a subset of nodes)
    and weight (use edge weights to compute degree).

    Parameters
    ==========
    graph : NetworkX graph-like class
    nbunch : node, container of nodes, or None meaning all nodes (default=None)
    weight : bool or string (default=None)

    Notes
    -----
    DegreeView can still lookup any node even if nbunch is specified.

    Examples
    --------
    >>> G = nx.path_graph(3)
    >>> DV = G.degree()
    >>> assert DV[2] == 1
    >>> assert sum(deg for n, deg in DV) == 4

    >>> DVweight = G.degree(weight="span")
    >>> G.add_edge(1, 2, span=34)
    >>> DVweight[2]
    34
    >>> DVweight[0]  #  default edge weight is 1
    1
    >>> sum(span for n, span in DVweight)  # sum weighted degrees
    70

    >>> DVnbunch = G.degree(nbunch=(1, 2))
    >>> assert len(list(DVnbunch)) == 2  # iteration over nbunch only
    '''
    _graph: Incomplete
    _succ: Incomplete
    _pred: Incomplete
    _nodes: Incomplete
    _weight: Incomplete
    def __init__(self, G, nbunch: Incomplete | None = None, weight: Incomplete | None = None) -> None: ...
    def __call__(self, nbunch: Incomplete | None = None, weight: Incomplete | None = None): ...
    def __getitem__(self, n): ...
    def __iter__(self): ...
    def __len__(self) -> int: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class DegreeView(DiDegreeView):
    '''A DegreeView class to act as G.degree for a NetworkX Graph

    Typical usage focuses on iteration over `(node, degree)` pairs.
    The degree is by default the number of edges incident to the node.
    Optional argument `weight` enables weighted degree using the edge
    attribute named in the `weight` argument.  Reporting and iteration
    can also be restricted to a subset of nodes using `nbunch`.

    Additional functionality include node lookup so that `G.degree[n]`
    reported the (possibly weighted) degree of node `n`. Calling the
    view creates a view with different arguments `nbunch` or `weight`.

    Parameters
    ==========
    graph : NetworkX graph-like class
    nbunch : node, container of nodes, or None meaning all nodes (default=None)
    weight : string or None (default=None)

    Notes
    -----
    DegreeView can still lookup any node even if nbunch is specified.

    Examples
    --------
    >>> G = nx.path_graph(3)
    >>> DV = G.degree()
    >>> assert DV[2] == 1
    >>> assert G.degree[2] == 1
    >>> assert sum(deg for n, deg in DV) == 4

    >>> DVweight = G.degree(weight="span")
    >>> G.add_edge(1, 2, span=34)
    >>> DVweight[2]
    34
    >>> DVweight[0]  #  default edge weight is 1
    1
    >>> sum(span for n, span in DVweight)  # sum weighted degrees
    70

    >>> DVnbunch = G.degree(nbunch=(1, 2))
    >>> assert len(list(DVnbunch)) == 2  # iteration over nbunch only
    '''
    def __getitem__(self, n): ...
    def __iter__(self): ...

class OutDegreeView(DiDegreeView):
    """A DegreeView class to report out_degree for a DiGraph; See DegreeView"""
    def __getitem__(self, n): ...
    def __iter__(self): ...

class InDegreeView(DiDegreeView):
    """A DegreeView class to report in_degree for a DiGraph; See DegreeView"""
    def __getitem__(self, n): ...
    def __iter__(self): ...

class MultiDegreeView(DiDegreeView):
    """A DegreeView class for undirected multigraphs; See DegreeView"""
    def __getitem__(self, n): ...
    def __iter__(self): ...

class DiMultiDegreeView(DiDegreeView):
    """A DegreeView class for MultiDiGraph; See DegreeView"""
    def __getitem__(self, n): ...
    def __iter__(self): ...

class InMultiDegreeView(DiDegreeView):
    """A DegreeView class for inward degree of MultiDiGraph; See DegreeView"""
    def __getitem__(self, n): ...
    def __iter__(self): ...

class OutMultiDegreeView(DiDegreeView):
    """A DegreeView class for outward degree of MultiDiGraph; See DegreeView"""
    def __getitem__(self, n): ...
    def __iter__(self): ...

class EdgeViewABC(ABC): ...

class OutEdgeDataView(EdgeViewABC):
    """EdgeDataView for outward edges of DiGraph; See EdgeDataView"""
    __slots__: Incomplete
    def __getstate__(self): ...
    def __setstate__(self, state) -> None: ...
    _viewer: Incomplete
    _nodes_nbrs: Incomplete
    _nbunch: Incomplete
    _data: Incomplete
    _default: Incomplete
    _report: Incomplete
    def __init__(self, viewer, nbunch: Incomplete | None = None, data: bool = False, *, default: Incomplete | None = None) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self): ...
    def __contains__(self, e) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class EdgeDataView(OutEdgeDataView):
    '''A EdgeDataView class for edges of Graph

    This view is primarily used to iterate over the edges reporting
    edges as node-tuples with edge data optionally reported. The
    argument `nbunch` allows restriction to edges incident to nodes
    in that container/singleton. The default (nbunch=None)
    reports all edges. The arguments `data` and `default` control
    what edge data is reported. The default `data is False` reports
    only node-tuples for each edge. If `data is True` the entire edge
    data dict is returned. Otherwise `data` is assumed to hold the name
    of the edge attribute to report with default `default` if  that
    edge attribute is not present.

    Parameters
    ----------
    nbunch : container of nodes, node or None (default None)
    data : False, True or string (default False)
    default : default value (default None)

    Examples
    --------
    >>> G = nx.path_graph(3)
    >>> G.add_edge(1, 2, foo="bar")
    >>> list(G.edges(data="foo", default="biz"))
    [(0, 1, \'biz\'), (1, 2, \'bar\')]
    >>> assert (0, 1, "biz") in G.edges(data="foo", default="biz")
    '''
    __slots__: Incomplete
    def __len__(self) -> int: ...
    def __iter__(self): ...
    def __contains__(self, e) -> bool: ...

class InEdgeDataView(OutEdgeDataView):
    """An EdgeDataView class for outward edges of DiGraph; See EdgeDataView"""
    __slots__: Incomplete
    def __iter__(self): ...
    def __contains__(self, e) -> bool: ...

class OutMultiEdgeDataView(OutEdgeDataView):
    """An EdgeDataView for outward edges of MultiDiGraph; See EdgeDataView"""
    __slots__: Incomplete
    def __getstate__(self): ...
    def __setstate__(self, state) -> None: ...
    _viewer: Incomplete
    keys: Incomplete
    _nodes_nbrs: Incomplete
    _nbunch: Incomplete
    _data: Incomplete
    _default: Incomplete
    _report: Incomplete
    def __init__(self, viewer, nbunch: Incomplete | None = None, data: bool = False, *, default: Incomplete | None = None, keys: bool = False) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self): ...
    def __contains__(self, e) -> bool: ...

class MultiEdgeDataView(OutMultiEdgeDataView):
    """An EdgeDataView class for edges of MultiGraph; See EdgeDataView"""
    __slots__: Incomplete
    def __iter__(self): ...
    def __contains__(self, e) -> bool: ...

class InMultiEdgeDataView(OutMultiEdgeDataView):
    """An EdgeDataView for inward edges of MultiDiGraph; See EdgeDataView"""
    __slots__: Incomplete
    def __iter__(self): ...
    def __contains__(self, e) -> bool: ...

class OutEdgeView(Set, Mapping, EdgeViewABC):
    """A EdgeView class for outward edges of a DiGraph"""
    __slots__: Incomplete
    def __getstate__(self): ...
    _graph: Incomplete
    _adjdict: Incomplete
    _nodes_nbrs: Incomplete
    def __setstate__(self, state) -> None: ...
    @classmethod
    def _from_iterable(cls, it): ...
    dataview = OutEdgeDataView
    def __init__(self, G) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self): ...
    def __contains__(self, e) -> bool: ...
    def __getitem__(self, e): ...
    def __call__(self, nbunch: Incomplete | None = None, data: bool = False, *, default: Incomplete | None = None): ...
    def data(self, data: bool = True, default: Incomplete | None = None, nbunch: Incomplete | None = None):
        '''
        Return a read-only view of edge data.

        Parameters
        ----------
        data : bool or edge attribute key
            If ``data=True``, then the data view maps each edge to a dictionary
            containing all of its attributes. If `data` is a key in the edge
            dictionary, then the data view maps each edge to its value for
            the keyed attribute. In this case, if the edge doesn\'t have the
            attribute, the `default` value is returned.
        default : object, default=None
            The value used when an edge does not have a specific attribute
        nbunch : container of nodes, optional (default=None)
            Allows restriction to edges only involving certain nodes. All edges
            are considered by default.

        Returns
        -------
        dataview
            Returns an `EdgeDataView` for undirected Graphs, `OutEdgeDataView`
            for DiGraphs, `MultiEdgeDataView` for MultiGraphs and
            `OutMultiEdgeDataView` for MultiDiGraphs.

        Notes
        -----
        If ``data=False``, returns an `EdgeView` without any edge data.

        See Also
        --------
        EdgeDataView
        OutEdgeDataView
        MultiEdgeDataView
        OutMultiEdgeDataView

        Examples
        --------
        >>> G = nx.Graph()
        >>> G.add_edges_from(
        ...     [
        ...         (0, 1, {"dist": 3, "capacity": 20}),
        ...         (1, 2, {"dist": 4}),
        ...         (2, 0, {"dist": 5}),
        ...     ]
        ... )

        Accessing edge data with ``data=True`` (the default) returns an
        edge data view object listing each edge with all of its attributes:

        >>> G.edges.data()
        EdgeDataView([(0, 1, {\'dist\': 3, \'capacity\': 20}), (0, 2, {\'dist\': 5}), (1, 2, {\'dist\': 4})])

        If `data` represents a key in the edge attribute dict, a dataview listing
        each edge with its value for that specific key is returned:

        >>> G.edges.data("dist")
        EdgeDataView([(0, 1, 3), (0, 2, 5), (1, 2, 4)])

        `nbunch` can be used to limit the edges:

        >>> G.edges.data("dist", nbunch=[0])
        EdgeDataView([(0, 1, 3), (0, 2, 5)])

        If a specific key is not found in an edge attribute dict, the value
        specified by `default` is used:

        >>> G.edges.data("capacity")
        EdgeDataView([(0, 1, 20), (0, 2, None), (1, 2, None)])

        Note that there is no check that the `data` key is present in any of
        the edge attribute dictionaries:

        >>> G.edges.data("speed")
        EdgeDataView([(0, 1, None), (0, 2, None), (1, 2, None)])
        '''
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class EdgeView(OutEdgeView):
    '''A EdgeView class for edges of a Graph

    This densely packed View allows iteration over edges, data lookup
    like a dict and set operations on edges represented by node-tuples.
    In addition, edge data can be controlled by calling this object
    possibly creating an EdgeDataView. Typically edges are iterated over
    and reported as `(u, v)` node tuples or `(u, v, key)` node/key tuples
    for multigraphs. Those edge representations can also be using to
    lookup the data dict for any edge. Set operations also are available
    where those tuples are the elements of the set.
    Calling this object with optional arguments `data`, `default` and `keys`
    controls the form of the tuple (see EdgeDataView). Optional argument
    `nbunch` allows restriction to edges only involving certain nodes.

    If `data is False` (the default) then iterate over 2-tuples `(u, v)`.
    If `data is True` iterate over 3-tuples `(u, v, datadict)`.
    Otherwise iterate over `(u, v, datadict.get(data, default))`.
    For Multigraphs, if `keys is True`, replace `u, v` with `u, v, key` above.

    Parameters
    ==========
    graph : NetworkX graph-like class
    nbunch : (default= all nodes in graph) only report edges with these nodes
    keys : (only for MultiGraph. default=False) report edge key in tuple
    data : bool or string (default=False) see above
    default : object (default=None)

    Examples
    ========
    >>> G = nx.path_graph(4)
    >>> EV = G.edges()
    >>> (2, 3) in EV
    True
    >>> for u, v in EV:
    ...     print((u, v))
    (0, 1)
    (1, 2)
    (2, 3)
    >>> assert EV & {(1, 2), (3, 4)} == {(1, 2)}

    >>> EVdata = G.edges(data="color", default="aqua")
    >>> G.add_edge(2, 3, color="blue")
    >>> assert (2, 3, "blue") in EVdata
    >>> for u, v, c in EVdata:
    ...     print(f"({u}, {v}) has color: {c}")
    (0, 1) has color: aqua
    (1, 2) has color: aqua
    (2, 3) has color: blue

    >>> EVnbunch = G.edges(nbunch=2)
    >>> assert (2, 3) in EVnbunch
    >>> assert (0, 1) not in EVnbunch
    >>> for u, v in EVnbunch:
    ...     assert u == 2 or v == 2

    >>> MG = nx.path_graph(4, create_using=nx.MultiGraph)
    >>> EVmulti = MG.edges(keys=True)
    >>> (2, 3, 0) in EVmulti
    True
    >>> (2, 3) in EVmulti  # 2-tuples work even when keys is True
    True
    >>> key = MG.add_edge(2, 3)
    >>> for u, v, k in EVmulti:
    ...     print((u, v, k))
    (0, 1, 0)
    (1, 2, 0)
    (2, 3, 0)
    (2, 3, 1)
    '''
    __slots__: Incomplete
    dataview = EdgeDataView
    def __len__(self) -> int: ...
    def __iter__(self): ...
    def __contains__(self, e) -> bool: ...

class InEdgeView(OutEdgeView):
    """A EdgeView class for inward edges of a DiGraph"""
    __slots__: Incomplete
    _graph: Incomplete
    _adjdict: Incomplete
    _nodes_nbrs: Incomplete
    def __setstate__(self, state) -> None: ...
    dataview = InEdgeDataView
    def __init__(self, G) -> None: ...
    def __iter__(self): ...
    def __contains__(self, e) -> bool: ...
    def __getitem__(self, e): ...

class OutMultiEdgeView(OutEdgeView):
    """A EdgeView class for outward edges of a MultiDiGraph"""
    __slots__: Incomplete
    dataview = OutMultiEdgeDataView
    def __len__(self) -> int: ...
    def __iter__(self): ...
    def __contains__(self, e) -> bool: ...
    def __getitem__(self, e): ...
    def __call__(self, nbunch: Incomplete | None = None, data: bool = False, *, default: Incomplete | None = None, keys: bool = False): ...
    def data(self, data: bool = True, default: Incomplete | None = None, nbunch: Incomplete | None = None, keys: bool = False): ...

class MultiEdgeView(OutMultiEdgeView):
    """A EdgeView class for edges of a MultiGraph"""
    __slots__: Incomplete
    dataview = MultiEdgeDataView
    def __len__(self) -> int: ...
    def __iter__(self): ...

class InMultiEdgeView(OutMultiEdgeView):
    """A EdgeView class for inward edges of a MultiDiGraph"""
    __slots__: Incomplete
    _graph: Incomplete
    _adjdict: Incomplete
    _nodes_nbrs: Incomplete
    def __setstate__(self, state) -> None: ...
    dataview = InMultiEdgeDataView
    def __init__(self, G) -> None: ...
    def __iter__(self): ...
    def __contains__(self, e) -> bool: ...
    def __getitem__(self, e): ...
