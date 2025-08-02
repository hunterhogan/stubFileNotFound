from collections.abc import Callable, Collection, Hashable, Iterable, Iterator, Mapping, MutableMapping
from functools import cached_property
from networkx.classes.coreviews import AdjacencyView, AtlasView
from networkx.classes.digraph import DiGraph
from networkx.classes.reportviews import DegreeView, EdgeView, NodeView
from typing import Any, ClassVar, overload, TypeVar
from typing_extensions import Self, TypeAlias
import numpy

_Node = TypeVar("_Node", bound=Hashable)
_NodeWithData: TypeAlias = tuple[_Node, dict[str, Any]]
_NodePlus: TypeAlias = _Node | _NodeWithData[_Node]
_Edge: TypeAlias = tuple[_Node, _Node]
_EdgeWithData: TypeAlias = tuple[_Node, _Node, dict[str, Any]]
_EdgePlus: TypeAlias = _Edge[_Node] | _EdgeWithData[_Node]
_MapFactory: TypeAlias = Callable[[], MutableMapping[str, Any]]
_NBunch: TypeAlias = _Node | Iterable[_Node] | None
_Data: TypeAlias = (
    Graph[_Node]
    | dict[_Node, dict[_Node, dict[str, Any]]]
    | dict[_Node, Iterable[_Node]]
    | Iterable[_EdgePlus[_Node]]
    | numpy.ndarray[Any, Any]
    # | scipy.sparse.base.spmatrix
)

__all__ = ['Graph']

class _CachedPropertyResetterAdj:
    """Data Descriptor class for _adj that resets ``adj`` cached_property when needed

    This assumes that the ``cached_property`` ``G.adj`` should be reset whenever ``G._adj`` is set to a new value.

    This object sits on a class and ensures that any instance of that class clears its cached property "adj" whenever the
    underlying instance attribute "_adj" is set to a new object. It only affects the set process of the obj._adj attribute. All
    get/del operations act as they normally would.

    For info on Data Descriptors see: https://docs.python.org/3/howto/descriptor.html
    """
    def __set__(self, obj: Any, value: Any) -> None: ...

class _CachedPropertyResetterNode:
    """Data Descriptor class for _node that resets ``nodes`` cached_property when needed

    This assumes that the ``cached_property`` ``G.node`` should be reset whenever ``G._node`` is set to a new value.

    This object sits on a class and ensures that any instance of that class clears its cached property "nodes" whenever the
    underlying instance attribute "_node" is set to a new object. It only affects the set process of the obj._adj attribute. All
    get/del operations act as they normally would.

    For info on Data Descriptors see: https://docs.python.org/3/howto/descriptor.html
    """
    def __set__(self, obj: Any, value: Any) -> None: ...

class Graph(Collection[_Node]):
    """
    Base class for undirected graphs.

    A `Graph` stores nodes and edges with optional data, or attributes. `Graphs` hold undirected edges. Self loops are allowed,
    but multiple (parallel) edges are not. Nodes can be arbitrary (hashable) Python objects with optional key/value attributes,
    except that `None` is not allowed as a node. Edges are represented as links between nodes with optional key/value attributes.

    Parameters
    ----------
    incoming_graph_data : input graph (optional, default: None)
        Data to initialize graph. If `None` (default), an empty graph is created. The data can be any format that is supported by
        the `to_networkx_graph()` function, currently including edge list, dict of dicts, dict of lists, `NetworkX` graph, 2D
        `NumPy` array, `SciPy` sparse matrix, or `PyGraphviz` graph.

    attr : keyword arguments, optional (default= no attributes)
        Attributes to add to graph as key=value pairs.

    See Also
    --------
    `DiGraph`
    `MultiGraph`
    `MultiDiGraph`
    """

    __networkx_backend__: ClassVar[str]
    node_dict_factory: ClassVar[_MapFactory]
    node_attr_dict_factory: ClassVar[_MapFactory]
    adjlist_outer_dict_factory: ClassVar[_MapFactory]
    adjlist_inner_dict_factory: ClassVar[_MapFactory]
    edge_attr_dict_factory: ClassVar[_MapFactory]
    graph_attr_dict_factory: ClassVar[_MapFactory]

    # Instance variables
    _adj: dict[_Node, dict[_Node, dict[str, Any]]]  # Type descriptor
    _node: dict[_Node, dict[str, Any]]  # Dictionary for node attributes
    graph: dict[str, Any]
    __networkx_cache__: dict[str, Any]

    def to_directed_class(self) -> type[DiGraph[_Node]]:
        """Returns the class to use for directed copies of this graph.

        When creating a directed version of the graph with functions like `to_directed()`, this provides the class that will be
        used.

        Returns
        -------
        type
            A directed graph class.
        """
        ...

    def to_undirected_class(self) -> type[Graph[_Node]]:
        """Returns the class to use for undirected copies of this graph.

        When creating an undirected version of the graph with functions like `to_undirected()`, this provides the class that will
        be used.

        Returns
        -------
        type
            An undirected graph class.
        """
        ...

    def __init__(self, incoming_graph_data: _Data[_Node] | None = None, **attr: Any) -> None:
        """Initialize a graph with edges, `name`, or graph attributes.

        Parameters
        ----------
        incoming_graph_data : input graph (optional, default: None)
            Data to initialize graph. If `None` (default), an empty graph is created. The data can be an edge list, or any
            `NetworkX` graph object. If the corresponding optional Python packages are installed, the data can also be a 2D
            `NumPy` array, a `SciPy` sparse array, or a `PyGraphviz` graph.

        attr : keyword arguments, optional (default= no attributes)
            Attributes to add to graph as key=value pairs.

        See Also
        --------
        convert
        """
        ...

    @cached_property
    def adj(self) -> AdjacencyView[_Node, _Node, Mapping[str, Any]]:
        """Graph adjacency object holding the neighbors of each node.

        This object is a read-only dict-like structure with node keys and neighbor-dict values. The neighbor-dict is keyed by
        neighbor to the edge-data-dict. So `G.adj[3][2]['color'] = 'blue'` sets the color of the edge `(3, 2)` to `"blue"`.

        Iterating over `G.adj` behaves like a dict. Useful idioms include `for nbr, datadict in G.adj[n].items():`.

        The neighbor information is also provided by subscripting the graph. So `for nbr, foovalue in G[node].data('foo',
        default=1):` works.

        For directed graphs, `G.adj` holds outgoing (successor) info.
        """
        ...

    @property
    def name(self) -> str:
        """String identifier of the graph.

        This graph attribute appears in the attribute dict G.graph keyed by the string `"name"`. As well as being stored as a
        graph attribute, this string is returned by the `__str__` method of the graph class. If `name=None`, a unique string
        identifier is chosen.
        """
        ...

    @name.setter
    def name(self, s: str) -> None:
        """Sets the name of the graph.

        Parameters
        ----------
        s : string
            The name of the graph.
        """
        ...

    def __getitem__(self, n: _Node) -> AtlasView[_Node, str, Any]:
        """Returns a dict of neighbors of node n.

        This allows the syntax G[n] to access the neighbors of node n.
        """
        ...

    def __iter__(self) -> Iterator[_Node]:
        """Iterate over the nodes. Use: 'for n in G'."""
        ...

    def __contains__(self, n: object) -> bool:
        """Returns True if n is a node, False otherwise. Use: 'n in G'."""
        ...

    def __len__(self) -> int:
        """Returns the number of nodes in the graph. Use: 'len(G)'."""
        ...

    def add_node(self, node_for_adding: _Node, **attr: Any) -> None:
        """Add a single node `node_for_adding` and update node attributes.

        Parameters
        ----------
        node_for_adding : node
            A node can be any hashable Python object except None.
        attr : keyword arguments, optional
            Set or change node attributes using key=value.
        """
        ...

    def add_nodes_from(self, nodes_for_adding: Iterable[_NodePlus[_Node]], **attr: Any) -> None:
        """Add multiple nodes.

        Parameters
        ----------
        nodes_for_adding : iterable container
            A container of nodes (list, dict, set, etc.). OR A container of (node, attribute dict) tuples. Node attributes are
            updated using the attribute dict.
        attr : keyword arguments, optional (default= no attributes)
            Update attributes for all nodes in nodes. Node attributes specified in nodes as a tuple take precedence over
            attributes specified via keyword arguments.
        """
        ...

    def remove_node(self, n: _Node) -> None:
        """Remove node `n`.

        Removes the node `n` and all adjacent edges. Attempting to remove a non-existent node will raise an exception.

        Parameters
        ----------
        n : node
            A node in the graph.

        Raises
        ------
        `NetworkXError`
            If `n` is not in the graph.
        """
        ...

    def remove_nodes_from(self, nodes: Iterable[_Node]) -> None:
        """Remove multiple nodes.

        Parameters
        ----------
        nodes : iterable container
            A container of nodes (list, dict, set, etc.). If a node in the container is not in the graph, it is silently ignored.
        """
        ...

    @cached_property
    def nodes(self) -> NodeView[_Node]:
        """A `NodeView` of the `Graph` as `G.nodes` or `G.nodes()`.

        The `NodeView` provides set-like operations, a dictionary interface, and a convenient way to iterate through node attributes.

        Parameters
        ----------
        data : string or bool, optional (default=False)
            The node attribute returned in 2-tuple (`n`, `ddict[data]`). If True, return entire node attribute dict as (`n`, `ddict`). If False, return just the nodes `n`.
        default : value, optional (default=None)
            Value used for nodes that don't have the requested attribute. Only relevant if `data` is not True or False.

        Returns
        -------
        `NodeView`
            Allows set-like operations over the nodes as well as node attribute dict lookup and iteration over nodes data.

        Examples
        --------
        There are two simple ways of getting a list of all nodes in the graph:

        >>> G = nx.path_graph(3)
        >>> list(G.nodes)
        [0, 1, 2]
        >>> list(G)
        [0, 1, 2]

        To get the node data along with the nodes:

        >>> G.add_node(1, time='5pm')
        >>> G.nodes[1]
        {'time': '5pm'}
        >>> list(G.nodes(data=True))
        [(0, {}), (1, {'time': '5pm'}), (2, {})]
        >>> list(G.nodes(data='time'))
        [(0, None), (1, '5pm'), (2, None)]
        >>> list(G.nodes(data='time', default='Not Available'))
        [(0, 'Not Available'), (1, '5pm'), (2, 'Not Available')]
        >>> list(G.nodes(data='weight', default=1))
        [(0, 1), (1, 1), (2, 1)]
        """
        ...

    def number_of_nodes(self) -> int:
        """Returns the number of nodes in the graph.

        Returns
        -------
        nnodes : int
            The number of nodes in the graph.

        See Also
        --------
        `order`: identical function.
        `__len__`: identical function.

        Examples
        --------
        >>> G = nx.path_graph(3)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G.number_of_nodes()
        3
        """
        ...

    def order(self) -> int:
        """Returns the number of nodes in the graph.

        Returns
        -------
        nnodes : int
            The number of nodes in the graph.

        See Also
        --------
        `number_of_nodes`: identical function.
        `__len__`: identical function.

        Examples
        --------
        >>> G = nx.path_graph(3)  # or `DiGraph`, `MultiGraph`, `MultiDiGraph`, etc.
        >>> G.order()
        3
        """
        ...

    def has_node(self, n: _Node) -> bool:
        """Returns True if the graph contains the node `n`.

        Identical to `n in G`.

        Parameters
        ----------
        n : node

        Examples
        --------
        >>> G = nx.path_graph(3)  # or `DiGraph`, `MultiGraph`, `MultiDiGraph`, etc.
        >>> G.has_node(0)
        True

        It is more readable and simpler to use:

        >>> 0 in G
        True
        """
        ...

    def add_edge(self, u_of_edge: _Node, v_of_edge: _Node, **attr: Any) -> None:
        """Add an edge between `u` and `v`.

        The nodes `u` and `v` will be automatically added if they are not already in the graph.

        Edge attributes can be specified with keywords or by directly accessing the edge's attribute dictionary. See examples below.

        Parameters
        ----------
        u_of_edge, v_of_edge : nodes
            Nodes can be, for example, strings or numbers. Nodes must be hashable (and not `None`) Python objects.
        attr : keyword arguments, optional
            Edge data (or labels or objects) can be assigned using keyword arguments.

        See Also
        --------
        `add_edges_from` : add a collection of edges.

        Notes
        -----
        Adding an edge that already exists updates the edge data.

        Many `NetworkX` algorithms designed for weighted graphs use an edge attribute (by default `weight`) to hold a numerical value.

        Examples
        --------
        The following all add the edge `e=(1, 2)` to graph `G`:

        >>> G = nx.Graph()   # or `DiGraph`, `MultiGraph`, `MultiDiGraph`, etc.
        >>> e = (1, 2)
        >>> G.add_edge(1, 2)           # explicit two-node form
        >>> G.add_edge(*e)             # single edge as tuple of two nodes
        >>> G.add_edges_from([(1, 2)])  # add edges from iterable container

        Associate data to edges using keywords:

        >>> G.add_edge(1, 2, weight=3)
        >>> G.add_edge(1, 3, weight=7, capacity=15, length=342.7)

        For non-string attribute keys, use subscript notation:

        >>> G.add_edge(1, 2)
        >>> G[1][2].update({0: 5})
        >>> G.edges[1, 2].update({0: 5})
        """
        ...

    def add_edges_from(self, ebunch_to_add: Iterable[_EdgePlus[_Node]], **attr: Any) -> None:
        """Add all the edges in `ebunch_to_add`.

        Parameters
        ----------
        ebunch_to_add : container of edges
            Each edge given in the container will be added to the graph. The edges must be given as 2-tuples (`u`, `v`) or 3-tuples
            (`u`, `v`, `d`) where `d` is a dictionary containing edge data.
        attr : keyword arguments, optional
            Edge data (or labels or objects) can be assigned using keyword arguments.

        See Also
        --------
        `add_edge` : add a single edge.
        `add_weighted_edges_from` : convenient way to add weighted edges.

        Notes
        -----
        Adding the same edge twice has no effect but any edge data will be updated when each duplicate edge is added.

        Edge attributes specified in `ebunch_to_add` as 3-tuples (`u`, `v`, `d`) take precedence over attributes specified via keyword arguments.

        Examples
        --------
        >>> G = nx.Graph()   # or `DiGraph`, `MultiGraph`, `MultiDiGraph`, etc.
        >>> G.add_edges_from([(0, 1), (1, 2)]) # using a list of edge tuples
        >>> e = zip(range(0, 3), range(1, 4))
        >>> G.add_edges_from(e) # Add the path graph 0-1-2-3

        Associate data to edges:

        >>> G.add_edges_from([(1, 2), (2, 3)], weight=3)
        >>> G.add_edges_from([(3, 4), (1, 4)], label='WN2898')
        """
        ...

    def add_weighted_edges_from(self, ebunch_to_add: Iterable[tuple[_Node, _Node, float | int]], weight: str = "weight", **attr: Any) -> None:
        """Add weighted edges in `ebunch_to_add` with specified weight attribute.

        Parameters
        ----------
        ebunch_to_add : container of edges
            Each edge given in the list or container will be added to the graph. The edges must be given as 3-tuples (`u`, `v`, `w`) where
            `w` is a number.
        weight : string, optional (default= 'weight')
            The attribute name for the edge weights to be added.
        attr : keyword arguments, optional (default= no attributes)
            Edge attributes to add/update for all edges.

        See Also
        --------
        `add_edge` : add a single edge.
        `add_edges_from` : add multiple edges.

        Notes
        -----
        Adding the same edge twice for `Graph`/`DiGraph` simply updates the edge data. For `MultiGraph`/`MultiDiGraph`, duplicate edges are stored.

        Examples
        --------
        >>> G = nx.Graph()   # or `DiGraph`, `MultiGraph`, `MultiDiGraph`, etc.
        >>> G.add_weighted_edges_from([(0, 1, 3.0), (1, 2, 7.5)])
        """
        ...

    def remove_edge(self, u: _Node, v: _Node) -> None:
        """Remove the edge between `u` and `v`.

        Parameters
        ----------
        u, v : nodes
            Remove the edge between nodes `u` and `v`.

        Raises
        ------
        `NetworkXError`
            If there is not an edge between `u` and `v`.

        See Also
        --------
        `remove_edges_from` : remove a collection of edges.

        Examples
        --------
        >>> G = nx.path_graph(4)  # or `DiGraph`, etc.
        >>> G.remove_edge(0, 1)
        >>> e = (1, 2)
        >>> G.remove_edge(*e) # unpacks `e` from an edge tuple.
        >>> e = (2, 3, {'weight': 7}) # an edge with attribute data.
        >>> G.remove_edge(*e[:2]) # select first part of edge tuple.
        """
        ...

    def remove_edges_from(self, ebunch: Iterable[_EdgePlus[_Node]]) -> None:
        """Remove all edges specified in `ebunch`.

        Parameters
        ----------
        ebunch: list or container of edge tuples
            Each edge given in the list or container will be removed from the graph. The edges can be:

                - 2-tuples (`u`, `v`) edge between `u` and `v`.
                - 3-tuples (`u`, `v`, `k`) where `k` is ignored.

        See Also
        --------
        `remove_edge` : remove a single edge.

        Notes
        -----
        Will fail silently if an edge in `ebunch` is not in the graph.

        Examples
        --------
        >>> G = nx.path_graph(4)  # or `DiGraph`, `MultiGraph`, `MultiDiGraph`, etc.
        >>> ebunch = [(1, 2), (2, 3)]
        >>> G.remove_edges_from(ebunch)
        """
        ...

    @overload
    def update(self, edges: Graph[_Node], nodes: None = None) -> None: ...

    @overload
    def update(self, edges: Graph[_Node] | Iterable[_EdgePlus[_Node]] | None = None, nodes: Iterable[_Node] | None = None) -> None: ...

    def update(self, edges: Any = None, nodes: Any = None) -> None:
        """Update the graph using nodes/edges/graphs as input.

        Like `dict.update`, this method takes a graph as input, adding the graph's nodes and edges to this graph. It can also take
        two inputs: edges and nodes. Finally, it can take either edges or nodes. To specify only nodes, the keyword `nodes` must be
        used.

        The collections of edges and nodes are treated similarly to the `add_edges_from`/`add_nodes_from` methods. When iterated, they
        should yield 2-tuples (`u`, `v`) or 3-tuples (`u`, `v`, `datadict`).

        Parameters
        ----------
        edges : `Graph` object, collection of edges, or None
            The first parameter can be a graph or some edges. If it has attributes `nodes` and `edges`, then it is taken to be a
            `Graph`-like object and those attributes are used as collections of nodes and edges to be added to the graph. If the
            first parameter does not have those attributes, it is treated as a collection of edges and added to the graph. If the
            first argument is None, no edges are added.
        nodes : collection of nodes, or None
            The second parameter is treated as a collection of nodes to be added to the graph unless it is None. If `edges is
            None` and `nodes is None`, an exception is raised. If the first parameter is a `Graph`, then `nodes` is ignored.

        Examples
        --------
        >>> G = nx.path_graph(5)
        >>> G.update(nx.complete_graph(range(4, 10)))
        >>> from itertools import combinations
        >>> edges = ((u, v, {'power': u * v})
        ...          for u, v in combinations(range(10, 20), 2)
        ...          if u * v < 225)
        >>> nodes = [1000]  # for singleton, use a container.
        >>> G.update(edges, nodes)

        Notes
        -----
        If you want to update the graph using an adjacency structure, it is straightforward to obtain the edges/nodes from
        adjacency. The following examples provide common cases; your adjacency may be slightly different and require tweaks of
        these examples::

        >>> # dict-of-set/list/tuple
        >>> adj = {1: {2, 3}, 2: {1, 3}, 3: {1, 2}}
        >>> e = [(u, v) for u, nbrs in adj.items() for v in nbrs]
        >>> G.update(edges=e, nodes=adj)

        >>> DG = nx.DiGraph()
        >>> # dict-of-dict-of-attribute
        >>> adj = {1: {2: 1.3, 3: 0.7}, 2: {1: 1.4}, 3: {1: 0.7}}
        >>> e = [(u, v, {'weight': d}) for u, nbrs in adj.items()
        ...      for v, d in nbrs.items()]
        >>> DG.update(edges=e, nodes=adj)

        >>> # dict-of-dict-of-dict
        >>> adj = {1: {2: {'weight': 1.3}, 3: {'color': 0.7, 'weight': 1.2}}}
        >>> e = [(u, v, {'weight': d}) for u, nbrs in adj.items()
        ...      for v, d in nbrs.items()]
        >>> DG.update(edges=e, nodes=adj)

        >>> # predecessor adjacency (dict-of-set)
        >>> pred = {1: {2, 3}, 2: {3}, 3: {}}
        >>> e = [(v, u) for u, nbrs in pred.items() for v in nbrs]

        >>> # MultiGraph dict-of-dict-of-dict-of-attribute
        >>> MDG = nx.MultiDiGraph()
        >>> adj = {1: {2: {0: {'weight': 1.3}, 1: {'weight': 1.2}}},
        ...        3: {2: {0: {'weight': 0.7}}}}
        >>> e = [(u, v, k, d) for u, nbrs in adj.items()
        ...      for v, kd in nbrs.items()
        ...      for k, d in kd.items()]
        >>> MDG.update(edges=e)

        See Also
        --------
        `add_edges_from`: add multiple edges to a graph.
        `add_nodes_from`: add multiple nodes to a graph.
        """
        ...

    def has_edge(self, u: _Node, v: _Node) -> bool:
        """Returns `True` if the edge `(u, v)` is in the graph.

        This is the same as `v in G[u]` without `KeyError` exceptions.

        Parameters
        ----------
        u, v : nodes
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not `None`) Python objects.

        Returns
        -------
        edge_ind : bool
            `True` if edge is in the graph, `False` otherwise.

        Examples
        --------
        >>> G = nx.path_graph(4)  # or `DiGraph`, `MultiGraph`, `MultiDiGraph`, etc.
        >>> G.has_edge(0, 1)  # using two nodes
        True
        >>> e = (0, 1)
        >>> G.has_edge(*e)  #  `e` is a 2-tuple `(u, v)`
        True
        >>> e = (0, 1, {'weight': 7})
        >>> G.has_edge(*e[:2])  # `e` is a 3-tuple `(u, v, data_dictionary)`
        True

        The following syntax are equivalent:

        >>> G.has_edge(0, 1)
        True
        >>> 1 in G[0]  # though this gives `KeyError` if `0` not in `G`
        True
        """
        ...

    def neighbors(self, n: _Node) -> Iterator[_Node]:
        """Returns an iterator over all neighbors of node `n`.

        This is identical to `iter(G[n])`.

        Parameters
        ----------
        n : node
            A node in the graph.

        Returns
        -------
        neighbors : iterator
            An iterator over all neighbors of node `n`.

        Raises
        ------
        `NetworkXError`
            If the node `n` is not in the graph.

        Examples
        --------
        >>> G = nx.path_graph(4)  # or `DiGraph`, `MultiGraph`, `MultiDiGraph`, etc.
        >>> [n for n in G.neighbors(0)]
        [1]

        Notes
        -----
        Alternate ways to access the neighbors are ``G.adj[n]`` or ``G[n]``:

        >>> G = nx.Graph()   # or `DiGraph`, `MultiGraph`, `MultiDiGraph`, etc.
        >>> G.add_edge('a', 'b', weight=7)
        >>> G['a']
        AtlasView({'b': {'weight': 7}})
        >>> G = nx.path_graph(4)
        >>> [n for n in G[0]]
        [1]
        """
        ...

    @cached_property
    def edges(self) -> EdgeView[_Node]:
        """An EdgeView of the Graph as G.edges or G.edges().

        edges(self, nbunch=None, data=False, default=None)

        The EdgeView provides set-like operations on the edge-tuples as well as edge attribute lookup. When called, it also
        provides an EdgeDataView object which allows control of access to edge attributes (but does not provide set-like
        operations). Hence, `G.edges[u, v]['color']` provides the value of the color attribute for edge `(u, v)` while `for (u, v,
        c) in G.edges.data('color', default='red'):` iterates through all the edges yielding the color attribute with default
        `'red'` if no color attribute exists.

        Parameters
        ----------
        nbunch : single node, container, or all nodes (default= all nodes)
            The view will only report edges incident to these nodes.
        data : string or bool, optional (default=False)
            The edge attribute returned in 3-tuple (u, v, ddict[data]). If True, return edge attribute dict in 3-tuple (u, v,
            ddict). If False, return 2-tuple (u, v).
        default : value, optional (default=None)
            Value used for edges that don't have the requested attribute. Only relevant if data is not True or False.

        Returns
        -------
        edges : EdgeView
            A view of edge attributes, usually it iterates over (u, v) or (u, v, d) tuples of edges, but can also be used for
            attribute lookup as `edges[u, v]['foo']`.

        Notes
        -----
        Nodes in nbunch that are not in the graph will be (quietly) ignored. For directed graphs this returns the out-edges.
        """
        ...

    def get_edge_data(self, u: _Node, v: _Node, default: Any = None) -> dict[str, Any]:
        """Returns the attribute dictionary associated with edge (u, v).

        This is identical to `G[u][v]` except the default is returned instead of an exception if the edge doesn't exist.

        Parameters
        ----------
        u, v : nodes
        default:  any Python object (default=None)
            Value to return if the edge (u, v) is not found.

        Returns
        -------
        edge_dict : dictionary
            The edge attribute dictionary.

        Examples
        --------
        >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G[0][1]
        {}

        Warning: Assigning to `G[u][v]` is not permitted.
        But it is safe to assign attributes `G[u][v]['foo']`

        >>> G[0][1]['weight'] = 7
        >>> G[0][1]['weight']
        7
        >>> G[1][2]['weight'] = 10
        >>> G.get_edge_data(1, 2)
        {'weight': 10}
        >>> G.get_edge_data(10, 20)
        >>>
        """
        ...

    def adjacency(self) -> Iterator[tuple[_Node, dict[_Node, dict[str, Any]]]]:
        """Returns an iterator over (node, adjacency dict) tuples for all nodes.

        For directed graphs, only outgoing neighbors/adjacencies are included.

        Returns
        -------
        adj_iter : iterator
            An iterator over (node, adjacency dictionary) for all nodes in
            the graph.

        Examples
        --------
        >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> [(n, nbrdict) for n, nbrdict in G.adjacency()]
        [(0, {1: {}}), (1, {0: {}, 2: {}}), (2, {1: {}, 3: {}}), (3, {2: {}})]

        """
        ...

    @cached_property
    def degree(self) -> int | DegreeView[_Node]:
        """A DegreeView for the Graph as G.degree or G.degree().

        The node degree is the number of edges adjacent to the node. The weighted node degree is the sum of the edge weights for
        edges incident to that node.

        This object provides an iterator for (node, degree) as well as lookup for the degree for a single node.

        Parameters
        ----------
        nbunch : single node, container, or all nodes (default= all nodes)
            The view will only report edges incident to these nodes.
        weight : string or None, optional (default=None)
            The name of an edge attribute that holds the numerical value used as a weight.  If None, then each edge has weight 1.

        Returns
        -------
        deg : int
            If a single node is requested: Degree of the node

        nd_view : DegreeView
            OR if multiple nodes are requested: A DegreeView object capable of iterating (node, degree) pairs

        Examples
        --------
        >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G.degree[0]  # node 0 has degree 1
        1
        >>> list(G.degree([0, 1, 2]))
        [(0, 1), (1, 2), (2, 2)]
        """
        ...

    def clear(self) -> None:
        """Remove all nodes and edges from the graph.

        This also removes the name, and all graph, node, and edge attributes.

        Examples
        --------
        >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G.clear()
        >>> list(G.nodes)
        []
        >>> list(G.edges)
        []
        """
        ...

    def clear_edges(self) -> None:
        """Remove all edges from the graph without altering nodes.

        Examples
        --------
        >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G.clear_edges()
        >>> list(G.nodes)
        [0, 1, 2, 3]
        >>> list(G.edges)
        []
        """
        ...

    def is_multigraph(self) -> bool:
        """Returns True if graph is a multigraph, False otherwise."""
        ...

    def is_directed(self) -> bool:
        """Returns True if graph is directed, False otherwise."""
        ...

    def copy(self, as_view: bool = False) -> Self:
        """Returns a copy of the graph.

        The copy method by default returns a deep copy of the graph and attributes
        using the Python's copy.deepcopy module. If `as_view` is True then a graph
        view is returned instead of a deep copy.

        Parameters
        ----------
        as_view : bool, optional (default=False)
            If True, the returned graph will be a view of the original graph.
            Otherwise, the returned graph is a deep copy of the original graph.

        Returns
        -------
        G : Graph
            A deep copy of the graph.

        Examples
        --------
        >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G[0][1]['weight'] = 3
        >>> H = G.copy()
        >>> H[0][1]['weight']
        3
        """
        ...

    def to_directed(self, as_view: bool = False) -> DiGraph[_Node]:
        """Returns a directed representation of the graph.

        Returns
        -------
        G : DiGraph
            A directed graph with the same name, same nodes, and with
            edges (u, v, data) if either (u, v, data) or (v, u, data)
            is in the graph.  If both edges exist in the graph, the edge data
            from the (u, v) edge is used.

        Notes
        -----
        This returns a "deepcopy" of the edge, node, and
        graph attributes which attempts to completely copy
        all of the data and references.

        This is in contrast to the similar D=DiGraph(G) which returns a
        shallow copy of the data.

        See the Python copy module for more information on shallow
        and deep copies, https://docs.python.org/3/library/copy.html.

        Warning: If you have subclassed Graph to use dict-like objects
        in the data structure, those changes do not transfer to the
        DiGraph created by this method.

        Examples
        --------
        >>> G = nx.Graph()  # or MultiGraph, etc
        >>> G.add_edge(0, 1)
        >>> G.add_edge(1, 0)
        >>> H = G.to_directed()
        >>> list(H.edges)
        [(0, 1), (1, 0)]

        Converting from a directed graph to an undirected graph
        removes all edges if they exist in both directions.

        >>> G = nx.DiGraph()
        >>> G.add_edge(0, 1)
        >>> G.add_edge(1, 0)
        >>> H = G.to_undirected()
        >>> list(H.edges)
        [(0, 1)]
        """
        ...

    def to_undirected(self, as_view: bool = False) -> Graph[_Node]:
        """Returns an undirected copy of the graph.

        Parameters
        ----------
        as_view : bool (optional, default=False)
        If True return a view of the original undirected graph.

        Returns
        -------
        G : Graph
            An undirected graph with the same name and nodes and
            with edge (u, v, data) if either (u, v, data) or (v, u, data)
            is in the original graph.

        Notes
        -----
        This returns a "deepcopy" of the edge, node, and
        graph attributes which attempts to completely copy
        all of the data and references.

        This is in contrast to the similar G=DiGraph(D) which returns a
        shallow copy of the data.

        See the Python copy module for more information on shallow
        and deep copies, https://docs.python.org/3/library/copy.html.

        Warning: If you have subclassed DiGraph to use dict-like objects
        in the data structure, those changes do not transfer to the
        Graph created by this method.

        Examples
        --------
        >>> G = nx.path_graph(2)   # or MultiGraph, etc
        >>> H = G.to_directed()
        >>> list(H.edges)
        [(0, 1), (1, 0)]
        >>> G2 = H.to_undirected()
        >>> list(G2.edges)
        [(0, 1)]
        """
        ...

    def subgraph(self, nodes: Iterable[_Node]) -> Graph[_Node]:
        """Returns a SubGraph view of the subgraph induced on `nodes`.

        The induced subgraph of the graph contains the nodes in `nodes`
        and the edges between those nodes.

        Parameters
        ----------
        nodes : list, iterable
            A container of nodes which will be iterated through once.

        Returns
        -------
        G : SubGraph View
            A subgraph view of the graph. The graph structure cannot be
            changed but node/edge attributes can and are shared with the
            original graph.

        Notes
        -----
        The graph, edge and node attributes are shared with the original graph.
        Changes to the graph structure is ruled out by the view, but changes
        to attributes are reflected in the original graph.

        To create a subgraph with its own copy of the edge/node attributes use:
        G.subgraph(nodes).copy()

        For an inplace reduction of a graph to a subgraph you can remove nodes:
        G.remove_nodes_from([n for n in G if n not in set(nodes)])

        Subgraph views are sometimes NOT what you want. In most cases where
        you want to check if node1 and node2 are connected in the original
        graph, use `G.has_edge(node1, node2)`. Otherwise `G.subgraph([node1, node2]).has_edge(node1, node2)`
        will test whether they are connected in the subgraph, i.e., if node1, node2,
        and atleast one edge connecting them, all exist in the original graph.

        Examples
        --------
        >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> H = G.subgraph([0, 1, 2])
        >>> list(H.edges)
        [(0, 1), (1, 2)]
        """
        ...

    def edge_subgraph(self, edges: Iterable[_Edge[_Node]]) -> Graph[_Node]:
        """Returns the subgraph induced by the specified edges.

        The induced subgraph contains each edge in `edges` and each
        node incident to any of those edges.

        Parameters
        ----------
        edges : iterable
            An iterable of edges in the graph.
            By default, this is an (u, v) tuple.

        Returns
        -------
        G : Graph
            An edge-induced subgraph of the graph with the same edge attributes.

        Notes
        -----
        The graph, edge, and node attributes in the returned subgraph view
        are references to the corresponding attributes in the original graph.
        The view is read-only.

        To create a full graph version of the subgraph with its own copy of the
        edge or node attributes, use::

            >>> G.edge_subgraph(edges).copy()  # doctest: +SKIP

        Examples
        --------
        >>> G = nx.path_graph(5)
        >>> H = G.edge_subgraph([(0, 1), (3, 4)])
        >>> list(H.nodes)
        [0, 1, 3, 4]
        >>> list(H.edges)
        [(0, 1), (3, 4)]
        """
        ...

    @overload
    def size(self, weight: None = None) -> int: ...

    @overload
    def size(self, weight: str) -> float: ...

    def size(self, weight: str | None = None) -> int | float:
        """Returns the number of edges or total of all edge weights.

        Parameters
        ----------
        weight : string or None, optional (default=None)
            The edge attribute that holds the numerical value used
            as a weight. If None, then each edge has weight 1.

        Returns
        -------
        size : numeric
            The number of edges or
            (if weight keyword is provided) the total weight sum.

            If weight is None, returns an int. Otherwise returns a float.

        Examples
        --------
        >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G.size()
        3

        >>> G = nx.Graph()   # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G.add_edge('a', 'b', weight=2)
        >>> G.add_edge('b', 'c', weight=4)
        >>> G.size()
        2
        >>> G.size(weight='weight')
        6.0
        """
        ...

    def number_of_edges(self, u: _Node | None = None, v: _Node | None = None) -> int:
        """Returns the number of edges between two nodes.

        Parameters
        ----------
        u, v : nodes, optional (default=all edges)
            If u and v are specified, return the number of edges between
            u and v. Otherwise return the total number of all edges.

        Returns
        -------
        nedges : int
            The number of edges in the graph.  If nodes u and v are specified
            return the number of edges between those nodes. If the graph is
            directed, this only returns the number of edges from u to v.

        Examples
        --------
        For undirected graphs, this method counts the total number of edges in
        the graph:

        >>> G = nx.path_graph(4)
        >>> G.number_of_edges()
        3

        If you specify two nodes, this counts the total number of edges joining
        the two nodes:

        >>> G.number_of_edges(0, 1)
        1

        For directed graphs, this method can count the total number of
        directed edges from `u` to `v`:

        >>> G = nx.DiGraph()
        >>> G.add_edge(0, 1)
        >>> G.add_edge(1, 0)
        >>> G.number_of_edges(0, 1)
        1
        """
        ...

    def nbunch_iter(self, nbunch: _NBunch[_Node] = None) -> Iterator[_Node]:
        """Returns an iterator over nodes contained in nbunch.

        The input nbunch can be any of:
            - None (all nodes)
            - a node
            - a container of nodes

        If nbunch is None, returns all nodes in the graph. This is the same
        as `G.nodes()`.

        If nbunch is a single node, returns an iterator with that node in it.

        If nbunch is a list, set, or other collection of nodes, returns an
        iterator over those nodes.

        If nbunch contains nodes that are not in the graph, they will not
        be included in the iterator. Prior to NetworkX 2.8, they were silently
        ignored. Use `contains=True` to verify node membership.

        Parameters
        ----------
        nbunch : single node, container, or all nodes (default= all nodes)
            The view will only report edges incident to these nodes.

        Returns
        -------
        node_iter : iterator
            An iterator over nodes in the graph.

        Raises
        ------
        NetworkXError
            If nbunch is not a node or a sequence of nodes.
            Use `contains=True` to verify node membership.
            Prior to NetworkX 2.8, non-existent nodes were silently ignored.

        See Also
        --------
        Graph.__iter__

        Examples
        --------
        >>> G = nx.path_graph(3)
        >>> sorted(G.nbunch_iter())
        [0, 1, 2]
        >>> sorted(G.nbunch_iter(0))
        [0]
        >>> sorted(G.nbunch_iter([0, 1]))
        [0, 1]
        >>> sorted(G.nbunch_iter([0, 10]))  # not in G
        [0]
        """
        ...
