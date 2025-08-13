from collections.abc import Iterable, Iterator
from functools import cached_property
from networkx.classes.coreviews import AdjacencyView
from networkx.classes.graph import _EdgePlus, _Node, _NodePlus, Graph  # type: ignore[reportPrivateUsage]
from networkx.classes.reportviews import (
	DiDegreeView, InDegreeView, InEdgeView, InMultiDegreeView, OutDegreeView, OutEdgeView, OutMultiDegreeView)
from typing import Any
from typing_extensions import Self

__all__ = ['DiGraph']

class _CachedPropertyResetterAdjAndSucc:
	"""Data Descriptor class that syncs and resets cached properties `adj` and `succ`.

	The cached properties `adj` and `succ` are reset whenever `_adj` or `_succ` are set to new objects. In addition, the
	attributes `_succ` and `_adj` are synced so these two names point to the same object.

	Warning: Most of the time, when `G._adj` is set, `G._pred` should also be set to maintain a valid data structure. They
	share datadicts.

	This object sits on a class and ensures that any instance of that class clears its cached properties `succ` and `adj` whenever
	the underlying instance attributes `_succ` or `_adj` are set to a new object. It only affects the set process of the `obj._adj`
	and `obj._succ` attribute. All get/del operations act as they normally would.

	For info on Data Descriptors, see https://docs.python.org/3/howto/descriptor.html.
	"""
	def __set__(self, obj: Any, value: Any) -> None: ...

class _CachedPropertyResetterPred:
	"""Data Descriptor class for `_pred` that resets `pred` cached_property when needed.

	This assumes that the `cached_property` `G.pred` should be reset whenever `G._pred` is set to a new value.

	Warning: Most of the time, when `G._pred` is set, `G._adj` should also be set to maintain a valid data structure. They
	share datadicts.

	This object sits on a class and ensures that any instance of that class clears its cached property `pred` whenever the
	underlying instance attribute `_pred` is set to a new object. It only affects the set process of the `obj._pred` attribute. All
	get/del operations act as they normally would.

	For info on Data Descriptors, see https://docs.python.org/3/howto/descriptor.html.
	"""
	def __set__(self, obj: Any, value: Any) -> None: ...

class DiGraph(Graph[_Node]):
	"""
	Base class for directed graphs.

	A `DiGraph` stores nodes and edges with optional data, or attributes.

	`DiGraph`s hold directed edges. Self loops are allowed but multiple (parallel) edges are not.

	Nodes can be arbitrary (hashable) Python objects with optional key/value attributes. By convention, `None` is not used as a
	node.

	Edges are represented as links between nodes with optional key/value attributes.

	Parameters
	----------
	incoming_graph_data : input graph (optional, default: `None`)
		Data to initialize graph. If `None` (default), an empty graph is created. The data can be any format that is supported by
		the `to_networkx_graph()` function, currently including edge list, dict of dicts, dict of lists, `NetworkX` graph, 2D `NumPy`
		array, `SciPy` sparse matrix, or `PyGraphviz` graph.

	attr : keyword arguments, optional (default= no attributes)
		Attributes to add to graph as key=value pairs.

	See Also
	--------
	`Graph`
	`MultiGraph`
	`MultiDiGraph`

	Examples
	--------
	Create an empty graph structure (a `null graph`) with no nodes and no edges.

	>>> G = nx.DiGraph()

	G can be grown in several ways.

	**Nodes:**

	Add one node at a time:

	>>> G.add_node(1)

	Add the nodes from any container (a list, dict, set or even the lines from a file or the nodes from another graph).

	>>> G.add_nodes_from([2, 3])
	>>> G.add_nodes_from(range(100, 110))
	>>> H = nx.path_graph(10)
	>>> G.add_nodes_from(H)

	In addition to strings and integers any hashable Python object (except None) can represent a node, e.g. a customized node
	object, or even another Graph.

	>>> G.add_node(H)

	**Edges:**

	G can also be grown by adding edges.

	Add one edge,

	>>> G.add_edge(1, 2)

	a list of edges,

	>>> G.add_edges_from([(1, 2), (1, 3)])

	or a collection of edges,

	>>> G.add_edges_from(H.edges)

	If some edges connect nodes not yet in the graph, the nodes are added automatically.  There are no errors when adding nodes or
	edges that already exist.

	**Attributes:**

	Each graph, node, and edge can hold key/value attribute pairs in an associated attribute dictionary (the keys must be
	hashable). By default these are empty, but can be added or changed using add_edge, add_node or direct manipulation of the
	attribute dictionaries named graph, node and edge respectively.

	>>> G = nx.DiGraph(day="Friday")
	>>> G.graph
	{'day': 'Friday'}

	Add node attributes using add_node(), add_nodes_from() or G.nodes

	>>> G.add_node(1, time="5pm")
	>>> G.add_nodes_from([3], time="2pm")
	>>> G.nodes[1]
	{'time': '5pm'}
	>>> G.nodes[1]["room"] = 714
	>>> del G.nodes[1]["room"]  # remove attribute
	>>> list(G.nodes(data=True))
	[(1, {'time': '5pm'}), (3, {'time': '2pm'})]

	Add edge attributes using add_edge(), add_edges_from(), subscript notation, or G.edges.

	>>> G.add_edge(1, 2, weight=4.7)
	>>> G.add_edges_from([(3, 4), (4, 5)], color="red")
	>>> G.add_edges_from([(1, 2, {"color": "blue"}), (2, 3, {"weight": 8})])
	>>> G[1][2]["weight"] = 4.7
	>>> G.edges[1, 2]["weight"] = 4

	Warning: we protect the graph data structure by making `G.edges[1, 2]` a read-only dict-like structure. However, you can
	assign to attributes in e.g. `G.edges[1, 2]`. Thus, use 2 sets of brackets to add/change data attributes: `G.edges[1,
	2]['weight'] = 4` (For multigraphs: `MG.edges[u, v, key][name] = value`).

	**Shortcuts:**

	Many common graph features allow python syntax to speed reporting.

	>>> 1 in G  # check if node in graph
	True
	>>> [n for n in G if n < 3]  # iterate through nodes
	[1, 2]
	>>> len(G)  # number of nodes in graph
	5

	Often the best way to traverse all edges of a graph is via the neighbors. The neighbors are reported as an adjacency-dict
	`G.adj` or `G.adjacency()`

	>>> for n, nbrsdict in G.adjacency():
	...     for nbr, eattr in nbrsdict.items():
	...         if "weight" in eattr:
	...             # Do something useful with the edges
	...             pass

	But the edges reporting object is often more convenient:

	>>> for u, v, weight in G.edges(data="weight"):
	...     if weight is not None:
	...         # Do something useful with the edges
	...         pass

	**Reporting:**

	Simple graph information is obtained using object-attributes and methods. Reporting usually provides views instead of
	containers to reduce memory usage. The views update as the graph is updated similarly to dict-views. The objects `nodes`,
	`edges` and `adj` provide access to data attributes via lookup (e.g. `nodes[n]`, `edges[u, v]`, `adj[u][v]`) and iteration
	(e.g. `nodes.items()`, `nodes.data('color')`, `nodes.data('color', default='blue')` and similarly for `edges`) Views exist for
	`nodes`, `edges`, `neighbors()`/`adj` and `degree`.

	For details on these and other miscellaneous methods, see below.

	**Subclasses (Advanced):**

	The Graph class uses a dict-of-dict-of-dict data structure. The outer dict (node_dict) holds adjacency information keyed by
	node. The next dict (adjlist_dict) represents the adjacency information and holds edge data keyed by neighbor.  The inner dict
	(edge_attr_dict) represents the edge data and holds edge attribute values keyed by attribute names.

	Each of these three dicts can be replaced in a subclass by a user defined dict-like object. In general, the dict-like features
	should be maintained but extra features can be added. To replace one of the dicts create a new graph class by changing the
	class(!) variable holding the factory for that dict-like structure. The variable names are node_dict_factory,
	node_attr_dict_factory, adjlist_inner_dict_factory, adjlist_outer_dict_factory, edge_attr_dict_factory and
	graph_attr_dict_factory.

	node_dict_factory : function, (default: dict)
		Factory function to be used to create the dict containing node attributes, keyed by node id. It should require no
		arguments and return a dict-like object.

	node_attr_dict_factory: function, (default: dict)
		Factory function to be used to create the node attribute dict which holds attribute values keyed by attribute name. It
		should require no arguments and return a dict-like object.

	adjlist_outer_dict_factory : function, (default: dict)
		Factory function to be used to create the outer-most dict in the data structure that holds adjacency info keyed by node.
		It should require no arguments and return a dict-like object.

	adjlist_inner_dict_factory : function, optional (default: dict)
		Factory function to be used to create the adjacency list dict which holds edge data keyed by neighbor. It should require
		no arguments and return a dict-like object.

	edge_attr_dict_factory : function, optional (default: dict)
		Factory function to be used to create the edge attribute dict which holds edge attribute values keyed by attribute name.
		It should require no arguments and return a dict-like object.

	graph_attr_dict_factory : function, (default: dict)
		Factory function to be used to create the graph attribute dict which holds attribute values keyed by attribute name. It
		should require no arguments and return a dict-like object.

	Typically, if your extension doesn't impact the data structure all methods will inherited without issue except:
	`to_directed/to_undirected`. By default these methods create a DiGraph/Graph class and you probably want them to create your
	extension of a DiGraph/Graph. To facilitate this we define two class variables that you can set in your subclass.

	to_directed_class : callable, (default: DiGraph or MultiDiGraph)
		Class to create a new graph structure in the `to_directed` method. If `None`, a NetworkX class (DiGraph or MultiDiGraph)
		is used.

	to_undirected_class : callable, (default: Graph or MultiGraph)
		Class to create a new graph structure in the `to_undirected` method. If `None`, a NetworkX class (Graph or MultiGraph) is
		used.

	**Subclassing Example**

	Create a low memory graph class that effectively disallows edge attributes by using a single attribute dict for all edges.
	This reduces the memory used, but you lose edge attributes.

	>>> class ThinGraph(nx.Graph):
	...     all_edge_dict = {"weight": 1}
	...
	...     def single_edge_dict(self):
	...         return self.all_edge_dict
	...
	...     edge_attr_dict_factory = single_edge_dict
	>>> G = ThinGraph()
	>>> G.add_edge(2, 1)
	>>> G[2][1]
	{'weight': 1}
	>>> G.add_edge(2, 2)
	>>> G[2][1] is G[2][2]
	True
	"""
	_adj: dict[_Node, dict[_Node, dict[str, Any]]]
	_succ: dict[_Node, dict[_Node, dict[str, Any]]]
	_pred: dict[_Node, dict[_Node, dict[str, Any]]]
	graph: dict[str, Any]
	_node: dict[_Node, dict[str, Any]]
	__networkx_cache__: dict[str, Any]

	def __init__(self, incoming_graph_data: Any | None = None, **attr: Any) -> None:
		"""Initialize a graph with edges, name, or graph attributes.

		Parameters
		----------
		incoming_graph_data : input graph (optional, default: None)
			Data to initialize graph.  If None (default) an empty graph is created.  The data can be an edge list, or any NetworkX
			graph object.  If the corresponding optional Python packages are installed the data can also be a 2D NumPy array, a
			SciPy sparse array, or a PyGraphviz graph.

		attr : keyword arguments, optional (default= no attributes)
			Attributes to add to graph as key=value pairs.
		"""
		...

	@cached_property
	def succ(self) -> AdjacencyView[_Node, _Node, dict[str, Any]]:
		"""Graph adjacency object holding the successors of each node.

		This object is a read-only dict-like structure with node keys and neighbor-dict values.  The neighbor-dict is keyed by
		neighbor to the edge-data-dict.  So `G.succ[3][2]['color'] = 'blue'` sets the color of the edge `(3, 2)` to `"blue"`.

		Iterating over G.succ behaves like a dict. Useful idioms include `for nbr, datadict in G.succ[n].items():`.  A data-view
		not provided by dicts also exists: `for nbr, foovalue in G.succ[node].data('foo'):` and a default can be set via a
		`default` argument to the `data` method.

		The neighbor information is also provided by subscripting the graph. So `for nbr, foovalue in G[node].data('foo',
		default=1):` works.

		For directed graphs, `G.adj` is identical to `G.succ`.
		"""
		...

	@cached_property
	def pred(self) -> AdjacencyView[_Node, _Node, dict[str, Any]]:
		"""Graph adjacency object holding the predecessors of each node.

		This object is a read-only dict-like structure with node keys and neighbor-dict values.  The neighbor-dict is keyed by
		neighbor to the edge-data-dict.  So `G.pred[2][3]['color'] = 'blue'` sets the color of the edge `(3, 2)` to `"blue"`.

		Iterating over G.pred behaves like a dict. Useful idioms include `for nbr, datadict in G.pred[n].items():`.  A data-view
		not provided by dicts also exists: `for nbr, foovalue in G.pred[node].data('foo'):` A default can be set via a `default`
		argument to the `data` method.
		"""
		...

	def has_successor(self, u: _Node, v: _Node) -> bool:
		"""Returns True if node u has successor v.

		This is true if graph has the edge u->v.
		"""
		...

	def has_predecessor(self, u: _Node, v: _Node) -> bool:
		"""Returns True if node u has predecessor v.

		This is true if graph has the edge v->u.
		"""
		...

	def successors(self, n: _Node) -> Iterator[_Node]:
		"""Returns an iterator over successor nodes of n.

		A successor of n is a node m such that there exists a directed edge from n to m.

		Parameters
		----------
		n : node
			A node in the graph

		Raises
		------
		NetworkXError
			If n is not in the graph.

		See Also
		--------
		predecessors

		Notes
		-----
		neighbors() and successors() are the same.
		"""
		...

	neighbors = successors

	def predecessors(self, n: _Node) -> Iterator[_Node]:
		"""Returns an iterator over predecessor nodes of n.

		A predecessor of n is a node m such that there exists a directed edge from m to n.

		Parameters
		----------
		n : node
			A node in the graph

		Raises
		------
		NetworkXError
			If n is not in the graph.

		See Also
		--------
		successors
		"""
		...

	@cached_property
	def out_edges(self) -> OutEdgeView[_Node]:
		"""An OutEdgeView of the DiGraph as G.out_edges or G.out_edges().

		out_edges(self, nbunch=None, data=False, default=None)

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
		edges : OutEdgeView
			A view of edge attributes, usually it iterates over (u, v) or (u, v, d) tuples of edges, but can also be used for
			attribute lookup as `edges[u, v]['foo']`.

		See Also
		--------
		in_edges
		"""
		...

	@cached_property
	def in_edges(self) -> InEdgeView[_Node]:
		"""An InEdgeView of the DiGraph as G.in_edges or G.in_edges().

		in_edges(self, nbunch=None, data=False, default=None)

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
		in_edges : InEdgeView
			A view of edge attributes, usually it iterates over (u, v) or (u, v, d) tuples of edges, but can also be used for
			attribute lookup as `edges[u, v]['foo']`.

		See Also
		--------
		`out_edges`
		"""
		...

	@cached_property
	def in_degree(self) -> int | InDegreeView[_Node] | InMultiDegreeView[_Node]:
		"""An InDegreeView for the DiGraph as G.in_degree or G.in_degree().

		The node in_degree is the number of edges pointing to the node. The weighted node degree is the sum of the edge weights
		for edges pointing to that node.

		This object provides an iterator for (node, in_degree) as well as lookup for the in_degree for a single node.

		Parameters
		----------
		nbunch : single node, container, or all nodes (default= all nodes)
			The view will only report edges incident to these nodes.
		weight : string or None, optional (default=None)
			The edge attribute that holds the numerical value used as a weight.  If None, then each edge has weight 1.

		Returns
		-------
		deg : int
			If a single node is requested: In_degree of the node

		nd_iter : iterator
			OR if multiple nodes are requested: The iterator returns two-tuples of (node, in_degree).

		See Also
		--------
		degree, out_degree

		Examples
		--------
		>>> G = nx.DiGraph()
		>>> nx.add_path(G, [0, 1, 2, 3])
		>>> G.in_degree(0) # node 0 with degree 0
		0
		>>> list(G.in_degree([0, 1, 2]))
		[(0, 0), (1, 1), (2, 1)]
		"""
		...

	@cached_property
	def out_degree(self) -> int | OutDegreeView[_Node] | OutMultiDegreeView[_Node]:
		"""An OutDegreeView for (node, out_degree) in the DiGraph as G.out_degree or G.out_degree().

		The node out_degree is the number of edges pointing out of the node. The weighted node degree is the sum of the edge
		weights for edges pointing out of that node.

		This object provides an iterator for (node, out_degree) as well as lookup for the out_degree for a single node.

		Parameters
		----------
		nbunch : single node, container, or all nodes (default= all nodes)
			The view will only report edges incident to these nodes.
		weight : string or None, optional (default=None)
			The edge attribute that holds the numerical value used as a weight.  If None, then each edge has weight 1.

		Returns
		-------
		deg : int
			If a single node is requested: Out_degree of the node

		nd_iter : iterator
			OR if multiple nodes are requested: The iterator returns two-tuples of (node, out_degree).

		See Also
		--------
		degree, in_degree

		Examples
		--------
		>>> G = nx.DiGraph()
		>>> nx.add_path(G, [0, 1, 2, 3])
		>>> G.out_degree(0) # node 0 with degree 1
		1
		>>> list(G.out_degree([0, 1, 2]))
		[(0, 1), (1, 1), (2, 1)]
		"""
		...

	def to_undirected(self, reciprocal: bool = False, as_view: bool = False) -> Graph[_Node]: # pyright: ignore[reportIncompatibleMethodOverride]
		"""Returns an undirected representation of the digraph.

		Parameters
		----------
		reciprocal : bool (optional, default=False)
			If True, only edges that appear in both directions in the original digraph will be kept in the undirected graph.
		as_view : bool (optional, default=False)
			If True return an undirected view of the original directed graph.

		Returns
		-------
		G : Graph
			An undirected graph with the same name and nodes and with edge (u, v, data) if either (u, v, data) or (v, u, data) is
			in the digraph.  If both edges exist in digraph and their edge data is different, only one edge is created with an
			arbitrary choice of which edge data to use. You must check and correct for this manually if desired.

		See Also
		--------
		`Graph`, `copy`, `add_edge`, `add_edges_from`

		Notes
		-----
		If edges in both directions (u, v) and (v, u) exist in the graph, attributes for the new undirected edge will be a
		combination of the attributes of the directed edges.  The edge data is updated in the (arbitrary) order that the edges are
		encountered.  For more customized control of the edge attributes use add_edge().

		This returns a "deepcopy" of the edge, node, and graph attributes which attempts to completely copy all of the data and
		references.

		This is in contrast to the similar G=DiGraph(D) which returns a shallow copy of the data.

		See the Python copy module for more information on shallow and deep copies, https://docs.python.org/3/library/copy.html.

		Warning: If you have subclassed DiGraph to use dict-like objects in the data structure, those changes do not transfer to
		the Graph created by this method.

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

	def reverse(self, copy: bool = True) -> Self:
		"""Returns the reverse of the graph.

		The reverse is a graph with the same nodes and edges but with the directions of the edges reversed.

		Parameters
		----------
		copy : bool optional (default=True)
			If True, return a new DiGraph holding the reversed edges. If False, the reverse graph is created using a view of the
			original graph.
		"""
		...

	def copy(self, as_view: bool = False) -> Self: ...

	@cached_property
	def adj(self) -> AdjacencyView[_Node, _Node, dict[str, Any]]:
		"""Graph adjacency object holding the neighbors of each node.

		This object is a read-only dict-like structure with node keys and neighbor-dict values.  The neighbor-dict is keyed by
		neighbor to the edge-data-dict.  So `G.adj[3][2]['color'] = 'blue'` sets the color of the edge `(3, 2)` to `"blue"`.

		Iterating over G.adj behaves like a dict. Useful idioms include `for nbr, datadict in G.adj[n].items():`.

		The neighbor information is also provided by subscripting the graph. So `for nbr, foovalue in G[node].data('foo',
		default=1):` works.

		For directed graphs, `G.adj` holds outgoing (successor) info.
		"""
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
		"""Remove node n.

		Removes the node n and all adjacent edges. Attempting to remove a nonexistent node will raise an exception.

		Parameters
		----------
		n : node
			A node in the graph

		Raises
		------
		NetworkXError
			If n is not in the graph.
		"""
		...

	def remove_nodes_from(self, nodes: Iterable[_Node]) -> None:
		"""Remove multiple nodes.

		Parameters
		----------
		nodes : iterable container
			A container of nodes (list, dict, set, etc.).  If a node in the container is not in the graph it is silently ignored.
		"""
		...

	def add_edge(self, u_of_edge: _Node, v_of_edge: _Node, **attr: Any) -> None:
		"""Add an edge between u and v.

		The nodes u and v will be automatically added if they are not already in the graph.

		Edge attributes can be specified with keywords or by directly accessing the edge's attribute dictionary. See examples
		below.

		Parameters
		----------
		u_of_edge, v_of_edge : nodes
			Nodes can be, for example, strings or numbers. Nodes must be hashable (and not None) Python objects.
		attr : keyword arguments, optional
			Edge data (or labels or objects) can be assigned using keyword arguments.

		See Also
		--------
		add_edges_from : add a collection of edges

		Notes
		-----
		Adding an edge that already exists updates the edge data.

		Many NetworkX algorithms designed for weighted graphs use an edge attribute (by default `weight`) to hold a numerical
		value.

		Examples
		--------
		The following all add the edge e=(1, 2) to graph G:

		>>> G = nx.Graph()   # or DiGraph, MultiGraph, MultiDiGraph, etc
		>>> e = (1, 2)
		>>> G.add_edge(1, 2)           # explicit two-node form
		>>> G.add_edge(*e)             # single edge as tuple of two nodes
		>>> G.add_edges_from([(1, 2)])  # add edges from iterable container

		Associate data to edges using keywords:

		>>> G.add_edge(1, 2, weight=3)
		>>> G.add_edge(1, 3, weight=7, capacity=15, length=342.7)

		For non-string attribute keys, use subscript notation.

		>>> G.add_edge(1, 2)
		>>> G[1][2].update({0: 5})
		>>> G.edges[1, 2].update({0: 5})
		"""
		...

	def add_edges_from(self, ebunch_to_add: Iterable[_EdgePlus[_Node]], **attr: Any) -> None:
		"""Add all the edges in ebunch_to_add.

		Parameters
		----------
		ebunch_to_add : container of edges
			Each edge given in the container will be added to the graph. The edges must be given as 2-tuples (u, v) or 3-tuples
			(u, v, d) where d is a dictionary containing edge data.
		attr : keyword arguments, optional
			Edge data (or labels or objects) can be assigned using keyword arguments.

		See Also
		--------
		add_edge : add a single edge
		add_weighted_edges_from : convenient way to add weighted edges

		Notes
		-----
		Adding the same edge twice has no effect but any edge data will be updated when each duplicate edge is added.

		Edge attributes specified in an ebunch take precedence over attributes specified via keyword arguments.

		Examples
		--------
		>>> G = nx.Graph()   # or DiGraph, MultiGraph, MultiDiGraph, etc
		>>> G.add_edges_from([(0, 1), (1, 2)]) # using a list of edge tuples
		>>> e = zip(range(0, 3), range(1, 4))
		>>> G.add_edges_from(e) # Add the path graph 0-1-2-3

		Associate data to edges

		>>> G.add_edges_from([(1, 2), (2, 3)], weight=3)
		>>> G.add_edges_from([(3, 4), (1, 4)], label='WN2898')
		"""
		...

	def remove_edge(self, u: _Node, v: _Node) -> None:
		"""Remove the edge between u and v.

		Parameters
		----------
		u, v : nodes
			Remove the edge between nodes u and v.

		Raises
		------
		NetworkXError
			If there is not an edge between u and v.

		See Also
		--------
		remove_edges_from : remove a collection of edges

		Examples
		--------
		>>> G = nx.path_graph(4)  # or DiGraph, etc
		>>> G.remove_edge(0, 1)
		>>> e = (1, 2)
		>>> G.remove_edge(*e) # unpacks e from an edge tuple
		>>> e = (2, 3, {'weight': 7}) # an edge with attribute data
		>>> G.remove_edge(*e[:2]) # select first part of edge tuple
		"""
		...

	def remove_edges_from(self, ebunch: Iterable[_EdgePlus[_Node]]) -> None:
		"""Remove all edges specified in ebunch.

		Parameters
		----------
		ebunch: iterable container of edge tuples
			Each edge given in the list or container will be removed from the graph. The edges can be:

				- 2-tuples (u, v) edge between u and v.
				- 3-tuples (u, v, k) where k is ignored.

		See Also
		--------
		remove_edge : remove a single edge

		Notes
		-----
		Will fail silently if an edge in ebunch is not in the graph.

		Examples
		--------
		>>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
		>>> ebunch = [(1, 2), (2, 3)]
		>>> G.remove_edges_from(ebunch)
		"""
		...

	@cached_property
	def edges(self) -> OutEdgeView[_Node]: # pyright: ignore[reportIncompatibleVariableOverride]
		"""An OutEdgeView of the DiGraph as G.edges or G.edges().

		edges(self, nbunch=None, data=False, default=None)

		The OutEdgeView provides set-like operations on the edge-tuples as well as edge attribute lookup. When called, it also
		provides an EdgeDataView object which allows control of access to edge attributes (but does not provide set-like
		operations). Hence, `G.edges[u, v]['color']` provides the value of the color attribute for edge (u, v) while `for (u, v,
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
		edges : OutEdgeView
			A view of edge attributes, usually it iterates over (u, v) or (u, v, d) tuples of edges, but can also be used for
			attribute lookup as `edges[u, v]['foo']`.

		See Also
		--------
		in_edges, out_edges

		Notes
		-----
		Nodes in nbunch that are not in the graph will be (quietly) ignored. For directed graphs this returns the out-edges.
		"""
		...

	@cached_property
	def degree(self) -> int | DiDegreeView[_Node]: # pyright: ignore[reportIncompatibleVariableOverride]
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

		nd_iter : iterator
			OR if multiple nodes are requested: The iterator returns two-tuples of (node, degree).

		See Also
		--------
		in_degree, out_degree

		Examples
		--------
		>>> G = nx.DiGraph()
		>>> nx.add_path(G, [0, 1, 2, 3])
		>>> G.degree(0) # node 0 with degree 1
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

