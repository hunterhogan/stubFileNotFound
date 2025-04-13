from _typeshed import Incomplete
from collections.abc import Generator

__all__ = ['ISMAGS']

class ISMAGS:
    '''
    Implements the ISMAGS subgraph matching algorithm. [1]_ ISMAGS stands for
    "Index-based Subgraph Matching Algorithm with General Symmetries". As the
    name implies, it is symmetry aware and will only generate non-symmetric
    isomorphisms.

    Notes
    -----
    The implementation imposes additional conditions compared to the VF2
    algorithm on the graphs provided and the comparison functions
    (:attr:`node_equality` and :attr:`edge_equality`):

     - Node keys in both graphs must be orderable as well as hashable.
     - Equality must be transitive: if A is equal to B, and B is equal to C,
       then A must be equal to C.

    Attributes
    ----------
    graph: networkx.Graph
    subgraph: networkx.Graph
    node_equality: collections.abc.Callable
        The function called to see if two nodes should be considered equal.
        It\'s signature looks like this:
        ``f(graph1: networkx.Graph, node1, graph2: networkx.Graph, node2) -> bool``.
        `node1` is a node in `graph1`, and `node2` a node in `graph2`.
        Constructed from the argument `node_match`.
    edge_equality: collections.abc.Callable
        The function called to see if two edges should be considered equal.
        It\'s signature looks like this:
        ``f(graph1: networkx.Graph, edge1, graph2: networkx.Graph, edge2) -> bool``.
        `edge1` is an edge in `graph1`, and `edge2` an edge in `graph2`.
        Constructed from the argument `edge_match`.

    References
    ----------
    .. [1] M. Houbraken, S. Demeyer, T. Michoel, P. Audenaert, D. Colle,
       M. Pickavet, "The Index-Based Subgraph Matching Algorithm with General
       Symmetries (ISMAGS): Exploiting Symmetry for Faster Subgraph
       Enumeration", PLoS One 9(5): e97896, 2014.
       https://doi.org/10.1371/journal.pone.0097896
    '''
    graph: Incomplete
    subgraph: Incomplete
    _symmetry_cache: Incomplete
    _sgn_partitions_: Incomplete
    _sge_partitions_: Incomplete
    _sgn_colors_: Incomplete
    _sge_colors_: Incomplete
    _gn_partitions_: Incomplete
    _ge_partitions_: Incomplete
    _gn_colors_: Incomplete
    _ge_colors_: Incomplete
    _node_compat_: Incomplete
    _edge_compat_: Incomplete
    node_equality: Incomplete
    edge_equality: Incomplete
    def __init__(self, graph, subgraph, node_match: Incomplete | None = None, edge_match: Incomplete | None = None, cache: Incomplete | None = None) -> None:
        """
        Parameters
        ----------
        graph: networkx.Graph
        subgraph: networkx.Graph
        node_match: collections.abc.Callable or None
            Function used to determine whether two nodes are equivalent. Its
            signature should look like ``f(n1: dict, n2: dict) -> bool``, with
            `n1` and `n2` node property dicts. See also
            :func:`~networkx.algorithms.isomorphism.categorical_node_match` and
            friends.
            If `None`, all nodes are considered equal.
        edge_match: collections.abc.Callable or None
            Function used to determine whether two edges are equivalent. Its
            signature should look like ``f(e1: dict, e2: dict) -> bool``, with
            `e1` and `e2` edge property dicts. See also
            :func:`~networkx.algorithms.isomorphism.categorical_edge_match` and
            friends.
            If `None`, all edges are considered equal.
        cache: collections.abc.Mapping
            A cache used for caching graph symmetries.
        """
    @property
    def _sgn_partitions(self): ...
    @property
    def _sge_partitions(self): ...
    @property
    def _gn_partitions(self): ...
    @property
    def _ge_partitions(self): ...
    @property
    def _sgn_colors(self): ...
    @property
    def _sge_colors(self): ...
    @property
    def _gn_colors(self): ...
    @property
    def _ge_colors(self): ...
    @property
    def _node_compatibility(self): ...
    @property
    def _edge_compatibility(self): ...
    @staticmethod
    def _node_match_maker(cmp): ...
    @staticmethod
    def _edge_match_maker(cmp): ...
    def find_isomorphisms(self, symmetry: bool = True) -> Generator[Incomplete, Incomplete, Incomplete]:
        """Find all subgraph isomorphisms between subgraph and graph

        Finds isomorphisms where :attr:`subgraph` <= :attr:`graph`.

        Parameters
        ----------
        symmetry: bool
            Whether symmetry should be taken into account. If False, found
            isomorphisms may be symmetrically equivalent.

        Yields
        ------
        dict
            The found isomorphism mappings of {graph_node: subgraph_node}.
        """
    @staticmethod
    def _find_neighbor_color_count(graph, node, node_color, edge_color):
        """
        For `node` in `graph`, count the number of edges of a specific color
        it has to nodes of a specific color.
        """
    def _get_lookahead_candidates(self):
        """
        Returns a mapping of {subgraph node: collection of graph nodes} for
        which the graph nodes are feasible candidates for the subgraph node, as
        determined by looking ahead one edge.
        """
    def largest_common_subgraph(self, symmetry: bool = True) -> Generator[Incomplete, Incomplete]:
        """
        Find the largest common induced subgraphs between :attr:`subgraph` and
        :attr:`graph`.

        Parameters
        ----------
        symmetry: bool
            Whether symmetry should be taken into account. If False, found
            largest common subgraphs may be symmetrically equivalent.

        Yields
        ------
        dict
            The found isomorphism mappings of {graph_node: subgraph_node}.
        """
    def analyze_symmetry(self, graph, node_partitions, edge_colors):
        """
        Find a minimal set of permutations and corresponding co-sets that
        describe the symmetry of `graph`, given the node and edge equalities
        given by `node_partitions` and `edge_colors`, respectively.

        Parameters
        ----------
        graph : networkx.Graph
            The graph whose symmetry should be analyzed.
        node_partitions : list of sets
            A list of sets containing node keys. Node keys in the same set
            are considered equivalent. Every node key in `graph` should be in
            exactly one of the sets. If all nodes are equivalent, this should
            be ``[set(graph.nodes)]``.
        edge_colors : dict mapping edges to their colors
            A dict mapping every edge in `graph` to its corresponding color.
            Edges with the same color are considered equivalent. If all edges
            are equivalent, this should be ``{e: 0 for e in graph.edges}``.


        Returns
        -------
        set[frozenset]
            The found permutations. This is a set of frozensets of pairs of node
            keys which can be exchanged without changing :attr:`subgraph`.
        dict[collections.abc.Hashable, set[collections.abc.Hashable]]
            The found co-sets. The co-sets is a dictionary of
            ``{node key: set of node keys}``.
            Every key-value pair describes which ``values`` can be interchanged
            without changing nodes less than ``key``.
        """
    def is_isomorphic(self, symmetry: bool = False):
        """
        Returns True if :attr:`graph` is isomorphic to :attr:`subgraph` and
        False otherwise.

        Returns
        -------
        bool
        """
    def subgraph_is_isomorphic(self, symmetry: bool = False):
        """
        Returns True if a subgraph of :attr:`graph` is isomorphic to
        :attr:`subgraph` and False otherwise.

        Returns
        -------
        bool
        """
    def isomorphisms_iter(self, symmetry: bool = True) -> Generator[Incomplete, Incomplete]:
        """
        Does the same as :meth:`find_isomorphisms` if :attr:`graph` and
        :attr:`subgraph` have the same number of nodes.
        """
    def subgraph_isomorphisms_iter(self, symmetry: bool = True):
        """Alternative name for :meth:`find_isomorphisms`."""
    def _find_nodecolor_candidates(self):
        """
        Per node in subgraph find all nodes in graph that have the same color.
        """
    @staticmethod
    def _make_constraints(cosets):
        """
        Turn cosets into constraints.
        """
    @staticmethod
    def _find_node_edge_color(graph, node_colors, edge_colors):
        """
        For every node in graph, come up with a color that combines 1) the
        color of the node, and 2) the number of edges of a color to each type
        of node.
        """
    @staticmethod
    def _get_permutations_by_length(items) -> Generator[Incomplete, Incomplete]:
        """
        Get all permutations of items, but only permute items with the same
        length.

        >>> found = list(ISMAGS._get_permutations_by_length([[1], [2], [3, 4], [4, 5]]))
        >>> answer = [
        ...     (([1], [2]), ([3, 4], [4, 5])),
        ...     (([1], [2]), ([4, 5], [3, 4])),
        ...     (([2], [1]), ([3, 4], [4, 5])),
        ...     (([2], [1]), ([4, 5], [3, 4])),
        ... ]
        >>> found == answer
        True
        """
    @classmethod
    def _refine_node_partitions(cls, graph, node_partitions, edge_colors, branch: bool = False) -> Generator[Incomplete, Incomplete, Incomplete]:
        """
        Given a partition of nodes in graph, make the partitions smaller such
        that all nodes in a partition have 1) the same color, and 2) the same
        number of edges to specific other partitions.
        """
    def _edges_of_same_color(self, sgn1, sgn2):
        """
        Returns all edges in :attr:`graph` that have the same colour as the
        edge between sgn1 and sgn2 in :attr:`subgraph`.
        """
    def _map_nodes(self, sgn, candidates, constraints, mapping: Incomplete | None = None, to_be_mapped: Incomplete | None = None) -> Generator[Incomplete, Incomplete, Incomplete]:
        """
        Find all subgraph isomorphisms honoring constraints.
        """
    def _largest_common_subgraph(self, candidates, constraints, to_be_mapped: Incomplete | None = None) -> Generator[Incomplete, Incomplete, Incomplete]:
        """
        Find all largest common subgraphs honoring constraints.
        """
    @staticmethod
    def _remove_node(node, nodes, constraints):
        """
        Returns a new set where node has been removed from nodes, subject to
        symmetry constraints. We know, that for every constraint we have
        those subgraph nodes are equal. So whenever we would remove the
        lower part of a constraint, remove the higher instead.
        """
    @staticmethod
    def _find_permutations(top_partitions, bottom_partitions):
        """
        Return the pairs of top/bottom partitions where the partitions are
        different. Ensures that all partitions in both top and bottom
        partitions have size 1.
        """
    @staticmethod
    def _update_orbits(orbits, permutations) -> None:
        """
        Update orbits based on permutations. Orbits is modified in place.
        For every pair of items in permutations their respective orbits are
        merged.
        """
    def _couple_nodes(self, top_partitions, bottom_partitions, pair_idx, t_node, b_node, graph, edge_colors) -> Generator[Incomplete]:
        """
        Generate new partitions from top and bottom_partitions where t_node is
        coupled to b_node. pair_idx is the index of the partitions where t_ and
        b_node can be found.
        """
    def _process_ordered_pair_partitions(self, graph, top_partitions, bottom_partitions, edge_colors, orbits: Incomplete | None = None, cosets: Incomplete | None = None):
        """
        Processes ordered pair partitions as per the reference paper. Finds and
        returns all permutations and cosets that leave the graph unchanged.
        """
