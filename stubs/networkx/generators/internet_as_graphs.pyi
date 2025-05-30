from _typeshed import Incomplete

__all__ = ['random_internet_as_graph']

class AS_graph_generator:
    """Generates random internet AS graphs."""
    seed: Incomplete
    n_t: Incomplete
    n_m: Incomplete
    n_cp: Incomplete
    n_c: Incomplete
    d_m: Incomplete
    d_cp: Incomplete
    d_c: Incomplete
    p_m_m: Incomplete
    p_cp_m: Incomplete
    p_cp_cp: Incomplete
    t_m: float
    t_cp: float
    t_c: float
    def __init__(self, n, seed) -> None:
        '''Initializes variables. Immediate numbers are taken from [1].

        Parameters
        ----------
        n: integer
            Number of graph nodes
        seed: random state
            Indicator of random number generation state.
            See :ref:`Randomness<randomness>`.

        Returns
        -------
        GG: AS_graph_generator object

        References
        ----------
        [1] A. Elmokashfi, A. Kvalbein and C. Dovrolis, "On the Scalability of
        BGP: The Role of Topology Growth," in IEEE Journal on Selected Areas
        in Communications, vol. 28, no. 8, pp. 1250-1261, October 2010.
        '''
    G: Incomplete
    def t_graph(self):
        """Generates the core mesh network of tier one nodes of a AS graph.

        Returns
        -------
        G: Networkx Graph
            Core network
        """
    def add_edge(self, i, j, kind) -> None: ...
    def choose_peer_pref_attach(self, node_list):
        """Pick a node with a probability weighted by its peer degree.

        Pick a node from node_list with preferential attachment
        computed only on their peer degree
        """
    def choose_node_pref_attach(self, node_list):
        """Pick a node with a probability weighted by its degree.

        Pick a node from node_list with preferential attachment
        computed on their degree
        """
    def add_customer(self, i, j) -> None:
        """Keep the dictionaries 'customers' and 'providers' consistent."""
    def add_node(self, i, kind, reg2prob, avg_deg, t_edge_prob):
        """Add a node and its customer transit edges to the graph.

        Parameters
        ----------
        i: object
            Identifier of the new node
        kind: string
            Type of the new node. Options are: 'M' for middle node, 'CP' for
            content provider and 'C' for customer.
        reg2prob: float
            Probability the new node can be in two different regions.
        avg_deg: float
            Average number of transit nodes of which node i is customer.
        t_edge_prob: float
            Probability node i establish a customer transit edge with a tier
            one (T) node

        Returns
        -------
        i: object
            Identifier of the new node
        """
    def add_m_peering_link(self, m, to_kind):
        """Add a peering link between two middle tier (M) nodes.

        Target node j is drawn considering a preferential attachment based on
        other M node peering degree.

        Parameters
        ----------
        m: object
            Node identifier
        to_kind: string
            type for target node j (must be always M)

        Returns
        -------
        success: boolean
        """
    def add_cp_peering_link(self, cp, to_kind):
        """Add a peering link to a content provider (CP) node.

        Target node j can be CP or M and it is drawn uniformly among the nodes
        belonging to the same region as cp.

        Parameters
        ----------
        cp: object
            Node identifier
        to_kind: string
            type for target node j (must be M or CP)

        Returns
        -------
        success: boolean
        """
    regions: Incomplete
    def graph_regions(self, rn) -> None:
        """Initializes AS network regions.

        Parameters
        ----------
        rn: integer
            Number of regions
        """
    def add_peering_links(self, from_kind, to_kind) -> None:
        """Utility function to add peering links among node groups."""
    customers: Incomplete
    providers: Incomplete
    nodes: Incomplete
    def generate(self):
        '''Generates a random AS network graph as described in [1].

        Returns
        -------
        G: Graph object

        Notes
        -----
        The process steps are the following: first we create the core network
        of tier one nodes, then we add the middle tier (M), the content
        provider (CP) and the customer (C) nodes along with their transit edges
        (link i,j means i is customer of j). Finally we add peering links
        between M nodes, between M and CP nodes and between CP node couples.
        For a detailed description of the algorithm, please refer to [1].

        References
        ----------
        [1] A. Elmokashfi, A. Kvalbein and C. Dovrolis, "On the Scalability of
        BGP: The Role of Topology Growth," in IEEE Journal on Selected Areas
        in Communications, vol. 28, no. 8, pp. 1250-1261, October 2010.
        '''

def random_internet_as_graph(n, seed: Incomplete | None = None):
    '''Generates a random undirected graph resembling the Internet AS network

    Parameters
    ----------
    n: integer in [1000, 10000]
        Number of graph nodes
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G: Networkx Graph object
        A randomly generated undirected graph

    Notes
    -----
    This algorithm returns an undirected graph resembling the Internet
    Autonomous System (AS) network, it uses the approach by Elmokashfi et al.
    [1]_ and it grants the properties described in the related paper [1]_.

    Each node models an autonomous system, with an attribute \'type\' specifying
    its kind; tier-1 (T), mid-level (M), customer (C) or content-provider (CP).
    Each edge models an ADV communication link (hence, bidirectional) with
    attributes:

      - type: transit|peer, the kind of commercial agreement between nodes;
      - customer: <node id>, the identifier of the node acting as customer
        (\'none\' if type is peer).

    References
    ----------
    .. [1] A. Elmokashfi, A. Kvalbein and C. Dovrolis, "On the Scalability of
       BGP: The Role of Topology Growth," in IEEE Journal on Selected Areas
       in Communications, vol. 28, no. 8, pp. 1250-1261, October 2010.
    '''
