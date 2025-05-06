from _typeshed import Incomplete
from sympy import Matrix as Matrix, cos as cos, pi as pi, sin as sin
from sympy.core.add import Add as Add
from sympy.core.evalf import INF as INF
from sympy.core.mul import Mul as Mul
from sympy.core.symbol import Symbol as Symbol
from sympy.core.sympify import sympify as sympify
from sympy.external.importtools import import_module as import_module
from sympy.functions.elementary.miscellaneous import sqrt as sqrt
from sympy.matrices.dense import zeros as zeros
from sympy.physics.units.quantities import Quantity as Quantity
from sympy.plotting import plot as plot
from sympy.utilities.decorator import doctest_depends_on as doctest_depends_on

__doctest_requires__: Incomplete
numpy: Incomplete

class Truss:
    '''
    A Truss is an assembly of members such as beams,
    connected by nodes, that create a rigid structure.
    In engineering, a truss is a structure that
    consists of two-force members only.

    Trusses are extremely important in engineering applications
    and can be seen in numerous real-world applications like bridges.

    Examples
    ========

    There is a Truss consisting of four nodes and five
    members connecting the nodes. A force P acts
    downward on the node D and there also exist pinned
    and roller joints on the nodes A and B respectively.

    .. image:: truss_example.png

    >>> from sympy.physics.continuum_mechanics.truss import Truss
    >>> t = Truss()
    >>> t.add_node(("node_1", 0, 0), ("node_2", 6, 0), ("node_3", 2, 2), ("node_4", 2, 0))
    >>> t.add_member(("member_1", "node_1", "node_4"), ("member_2", "node_2", "node_4"), ("member_3", "node_1", "node_3"))
    >>> t.add_member(("member_4", "node_2", "node_3"), ("member_5", "node_3", "node_4"))
    >>> t.apply_load(("node_4", 10, 270))
    >>> t.apply_support(("node_1", "pinned"), ("node_2", "roller"))
    '''
    _nodes: Incomplete
    _members: Incomplete
    _loads: Incomplete
    _supports: Incomplete
    _node_labels: Incomplete
    _node_positions: Incomplete
    _node_position_x: Incomplete
    _node_position_y: Incomplete
    _nodes_occupied: Incomplete
    _member_lengths: Incomplete
    _reaction_loads: Incomplete
    _internal_forces: Incomplete
    _node_coordinates: Incomplete
    def __init__(self) -> None:
        """
        Initializes the class
        """
    @property
    def nodes(self):
        """
        Returns the nodes of the truss along with their positions.
        """
    @property
    def node_labels(self):
        """
        Returns the node labels of the truss.
        """
    @property
    def node_positions(self):
        """
        Returns the positions of the nodes of the truss.
        """
    @property
    def members(self):
        """
        Returns the members of the truss along with the start and end points.
        """
    @property
    def member_lengths(self):
        """
        Returns the length of each member of the truss.
        """
    @property
    def supports(self):
        """
        Returns the nodes with provided supports along with the kind of support provided i.e.
        pinned or roller.
        """
    @property
    def loads(self):
        """
        Returns the loads acting on the truss.
        """
    @property
    def reaction_loads(self):
        """
        Returns the reaction forces for all supports which are all initialized to 0.
        """
    @property
    def internal_forces(self):
        """
        Returns the internal forces for all members which are all initialized to 0.
        """
    def add_node(self, *args) -> None:
        """
        This method adds a node to the truss along with its name/label and its location.
        Multiple nodes can be added at the same time.

        Parameters
        ==========
        The input(s) for this method are tuples of the form (label, x, y).

        label:  String or a Symbol
            The label for a node. It is the only way to identify a particular node.

        x: Sympifyable
            The x-coordinate of the position of the node.

        y: Sympifyable
            The y-coordinate of the position of the node.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node(('A', 0, 0))
        >>> t.nodes
        [('A', 0, 0)]
        >>> t.add_node(('B', 3, 0), ('C', 4, 1))
        >>> t.nodes
        [('A', 0, 0), ('B', 3, 0), ('C', 4, 1)]
        """
    def remove_node(self, *args) -> None:
        """
        This method removes a node from the truss.
        Multiple nodes can be removed at the same time.

        Parameters
        ==========
        The input(s) for this method are the labels of the nodes to be removed.

        label:  String or Symbol
            The label of the node to be removed.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node(('A', 0, 0), ('B', 3, 0), ('C', 5, 0))
        >>> t.nodes
        [('A', 0, 0), ('B', 3, 0), ('C', 5, 0)]
        >>> t.remove_node('A', 'C')
        >>> t.nodes
        [('B', 3, 0)]
        """
    def add_member(self, *args) -> None:
        """
        This method adds a member between any two nodes in the given truss.

        Parameters
        ==========
        The input(s) of the method are tuple(s) of the form (label, start, end).

        label: String or Symbol
            The label for a member. It is the only way to identify a particular member.

        start: String or Symbol
            The label of the starting point/node of the member.

        end: String or Symbol
            The label of the ending point/node of the member.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node(('A', 0, 0), ('B', 3, 0), ('C', 2, 2))
        >>> t.add_member(('AB', 'A', 'B'), ('BC', 'B', 'C'))
        >>> t.members
        {'AB': ['A', 'B'], 'BC': ['B', 'C']}
        """
    def remove_member(self, *args) -> None:
        """
        This method removes members from the given truss.

        Parameters
        ==========
        labels: String or Symbol
            The label for the member to be removed.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node(('A', 0, 0), ('B', 3, 0), ('C', 2, 2))
        >>> t.add_member(('AB', 'A', 'B'), ('AC', 'A', 'C'), ('BC', 'B', 'C'))
        >>> t.members
        {'AB': ['A', 'B'], 'AC': ['A', 'C'], 'BC': ['B', 'C']}
        >>> t.remove_member('AC', 'BC')
        >>> t.members
        {'AB': ['A', 'B']}
        """
    def change_node_label(self, *args) -> None:
        """
        This method changes the label(s) of the specified node(s).

        Parameters
        ==========
        The input(s) of this method are tuple(s) of the form (label, new_label).

        label: String or Symbol
            The label of the node for which the label has
            to be changed.

        new_label: String or Symbol
            The new label of the node.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node(('A', 0, 0), ('B', 3, 0))
        >>> t.nodes
        [('A', 0, 0), ('B', 3, 0)]
        >>> t.change_node_label(('A', 'C'), ('B', 'D'))
        >>> t.nodes
        [('C', 0, 0), ('D', 3, 0)]
        """
    def change_member_label(self, *args) -> None:
        """
        This method changes the label(s) of the specified member(s).

        Parameters
        ==========
        The input(s) of this method are tuple(s) of the form (label, new_label)

        label: String or Symbol
            The label of the member for which the label has
            to be changed.

        new_label: String or Symbol
            The new label of the member.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node(('A', 0, 0), ('B', 3, 0), ('D', 5, 0))
        >>> t.nodes
        [('A', 0, 0), ('B', 3, 0), ('D', 5, 0)]
        >>> t.change_node_label(('A', 'C'))
        >>> t.nodes
        [('C', 0, 0), ('B', 3, 0), ('D', 5, 0)]
        >>> t.add_member(('BC', 'B', 'C'), ('BD', 'B', 'D'))
        >>> t.members
        {'BC': ['B', 'C'], 'BD': ['B', 'D']}
        >>> t.change_member_label(('BC', 'BC_new'), ('BD', 'BD_new'))
        >>> t.members
        {'BC_new': ['B', 'C'], 'BD_new': ['B', 'D']}
        """
    def apply_load(self, *args) -> None:
        """
        This method applies external load(s) at the specified node(s).

        Parameters
        ==========
        The input(s) of the method are tuple(s) of the form (location, magnitude, direction).

        location: String or Symbol
            Label of the Node at which load is applied.

        magnitude: Sympifyable
            Magnitude of the load applied. It must always be positive and any changes in
            the direction of the load are not reflected here.

        direction: Sympifyable
            The angle, in degrees, that the load vector makes with the horizontal
            in the counter-clockwise direction. It takes the values 0 to 360,
            inclusive.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> from sympy import symbols
        >>> t = Truss()
        >>> t.add_node(('A', 0, 0), ('B', 3, 0))
        >>> P = symbols('P')
        >>> t.apply_load(('A', P, 90), ('A', P/2, 45), ('A', P/4, 90))
        >>> t.loads
        {'A': [[P, 90], [P/2, 45], [P/4, 90]]}
        """
    def remove_load(self, *args) -> None:
        """
        This method removes already
        present external load(s) at specified node(s).

        Parameters
        ==========
        The input(s) of this method are tuple(s) of the form (location, magnitude, direction).

        location: String or Symbol
            Label of the Node at which load is applied and is to be removed.

        magnitude: Sympifyable
            Magnitude of the load applied.

        direction: Sympifyable
            The angle, in degrees, that the load vector makes with the horizontal
            in the counter-clockwise direction. It takes the values 0 to 360,
            inclusive.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> from sympy import symbols
        >>> t = Truss()
        >>> t.add_node(('A', 0, 0), ('B', 3, 0))
        >>> P = symbols('P')
        >>> t.apply_load(('A', P, 90), ('A', P/2, 45), ('A', P/4, 90))
        >>> t.loads
        {'A': [[P, 90], [P/2, 45], [P/4, 90]]}
        >>> t.remove_load(('A', P/4, 90), ('A', P/2, 45))
        >>> t.loads
        {'A': [[P, 90]]}
        """
    def apply_support(self, *args) -> None:
        """
        This method adds a pinned or roller support at specified node(s).

        Parameters
        ==========
        The input(s) of this method are of the form (location, type).

        location: String or Symbol
            Label of the Node at which support is added.

        type: String
            Type of the support being provided at the node.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node(('A', 0, 0), ('B', 3, 0))
        >>> t.apply_support(('A', 'pinned'), ('B', 'roller'))
        >>> t.supports
        {'A': 'pinned', 'B': 'roller'}
        """
    def remove_support(self, *args) -> None:
        """
        This method removes support from specified node(s.)

        Parameters
        ==========

        locations: String or Symbol
            Label of the Node(s) at which support is to be removed.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node(('A', 0, 0), ('B', 3, 0))
        >>> t.apply_support(('A', 'pinned'), ('B', 'roller'))
        >>> t.supports
        {'A': 'pinned', 'B': 'roller'}
        >>> t.remove_support('A','B')
        >>> t.supports
        {}
        """
    def solve(self) -> None:
        '''
        This method solves for all reaction forces of all supports and all internal forces
        of all the members in the truss, provided the Truss is solvable.

        A Truss is solvable if the following condition is met,

        2n >= r + m

        Where n is the number of nodes, r is the number of reaction forces, where each pinned
        support has 2 reaction forces and each roller has 1, and m is the number of members.

        The given condition is derived from the fact that a system of equations is solvable
        only when the number of variables is lesser than or equal to the number of equations.
        Equilibrium Equations in x and y directions give two equations per node giving 2n number
        equations. However, the truss needs to be stable as well and may be unstable if 2n > r + m.
        The number of variables is simply the sum of the number of reaction forces and member
        forces.

        .. note::
           The sign convention for the internal forces present in a member revolves around whether each
           force is compressive or tensile. While forming equations for each node, internal force due
           to a member on the node is assumed to be away from the node i.e. each force is assumed to
           be compressive by default. Hence, a positive value for an internal force implies the
           presence of compressive force in the member and a negative value implies a tensile force.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node(("node_1", 0, 0), ("node_2", 6, 0), ("node_3", 2, 2), ("node_4", 2, 0))
        >>> t.add_member(("member_1", "node_1", "node_4"), ("member_2", "node_2", "node_4"), ("member_3", "node_1", "node_3"))
        >>> t.add_member(("member_4", "node_2", "node_3"), ("member_5", "node_3", "node_4"))
        >>> t.apply_load(("node_4", 10, 270))
        >>> t.apply_support(("node_1", "pinned"), ("node_2", "roller"))
        >>> t.solve()
        >>> t.reaction_loads
        {\'R_node_1_x\': 0, \'R_node_1_y\': 20/3, \'R_node_2_y\': 10/3}
        >>> t.internal_forces
        {\'member_1\': 20/3, \'member_2\': 20/3, \'member_3\': -20*sqrt(2)/3, \'member_4\': -10*sqrt(5)/3, \'member_5\': 10}
        '''
    def draw(self, subs_dict: Incomplete | None = None):
        '''
        Returns a plot object of the Truss with all its nodes, members,
        supports and loads.

        .. note::
            The user must be careful while entering load values in their
            directions. The draw function assumes a sign convention that
            is used for plotting loads.

            Given a right-handed coordinate system with XYZ coordinates,
            the supports are assumed to be such that the reaction forces of a
            pinned support is in the +X and +Y direction while those of a
            roller support is in the +Y direction. For the load, the range
            of angles, one can input goes all the way to 360 degrees which, in the
            the plot is the angle that the load vector makes with the positive x-axis in the anticlockwise direction.

            For example, for a 90-degree angle, the load will be a vertically
            directed along +Y while a 270-degree angle denotes a vertical
            load as well but along -Y.

        Examples
        ========

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.truss import Truss
            >>> import math
            >>> t = Truss()
            >>> t.add_node(("A", -4, 0), ("B", 0, 0), ("C", 4, 0), ("D", 8, 0))
            >>> t.add_node(("E", 6, 2/math.sqrt(3)))
            >>> t.add_node(("F", 2, 2*math.sqrt(3)))
            >>> t.add_node(("G", -2, 2/math.sqrt(3)))
            >>> t.add_member(("AB","A","B"), ("BC","B","C"), ("CD","C","D"))
            >>> t.add_member(("AG","A","G"), ("GB","G","B"), ("GF","G","F"))
            >>> t.add_member(("BF","B","F"), ("FC","F","C"), ("CE","C","E"))
            >>> t.add_member(("FE","F","E"), ("DE","D","E"))
            >>> t.apply_support(("A","pinned"), ("D","roller"))
            >>> t.apply_load(("G", 3, 90), ("E", 3, 90), ("F", 2, 90))
            >>> p = t.draw()
            >>> p  # doctest: +ELLIPSIS
            Plot object containing:
            [0]: cartesian line: 1 for x over (1.0, 1.0)
            ...
            >>> p.show()
        '''
    def _draw_nodes(self, subs_dict): ...
    def _draw_members(self): ...
    def _draw_supports(self): ...
    def _draw_loads(self): ...
