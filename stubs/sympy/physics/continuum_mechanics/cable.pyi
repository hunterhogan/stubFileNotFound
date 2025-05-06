from _typeshed import Incomplete
from sympy import atan as atan, cos as cos, diff as diff, pi as pi, sin as sin

class Cable:
    """
    Cables are structures in engineering that support
    the applied transverse loads through the tensile
    resistance developed in its members.

    Cables are widely used in suspension bridges, tension
    leg offshore platforms, transmission lines, and find
    use in several other engineering applications.

    Examples
    ========
    A cable is supported at (0, 10) and (10, 10). Two point loads
    acting vertically downwards act on the cable, one with magnitude 3 kN
    and acting 2 meters from the left support and 3 meters below it, while
    the other with magnitude 2 kN is 6 meters from the left support and
    6 meters below it.

    >>> from sympy.physics.continuum_mechanics.cable import Cable
    >>> c = Cable(('A', 0, 10), ('B', 10, 10))
    >>> c.apply_load(-1, ('P', 2, 7, 3, 270))
    >>> c.apply_load(-1, ('Q', 6, 4, 2, 270))
    >>> c.loads
    {'distributed': {}, 'point_load': {'P': [3, 270], 'Q': [2, 270]}}
    >>> c.loads_position
    {'P': [2, 7], 'Q': [6, 4]}
    """
    _left_support: Incomplete
    _right_support: Incomplete
    _supports: Incomplete
    _support_labels: Incomplete
    _loads: Incomplete
    _loads_position: Incomplete
    _length: int
    _reaction_loads: Incomplete
    _tension: Incomplete
    _lowest_x_global: Incomplete
    def __init__(self, support_1, support_2) -> None:
        """
        Initializes the class.

        Parameters
        ==========

        support_1 and support_2 are tuples of the form
        (label, x, y), where

        label : String or symbol
            The label of the support

        x : Sympifyable
            The x coordinate of the position of the support

        y : Sympifyable
            The y coordinate of the position of the support
        """
    @property
    def supports(self):
        """
        Returns the supports of the cable along with their
        positions.
        """
    @property
    def left_support(self):
        """
        Returns the position of the left support.
        """
    @property
    def right_support(self):
        """
        Returns the position of the right support.
        """
    @property
    def loads(self):
        """
        Returns the magnitude and direction of the loads
        acting on the cable.
        """
    @property
    def loads_position(self):
        """
        Returns the position of the point loads acting on the
        cable.
        """
    @property
    def length(self):
        """
        Returns the length of the cable.
        """
    @property
    def reaction_loads(self):
        """
        Returns the reaction forces at the supports, which are
        initialized to 0.
        """
    @property
    def tension(self):
        """
        Returns the tension developed in the cable due to the loads
        applied.
        """
    def tension_at(self, x):
        """
        Returns the tension at a given value of x developed due to
        distributed load.
        """
    def apply_length(self, length) -> None:
        """
        This method specifies the length of the cable

        Parameters
        ==========

        length : Sympifyable
            The length of the cable

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.cable import Cable
        >>> c = Cable(('A', 0, 10), ('B', 10, 10))
        >>> c.apply_length(20)
        >>> c.length
        20
        """
    def change_support(self, label, new_support) -> None:
        """
        This method changes the mentioned support with a new support.

        Parameters
        ==========
        label: String or symbol
            The label of the support to be changed

        new_support: Tuple of the form (new_label, x, y)
            new_label: String or symbol
                The label of the new support

            x: Sympifyable
                The x-coordinate of the position of the new support.

            y: Sympifyable
                The y-coordinate of the position of the new support.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.cable import Cable
        >>> c = Cable(('A', 0, 10), ('B', 10, 10))
        >>> c.supports
        {'A': [0, 10], 'B': [10, 10]}
        >>> c.change_support('B', ('C', 5, 6))
        >>> c.supports
        {'A': [0, 10], 'C': [5, 6]}
        """
    def apply_load(self, order, load) -> None:
        """
        This method adds load to the cable.

        Parameters
        ==========

        order : Integer
            The order of the applied load.

                - For point loads, order = -1
                - For distributed load, order = 0

        load : tuple

            * For point loads, load is of the form (label, x, y, magnitude, direction), where:

            label : String or symbol
                The label of the load

            x : Sympifyable
                The x coordinate of the position of the load

            y : Sympifyable
                The y coordinate of the position of the load

            magnitude : Sympifyable
                The magnitude of the load. It must always be positive

            direction : Sympifyable
                The angle, in degrees, that the load vector makes with the horizontal
                in the counter-clockwise direction. It takes the values 0 to 360,
                inclusive.


            * For uniformly distributed load, load is of the form (label, magnitude)

            label : String or symbol
                The label of the load

            magnitude : Sympifyable
                The magnitude of the load. It must always be positive

        Examples
        ========

        For a point load of magnitude 12 units inclined at 30 degrees with the horizontal:

        >>> from sympy.physics.continuum_mechanics.cable import Cable
        >>> c = Cable(('A', 0, 10), ('B', 10, 10))
        >>> c.apply_load(-1, ('Z', 5, 5, 12, 30))
        >>> c.loads
        {'distributed': {}, 'point_load': {'Z': [12, 30]}}
        >>> c.loads_position
        {'Z': [5, 5]}


        For a uniformly distributed load of magnitude 9 units:

        >>> from sympy.physics.continuum_mechanics.cable import Cable
        >>> c = Cable(('A', 0, 10), ('B', 10, 10))
        >>> c.apply_load(0, ('X', 9))
        >>> c.loads
        {'distributed': {'X': 9}, 'point_load': {}}
        """
    def remove_loads(self, *args) -> None:
        """
        This methods removes the specified loads.

        Parameters
        ==========
        This input takes multiple label(s) as input
        label(s): String or symbol
            The label(s) of the loads to be removed.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.cable import Cable
        >>> c = Cable(('A', 0, 10), ('B', 10, 10))
        >>> c.apply_load(-1, ('Z', 5, 5, 12, 30))
        >>> c.loads
        {'distributed': {}, 'point_load': {'Z': [12, 30]}}
        >>> c.remove_loads('Z')
        >>> c.loads
        {'distributed': {}, 'point_load': {}}
        """
    def solve(self, *args):
        '''
        This method solves for the reaction forces at the supports, the tension developed in
        the cable, and updates the length of the cable.

        Parameters
        ==========
        This method requires no input when solving for point loads
        For distributed load, the x and y coordinates of the lowest point of the cable are
        required as

        x: Sympifyable
            The x coordinate of the lowest point

        y: Sympifyable
            The y coordinate of the lowest point

        Examples
        ========
        For point loads,

        >>> from sympy.physics.continuum_mechanics.cable import Cable
        >>> c = Cable(("A", 0, 10), ("B", 10, 10))
        >>> c.apply_load(-1, (\'Z\', 2, 7.26, 3, 270))
        >>> c.apply_load(-1, (\'X\', 4, 6, 8, 270))
        >>> c.solve()
        >>> c.tension
        {A_Z: 8.91403453669861, X_B: 19*sqrt(13)/10, Z_X: 4.79150773600774}
        >>> c.reaction_loads
        {R_A_x: -5.25547445255474, R_A_y: 7.2, R_B_x: 5.25547445255474, R_B_y: 3.8}
        >>> c.length
        5.7560958484519 + 2*sqrt(13)

        For distributed load,

        >>> from sympy.physics.continuum_mechanics.cable import Cable
        >>> c=Cable(("A", 0, 40),("B", 100, 20))
        >>> c.apply_load(0, ("X", 850))
        >>> c.solve(58.58, 0)
        >>> c.tension
        {\'distributed\': 36456.8485*sqrt(0.000543529004799705*(X + 0.00135624381275735)**2 + 1)}
        >>> c.tension_at(0)
        61709.0363315913
        >>> c.reaction_loads
        {R_A_x: 36456.8485, R_A_y: -49788.5866682485, R_B_x: 44389.8401587246, R_B_y: 42866.621696333}
        '''
