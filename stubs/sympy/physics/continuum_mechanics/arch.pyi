from _typeshed import Incomplete
from sympy import Min as Min, atan as atan, cos as cos, diff as diff, limit as limit, rad as rad, sin as sin, sqrt as sqrt
from sympy.core.relational import Eq as Eq
from sympy.core.symbol import Symbol as Symbol, symbols as symbols
from sympy.core.sympify import sympify as sympify
from sympy.external.importtools import import_module as import_module
from sympy.functions import Piecewise as Piecewise
from sympy.plotting import plot as plot
from sympy.solvers.solvers import solve as solve
from sympy.utilities.decorator import doctest_depends_on as doctest_depends_on

numpy: Incomplete

class Arch:
    """
    This class is used to solve problems related to a three hinged arch(determinate) structure.

    An arch is a curved vertical structure spanning an open space underneath it.

    Arches can be used to reduce the bending moments in long-span structures.


    Arches are used in structural engineering(over windows, door and even bridges)

    because they can support a very large mass placed on top of them.

    Example
    ========
    >>> from sympy.physics.continuum_mechanics.arch import Arch
    >>> a = Arch((0,0),(10,0),crown_x=5,crown_y=5)
    >>> a.get_shape_eqn
    5 - (x - 5)**2/5

    >>> from sympy.physics.continuum_mechanics.arch import Arch
    >>> a = Arch((0,0),(10,1),crown_x=6)
    >>> a.get_shape_eqn
    9/5 - (x - 6)**2/20
    """
    _shape_eqn: Incomplete
    _left_support: Incomplete
    _right_support: Incomplete
    _crown_x: Incomplete
    _crown_y: Incomplete
    _conc_loads: Incomplete
    _distributed_loads: Incomplete
    _loads: Incomplete
    _loads_applied: Incomplete
    _supports: Incomplete
    _member: Incomplete
    _member_force: Incomplete
    _reaction_force: Incomplete
    _points_disc_x: Incomplete
    _points_disc_y: Incomplete
    _moment_x: Incomplete
    _moment_y: Incomplete
    _load_x: Incomplete
    _load_y: Incomplete
    _moment_x_func: Incomplete
    _moment_y_func: Incomplete
    _load_x_func: Incomplete
    _load_y_func: Incomplete
    _bending_moment: Incomplete
    _shear_force: Incomplete
    _axial_force: Incomplete
    def __init__(self, left_support, right_support, **kwargs) -> None: ...
    @property
    def get_shape_eqn(self):
        """returns the equation of the shape of arch developed"""
    @property
    def get_loads(self):
        """
        return the position of the applied load and angle (for concentrated loads)
        """
    @property
    def supports(self):
        """
        Returns the type of support
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
    def reaction_force(self):
        """
        return the reaction forces generated
        """
    def apply_load(self, order, label, start, mag, end=None, angle=None) -> None:
        """
        This method adds load to the Arch.

        Parameters
        ==========

            order : Integer
                Order of the applied load.

                    - For point/concentrated loads, order = -1
                    - For distributed load, order = 0

            label : String or Symbol
                The label of the load
                - should not use 'A' or 'B' as it is used for supports.

            start : Float

                    - For concentrated/point loads, start is the x coordinate
                    - For distributed loads, start is the starting position of distributed load

            mag : Sympifyable
                Magnitude of the applied load. Must be positive

            end : Float
                Required for distributed loads

                    - For concentrated/point load , end is None(may not be given)
                    - For distributed loads, end is the end position of distributed load

            angle: Sympifyable
                The angle in degrees, the load vector makes with the horizontal
                in the counter-clockwise direction.

        Examples
        ========
        For applying distributed load

        >>> from sympy.physics.continuum_mechanics.arch import Arch
        >>> a = Arch((0,0),(10,0),crown_x=5,crown_y=5)
        >>> a.apply_load(0,'C',start=3,end=5,mag=-10)

        For applying point/concentrated_loads

        >>> from sympy.physics.continuum_mechanics.arch import Arch
        >>> a = Arch((0,0),(10,0),crown_x=5,crown_y=5)
        >>> a.apply_load(-1,'C',start=2,mag=15,angle=45)

        """
    def remove_load(self, label) -> None:
        """
        This methods removes the load applied to the arch

        Parameters
        ==========

        label : String or Symbol
            The label of the applied load

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.arch import Arch
        >>> a = Arch((0,0),(10,0),crown_x=5,crown_y=5)
        >>> a.apply_load(0,'C',start=3,end=5,mag=-10)
        >>> a.remove_load('C')
        removed load C: {'start': 3, 'end': 5, 'f_y': -10}
        """
    def change_support_position(self, left_support=None, right_support=None) -> None:
        """
        Change position of supports.
        If not provided , defaults to the old value.
        Parameters
        ==========

            left_support: tuple (x, y)
                x: float
                    x-coordinate value of the left_support

                y: float
                    y-coordinate value of the left_support

            right_support: tuple (x, y)
                x: float
                    x-coordinate value of the right_support

                y: float
                    y-coordinate value of the right_support
        """
    def change_crown_position(self, crown_x=None, crown_y=None) -> None:
        """
        Change the position of the crown/hinge of the arch

        Parameters
        ==========

            crown_x: Float
                The x coordinate of the position of the hinge
                - if not provided, defaults to old value

            crown_y: Float
                The y coordinate of the position of the hinge
                - if not provided defaults to None
        """
    def change_support_type(self, left_support=None, right_support=None) -> None:
        '''
        Add the type for support at each end.
        Can use roller or hinge support at each end.

        Parameters
        ==========

            left_support, right_support : string
                Type of support at respective end

                    - For roller support , left_support/right_support = "roller"
                    - For hinged support, left_support/right_support = "hinge"
                    - defaults to hinge if value not provided

        Examples
        ========

        For applying roller support at right end

        >>> from sympy.physics.continuum_mechanics.arch import Arch
        >>> a = Arch((0,0),(10,0),crown_x=5,crown_y=5)
        >>> a.change_support_type(right_support="roller")

        '''
    def add_member(self, y) -> None:
        """
        This method adds a member/rod at a particular height y.
        A rod is used for stability of the structure in case of a roller support.
        """
    def shear_force_at(self, pos=None, **kwargs):
        """
        return the shear at some x-coordinates
        if no x value provided, returns the formula
        """
    def bending_moment_at(self, pos=None, **kwargs):
        """
        return the bending moment at some x-coordinates
        if no x value provided, returns the formula
        """
    def axial_force_at(self, pos=None, **kwargs):
        """
        return the axial/normal force generated at some x-coordinate
        if no x value provided, returns the formula
        """
    def solve(self) -> None:
        """
        This method solves for the reaction forces generated at the supports,

        and bending moment and generated in the arch and tension produced in the member if used.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.arch import Arch
        >>> a = Arch((0,0),(10,0),crown_x=5,crown_y=5)
        >>> a.apply_load(0,'C',start=3,end=5,mag=-10)
        >>> a.solve()
        >>> a.reaction_force
        {R_A_x: 8, R_A_y: 12, R_B_x: -8, R_B_y: 8}

        >>> from sympy import Symbol
        >>> t = Symbol('t')
        >>> from sympy.physics.continuum_mechanics.arch import Arch
        >>> a = Arch((0,0),(16,0),crown_x=8,crown_y=5)
        >>> a.apply_load(0,'C',start=3,end=5,mag=t)
        >>> a.solve()
        >>> a.reaction_force
        {R_A_x: -4*t/5, R_A_y: -3*t/2, R_B_x: 4*t/5, R_B_y: -t/2}

        >>> a.bending_moment_at(4)
        -5*t/2
        """
    def draw(self):
        """
        This method returns a plot object containing the diagram of the specified arch along with the supports
        and forces applied to the structure.

        Examples
        ========

        >>> from sympy import Symbol
        >>> t = Symbol('t')
        >>> from sympy.physics.continuum_mechanics.arch import Arch
        >>> a = Arch((0,0),(40,0),crown_x=20,crown_y=12)
        >>> a.apply_load(-1,'C',8,150,angle=270)
        >>> a.apply_load(0,'D',start=20,end=40,mag=-4)
        >>> a.apply_load(-1,'E',10,t,angle=300)
        >>> p = a.draw()
        >>> p # doctest: +ELLIPSIS
        Plot object containing:
        [0]: cartesian line: 11.325 - 3*(x - 20)**2/100 for x over (0.0, 40.0)
        [1]: cartesian line: 12 - 3*(x - 20)**2/100 for x over (0.0, 40.0)
        ...
        >>> p.show()

        """
    def _draw_supports(self): ...
    def _draw_rectangles(self): ...
    def _draw_loads(self): ...
    def _draw_filler(self): ...
