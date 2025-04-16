from _typeshed import Incomplete
from sympy.physics.mechanics.body_base import BodyBase

__all__ = ['RigidBody']

class RigidBody(BodyBase):
    """An idealized rigid body.

    Explanation
    ===========

    This is essentially a container which holds the various components which
    describe a rigid body: a name, mass, center of mass, reference frame, and
    inertia.

    All of these need to be supplied on creation, but can be changed
    afterwards.

    Attributes
    ==========

    name : string
        The body's name.
    masscenter : Point
        The point which represents the center of mass of the rigid body.
    frame : ReferenceFrame
        The ReferenceFrame which the rigid body is fixed in.
    mass : Sympifyable
        The body's mass.
    inertia : (Dyadic, Point)
        The body's inertia about a point; stored in a tuple as shown above.
    potential_energy : Sympifyable
        The potential energy of the RigidBody.

    Examples
    ========

    >>> from sympy import Symbol
    >>> from sympy.physics.mechanics import ReferenceFrame, Point, RigidBody
    >>> from sympy.physics.mechanics import outer
    >>> m = Symbol('m')
    >>> A = ReferenceFrame('A')
    >>> P = Point('P')
    >>> I = outer (A.x, A.x)
    >>> inertia_tuple = (I, P)
    >>> B = RigidBody('B', P, A, m, inertia_tuple)
    >>> # Or you could change them afterwards
    >>> m2 = Symbol('m2')
    >>> B.mass = m2

    """
    def __init__(self, name, masscenter: Incomplete | None = None, frame: Incomplete | None = None, mass: Incomplete | None = None, inertia: Incomplete | None = None) -> None: ...
    def __repr__(self) -> str: ...
    @property
    def frame(self):
        """The ReferenceFrame fixed to the body."""
    _frame: Incomplete
    @frame.setter
    def frame(self, F) -> None: ...
    @property
    def x(self):
        """The basis Vector for the body, in the x direction. """
    @property
    def y(self):
        """The basis Vector for the body, in the y direction. """
    @property
    def z(self):
        """The basis Vector for the body, in the z direction. """
    @property
    def inertia(self):
        """The body's inertia about a point; stored as (Dyadic, Point)."""
    _inertia: Incomplete
    _central_inertia: Incomplete
    @inertia.setter
    def inertia(self, I) -> None: ...
    @property
    def central_inertia(self):
        """The body's central inertia dyadic."""
    @central_inertia.setter
    def central_inertia(self, I) -> None: ...
    def linear_momentum(self, frame):
        """ Linear momentum of the rigid body.

        Explanation
        ===========

        The linear momentum L, of a rigid body B, with respect to frame N is
        given by:

        ``L = m * v``

        where m is the mass of the rigid body, and v is the velocity of the mass
        center of B in the frame N.

        Parameters
        ==========

        frame : ReferenceFrame
            The frame in which linear momentum is desired.

        Examples
        ========

        >>> from sympy.physics.mechanics import Point, ReferenceFrame, outer
        >>> from sympy.physics.mechanics import RigidBody, dynamicsymbols
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> m, v = dynamicsymbols('m v')
        >>> N = ReferenceFrame('N')
        >>> P = Point('P')
        >>> P.set_vel(N, v * N.x)
        >>> I = outer (N.x, N.x)
        >>> Inertia_tuple = (I, P)
        >>> B = RigidBody('B', P, N, m, Inertia_tuple)
        >>> B.linear_momentum(N)
        m*v*N.x

        """
    def angular_momentum(self, point, frame):
        """Returns the angular momentum of the rigid body about a point in the
        given frame.

        Explanation
        ===========

        The angular momentum H of a rigid body B about some point O in a frame N
        is given by:

        ``H = dot(I, w) + cross(r, m * v)``

        where I and m are the central inertia dyadic and mass of rigid body B, w
        is the angular velocity of body B in the frame N, r is the position
        vector from point O to the mass center of B, and v is the velocity of
        the mass center in the frame N.

        Parameters
        ==========

        point : Point
            The point about which angular momentum is desired.
        frame : ReferenceFrame
            The frame in which angular momentum is desired.

        Examples
        ========

        >>> from sympy.physics.mechanics import Point, ReferenceFrame, outer
        >>> from sympy.physics.mechanics import RigidBody, dynamicsymbols
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> m, v, r, omega = dynamicsymbols('m v r omega')
        >>> N = ReferenceFrame('N')
        >>> b = ReferenceFrame('b')
        >>> b.set_ang_vel(N, omega * b.x)
        >>> P = Point('P')
        >>> P.set_vel(N, 1 * N.x)
        >>> I = outer(b.x, b.x)
        >>> B = RigidBody('B', P, b, m, (I, P))
        >>> B.angular_momentum(P, N)
        omega*b.x

        """
    def kinetic_energy(self, frame):
        """Kinetic energy of the rigid body.

        Explanation
        ===========

        The kinetic energy, T, of a rigid body, B, is given by:

        ``T = 1/2 * (dot(dot(I, w), w) + dot(m * v, v))``

        where I and m are the central inertia dyadic and mass of rigid body B
        respectively, w is the body's angular velocity, and v is the velocity of
        the body's mass center in the supplied ReferenceFrame.

        Parameters
        ==========

        frame : ReferenceFrame
            The RigidBody's angular velocity and the velocity of it's mass
            center are typically defined with respect to an inertial frame but
            any relevant frame in which the velocities are known can be
            supplied.

        Examples
        ========

        >>> from sympy.physics.mechanics import Point, ReferenceFrame, outer
        >>> from sympy.physics.mechanics import RigidBody
        >>> from sympy import symbols
        >>> m, v, r, omega = symbols('m v r omega')
        >>> N = ReferenceFrame('N')
        >>> b = ReferenceFrame('b')
        >>> b.set_ang_vel(N, omega * b.x)
        >>> P = Point('P')
        >>> P.set_vel(N, v * N.x)
        >>> I = outer (b.x, b.x)
        >>> inertia_tuple = (I, P)
        >>> B = RigidBody('B', P, b, m, inertia_tuple)
        >>> B.kinetic_energy(N)
        m*v**2/2 + omega**2/2

        """
    potential_energy: Incomplete
    def set_potential_energy(self, scalar) -> None: ...
    def parallel_axis(self, point, frame: Incomplete | None = None):
        """Returns the inertia dyadic of the body with respect to another point.

        Parameters
        ==========

        point : sympy.physics.vector.Point
            The point to express the inertia dyadic about.
        frame : sympy.physics.vector.ReferenceFrame
            The reference frame used to construct the dyadic.

        Returns
        =======

        inertia : sympy.physics.vector.Dyadic
            The inertia dyadic of the rigid body expressed about the provided
            point.

        """
