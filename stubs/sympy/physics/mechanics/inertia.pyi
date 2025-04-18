from _typeshed import Incomplete
from typing import NamedTuple

__all__ = ['inertia', 'inertia_of_point_mass', 'Inertia']

def inertia(frame, ixx, iyy, izz, ixy: int = 0, iyz: int = 0, izx: int = 0):
    """Simple way to create inertia Dyadic object.

    Explanation
    ===========

    Creates an inertia Dyadic based on the given tensor values and a body-fixed
    reference frame.

    Parameters
    ==========

    frame : ReferenceFrame
        The frame the inertia is defined in.
    ixx : Sympifyable
        The xx element in the inertia dyadic.
    iyy : Sympifyable
        The yy element in the inertia dyadic.
    izz : Sympifyable
        The zz element in the inertia dyadic.
    ixy : Sympifyable
        The xy element in the inertia dyadic.
    iyz : Sympifyable
        The yz element in the inertia dyadic.
    izx : Sympifyable
        The zx element in the inertia dyadic.

    Examples
    ========

    >>> from sympy.physics.mechanics import ReferenceFrame, inertia
    >>> N = ReferenceFrame('N')
    >>> inertia(N, 1, 2, 3)
    (N.x|N.x) + 2*(N.y|N.y) + 3*(N.z|N.z)

    """
def inertia_of_point_mass(mass, pos_vec, frame):
    """Inertia dyadic of a point mass relative to point O.

    Parameters
    ==========

    mass : Sympifyable
        Mass of the point mass
    pos_vec : Vector
        Position from point O to point mass
    frame : ReferenceFrame
        Reference frame to express the dyadic in

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import ReferenceFrame, inertia_of_point_mass
    >>> N = ReferenceFrame('N')
    >>> r, m = symbols('r m')
    >>> px = r * N.x
    >>> inertia_of_point_mass(m, px, N)
    m*r**2*(N.y|N.y) + m*r**2*(N.z|N.z)

    """

class Inertia(NamedTuple('Inertia', [('dyadic', Incomplete), ('point', Incomplete)])):
    """Inertia object consisting of a Dyadic and a Point of reference.

    Explanation
    ===========

    This is a simple class to store the Point and Dyadic, belonging to an
    inertia.

    Attributes
    ==========

    dyadic : Dyadic
        The dyadic of the inertia.
    point : Point
        The reference point of the inertia.

    Examples
    ========

    >>> from sympy.physics.mechanics import ReferenceFrame, Point, Inertia
    >>> N = ReferenceFrame('N')
    >>> Po = Point('Po')
    >>> Inertia(N.x.outer(N.x) + N.y.outer(N.y) + N.z.outer(N.z), Po)
    ((N.x|N.x) + (N.y|N.y) + (N.z|N.z), Po)

    In the example above the Dyadic was created manually, one can however also
    use the ``inertia`` function for this or the class method ``from_tensor`` as
    shown below.

    >>> Inertia.from_inertia_scalars(Po, N, 1, 1, 1)
    ((N.x|N.x) + (N.y|N.y) + (N.z|N.z), Po)

    """
    def __new__(cls, dyadic, point): ...
    @classmethod
    def from_inertia_scalars(cls, point, frame, ixx, iyy, izz, ixy: int = 0, iyz: int = 0, izx: int = 0):
        """Simple way to create an Inertia object based on the tensor values.

        Explanation
        ===========

        This class method uses the :func`~.inertia` to create the Dyadic based
        on the tensor values.

        Parameters
        ==========

        point : Point
            The reference point of the inertia.
        frame : ReferenceFrame
            The frame the inertia is defined in.
        ixx : Sympifyable
            The xx element in the inertia dyadic.
        iyy : Sympifyable
            The yy element in the inertia dyadic.
        izz : Sympifyable
            The zz element in the inertia dyadic.
        ixy : Sympifyable
            The xy element in the inertia dyadic.
        iyz : Sympifyable
            The yz element in the inertia dyadic.
        izx : Sympifyable
            The zx element in the inertia dyadic.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.mechanics import ReferenceFrame, Point, Inertia
        >>> ixx, iyy, izz, ixy, iyz, izx = symbols('ixx iyy izz ixy iyz izx')
        >>> N = ReferenceFrame('N')
        >>> P = Point('P')
        >>> I = Inertia.from_inertia_scalars(P, N, ixx, iyy, izz, ixy, iyz, izx)

        The tensor values can easily be seen when converting the dyadic to a
        matrix.

        >>> I.dyadic.to_matrix(N)
        Matrix([
        [ixx, ixy, izx],
        [ixy, iyy, iyz],
        [izx, iyz, izz]])

        """
    def __add__(self, other) -> None: ...
    def __mul__(self, other) -> None: ...
    __radd__ = __add__
    __rmul__ = __mul__
