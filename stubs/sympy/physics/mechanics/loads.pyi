from _typeshed import Incomplete
from abc import ABC
from typing import NamedTuple

__all__ = ['LoadBase', 'Force', 'Torque']

class LoadBase(ABC, NamedTuple('LoadBase', [('location', Incomplete), ('vector', Incomplete)])):
    """Abstract base class for the various loading types."""
    def __add__(self, other) -> None: ...
    def __mul__(self, other) -> None: ...
    __radd__ = __add__
    __rmul__ = __mul__

class Force(LoadBase):
    """Force acting upon a point.

    Explanation
    ===========

    A force is a vector that is bound to a line of action. This class stores
    both a point, which lies on the line of action, and the vector. A tuple can
    also be used, with the location as the first entry and the vector as second
    entry.

    Examples
    ========

    A force of magnitude 2 along N.x acting on a point Po can be created as
    follows:

    >>> from sympy.physics.mechanics import Point, ReferenceFrame, Force
    >>> N = ReferenceFrame('N')
    >>> Po = Point('Po')
    >>> Force(Po, 2 * N.x)
    (Po, 2*N.x)

    If a body is supplied, then the center of mass of that body is used.

    >>> from sympy.physics.mechanics import Particle
    >>> P = Particle('P', point=Po)
    >>> Force(P, 2 * N.x)
    (Po, 2*N.x)

    """
    def __new__(cls, point, force): ...
    def __repr__(self) -> str: ...
    @property
    def point(self): ...
    @property
    def force(self): ...

class Torque(LoadBase):
    """Torque acting upon a frame.

    Explanation
    ===========

    A torque is a free vector that is acting on a reference frame, which is
    associated with a rigid body. This class stores both the frame and the
    vector. A tuple can also be used, with the location as the first item and
    the vector as second item.

    Examples
    ========

    A torque of magnitude 2 about N.x acting on a frame N can be created as
    follows:

    >>> from sympy.physics.mechanics import ReferenceFrame, Torque
    >>> N = ReferenceFrame('N')
    >>> Torque(N, 2 * N.x)
    (N, 2*N.x)

    If a body is supplied, then the frame fixed to that body is used.

    >>> from sympy.physics.mechanics import RigidBody
    >>> rb = RigidBody('rb', frame=N)
    >>> Torque(rb, 2 * N.x)
    (N, 2*N.x)

    """
    def __new__(cls, frame, torque): ...
    def __repr__(self) -> str: ...
    @property
    def frame(self): ...
    @property
    def torque(self): ...
