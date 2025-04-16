import abc
from _typeshed import Incomplete
from abc import ABC, abstractmethod

__all__ = ['WrappingGeometryBase', 'WrappingCylinder', 'WrappingSphere']

class WrappingGeometryBase(ABC, metaclass=abc.ABCMeta):
    """Abstract base class for all geometry classes to inherit from.

    Notes
    =====

    Instances of this class cannot be directly instantiated by users. However,
    it can be used to created custom geometry types through subclassing.

    """
    @property
    @abstractmethod
    def point(cls):
        """The point with which the geometry is associated."""
    @abstractmethod
    def point_on_surface(self, point):
        """Returns ``True`` if a point is on the geometry's surface.

        Parameters
        ==========
        point : Point
            The point for which it's to be ascertained if it's on the
            geometry's surface or not.

        """
    @abstractmethod
    def geodesic_length(self, point_1, point_2):
        """Returns the shortest distance between two points on a geometry's
        surface.

        Parameters
        ==========

        point_1 : Point
            The point from which the geodesic length should be calculated.
        point_2 : Point
            The point to which the geodesic length should be calculated.

        """
    @abstractmethod
    def geodesic_end_vectors(self, point_1, point_2):
        """The vectors parallel to the geodesic at the two end points.

        Parameters
        ==========

        point_1 : Point
            The point from which the geodesic originates.
        point_2 : Point
            The point at which the geodesic terminates.

        """
    def __repr__(self) -> str:
        """Default representation of a geometry object."""

class WrappingSphere(WrappingGeometryBase):
    """A solid spherical object.

    Explanation
    ===========

    A wrapping geometry that allows for circular arcs to be defined between
    pairs of points. These paths are always geodetic (the shortest possible).

    Examples
    ========

    To create a ``WrappingSphere`` instance, a ``Symbol`` denoting its radius
    and ``Point`` at which its center will be located are needed:

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import Point, WrappingSphere
    >>> r = symbols('r')
    >>> pO = Point('pO')

    A sphere with radius ``r`` centered on ``pO`` can be instantiated with:

    >>> WrappingSphere(r, pO)
    WrappingSphere(radius=r, point=pO)

    Parameters
    ==========

    radius : Symbol
        Radius of the sphere. This symbol must represent a value that is
        positive and constant, i.e. it cannot be a dynamic symbol, nor can it
        be an expression.
    point : Point
        A point at which the sphere is centered.

    See Also
    ========

    WrappingCylinder: Cylindrical geometry where the wrapping direction can be
        defined.

    """
    def __init__(self, radius, point) -> None:
        """Initializer for ``WrappingSphere``.

        Parameters
        ==========

        radius : Symbol
            The radius of the sphere.
        point : Point
            A point on which the sphere is centered.

        """
    @property
    def radius(self):
        """Radius of the sphere."""
    _radius: Incomplete
    @radius.setter
    def radius(self, radius) -> None: ...
    @property
    def point(self):
        """A point on which the sphere is centered."""
    _point: Incomplete
    @point.setter
    def point(self, point) -> None: ...
    def point_on_surface(self, point):
        """Returns ``True`` if a point is on the sphere's surface.

        Parameters
        ==========

        point : Point
            The point for which it's to be ascertained if it's on the sphere's
            surface or not. This point's position relative to the sphere's
            center must be a simple expression involving the radius of the
            sphere, otherwise this check will likely not work.

        """
    def geodesic_length(self, point_1, point_2):
        """Returns the shortest distance between two points on the sphere's
        surface.

        Explanation
        ===========

        The geodesic length, i.e. the shortest arc along the surface of a
        sphere, connecting two points can be calculated using the formula:

        .. math::

           l = \\arccos\\left(\\mathbf{v}_1 \\cdot \\mathbf{v}_2\\right)

        where $\\mathbf{v}_1$ and $\\mathbf{v}_2$ are the unit vectors from the
        sphere's center to the first and second points on the sphere's surface
        respectively. Note that the actual path that the geodesic will take is
        undefined when the two points are directly opposite one another.

        Examples
        ========

        A geodesic length can only be calculated between two points on the
        sphere's surface. Firstly, a ``WrappingSphere`` instance must be
        created along with two points that will lie on its surface:

        >>> from sympy import symbols
        >>> from sympy.physics.mechanics import (Point, ReferenceFrame,
        ...     WrappingSphere)
        >>> N = ReferenceFrame('N')
        >>> r = symbols('r')
        >>> pO = Point('pO')
        >>> pO.set_vel(N, 0)
        >>> sphere = WrappingSphere(r, pO)
        >>> p1 = Point('p1')
        >>> p2 = Point('p2')

        Let's assume that ``p1`` lies at a distance of ``r`` in the ``N.x``
        direction from ``pO`` and that ``p2`` is located on the sphere's
        surface in the ``N.y + N.z`` direction from ``pO``. These positions can
        be set with:

        >>> p1.set_pos(pO, r*N.x)
        >>> p1.pos_from(pO)
        r*N.x
        >>> p2.set_pos(pO, r*(N.y + N.z).normalize())
        >>> p2.pos_from(pO)
        sqrt(2)*r/2*N.y + sqrt(2)*r/2*N.z

        The geodesic length, which is in this case is a quarter of the sphere's
        circumference, can be calculated using the ``geodesic_length`` method:

        >>> sphere.geodesic_length(p1, p2)
        pi*r/2

        If the ``geodesic_length`` method is passed an argument, the ``Point``
        that doesn't lie on the sphere's surface then a ``ValueError`` is
        raised because it's not possible to calculate a value in this case.

        Parameters
        ==========

        point_1 : Point
            Point from which the geodesic length should be calculated.
        point_2 : Point
            Point to which the geodesic length should be calculated.

        """
    def geodesic_end_vectors(self, point_1, point_2):
        """The vectors parallel to the geodesic at the two end points.

        Parameters
        ==========

        point_1 : Point
            The point from which the geodesic originates.
        point_2 : Point
            The point at which the geodesic terminates.

        """
    def __repr__(self) -> str:
        """Representation of a ``WrappingSphere``."""

class WrappingCylinder(WrappingGeometryBase):
    """A solid (infinite) cylindrical object.

    Explanation
    ===========

    A wrapping geometry that allows for circular arcs to be defined between
    pairs of points. These paths are always geodetic (the shortest possible) in
    the sense that they will be a straight line on the unwrapped cylinder's
    surface. However, it is also possible for a direction to be specified, i.e.
    paths can be influenced such that they either wrap along the shortest side
    or the longest side of the cylinder. To define these directions, rotations
    are in the positive direction following the right-hand rule.

    Examples
    ========

    To create a ``WrappingCylinder`` instance, a ``Symbol`` denoting its
    radius, a ``Vector`` defining its axis, and a ``Point`` through which its
    axis passes are needed:

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import (Point, ReferenceFrame,
    ...     WrappingCylinder)
    >>> N = ReferenceFrame('N')
    >>> r = symbols('r')
    >>> pO = Point('pO')
    >>> ax = N.x

    A cylinder with radius ``r``, and axis parallel to ``N.x`` passing through
    ``pO`` can be instantiated with:

    >>> WrappingCylinder(r, pO, ax)
    WrappingCylinder(radius=r, point=pO, axis=N.x)

    Parameters
    ==========

    radius : Symbol
        The radius of the cylinder.
    point : Point
        A point through which the cylinder's axis passes.
    axis : Vector
        The axis along which the cylinder is aligned.

    See Also
    ========

    WrappingSphere: Spherical geometry where the wrapping direction is always
        geodetic.

    """
    def __init__(self, radius, point, axis) -> None:
        """Initializer for ``WrappingCylinder``.

        Parameters
        ==========

        radius : Symbol
            The radius of the cylinder. This symbol must represent a value that
            is positive and constant, i.e. it cannot be a dynamic symbol.
        point : Point
            A point through which the cylinder's axis passes.
        axis : Vector
            The axis along which the cylinder is aligned.

        """
    @property
    def radius(self):
        """Radius of the cylinder."""
    _radius: Incomplete
    @radius.setter
    def radius(self, radius) -> None: ...
    @property
    def point(self):
        """A point through which the cylinder's axis passes."""
    _point: Incomplete
    @point.setter
    def point(self, point) -> None: ...
    @property
    def axis(self):
        """Axis along which the cylinder is aligned."""
    _axis: Incomplete
    @axis.setter
    def axis(self, axis) -> None: ...
    def point_on_surface(self, point):
        """Returns ``True`` if a point is on the cylinder's surface.

        Parameters
        ==========

        point : Point
            The point for which it's to be ascertained if it's on the
            cylinder's surface or not. This point's position relative to the
            cylinder's axis must be a simple expression involving the radius of
            the sphere, otherwise this check will likely not work.

        """
    def geodesic_length(self, point_1, point_2):
        """The shortest distance between two points on a geometry's surface.

        Explanation
        ===========

        The geodesic length, i.e. the shortest arc along the surface of a
        cylinder, connecting two points. It can be calculated using Pythagoras'
        theorem. The first short side is the distance between the two points on
        the cylinder's surface parallel to the cylinder's axis. The second
        short side is the arc of a circle between the two points of the
        cylinder's surface perpendicular to the cylinder's axis. The resulting
        hypotenuse is the geodesic length.

        Examples
        ========

        A geodesic length can only be calculated between two points on the
        cylinder's surface. Firstly, a ``WrappingCylinder`` instance must be
        created along with two points that will lie on its surface:

        >>> from sympy import symbols, cos, sin
        >>> from sympy.physics.mechanics import (Point, ReferenceFrame,
        ...     WrappingCylinder, dynamicsymbols)
        >>> N = ReferenceFrame('N')
        >>> r = symbols('r')
        >>> pO = Point('pO')
        >>> pO.set_vel(N, 0)
        >>> cylinder = WrappingCylinder(r, pO, N.x)
        >>> p1 = Point('p1')
        >>> p2 = Point('p2')

        Let's assume that ``p1`` is located at ``N.x + r*N.y`` relative to
        ``pO`` and that ``p2`` is located at ``r*(cos(q)*N.y + sin(q)*N.z)``
        relative to ``pO``, where ``q(t)`` is a generalized coordinate
        specifying the angle rotated around the ``N.x`` axis according to the
        right-hand rule where ``N.y`` is zero. These positions can be set with:

        >>> q = dynamicsymbols('q')
        >>> p1.set_pos(pO, N.x + r*N.y)
        >>> p1.pos_from(pO)
        N.x + r*N.y
        >>> p2.set_pos(pO, r*(cos(q)*N.y + sin(q)*N.z).normalize())
        >>> p2.pos_from(pO).simplify()
        r*cos(q(t))*N.y + r*sin(q(t))*N.z

        The geodesic length, which is in this case a is the hypotenuse of a
        right triangle where the other two side lengths are ``1`` (parallel to
        the cylinder's axis) and ``r*q(t)`` (parallel to the cylinder's cross
        section), can be calculated using the ``geodesic_length`` method:

        >>> cylinder.geodesic_length(p1, p2).simplify()
        sqrt(r**2*q(t)**2 + 1)

        If the ``geodesic_length`` method is passed an argument ``Point`` that
        doesn't lie on the sphere's surface then a ``ValueError`` is raised
        because it's not possible to calculate a value in this case.

        Parameters
        ==========

        point_1 : Point
            Point from which the geodesic length should be calculated.
        point_2 : Point
            Point to which the geodesic length should be calculated.

        """
    def geodesic_end_vectors(self, point_1, point_2):
        """The vectors parallel to the geodesic at the two end points.

        Parameters
        ==========

        point_1 : Point
            The point from which the geodesic originates.
        point_2 : Point
            The point at which the geodesic terminates.

        """
    def __repr__(self) -> str:
        """Representation of a ``WrappingCylinder``."""
