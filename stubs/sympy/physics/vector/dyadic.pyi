from _typeshed import Incomplete
from sympy.core.evalf import EvalfMixin
from sympy.printing.defaults import Printable

__all__ = ['Dyadic']

class Dyadic(Printable, EvalfMixin):
    """A Dyadic object.

    See:
    https://en.wikipedia.org/wiki/Dyadic_tensor
    Kane, T., Levinson, D. Dynamics Theory and Applications. 1985 McGraw-Hill

    A more powerful way to represent a rigid body's inertia. While it is more
    complex, by choosing Dyadic components to be in body fixed basis vectors,
    the resulting matrix is equivalent to the inertia tensor.

    """
    is_number: bool
    args: Incomplete
    def __init__(self, inlist) -> None:
        """
        Just like Vector's init, you should not call this unless creating a
        zero dyadic.

        zd = Dyadic(0)

        Stores a Dyadic as a list of lists; the inner list has the measure
        number and the two unit vectors; the outerlist holds each unique
        unit vector pair.

        """
    @property
    def func(self):
        """Returns the class Dyadic. """
    def __add__(self, other):
        """The add operator for Dyadic. """
    __radd__ = __add__
    def __mul__(self, other):
        """Multiplies the Dyadic by a sympifyable expression.

        Parameters
        ==========

        other : Sympafiable
            The scalar to multiply this Dyadic with

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame, outer
        >>> N = ReferenceFrame('N')
        >>> d = outer(N.x, N.x)
        >>> 5 * d
        5*(N.x|N.x)

        """
    __rmul__ = __mul__
    def dot(self, other):
        """The inner product operator for a Dyadic and a Dyadic or Vector.

        Parameters
        ==========

        other : Dyadic or Vector
            The other Dyadic or Vector to take the inner product with

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame, outer
        >>> N = ReferenceFrame('N')
        >>> D1 = outer(N.x, N.y)
        >>> D2 = outer(N.y, N.y)
        >>> D1.dot(D2)
        (N.x|N.y)
        >>> D1.dot(N.y)
        N.x

        """
    __and__ = dot
    def __truediv__(self, other):
        """Divides the Dyadic by a sympifyable expression. """
    def __eq__(self, other):
        """Tests for equality.

        Is currently weak; needs stronger comparison testing

        """
    def __ne__(self, other): ...
    def __neg__(self): ...
    def _latex(self, printer): ...
    def _pretty(self, printer): ...
    def __rsub__(self, other): ...
    def _sympystr(self, printer):
        """Printing method. """
    def __sub__(self, other):
        """The subtraction operator. """
    def cross(self, other):
        """Returns the dyadic resulting from the dyadic vector cross product:
        Dyadic x Vector.

        Parameters
        ==========
        other : Vector
            Vector to cross with.

        Examples
        ========
        >>> from sympy.physics.vector import ReferenceFrame, outer, cross
        >>> N = ReferenceFrame('N')
        >>> d = outer(N.x, N.x)
        >>> cross(d, N.y)
        (N.x|N.z)

        """
    __xor__ = cross
    def express(self, frame1, frame2=None):
        """Expresses this Dyadic in alternate frame(s)

        The first frame is the list side expression, the second frame is the
        right side; if Dyadic is in form A.x|B.y, you can express it in two
        different frames. If no second frame is given, the Dyadic is
        expressed in only one frame.

        Calls the global express function

        Parameters
        ==========

        frame1 : ReferenceFrame
            The frame to express the left side of the Dyadic in
        frame2 : ReferenceFrame
            If provided, the frame to express the right side of the Dyadic in

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame, outer, dynamicsymbols
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> N = ReferenceFrame('N')
        >>> q = dynamicsymbols('q')
        >>> B = N.orientnew('B', 'Axis', [q, N.z])
        >>> d = outer(N.x, N.x)
        >>> d.express(B, N)
        cos(q)*(B.x|N.x) - sin(q)*(B.y|N.x)

        """
    def to_matrix(self, reference_frame, second_reference_frame=None):
        """Returns the matrix form of the dyadic with respect to one or two
        reference frames.

        Parameters
        ----------
        reference_frame : ReferenceFrame
            The reference frame that the rows and columns of the matrix
            correspond to. If a second reference frame is provided, this
            only corresponds to the rows of the matrix.
        second_reference_frame : ReferenceFrame, optional, default=None
            The reference frame that the columns of the matrix correspond
            to.

        Returns
        -------
        matrix : ImmutableMatrix, shape(3,3)
            The matrix that gives the 2D tensor form.

        Examples
        ========

        >>> from sympy import symbols, trigsimp
        >>> from sympy.physics.vector import ReferenceFrame
        >>> from sympy.physics.mechanics import inertia
        >>> Ixx, Iyy, Izz, Ixy, Iyz, Ixz = symbols('Ixx, Iyy, Izz, Ixy, Iyz, Ixz')
        >>> N = ReferenceFrame('N')
        >>> inertia_dyadic = inertia(N, Ixx, Iyy, Izz, Ixy, Iyz, Ixz)
        >>> inertia_dyadic.to_matrix(N)
        Matrix([
        [Ixx, Ixy, Ixz],
        [Ixy, Iyy, Iyz],
        [Ixz, Iyz, Izz]])
        >>> beta = symbols('beta')
        >>> A = N.orientnew('A', 'Axis', (beta, N.x))
        >>> trigsimp(inertia_dyadic.to_matrix(A))
        Matrix([
        [                           Ixx,                                           Ixy*cos(beta) + Ixz*sin(beta),                                           -Ixy*sin(beta) + Ixz*cos(beta)],
        [ Ixy*cos(beta) + Ixz*sin(beta), Iyy*cos(2*beta)/2 + Iyy/2 + Iyz*sin(2*beta) - Izz*cos(2*beta)/2 + Izz/2,                 -Iyy*sin(2*beta)/2 + Iyz*cos(2*beta) + Izz*sin(2*beta)/2],
        [-Ixy*sin(beta) + Ixz*cos(beta),                -Iyy*sin(2*beta)/2 + Iyz*cos(2*beta) + Izz*sin(2*beta)/2, -Iyy*cos(2*beta)/2 + Iyy/2 - Iyz*sin(2*beta) + Izz*cos(2*beta)/2 + Izz/2]])

        """
    def doit(self, **hints):
        """Calls .doit() on each term in the Dyadic"""
    def dt(self, frame):
        """Take the time derivative of this Dyadic in a frame.

        This function calls the global time_derivative method

        Parameters
        ==========

        frame : ReferenceFrame
            The frame to take the time derivative in

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame, outer, dynamicsymbols
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> N = ReferenceFrame('N')
        >>> q = dynamicsymbols('q')
        >>> B = N.orientnew('B', 'Axis', [q, N.z])
        >>> d = outer(N.x, N.x)
        >>> d.dt(B)
        - q'*(N.y|N.x) - q'*(N.x|N.y)

        """
    def simplify(self):
        """Returns a simplified Dyadic."""
    def subs(self, *args, **kwargs):
        """Substitution on the Dyadic.

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame
        >>> from sympy import Symbol
        >>> N = ReferenceFrame('N')
        >>> s = Symbol('s')
        >>> a = s*(N.x|N.x)
        >>> a.subs({s: 2})
        2*(N.x|N.x)

        """
    def applyfunc(self, f):
        """Apply a function to each component of a Dyadic."""
    def _eval_evalf(self, prec): ...
    def xreplace(self, rule):
        """
        Replace occurrences of objects within the measure numbers of the
        Dyadic.

        Parameters
        ==========

        rule : dict-like
            Expresses a replacement rule.

        Returns
        =======

        Dyadic
            Result of the replacement.

        Examples
        ========

        >>> from sympy import symbols, pi
        >>> from sympy.physics.vector import ReferenceFrame, outer
        >>> N = ReferenceFrame('N')
        >>> D = outer(N.x, N.x)
        >>> x, y, z = symbols('x y z')
        >>> ((1 + x*y) * D).xreplace({x: pi})
        (pi*y + 1)*(N.x|N.x)
        >>> ((1 + x*y) * D).xreplace({x: pi, y: 2})
        (1 + 2*pi)*(N.x|N.x)

        Replacements occur only if an entire node in the expression tree is
        matched:

        >>> ((x*y + z) * D).xreplace({x*y: pi})
        (z + pi)*(N.x|N.x)
        >>> ((x*y*z) * D).xreplace({x*y: pi})
        x*y*z*(N.x|N.x)

        """
