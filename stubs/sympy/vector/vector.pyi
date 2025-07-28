from sympy.core import Add as Add, Basic as Basic
from sympy.core.assumptions import StdFactKB as StdFactKB
from sympy.core.expr import AtomicExpr as AtomicExpr, Expr as Expr
from sympy.core.power import Pow as Pow
from sympy.core.singleton import S as S
from sympy.core.sorting import default_sort_key as default_sort_key
from sympy.core.sympify import sympify as sympify
from sympy.functions.elementary.miscellaneous import sqrt as sqrt
from sympy.vector.basisdependent import BasisDependent as BasisDependent, BasisDependentAdd as BasisDependentAdd, BasisDependentMul as BasisDependentMul, BasisDependentZero as BasisDependentZero
from sympy.vector.coordsysrect import CoordSys3D as CoordSys3D
from sympy.vector.dyadic import BaseDyadic as BaseDyadic, Dyadic as Dyadic, DyadicAdd as DyadicAdd
from sympy.vector.kind import VectorKind as VectorKind

class Vector(BasisDependent):
    """
    Super class for all Vector classes.
    Ideally, neither this class nor any of its subclasses should be
    instantiated by the user.
    """
    is_scalar: bool
    is_Vector: bool
    _op_priority: float
    _expr_type: type[Vector]
    _mul_func: type[Vector]
    _add_func: type[Vector]
    _zero_func: type[Vector]
    _base_func: type[Vector]
    zero: VectorZero
    kind: VectorKind
    @property
    def components(self):
        """
        Returns the components of this vector in the form of a
        Python dictionary mapping BaseVector instances to the
        corresponding measure numbers.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> C = CoordSys3D('C')
        >>> v = 3*C.i + 4*C.j + 5*C.k
        >>> v.components
        {C.i: 3, C.j: 4, C.k: 5}

        """
    def magnitude(self):
        """
        Returns the magnitude of this vector.
        """
    def normalize(self):
        """
        Returns the normalized version of this vector.
        """
    def equals(self, other):
        """
        Check if ``self`` and ``other`` are identically equal vectors.

        Explanation
        ===========

        Checks if two vector expressions are equal for all possible values of
        the symbols present in the expressions.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> from sympy.abc import x, y
        >>> from sympy import pi
        >>> C = CoordSys3D('C')

        Compare vectors that are equal or not:

        >>> C.i.equals(C.j)
        False
        >>> C.i.equals(C.i)
        True

        These two vectors are equal if `x = y` but are not identically equal
        as expressions since for some values of `x` and `y` they are unequal:

        >>> v1 = x*C.i + C.j
        >>> v2 = y*C.i + C.j
        >>> v1.equals(v1)
        True
        >>> v1.equals(v2)
        False

        Vectors from different coordinate systems can be compared:

        >>> D = C.orient_new_axis('D', pi/2, C.i)
        >>> D.j.equals(C.j)
        False
        >>> D.j.equals(C.k)
        True

        Parameters
        ==========

        other: Vector
            The other vector expression to compare with.

        Returns
        =======

        ``True``, ``False`` or ``None``. A return value of ``True`` indicates
        that the two vectors are identically equal. A return value of ``False``
        indicates that they are not. In some cases it is not possible to
        determine if the two vectors are identically equal and ``None`` is
        returned.

        See Also
        ========

        sympy.core.expr.Expr.equals
        """
    def dot(self, other):
        """
        Returns the dot product of this Vector, either with another
        Vector, or a Dyadic, or a Del operator.
        If 'other' is a Vector, returns the dot product scalar (SymPy
        expression).
        If 'other' is a Dyadic, the dot product is returned as a Vector.
        If 'other' is an instance of Del, returns the directional
        derivative operator as a Python function. If this function is
        applied to a scalar expression, it returns the directional
        derivative of the scalar field wrt this Vector.

        Parameters
        ==========

        other: Vector/Dyadic/Del
            The Vector or Dyadic we are dotting with, or a Del operator .

        Examples
        ========

        >>> from sympy.vector import CoordSys3D, Del
        >>> C = CoordSys3D('C')
        >>> delop = Del()
        >>> C.i.dot(C.j)
        0
        >>> C.i & C.i
        1
        >>> v = 3*C.i + 4*C.j + 5*C.k
        >>> v.dot(C.k)
        5
        >>> (C.i & delop)(C.x*C.y*C.z)
        C.y*C.z
        >>> d = C.i.outer(C.i)
        >>> C.i.dot(d)
        C.i

        """
    def __and__(self, other): ...
    def cross(self, other):
        """
        Returns the cross product of this Vector with another Vector or
        Dyadic instance.
        The cross product is a Vector, if 'other' is a Vector. If 'other'
        is a Dyadic, this returns a Dyadic instance.

        Parameters
        ==========

        other: Vector/Dyadic
            The Vector or Dyadic we are crossing with.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> C = CoordSys3D('C')
        >>> C.i.cross(C.j)
        C.k
        >>> C.i ^ C.i
        0
        >>> v = 3*C.i + 4*C.j + 5*C.k
        >>> v ^ C.i
        5*C.j + (-4)*C.k
        >>> d = C.i.outer(C.i)
        >>> C.j.cross(d)
        (-1)*(C.k|C.i)

        """
    def __xor__(self, other): ...
    def outer(self, other):
        """
        Returns the outer product of this vector with another, in the
        form of a Dyadic instance.

        Parameters
        ==========

        other : Vector
            The Vector with respect to which the outer product is to
            be computed.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> N = CoordSys3D('N')
        >>> N.i.outer(N.j)
        (N.i|N.j)

        """
    def projection(self, other, scalar: bool = False):
        """
        Returns the vector or scalar projection of the 'other' on 'self'.

        Examples
        ========

        >>> from sympy.vector.coordsysrect import CoordSys3D
        >>> C = CoordSys3D('C')
        >>> i, j, k = C.base_vectors()
        >>> v1 = i + j + k
        >>> v2 = 3*i + 4*j
        >>> v1.projection(v2)
        7/3*C.i + 7/3*C.j + 7/3*C.k
        >>> v1.projection(v2, scalar=True)
        7/3

        """
    @property
    def _projections(self):
        """
        Returns the components of this vector but the output includes
        also zero values components.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D, Vector
        >>> C = CoordSys3D('C')
        >>> v1 = 3*C.i + 4*C.j + 5*C.k
        >>> v1._projections
        (3, 4, 5)
        >>> v2 = C.x*C.y*C.z*C.i
        >>> v2._projections
        (C.x*C.y*C.z, 0, 0)
        >>> v3 = Vector.zero
        >>> v3._projections
        (0, 0, 0)
        """
    def __or__(self, other): ...
    def to_matrix(self, system):
        """
        Returns the matrix form of this vector with respect to the
        specified coordinate system.

        Parameters
        ==========

        system : CoordSys3D
            The system wrt which the matrix form is to be computed

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> C = CoordSys3D('C')
        >>> from sympy.abc import a, b, c
        >>> v = a*C.i + b*C.j + c*C.k
        >>> v.to_matrix(C)
        Matrix([
        [a],
        [b],
        [c]])

        """
    def separate(self):
        """
        The constituents of this vector in different coordinate systems,
        as per its definition.

        Returns a dict mapping each CoordSys3D to the corresponding
        constituent Vector.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> R1 = CoordSys3D('R1')
        >>> R2 = CoordSys3D('R2')
        >>> v = R1.i + R2.i
        >>> v.separate() == {R1: R1.i, R2: R2.i}
        True

        """
    def _div_helper(one, other):
        """ Helper for division involving vectors. """

def get_postprocessor(cls): ...

class BaseVector(Vector, AtomicExpr):
    """
    Class to denote a base vector.

    """
    def __new__(cls, index, system, pretty_str=None, latex_str=None): ...
    @property
    def system(self): ...
    def _sympystr(self, printer): ...
    def _sympyrepr(self, printer): ...
    @property
    def free_symbols(self): ...
    def _eval_conjugate(self): ...

class VectorAdd(BasisDependentAdd, Vector):
    """
    Class to denote sum of Vector instances.
    """
    def __new__(cls, *args, **options): ...
    def _sympystr(self, printer): ...

class VectorMul(BasisDependentMul, Vector):
    """
    Class to denote products of scalars and BaseVectors.
    """
    def __new__(cls, *args, **options): ...
    @property
    def base_vector(self):
        """ The BaseVector involved in the product. """
    @property
    def measure_number(self):
        """ The scalar expression involved in the definition of
        this VectorMul.
        """

class VectorZero(BasisDependentZero, Vector):
    """
    Class to denote a zero vector
    """
    _op_priority: float
    _pretty_form: str
    _latex_form: str
    def __new__(cls): ...

class Cross(Vector):
    """
    Represents unevaluated Cross product.

    Examples
    ========

    >>> from sympy.vector import CoordSys3D, Cross
    >>> R = CoordSys3D('R')
    >>> v1 = R.i + R.j + R.k
    >>> v2 = R.x * R.i + R.y * R.j + R.z * R.k
    >>> Cross(v1, v2)
    Cross(R.i + R.j + R.k, R.x*R.i + R.y*R.j + R.z*R.k)
    >>> Cross(v1, v2).doit()
    (-R.y + R.z)*R.i + (R.x - R.z)*R.j + (-R.x + R.y)*R.k

    """
    def __new__(cls, expr1, expr2): ...
    def doit(self, **hints): ...

class Dot(Expr):
    """
    Represents unevaluated Dot product.

    Examples
    ========

    >>> from sympy.vector import CoordSys3D, Dot
    >>> from sympy import symbols
    >>> R = CoordSys3D('R')
    >>> a, b, c = symbols('a b c')
    >>> v1 = R.i + R.j + R.k
    >>> v2 = a * R.i + b * R.j + c * R.k
    >>> Dot(v1, v2)
    Dot(R.i + R.j + R.k, a*R.i + b*R.j + c*R.k)
    >>> Dot(v1, v2).doit()
    a + b + c

    """
    def __new__(cls, expr1, expr2): ...
    def doit(self, **hints): ...

def cross(vect1, vect2):
    """
    Returns cross product of two vectors.

    Examples
    ========

    >>> from sympy.vector import CoordSys3D
    >>> from sympy.vector.vector import cross
    >>> R = CoordSys3D('R')
    >>> v1 = R.i + R.j + R.k
    >>> v2 = R.x * R.i + R.y * R.j + R.z * R.k
    >>> cross(v1, v2)
    (-R.y + R.z)*R.i + (R.x - R.z)*R.j + (-R.x + R.y)*R.k

    """
def dot(vect1, vect2):
    """
    Returns dot product of two vectors.

    Examples
    ========

    >>> from sympy.vector import CoordSys3D
    >>> from sympy.vector.vector import dot
    >>> R = CoordSys3D('R')
    >>> v1 = R.i + R.j + R.k
    >>> v2 = R.x * R.i + R.y * R.j + R.z * R.k
    >>> dot(v1, v2)
    R.x + R.y + R.z

    """
