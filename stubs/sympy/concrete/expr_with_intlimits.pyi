from _typeshed import Incomplete
from sympy.concrete.expr_with_limits import ExprWithLimits as ExprWithLimits
from sympy.core.relational import Eq as Eq
from sympy.core.singleton import S as S

class ReorderError(NotImplementedError):
    """
    Exception raised when trying to reorder dependent limits.
    """
    def __init__(self, expr, msg) -> None: ...

class ExprWithIntLimits(ExprWithLimits):
    """
    Superclass for Product and Sum.

    See Also
    ========

    sympy.concrete.expr_with_limits.ExprWithLimits
    sympy.concrete.products.Product
    sympy.concrete.summations.Sum
    """
    __slots__: Incomplete
    def change_index(self, var, trafo, newvar: Incomplete | None = None):
        """
        Change index of a Sum or Product.

        Perform a linear transformation `x \\mapsto a x + b` on the index variable
        `x`. For `a` the only values allowed are `\\pm 1`. A new variable to be used
        after the change of index can also be specified.

        Explanation
        ===========

        ``change_index(expr, var, trafo, newvar=None)`` where ``var`` specifies the
        index variable `x` to transform. The transformation ``trafo`` must be linear
        and given in terms of ``var``. If the optional argument ``newvar`` is
        provided then ``var`` gets replaced by ``newvar`` in the final expression.

        Examples
        ========

        >>> from sympy import Sum, Product, simplify
        >>> from sympy.abc import x, y, a, b, c, d, u, v, i, j, k, l

        >>> S = Sum(x, (x, a, b))
        >>> S.doit()
        -a**2/2 + a/2 + b**2/2 + b/2

        >>> Sn = S.change_index(x, x + 1, y)
        >>> Sn
        Sum(y - 1, (y, a + 1, b + 1))
        >>> Sn.doit()
        -a**2/2 + a/2 + b**2/2 + b/2

        >>> Sn = S.change_index(x, -x, y)
        >>> Sn
        Sum(-y, (y, -b, -a))
        >>> Sn.doit()
        -a**2/2 + a/2 + b**2/2 + b/2

        >>> Sn = S.change_index(x, x+u)
        >>> Sn
        Sum(-u + x, (x, a + u, b + u))
        >>> Sn.doit()
        -a**2/2 - a*u + a/2 + b**2/2 + b*u + b/2 - u*(-a + b + 1) + u
        >>> simplify(Sn.doit())
        -a**2/2 + a/2 + b**2/2 + b/2

        >>> Sn = S.change_index(x, -x - u, y)
        >>> Sn
        Sum(-u - y, (y, -b - u, -a - u))
        >>> Sn.doit()
        -a**2/2 - a*u + a/2 + b**2/2 + b*u + b/2 - u*(-a + b + 1) + u
        >>> simplify(Sn.doit())
        -a**2/2 + a/2 + b**2/2 + b/2

        >>> P = Product(i*j**2, (i, a, b), (j, c, d))
        >>> P
        Product(i*j**2, (i, a, b), (j, c, d))
        >>> P2 = P.change_index(i, i+3, k)
        >>> P2
        Product(j**2*(k - 3), (k, a + 3, b + 3), (j, c, d))
        >>> P3 = P2.change_index(j, -j, l)
        >>> P3
        Product(l**2*(k - 3), (k, a + 3, b + 3), (l, -d, -c))

        When dealing with symbols only, we can make a
        general linear transformation:

        >>> Sn = S.change_index(x, u*x+v, y)
        >>> Sn
        Sum((-v + y)/u, (y, b*u + v, a*u + v))
        >>> Sn.doit()
        -v*(a*u - b*u + 1)/u + (a**2*u**2/2 + a*u*v + a*u/2 - b**2*u**2/2 - b*u*v + b*u/2 + v)/u
        >>> simplify(Sn.doit())
        a**2*u/2 + a/2 - b**2*u/2 + b/2

        However, the last result can be inconsistent with usual
        summation where the index increment is always 1. This is
        obvious as we get back the original value only for ``u``
        equal +1 or -1.

        See Also
        ========

        sympy.concrete.expr_with_intlimits.ExprWithIntLimits.index,
        reorder_limit,
        sympy.concrete.expr_with_intlimits.ExprWithIntLimits.reorder,
        sympy.concrete.summations.Sum.reverse_order,
        sympy.concrete.products.Product.reverse_order
        """
    def index(expr, x):
        """
        Return the index of a dummy variable in the list of limits.

        Explanation
        ===========

        ``index(expr, x)``  returns the index of the dummy variable ``x`` in the
        limits of ``expr``. Note that we start counting with 0 at the inner-most
        limits tuple.

        Examples
        ========

        >>> from sympy.abc import x, y, a, b, c, d
        >>> from sympy import Sum, Product
        >>> Sum(x*y, (x, a, b), (y, c, d)).index(x)
        0
        >>> Sum(x*y, (x, a, b), (y, c, d)).index(y)
        1
        >>> Product(x*y, (x, a, b), (y, c, d)).index(x)
        0
        >>> Product(x*y, (x, a, b), (y, c, d)).index(y)
        1

        See Also
        ========

        reorder_limit, reorder, sympy.concrete.summations.Sum.reverse_order,
        sympy.concrete.products.Product.reverse_order
        """
    def reorder(expr, *arg):
        """
        Reorder limits in a expression containing a Sum or a Product.

        Explanation
        ===========

        ``expr.reorder(*arg)`` reorders the limits in the expression ``expr``
        according to the list of tuples given by ``arg``. These tuples can
        contain numerical indices or index variable names or involve both.

        Examples
        ========

        >>> from sympy import Sum, Product
        >>> from sympy.abc import x, y, z, a, b, c, d, e, f

        >>> Sum(x*y, (x, a, b), (y, c, d)).reorder((x, y))
        Sum(x*y, (y, c, d), (x, a, b))

        >>> Sum(x*y*z, (x, a, b), (y, c, d), (z, e, f)).reorder((x, y), (x, z), (y, z))
        Sum(x*y*z, (z, e, f), (y, c, d), (x, a, b))

        >>> P = Product(x*y*z, (x, a, b), (y, c, d), (z, e, f))
        >>> P.reorder((x, y), (x, z), (y, z))
        Product(x*y*z, (z, e, f), (y, c, d), (x, a, b))

        We can also select the index variables by counting them, starting
        with the inner-most one:

        >>> Sum(x**2, (x, a, b), (x, c, d)).reorder((0, 1))
        Sum(x**2, (x, c, d), (x, a, b))

        And of course we can mix both schemes:

        >>> Sum(x*y, (x, a, b), (y, c, d)).reorder((y, x))
        Sum(x*y, (y, c, d), (x, a, b))
        >>> Sum(x*y, (x, a, b), (y, c, d)).reorder((y, 0))
        Sum(x*y, (y, c, d), (x, a, b))

        See Also
        ========

        reorder_limit, index, sympy.concrete.summations.Sum.reverse_order,
        sympy.concrete.products.Product.reverse_order
        """
    def reorder_limit(expr, x, y):
        """
        Interchange two limit tuples of a Sum or Product expression.

        Explanation
        ===========

        ``expr.reorder_limit(x, y)`` interchanges two limit tuples. The
        arguments ``x`` and ``y`` are integers corresponding to the index
        variables of the two limits which are to be interchanged. The
        expression ``expr`` has to be either a Sum or a Product.

        Examples
        ========

        >>> from sympy.abc import x, y, z, a, b, c, d, e, f
        >>> from sympy import Sum, Product

        >>> Sum(x*y*z, (x, a, b), (y, c, d), (z, e, f)).reorder_limit(0, 2)
        Sum(x*y*z, (z, e, f), (y, c, d), (x, a, b))
        >>> Sum(x**2, (x, a, b), (x, c, d)).reorder_limit(1, 0)
        Sum(x**2, (x, c, d), (x, a, b))

        >>> Product(x*y*z, (x, a, b), (y, c, d), (z, e, f)).reorder_limit(0, 2)
        Product(x*y*z, (z, e, f), (y, c, d), (x, a, b))

        See Also
        ========

        index, reorder, sympy.concrete.summations.Sum.reverse_order,
        sympy.concrete.products.Product.reverse_order
        """
    @property
    def has_empty_sequence(self):
        """
        Returns True if the Sum or Product is computed for an empty sequence.

        Examples
        ========

        >>> from sympy import Sum, Product, Symbol
        >>> m = Symbol('m')
        >>> Sum(m, (m, 1, 0)).has_empty_sequence
        True

        >>> Sum(m, (m, 1, 1)).has_empty_sequence
        False

        >>> M = Symbol('M', integer=True, positive=True)
        >>> Product(m, (m, 1, M)).has_empty_sequence
        False

        >>> Product(m, (m, 2, M)).has_empty_sequence

        >>> Product(m, (m, M + 1, M)).has_empty_sequence
        True

        >>> N = Symbol('N', integer=True, positive=True)
        >>> Sum(m, (m, N, M)).has_empty_sequence

        >>> N = Symbol('N', integer=True, negative=True)
        >>> Sum(m, (m, N, M)).has_empty_sequence
        False

        See Also
        ========

        has_reversed_limits
        has_finite_limits

        """
