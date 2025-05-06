from _typeshed import Incomplete
from collections.abc import Generator
from sympy.core import Function as Function, NumberKind as NumberKind, S as S, sympify as sympify
from sympy.core.expr import Expr as Expr
from sympy.core.function import Application as Application, ArgumentIndexError as ArgumentIndexError, Lambda as Lambda
from sympy.core.logic import _torf as _torf, fuzzy_and as fuzzy_and, fuzzy_or as fuzzy_or
from sympy.core.numbers import Integer as Integer, Rational as Rational
from sympy.core.operations import LatticeOp as LatticeOp, ShortCircuit as ShortCircuit
from sympy.core.relational import Eq as Eq, Relational as Relational
from sympy.core.singleton import Singleton as Singleton
from sympy.logic.boolalg import And as And, Or as Or

def _minmax_as_Piecewise(op, *args): ...

class IdentityFunction(Lambda, metaclass=Singleton):
    """
    The identity function

    Examples
    ========

    >>> from sympy import Id, Symbol
    >>> x = Symbol('x')
    >>> Id(x)
    x

    """
    _symbol: Incomplete
    @property
    def signature(self): ...
    @property
    def expr(self): ...

Id: Incomplete

def sqrt(arg, evaluate: Incomplete | None = None):
    """Returns the principal square root.

    Parameters
    ==========

    evaluate : bool, optional
        The parameter determines if the expression should be evaluated.
        If ``None``, its value is taken from
        ``global_parameters.evaluate``.

    Examples
    ========

    >>> from sympy import sqrt, Symbol, S
    >>> x = Symbol('x')

    >>> sqrt(x)
    sqrt(x)

    >>> sqrt(x)**2
    x

    Note that sqrt(x**2) does not simplify to x.

    >>> sqrt(x**2)
    sqrt(x**2)

    This is because the two are not equal to each other in general.
    For example, consider x == -1:

    >>> from sympy import Eq
    >>> Eq(sqrt(x**2), x).subs(x, -1)
    False

    This is because sqrt computes the principal square root, so the square may
    put the argument in a different branch.  This identity does hold if x is
    positive:

    >>> y = Symbol('y', positive=True)
    >>> sqrt(y**2)
    y

    You can force this simplification by using the powdenest() function with
    the force option set to True:

    >>> from sympy import powdenest
    >>> sqrt(x**2)
    sqrt(x**2)
    >>> powdenest(sqrt(x**2), force=True)
    x

    To get both branches of the square root you can use the rootof function:

    >>> from sympy import rootof

    >>> [rootof(x**2-3,i) for i in (0,1)]
    [-sqrt(3), sqrt(3)]

    Although ``sqrt`` is printed, there is no ``sqrt`` function so looking for
    ``sqrt`` in an expression will fail:

    >>> from sympy.utilities.misc import func_name
    >>> func_name(sqrt(x))
    'Pow'
    >>> sqrt(x).has(sqrt)
    False

    To find ``sqrt`` look for ``Pow`` with an exponent of ``1/2``:

    >>> (x + 1/sqrt(x)).find(lambda i: i.is_Pow and abs(i.exp) is S.Half)
    {1/sqrt(x)}

    See Also
    ========

    sympy.polys.rootoftools.rootof, root, real_root

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Square_root
    .. [2] https://en.wikipedia.org/wiki/Principal_value
    """
def cbrt(arg, evaluate: Incomplete | None = None):
    """Returns the principal cube root.

    Parameters
    ==========

    evaluate : bool, optional
        The parameter determines if the expression should be evaluated.
        If ``None``, its value is taken from
        ``global_parameters.evaluate``.

    Examples
    ========

    >>> from sympy import cbrt, Symbol
    >>> x = Symbol('x')

    >>> cbrt(x)
    x**(1/3)

    >>> cbrt(x)**3
    x

    Note that cbrt(x**3) does not simplify to x.

    >>> cbrt(x**3)
    (x**3)**(1/3)

    This is because the two are not equal to each other in general.
    For example, consider `x == -1`:

    >>> from sympy import Eq
    >>> Eq(cbrt(x**3), x).subs(x, -1)
    False

    This is because cbrt computes the principal cube root, this
    identity does hold if `x` is positive:

    >>> y = Symbol('y', positive=True)
    >>> cbrt(y**3)
    y

    See Also
    ========

    sympy.polys.rootoftools.rootof, root, real_root

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Cube_root
    .. [2] https://en.wikipedia.org/wiki/Principal_value

    """
def root(arg, n, k: int = 0, evaluate: Incomplete | None = None):
    """Returns the *k*-th *n*-th root of ``arg``.

    Parameters
    ==========

    k : int, optional
        Should be an integer in $\\{0, 1, ..., n-1\\}$.
        Defaults to the principal root if $0$.

    evaluate : bool, optional
        The parameter determines if the expression should be evaluated.
        If ``None``, its value is taken from
        ``global_parameters.evaluate``.

    Examples
    ========

    >>> from sympy import root, Rational
    >>> from sympy.abc import x, n

    >>> root(x, 2)
    sqrt(x)

    >>> root(x, 3)
    x**(1/3)

    >>> root(x, n)
    x**(1/n)

    >>> root(x, -Rational(2, 3))
    x**(-3/2)

    To get the k-th n-th root, specify k:

    >>> root(-2, 3, 2)
    -(-1)**(2/3)*2**(1/3)

    To get all n n-th roots you can use the rootof function.
    The following examples show the roots of unity for n
    equal 2, 3 and 4:

    >>> from sympy import rootof

    >>> [rootof(x**2 - 1, i) for i in range(2)]
    [-1, 1]

    >>> [rootof(x**3 - 1,i) for i in range(3)]
    [1, -1/2 - sqrt(3)*I/2, -1/2 + sqrt(3)*I/2]

    >>> [rootof(x**4 - 1,i) for i in range(4)]
    [-1, 1, -I, I]

    SymPy, like other symbolic algebra systems, returns the
    complex root of negative numbers. This is the principal
    root and differs from the text-book result that one might
    be expecting. For example, the cube root of -8 does not
    come back as -2:

    >>> root(-8, 3)
    2*(-1)**(1/3)

    The real_root function can be used to either make the principal
    result real (or simply to return the real root directly):

    >>> from sympy import real_root
    >>> real_root(_)
    -2
    >>> real_root(-32, 5)
    -2

    Alternatively, the n//2-th n-th root of a negative number can be
    computed with root:

    >>> root(-32, 5, 5//2)
    -2

    See Also
    ========

    sympy.polys.rootoftools.rootof
    sympy.core.intfunc.integer_nthroot
    sqrt, real_root

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Square_root
    .. [2] https://en.wikipedia.org/wiki/Real_root
    .. [3] https://en.wikipedia.org/wiki/Root_of_unity
    .. [4] https://en.wikipedia.org/wiki/Principal_value
    .. [5] https://mathworld.wolfram.com/CubeRoot.html

    """
def real_root(arg, n: Incomplete | None = None, evaluate: Incomplete | None = None):
    """Return the real *n*'th-root of *arg* if possible.

    Parameters
    ==========

    n : int or None, optional
        If *n* is ``None``, then all instances of
        $(-n)^{1/\\text{odd}}$ will be changed to $-n^{1/\\text{odd}}$.
        This will only create a real root of a principal root.
        The presence of other factors may cause the result to not be
        real.

    evaluate : bool, optional
        The parameter determines if the expression should be evaluated.
        If ``None``, its value is taken from
        ``global_parameters.evaluate``.

    Examples
    ========

    >>> from sympy import root, real_root

    >>> real_root(-8, 3)
    -2
    >>> root(-8, 3)
    2*(-1)**(1/3)
    >>> real_root(_)
    -2

    If one creates a non-principal root and applies real_root, the
    result will not be real (so use with caution):

    >>> root(-8, 3, 2)
    -2*(-1)**(2/3)
    >>> real_root(_)
    -2*(-1)**(2/3)

    See Also
    ========

    sympy.polys.rootoftools.rootof
    sympy.core.intfunc.integer_nthroot
    root, sqrt
    """

class MinMaxBase(Expr, LatticeOp):
    def __new__(cls, *args, **assumptions): ...
    @classmethod
    def _collapse_arguments(cls, args, **assumptions):
        """Remove redundant args.

        Examples
        ========

        >>> from sympy import Min, Max
        >>> from sympy.abc import a, b, c, d, e

        Any arg in parent that appears in any
        parent-like function in any of the flat args
        of parent can be removed from that sub-arg:

        >>> Min(a, Max(b, Min(a, c, d)))
        Min(a, Max(b, Min(c, d)))

        If the arg of parent appears in an opposite-than parent
        function in any of the flat args of parent that function
        can be replaced with the arg:

        >>> Min(a, Max(b, Min(c, d, Max(a, e))))
        Min(a, Max(b, Min(a, c, d)))
        """
    @classmethod
    def _new_args_filter(cls, arg_sequence) -> Generator[Incomplete, Incomplete]:
        """
        Generator filtering args.

        first standard filter, for cls.zero and cls.identity.
        Also reshape ``Max(a, Max(b, c))`` to ``Max(a, b, c)``,
        and check arguments for comparability
        """
    @classmethod
    def _find_localzeros(cls, values, **options):
        """
        Sequentially allocate values to localzeros.

        When a value is identified as being more extreme than another member it
        replaces that member; if this is never true, then the value is simply
        appended to the localzeros.
        """
    @classmethod
    def _is_connected(cls, x, y):
        """
        Check if x and y are connected somehow.
        """
    def _eval_derivative(self, s): ...
    def _eval_rewrite_as_Abs(self, *args, **kwargs): ...
    def evalf(self, n: int = 15, **options): ...
    def n(self, *args, **kwargs): ...
    _eval_is_algebraic: Incomplete
    _eval_is_antihermitian: Incomplete
    _eval_is_commutative: Incomplete
    _eval_is_complex: Incomplete
    _eval_is_composite: Incomplete
    _eval_is_even: Incomplete
    _eval_is_finite: Incomplete
    _eval_is_hermitian: Incomplete
    _eval_is_imaginary: Incomplete
    _eval_is_infinite: Incomplete
    _eval_is_integer: Incomplete
    _eval_is_irrational: Incomplete
    _eval_is_negative: Incomplete
    _eval_is_noninteger: Incomplete
    _eval_is_nonnegative: Incomplete
    _eval_is_nonpositive: Incomplete
    _eval_is_nonzero: Incomplete
    _eval_is_odd: Incomplete
    _eval_is_polar: Incomplete
    _eval_is_positive: Incomplete
    _eval_is_prime: Incomplete
    _eval_is_rational: Incomplete
    _eval_is_real: Incomplete
    _eval_is_extended_real: Incomplete
    _eval_is_transcendental: Incomplete
    _eval_is_zero: Incomplete

class Max(MinMaxBase, Application):
    '''
    Return, if possible, the maximum value of the list.

    When number of arguments is equal one, then
    return this argument.

    When number of arguments is equal two, then
    return, if possible, the value from (a, b) that is $\\ge$ the other.

    In common case, when the length of list greater than 2, the task
    is more complicated. Return only the arguments, which are greater
    than others, if it is possible to determine directional relation.

    If is not possible to determine such a relation, return a partially
    evaluated result.

    Assumptions are used to make the decision too.

    Also, only comparable arguments are permitted.

    It is named ``Max`` and not ``max`` to avoid conflicts
    with the built-in function ``max``.


    Examples
    ========

    >>> from sympy import Max, Symbol, oo
    >>> from sympy.abc import x, y, z
    >>> p = Symbol(\'p\', positive=True)
    >>> n = Symbol(\'n\', negative=True)

    >>> Max(x, -2)
    Max(-2, x)
    >>> Max(x, -2).subs(x, 3)
    3
    >>> Max(p, -2)
    p
    >>> Max(x, y)
    Max(x, y)
    >>> Max(x, y) == Max(y, x)
    True
    >>> Max(x, Max(y, z))
    Max(x, y, z)
    >>> Max(n, 8, p, 7, -oo)
    Max(8, p)
    >>> Max (1, x, oo)
    oo

    * Algorithm

    The task can be considered as searching of supremums in the
    directed complete partial orders [1]_.

    The source values are sequentially allocated by the isolated subsets
    in which supremums are searched and result as Max arguments.

    If the resulted supremum is single, then it is returned.

    The isolated subsets are the sets of values which are only the comparable
    with each other in the current set. E.g. natural numbers are comparable with
    each other, but not comparable with the `x` symbol. Another example: the
    symbol `x` with negative assumption is comparable with a natural number.

    Also there are "least" elements, which are comparable with all others,
    and have a zero property (maximum or minimum for all elements).
    For example, in case of $\\infty$, the allocation operation is terminated
    and only this value is returned.

    Assumption:
       - if $A > B > C$ then $A > C$
       - if $A = B$ then $B$ can be removed

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Directed_complete_partial_order
    .. [2] https://en.wikipedia.org/wiki/Lattice_%28order%29

    See Also
    ========

    Min : find minimum values
    '''
    zero: Incomplete
    identity: Incomplete
    def fdiff(self, argindex): ...
    def _eval_rewrite_as_Heaviside(self, *args, **kwargs): ...
    def _eval_rewrite_as_Piecewise(self, *args, **kwargs): ...
    def _eval_is_positive(self): ...
    def _eval_is_nonnegative(self): ...
    def _eval_is_negative(self): ...

class Min(MinMaxBase, Application):
    """
    Return, if possible, the minimum value of the list.
    It is named ``Min`` and not ``min`` to avoid conflicts
    with the built-in function ``min``.

    Examples
    ========

    >>> from sympy import Min, Symbol, oo
    >>> from sympy.abc import x, y
    >>> p = Symbol('p', positive=True)
    >>> n = Symbol('n', negative=True)

    >>> Min(x, -2)
    Min(-2, x)
    >>> Min(x, -2).subs(x, 3)
    -2
    >>> Min(p, -3)
    -3
    >>> Min(x, y)
    Min(x, y)
    >>> Min(n, 8, p, -7, p, oo)
    Min(-7, n)

    See Also
    ========

    Max : find maximum values
    """
    zero: Incomplete
    identity: Incomplete
    def fdiff(self, argindex): ...
    def _eval_rewrite_as_Heaviside(self, *args, **kwargs): ...
    def _eval_rewrite_as_Piecewise(self, *args, **kwargs): ...
    def _eval_is_positive(self): ...
    def _eval_is_nonnegative(self): ...
    def _eval_is_negative(self): ...

class Rem(Function):
    """Returns the remainder when ``p`` is divided by ``q`` where ``p`` is finite
    and ``q`` is not equal to zero. The result, ``p - int(p/q)*q``, has the same sign
    as the divisor.

    Parameters
    ==========

    p : Expr
        Dividend.

    q : Expr
        Divisor.

    Notes
    =====

    ``Rem`` corresponds to the ``%`` operator in C.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy import Rem
    >>> Rem(x**3, y)
    Rem(x**3, y)
    >>> Rem(x**3, y).subs({x: -5, y: 3})
    -2

    See Also
    ========

    Mod
    """
    kind = NumberKind
    @classmethod
    def eval(cls, p, q):
        """Return the function remainder if both p, q are numbers and q is not
        zero.
        """
