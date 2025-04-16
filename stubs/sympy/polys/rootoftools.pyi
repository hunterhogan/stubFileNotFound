from _typeshed import Incomplete
from sympy.core import Expr

__all__ = ['CRootOf', 'rootof', 'RootOf', 'ComplexRootOf', 'RootSum']

class _pure_key_dict:
    """A minimal dictionary that makes sure that the key is a
    univariate PurePoly instance.

    Examples
    ========

    Only the following actions are guaranteed:

    >>> from sympy.polys.rootoftools import _pure_key_dict
    >>> from sympy import PurePoly
    >>> from sympy.abc import x, y

    1) creation

    >>> P = _pure_key_dict()

    2) assignment for a PurePoly or univariate polynomial

    >>> P[x] = 1
    >>> P[PurePoly(x - y, x)] = 2

    3) retrieval based on PurePoly key comparison (use this
       instead of the get method)

    >>> P[y]
    1

    4) KeyError when trying to retrieve a nonexisting key

    >>> P[y + 1]
    Traceback (most recent call last):
    ...
    KeyError: PurePoly(y + 1, y, domain='ZZ')

    5) ability to query with ``in``

    >>> x + 1 in P
    False

    NOTE: this is a *not* a dictionary. It is a very basic object
    for internal use that makes sure to always address its cache
    via PurePoly instances. It does not, for example, implement
    ``get`` or ``setdefault``.
    """
    _dict: Incomplete
    def __init__(self) -> None: ...
    def __getitem__(self, k): ...
    def __setitem__(self, k, v) -> None: ...
    def __contains__(self, k) -> bool: ...

def rootof(f, x, index: Incomplete | None = None, radicals: bool = True, expand: bool = True):
    """An indexed root of a univariate polynomial.

    Returns either a :obj:`ComplexRootOf` object or an explicit
    expression involving radicals.

    Parameters
    ==========

    f : Expr
        Univariate polynomial.
    x : Symbol, optional
        Generator for ``f``.
    index : int or Integer
    radicals : bool
               Return a radical expression if possible.
    expand : bool
             Expand ``f``.
    """

class RootOf(Expr):
    """Represents a root of a univariate polynomial.

    Base class for roots of different kinds of polynomials.
    Only complex roots are currently supported.
    """
    __slots__: Incomplete
    def __new__(cls, f, x, index: Incomplete | None = None, radicals: bool = True, expand: bool = True):
        """Construct a new ``CRootOf`` object for ``k``-th root of ``f``."""

class ComplexRootOf(RootOf):
    """Represents an indexed complex root of a polynomial.

    Roots of a univariate polynomial separated into disjoint
    real or complex intervals and indexed in a fixed order:

    * real roots come first and are sorted in increasing order;
    * complex roots come next and are sorted primarily by increasing
      real part, secondarily by increasing imaginary part.

    Currently only rational coefficients are allowed.
    Can be imported as ``CRootOf``. To avoid confusion, the
    generator must be a Symbol.


    Examples
    ========

    >>> from sympy import CRootOf, rootof
    >>> from sympy.abc import x

    CRootOf is a way to reference a particular root of a
    polynomial. If there is a rational root, it will be returned:

    >>> CRootOf.clear_cache()  # for doctest reproducibility
    >>> CRootOf(x**2 - 4, 0)
    -2

    Whether roots involving radicals are returned or not
    depends on whether the ``radicals`` flag is true (which is
    set to True with rootof):

    >>> CRootOf(x**2 - 3, 0)
    CRootOf(x**2 - 3, 0)
    >>> CRootOf(x**2 - 3, 0, radicals=True)
    -sqrt(3)
    >>> rootof(x**2 - 3, 0)
    -sqrt(3)

    The following cannot be expressed in terms of radicals:

    >>> r = rootof(4*x**5 + 16*x**3 + 12*x**2 + 7, 0); r
    CRootOf(4*x**5 + 16*x**3 + 12*x**2 + 7, 0)

    The root bounds can be seen, however, and they are used by the
    evaluation methods to get numerical approximations for the root.

    >>> interval = r._get_interval(); interval
    (-1, 0)
    >>> r.evalf(2)
    -0.98

    The evalf method refines the width of the root bounds until it
    guarantees that any decimal approximation within those bounds
    will satisfy the desired precision. It then stores the refined
    interval so subsequent requests at or below the requested
    precision will not have to recompute the root bounds and will
    return very quickly.

    Before evaluation above, the interval was

    >>> interval
    (-1, 0)

    After evaluation it is now

    >>> r._get_interval() # doctest: +SKIP
    (-165/169, -206/211)

    To reset all intervals for a given polynomial, the :meth:`_reset` method
    can be called from any CRootOf instance of the polynomial:

    >>> r._reset()
    >>> r._get_interval()
    (-1, 0)

    The :meth:`eval_approx` method will also find the root to a given
    precision but the interval is not modified unless the search
    for the root fails to converge within the root bounds. And
    the secant method is used to find the root. (The ``evalf``
    method uses bisection and will always update the interval.)

    >>> r.eval_approx(2)
    -0.98

    The interval needed to be slightly updated to find that root:

    >>> r._get_interval()
    (-1, -1/2)

    The ``evalf_rational`` will compute a rational approximation
    of the root to the desired accuracy or precision.

    >>> r.eval_rational(n=2)
    -69629/71318

    >>> t = CRootOf(x**3 + 10*x + 1, 1)
    >>> t.eval_rational(1e-1)
    15/256 - 805*I/256
    >>> t.eval_rational(1e-1, 1e-4)
    3275/65536 - 414645*I/131072
    >>> t.eval_rational(1e-4, 1e-4)
    6545/131072 - 414645*I/131072
    >>> t.eval_rational(n=2)
    104755/2097152 - 6634255*I/2097152

    Notes
    =====

    Although a PurePoly can be constructed from a non-symbol generator
    RootOf instances of non-symbols are disallowed to avoid confusion
    over what root is being represented.

    >>> from sympy import exp, PurePoly
    >>> PurePoly(x) == PurePoly(exp(x))
    True
    >>> CRootOf(x - 1, 0)
    1
    >>> CRootOf(exp(x) - 1, 0)  # would correspond to x == 0
    Traceback (most recent call last):
    ...
    sympy.polys.polyerrors.PolynomialError: generator must be a Symbol

    See Also
    ========

    eval_approx
    eval_rational

    """
    __slots__: Incomplete
    is_complex: bool
    is_number: bool
    is_finite: bool
    def __new__(cls, f, x, index: Incomplete | None = None, radicals: bool = False, expand: bool = True):
        """ Construct an indexed complex root of a polynomial.

        See ``rootof`` for the parameters.

        The default value of ``radicals`` is ``False`` to satisfy
        ``eval(srepr(expr) == expr``.
        """
    @classmethod
    def _new(cls, poly, index):
        """Construct new ``CRootOf`` object from raw data. """
    def _hashable_content(self): ...
    @property
    def expr(self): ...
    @property
    def args(self): ...
    @property
    def free_symbols(self): ...
    def _eval_is_real(self):
        """Return ``True`` if the root is real. """
    def _eval_is_imaginary(self):
        """Return ``True`` if the root is imaginary. """
    @classmethod
    def real_roots(cls, poly, radicals: bool = True):
        """Get real roots of a polynomial. """
    @classmethod
    def all_roots(cls, poly, radicals: bool = True):
        """Get real and complex roots of a polynomial. """
    @classmethod
    def _get_reals_sqf(cls, currentfactor, use_cache: bool = True):
        """Get real root isolating intervals for a square-free factor."""
    @classmethod
    def _get_complexes_sqf(cls, currentfactor, use_cache: bool = True):
        """Get complex root isolating intervals for a square-free factor."""
    @classmethod
    def _get_reals(cls, factors, use_cache: bool = True):
        """Compute real root isolating intervals for a list of factors. """
    @classmethod
    def _get_complexes(cls, factors, use_cache: bool = True):
        """Compute complex root isolating intervals for a list of factors. """
    @classmethod
    def _reals_sorted(cls, reals):
        """Make real isolating intervals disjoint and sort roots. """
    @classmethod
    def _refine_imaginary(cls, complexes): ...
    @classmethod
    def _refine_complexes(cls, complexes):
        """return complexes such that no bounding rectangles of non-conjugate
        roots would intersect. In addition, assure that neither ay nor by is
        0 to guarantee that non-real roots are distinct from real roots in
        terms of the y-bounds.
        """
    @classmethod
    def _complexes_sorted(cls, complexes):
        """Make complex isolating intervals disjoint and sort roots. """
    @classmethod
    def _reals_index(cls, reals, index):
        """
        Map initial real root index to an index in a factor where
        the root belongs.
        """
    @classmethod
    def _complexes_index(cls, complexes, index):
        """
        Map initial complex root index to an index in a factor where
        the root belongs.
        """
    @classmethod
    def _count_roots(cls, roots):
        """Count the number of real or complex roots with multiplicities."""
    @classmethod
    def _indexed_root(cls, poly, index, lazy: bool = False):
        """Get a root of a composite polynomial by index. """
    def _ensure_reals_init(self) -> None:
        """Ensure that our poly has entries in the reals cache. """
    def _ensure_complexes_init(self) -> None:
        """Ensure that our poly has entries in the complexes cache. """
    @classmethod
    def _real_roots(cls, poly):
        """Get real roots of a composite polynomial. """
    def _reset(self) -> None:
        """
        Reset all intervals
        """
    @classmethod
    def _all_roots(cls, poly, use_cache: bool = True):
        """Get real and complex roots of a composite polynomial. """
    @classmethod
    def _roots_trivial(cls, poly, radicals):
        """Compute roots in linear, quadratic and binomial cases. """
    @classmethod
    def _preprocess_roots(cls, poly):
        """Take heroic measures to make ``poly`` compatible with ``CRootOf``."""
    @classmethod
    def _postprocess_root(cls, root, radicals):
        """Return the root if it is trivial or a ``CRootOf`` object. """
    @classmethod
    def _get_roots(cls, method, poly, radicals):
        """Return postprocessed roots of specified kind. """
    @classmethod
    def clear_cache(cls) -> None:
        """Reset cache for reals and complexes.

        The intervals used to approximate a root instance are updated
        as needed. When a request is made to see the intervals, the
        most current values are shown. `clear_cache` will reset all
        CRootOf instances back to their original state.

        See Also
        ========

        _reset
        """
    def _get_interval(self):
        """Internal function for retrieving isolation interval from cache. """
    def _set_interval(self, interval) -> None:
        """Internal function for updating isolation interval in cache. """
    def _eval_subs(self, old, new): ...
    def _eval_conjugate(self): ...
    def eval_approx(self, n, return_mpmath: bool = False):
        """Evaluate this complex root to the given precision.

        This uses secant method and root bounds are used to both
        generate an initial guess and to check that the root
        returned is valid. If ever the method converges outside the
        root bounds, the bounds will be made smaller and updated.
        """
    def _eval_evalf(self, prec, **kwargs):
        """Evaluate this complex root to the given precision."""
    def eval_rational(self, dx: Incomplete | None = None, dy: Incomplete | None = None, n: int = 15):
        '''
        Return a Rational approximation of ``self`` that has real
        and imaginary component approximations that are within ``dx``
        and ``dy`` of the true values, respectively. Alternatively,
        ``n`` digits of precision can be specified.

        The interval is refined with bisection and is sure to
        converge. The root bounds are updated when the refinement
        is complete so recalculation at the same or lesser precision
        will not have to repeat the refinement and should be much
        faster.

        The following example first obtains Rational approximation to
        1e-8 accuracy for all roots of the 4-th order Legendre
        polynomial. Since the roots are all less than 1, this will
        ensure the decimal representation of the approximation will be
        correct (including rounding) to 6 digits:

        >>> from sympy import legendre_poly, Symbol
        >>> x = Symbol("x")
        >>> p = legendre_poly(4, x, polys=True)
        >>> r = p.real_roots()[-1]
        >>> r.eval_rational(10**-8).n(6)
        0.861136

        It is not necessary to a two-step calculation, however: the
        decimal representation can be computed directly:

        >>> r.evalf(17)
        0.86113631159405258

        '''
CRootOf = ComplexRootOf

class RootSum(Expr):
    """Represents a sum of all roots of a univariate polynomial. """
    __slots__: Incomplete
    def __new__(cls, expr, func: Incomplete | None = None, x: Incomplete | None = None, auto: bool = True, quadratic: bool = False):
        """Construct a new ``RootSum`` instance of roots of a polynomial."""
    @classmethod
    def _new(cls, poly, func, auto: bool = True):
        """Construct new raw ``RootSum`` instance. """
    @classmethod
    def new(cls, poly, func, auto: bool = True):
        """Construct new ``RootSum`` instance. """
    @classmethod
    def _transform(cls, expr, x):
        """Transform an expression to a polynomial. """
    @classmethod
    def _is_func_rational(cls, poly, func):
        """Check if a lambda is a rational function. """
    @classmethod
    def _rational_case(cls, poly, func):
        """Handle the rational function case. """
    def _hashable_content(self): ...
    @property
    def expr(self): ...
    @property
    def args(self): ...
    @property
    def free_symbols(self): ...
    @property
    def is_commutative(self): ...
    def doit(self, **hints): ...
    def _eval_evalf(self, prec): ...
    def _eval_derivative(self, x): ...
