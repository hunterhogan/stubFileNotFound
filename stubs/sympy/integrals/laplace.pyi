from _typeshed import Incomplete
from sympy.core import I as I, S as S, pi as pi
from sympy.core.add import Add as Add
from sympy.core.cache import cacheit as cacheit
from sympy.core.expr import Expr as Expr
from sympy.core.function import AppliedUndef as AppliedUndef, Derivative as Derivative, Lambda as Lambda, Subs as Subs, WildFunction as WildFunction, diff as diff, expand as expand, expand_complex as expand_complex, expand_mul as expand_mul, expand_trig as expand_trig
from sympy.core.mul import Mul as Mul, prod as prod
from sympy.core.relational import Eq as Eq, Ge as Ge, Gt as Gt, Lt as Lt, Ne as Ne, Relational as Relational, Unequality as Unequality, _canonical as _canonical
from sympy.core.sorting import ordered as ordered
from sympy.core.symbol import Dummy as Dummy, Wild as Wild, symbols as symbols
from sympy.functions.elementary.complexes import Abs as Abs, arg as arg, im as im, periodic_argument as periodic_argument, polar_lift as polar_lift, re as re
from sympy.functions.elementary.exponential import exp as exp, log as log
from sympy.functions.elementary.hyperbolic import asinh as asinh, cosh as cosh, coth as coth, sinh as sinh
from sympy.functions.elementary.miscellaneous import Max as Max, Min as Min, sqrt as sqrt
from sympy.functions.elementary.piecewise import Piecewise as Piecewise, piecewise_exclusive as piecewise_exclusive
from sympy.functions.elementary.trigonometric import atan as atan, cos as cos, sin as sin, sinc as sinc
from sympy.functions.special.bessel import besseli as besseli, besselj as besselj, besselk as besselk, bessely as bessely
from sympy.functions.special.delta_functions import DiracDelta as DiracDelta, Heaviside as Heaviside
from sympy.functions.special.error_functions import Ei as Ei, erf as erf, erfc as erfc
from sympy.functions.special.gamma_functions import digamma as digamma, gamma as gamma, lowergamma as lowergamma, uppergamma as uppergamma
from sympy.functions.special.singularity_functions import SingularityFunction as SingularityFunction
from sympy.integrals import Integral as Integral, integrate as integrate
from sympy.integrals.transforms import IntegralTransform as IntegralTransform, IntegralTransformError as IntegralTransformError, _simplify as _simplify
from sympy.logic.boolalg import And as And, Or as Or, conjuncts as conjuncts, disjuncts as disjuncts, to_cnf as to_cnf
from sympy.matrices.matrixbase import MatrixBase as MatrixBase
from sympy.polys.matrices.linsolve import _lin_eq2dict as _lin_eq2dict
from sympy.polys.polyerrors import PolynomialError as PolynomialError
from sympy.polys.polyroots import roots as roots
from sympy.polys.polytools import Poly as Poly
from sympy.polys.rationaltools import together as together
from sympy.polys.rootoftools import RootSum as RootSum
from sympy.utilities.exceptions import SymPyDeprecationWarning as SymPyDeprecationWarning, ignore_warnings as ignore_warnings, sympy_deprecation_warning as sympy_deprecation_warning
from sympy.utilities.misc import debugf as debugf

_LT_level: int

def DEBUG_WRAP(func): ...
def _debug(text) -> None: ...
def _simplifyconds(expr, s, a):
    """
    Naively simplify some conditions occurring in ``expr``,
    given that `\\operatorname{Re}(s) > a`.

    Examples
    ========

    >>> from sympy.integrals.laplace import _simplifyconds
    >>> from sympy.abc import x
    >>> from sympy import sympify as S
    >>> _simplifyconds(abs(x**2) < 1, x, 1)
    False
    >>> _simplifyconds(abs(x**2) < 1, x, 2)
    False
    >>> _simplifyconds(abs(x**2) < 1, x, 0)
    Abs(x**2) < 1
    >>> _simplifyconds(abs(1/x**2) < 1, x, 1)
    True
    >>> _simplifyconds(S(1) < abs(x), x, 1)
    True
    >>> _simplifyconds(S(1) < abs(1/x), x, 1)
    False

    >>> from sympy import Ne
    >>> _simplifyconds(Ne(1, x**3), x, 1)
    True
    >>> _simplifyconds(Ne(1, x**3), x, 2)
    True
    >>> _simplifyconds(Ne(1, x**3), x, 0)
    Ne(1, x**3)
    """
@DEBUG_WRAP
def expand_dirac_delta(expr):
    """
    Expand an expression involving DiractDelta to get it as a linear
    combination of DiracDelta functions.
    """
@DEBUG_WRAP
def _laplace_transform_integration(f, t, s_, *, simplify):
    """ The backend function for doing Laplace transforms by integration.

    This backend assumes that the frontend has already split sums
    such that `f` is to an addition anymore.
    """
@DEBUG_WRAP
def _laplace_deep_collect(f, t):
    """
    This is an internal helper function that traverses through the expression
    tree of `f(t)` and collects arguments. The purpose of it is that
    anything like `f(w*t-1*t-c)` will be written as `f((w-1)*t-c)` such that
    it can match `f(a*t+b)`.
    """
@cacheit
def _laplace_build_rules():
    """
    This is an internal helper function that returns the table of Laplace
    transform rules in terms of the time variable `t` and the frequency
    variable `s`.  It is used by ``_laplace_apply_rules``.  Each entry is a
    tuple containing:

        (time domain pattern,
         frequency-domain replacement,
         condition for the rule to be applied,
         convergence plane,
         preparation function)

    The preparation function is a function with one argument that is applied
    to the expression before matching. For most rules it should be
    ``_laplace_deep_collect``.
    """
@DEBUG_WRAP
def _laplace_rule_timescale(f, t, s):
    """
    This function applies the time-scaling rule of the Laplace transform in
    a straight-forward way. For example, if it gets ``(f(a*t), t, s)``, it will
    compute ``LaplaceTransform(f(t)/a, t, s/a)`` if ``a>0``.
    """
@DEBUG_WRAP
def _laplace_rule_heaviside(f, t, s):
    """
    This function deals with time-shifted Heaviside step functions. If the time
    shift is positive, it applies the time-shift rule of the Laplace transform.
    For example, if it gets ``(Heaviside(t-a)*f(t), t, s)``, it will compute
    ``exp(-a*s)*LaplaceTransform(f(t+a), t, s)``.

    If the time shift is negative, the Heaviside function is simply removed
    as it means nothing to the Laplace transform.

    The function does not remove a factor ``Heaviside(t)``; this is done by
    the simple rules.
    """
@DEBUG_WRAP
def _laplace_rule_exp(f, t, s):
    """
    If this function finds a factor ``exp(a*t)``, it applies the
    frequency-shift rule of the Laplace transform and adjusts the convergence
    plane accordingly.  For example, if it gets ``(exp(-a*t)*f(t), t, s)``, it
    will compute ``LaplaceTransform(f(t), t, s+a)``.
    """
@DEBUG_WRAP
def _laplace_rule_delta(f, t, s):
    """
    If this function finds a factor ``DiracDelta(b*t-a)``, it applies the
    masking property of the delta distribution. For example, if it gets
    ``(DiracDelta(t-a)*f(t), t, s)``, it will return
    ``(f(a)*exp(-a*s), -a, True)``.
    """
@DEBUG_WRAP
def _laplace_trig_split(fn):
    """
    Helper function for `_laplace_rule_trig`.  This function returns two terms
    `f` and `g`.  `f` contains all product terms with sin, cos, sinh, cosh in
    them; `g` contains everything else.
    """
@DEBUG_WRAP
def _laplace_trig_expsum(f, t):
    """
    Helper function for `_laplace_rule_trig`.  This function expects the `f`
    from `_laplace_trig_split`.  It returns two lists `xm` and `xn`.  `xm` is
    a list of dictionaries with keys `k` and `a` representing a function
    `k*exp(a*t)`.  `xn` is a list of all terms that cannot be brought into
    that form, which may happen, e.g., when a trigonometric function has
    another function in its argument.
    """
@DEBUG_WRAP
def _laplace_trig_ltex(xm, t, s):
    """
    Helper function for `_laplace_rule_trig`.  This function takes the list of
    exponentials `xm` from `_laplace_trig_expsum` and simplifies complex
    conjugate and real symmetric poles.  It returns the result as a sum and
    the convergence plane.
    """
@DEBUG_WRAP
def _laplace_rule_trig(fn, t_, s):
    """
    This rule covers trigonometric factors by splitting everything into a
    sum of exponential functions and collecting complex conjugate poles and
    real symmetric poles.
    """
@DEBUG_WRAP
def _laplace_rule_diff(f, t, s):
    """
    This function looks for derivatives in the time domain and replaces it
    by factors of `s` and initial conditions in the frequency domain. For
    example, if it gets ``(diff(f(t), t), t, s)``, it will compute
    ``s*LaplaceTransform(f(t), t, s) - f(0)``.
    """
@DEBUG_WRAP
def _laplace_rule_sdiff(f, t, s):
    """
    This function looks for multiplications with polynoimials in `t` as they
    correspond to differentiation in the frequency domain. For example, if it
    gets ``(t*f(t), t, s)``, it will compute
    ``-Derivative(LaplaceTransform(f(t), t, s), s)``.
    """
@DEBUG_WRAP
def _laplace_expand(f, t, s):
    """
    This function tries to expand its argument with successively stronger
    methods: first it will expand on the top level, then it will expand any
    multiplications in depth, then it will try all available expansion methods,
    and finally it will try to expand trigonometric functions.

    If it can expand, it will then compute the Laplace transform of the
    expanded term.
    """
@DEBUG_WRAP
def _laplace_apply_prog_rules(f, t, s):
    """
    This function applies all program rules and returns the result if one
    of them gives a result.
    """
@DEBUG_WRAP
def _laplace_apply_simple_rules(f, t, s):
    """
    This function applies all simple rules and returns the result if one
    of them gives a result.
    """
@DEBUG_WRAP
def _piecewise_to_heaviside(f, t):
    """
    This function converts a Piecewise expression to an expression written
    with Heaviside. It is not exact, but valid in the context of the Laplace
    transform.
    """
def laplace_correspondence(f, fdict, /):
    '''
    This helper function takes a function `f` that is the result of a
    ``laplace_transform`` or an ``inverse_laplace_transform``.  It replaces all
    unevaluated ``LaplaceTransform(y(t), t, s)`` by `Y(s)` for any `s` and
    all ``InverseLaplaceTransform(Y(s), s, t)`` by `y(t)` for any `t` if
    ``fdict`` contains a correspondence ``{y: Y}``.

    Parameters
    ==========

    f : sympy expression
        Expression containing unevaluated ``LaplaceTransform`` or
        ``LaplaceTransform`` objects.
    fdict : dictionary
        Dictionary containing one or more function correspondences,
        e.g., ``{x: X, y: Y}`` meaning that ``X`` and ``Y`` are the
        Laplace transforms of ``x`` and ``y``, respectively.

    Examples
    ========

    >>> from sympy import laplace_transform, diff, Function
    >>> from sympy import laplace_correspondence, inverse_laplace_transform
    >>> from sympy.abc import t, s
    >>> y = Function("y")
    >>> Y = Function("Y")
    >>> z = Function("z")
    >>> Z = Function("Z")
    >>> f = laplace_transform(diff(y(t), t, 1) + z(t), t, s, noconds=True)
    >>> laplace_correspondence(f, {y: Y, z: Z})
    s*Y(s) + Z(s) - y(0)
    >>> f = inverse_laplace_transform(Y(s), s, t)
    >>> laplace_correspondence(f, {y: Y})
    y(t)
    '''
def laplace_initial_conds(f, t, fdict, /):
    '''
    This helper function takes a function `f` that is the result of a
    ``laplace_transform``.  It takes an fdict of the form ``{y: [1, 4, 2]}``,
    where the values in the list are the initial value, the initial slope, the
    initial second derivative, etc., of the function `y(t)`, and replaces all
    unevaluated initial conditions.

    Parameters
    ==========

    f : sympy expression
        Expression containing initial conditions of unevaluated functions.
    t : sympy expression
        Variable for which the initial conditions are to be applied.
    fdict : dictionary
        Dictionary containing a list of initial conditions for every
        function, e.g., ``{y: [0, 1, 2], x: [3, 4, 5]}``. The order
        of derivatives is ascending, so `0`, `1`, `2` are `y(0)`, `y\'(0)`,
        and `y\'\'(0)`, respectively.

    Examples
    ========

    >>> from sympy import laplace_transform, diff, Function
    >>> from sympy import laplace_correspondence, laplace_initial_conds
    >>> from sympy.abc import t, s
    >>> y = Function("y")
    >>> Y = Function("Y")
    >>> f = laplace_transform(diff(y(t), t, 3), t, s, noconds=True)
    >>> g = laplace_correspondence(f, {y: Y})
    >>> laplace_initial_conds(g, t, {y: [2, 4, 8, 16, 32]})
    s**3*Y(s) - 2*s**2 - 4*s - 8
    '''
@DEBUG_WRAP
def _laplace_transform(fn, t_, s_, *, simplify):
    """
    Front-end function of the Laplace transform. It tries to apply all known
    rules recursively, and if everything else fails, it tries to integrate.
    """

class LaplaceTransform(IntegralTransform):
    """
    Class representing unevaluated Laplace transforms.

    For usage of this class, see the :class:`IntegralTransform` docstring.

    For how to compute Laplace transforms, see the :func:`laplace_transform`
    docstring.

    If this is called with ``.doit()``, it returns the Laplace transform as an
    expression. If it is called with ``.doit(noconds=False)``, it returns a
    tuple containing the same expression, a convergence plane, and conditions.
    """
    _name: str
    def _compute_transform(self, f, t, s, **hints): ...
    def _as_integral(self, f, t, s): ...
    def doit(self, **hints):
        """
        Try to evaluate the transform in closed form.

        Explanation
        ===========

        Standard hints are the following:
        - ``noconds``:  if True, do not return convergence conditions. The
        default setting is `True`.
        - ``simplify``: if True, it simplifies the final result. The
        default setting is `False`.
        """

def laplace_transform(f, t, s, legacy_matrix: bool = True, **hints):
    """
    Compute the Laplace Transform `F(s)` of `f(t)`,

    .. math :: F(s) = \\int_{0^{-}}^\\infty e^{-st} f(t) \\mathrm{d}t.

    Explanation
    ===========

    For all sensible functions, this converges absolutely in a
    half-plane

    .. math :: a < \\operatorname{Re}(s)

    This function returns ``(F, a, cond)`` where ``F`` is the Laplace
    transform of ``f``, `a` is the half-plane of convergence, and `cond` are
    auxiliary convergence conditions.

    The implementation is rule-based, and if you are interested in which
    rules are applied, and whether integration is attempted, you can switch
    debug information on by setting ``sympy.SYMPY_DEBUG=True``. The numbers
    of the rules in the debug information (and the code) refer to Bateman's
    Tables of Integral Transforms [1].

    The lower bound is `0-`, meaning that this bound should be approached
    from the lower side. This is only necessary if distributions are involved.
    At present, it is only done if `f(t)` contains ``DiracDelta``, in which
    case the Laplace transform is computed implicitly as

    .. math ::
        F(s) = \\lim_{\\tau\\to 0^{-}} \\int_{\\tau}^\\infty e^{-st}
        f(t) \\mathrm{d}t

    by applying rules.

    If the Laplace transform cannot be fully computed in closed form, this
    function returns expressions containing unevaluated
    :class:`LaplaceTransform` objects.

    For a description of possible hints, refer to the docstring of
    :func:`sympy.integrals.transforms.IntegralTransform.doit`. If
    ``noconds=True``, only `F` will be returned (i.e. not ``cond``, and also
    not the plane ``a``).

    .. deprecated:: 1.9
        Legacy behavior for matrices where ``laplace_transform`` with
        ``noconds=False`` (the default) returns a Matrix whose elements are
        tuples. The behavior of ``laplace_transform`` for matrices will change
        in a future release of SymPy to return a tuple of the transformed
        Matrix and the convergence conditions for the matrix as a whole. Use
        ``legacy_matrix=False`` to enable the new behavior.

    Examples
    ========

    >>> from sympy import DiracDelta, exp, laplace_transform
    >>> from sympy.abc import t, s, a
    >>> laplace_transform(t**4, t, s)
    (24/s**5, 0, True)
    >>> laplace_transform(t**a, t, s)
    (gamma(a + 1)/(s*s**a), 0, re(a) > -1)
    >>> laplace_transform(DiracDelta(t)-a*exp(-a*t), t, s, simplify=True)
    (s/(a + s), -re(a), True)

    There are also helper functions that make it easy to solve differential
    equations by Laplace transform. For example, to solve

    .. math :: m x''(t) + d x'(t) + k x(t) = 0

    with initial value `0` and initial derivative `v`:

    >>> from sympy import Function, laplace_correspondence, diff, solve
    >>> from sympy import laplace_initial_conds, inverse_laplace_transform
    >>> from sympy.abc import d, k, m, v
    >>> x = Function('x')
    >>> X = Function('X')
    >>> f = m*diff(x(t), t, 2) + d*diff(x(t), t) + k*x(t)
    >>> F = laplace_transform(f, t, s, noconds=True)
    >>> F = laplace_correspondence(F, {x: X})
    >>> F = laplace_initial_conds(F, t, {x: [0, v]})
    >>> F
    d*s*X(s) + k*X(s) + m*(s**2*X(s) - v)
    >>> Xs = solve(F, X(s))[0]
    >>> Xs
    m*v/(d*s + k + m*s**2)
    >>> inverse_laplace_transform(Xs, s, t)
    2*v*exp(-d*t/(2*m))*sin(t*sqrt((-d**2 + 4*k*m)/m**2)/2)*Heaviside(t)/sqrt((-d**2 + 4*k*m)/m**2)

    References
    ==========

    .. [1] Erdelyi, A. (ed.), Tables of Integral Transforms, Volume 1,
           Bateman Manuscript Prooject, McGraw-Hill (1954), available:
           https://resolver.caltech.edu/CaltechAUTHORS:20140123-101456353

    See Also
    ========

    inverse_laplace_transform, mellin_transform, fourier_transform
    hankel_transform, inverse_hankel_transform

    """
@DEBUG_WRAP
def _inverse_laplace_transform_integration(F, s, t_, plane, *, simplify):
    """ The backend function for inverse Laplace transforms. """
@DEBUG_WRAP
def _complete_the_square_in_denom(f, s): ...
@cacheit
def _inverse_laplace_build_rules():
    """
    This is an internal helper function that returns the table of inverse
    Laplace transform rules in terms of the time variable `t` and the
    frequency variable `s`.  It is used by `_inverse_laplace_apply_rules`.
    """
@DEBUG_WRAP
def _inverse_laplace_apply_simple_rules(f, s, t):
    """
    Helper function for the class InverseLaplaceTransform.
    """
@DEBUG_WRAP
def _inverse_laplace_diff(f, s, t, plane):
    """
    Helper function for the class InverseLaplaceTransform.
    """
@DEBUG_WRAP
def _inverse_laplace_time_shift(F, s, t, plane):
    """
    Helper function for the class InverseLaplaceTransform.
    """
@DEBUG_WRAP
def _inverse_laplace_freq_shift(F, s, t, plane):
    """
    Helper function for the class InverseLaplaceTransform.
    """
@DEBUG_WRAP
def _inverse_laplace_time_diff(F, s, t, plane):
    """
    Helper function for the class InverseLaplaceTransform.
    """
@DEBUG_WRAP
def _inverse_laplace_irrational(fn, s, t, plane):
    """
    Helper function for the class InverseLaplaceTransform.
    """
@DEBUG_WRAP
def _inverse_laplace_early_prog_rules(F, s, t, plane):
    """
    Helper function for the class InverseLaplaceTransform.
    """
@DEBUG_WRAP
def _inverse_laplace_apply_prog_rules(F, s, t, plane):
    """
    Helper function for the class InverseLaplaceTransform.
    """
@DEBUG_WRAP
def _inverse_laplace_expand(fn, s, t, plane):
    """
    Helper function for the class InverseLaplaceTransform.
    """
@DEBUG_WRAP
def _inverse_laplace_rational(fn, s, t, plane, *, simplify):
    """
    Helper function for the class InverseLaplaceTransform.
    """
@DEBUG_WRAP
def _inverse_laplace_transform(fn, s_, t_, plane, *, simplify, dorational):
    """
    Front-end function of the inverse Laplace transform. It tries to apply all
    known rules recursively.  If everything else fails, it tries to integrate.
    """

class InverseLaplaceTransform(IntegralTransform):
    """
    Class representing unevaluated inverse Laplace transforms.

    For usage of this class, see the :class:`IntegralTransform` docstring.

    For how to compute inverse Laplace transforms, see the
    :func:`inverse_laplace_transform` docstring.
    """
    _name: str
    _none_sentinel: Incomplete
    _c: Incomplete
    def __new__(cls, F, s, x, plane, **opts): ...
    @property
    def fundamental_plane(self): ...
    def _compute_transform(self, F, s, t, **hints): ...
    def _as_integral(self, F, s, t): ...
    def doit(self, **hints):
        """
        Try to evaluate the transform in closed form.

        Explanation
        ===========

        Standard hints are the following:
        - ``noconds``:  if True, do not return convergence conditions. The
        default setting is `True`.
        - ``simplify``: if True, it simplifies the final result. The
        default setting is `False`.
        """

def inverse_laplace_transform(F, s, t, plane=None, **hints):
    """
    Compute the inverse Laplace transform of `F(s)`, defined as

    .. math ::
        f(t) = \\frac{1}{2\\pi i} \\int_{c-i\\infty}^{c+i\\infty} e^{st}
        F(s) \\mathrm{d}s,

    for `c` so large that `F(s)` has no singularites in the
    half-plane `\\operatorname{Re}(s) > c-\\epsilon`.

    Explanation
    ===========

    The plane can be specified by
    argument ``plane``, but will be inferred if passed as None.

    Under certain regularity conditions, this recovers `f(t)` from its
    Laplace Transform `F(s)`, for non-negative `t`, and vice
    versa.

    If the integral cannot be computed in closed form, this function returns
    an unevaluated :class:`InverseLaplaceTransform` object.

    Note that this function will always assume `t` to be real,
    regardless of the SymPy assumption on `t`.

    For a description of possible hints, refer to the docstring of
    :func:`sympy.integrals.transforms.IntegralTransform.doit`.

    Examples
    ========

    >>> from sympy import inverse_laplace_transform, exp, Symbol
    >>> from sympy.abc import s, t
    >>> a = Symbol('a', positive=True)
    >>> inverse_laplace_transform(exp(-a*s)/s, s, t)
    Heaviside(-a + t)

    See Also
    ========

    laplace_transform
    hankel_transform, inverse_hankel_transform
    """
def _fast_inverse_laplace(e, s, t):
    """Fast inverse Laplace transform of rational function including RootSum"""
