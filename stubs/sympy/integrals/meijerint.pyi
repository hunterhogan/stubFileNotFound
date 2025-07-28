from .integrals import Integral as Integral
from _typeshed import Incomplete
from sympy import SYMPY_DEBUG as SYMPY_DEBUG
from sympy.core import Expr as Expr, S as S
from sympy.core.add import Add as Add
from sympy.core.basic import Basic as Basic
from sympy.core.cache import cacheit as cacheit
from sympy.core.containers import Tuple as Tuple
from sympy.core.exprtools import factor_terms as factor_terms
from sympy.core.function import Function as Function, expand as expand, expand_mul as expand_mul, expand_power_base as expand_power_base, expand_trig as expand_trig
from sympy.core.intfunc import ilcm as ilcm
from sympy.core.mul import Mul as Mul
from sympy.core.numbers import Rational as Rational, pi as pi
from sympy.core.relational import Eq as Eq, Ne as Ne, _canonical_coeff as _canonical_coeff
from sympy.core.sorting import default_sort_key as default_sort_key, ordered as ordered
from sympy.core.symbol import Dummy as Dummy, Symbol as Symbol, Wild as Wild, symbols as symbols
from sympy.core.sympify import sympify as sympify
from sympy.functions.combinatorial.factorials import factorial as factorial
from sympy.functions.elementary.complexes import Abs as Abs, arg as arg, im as im, periodic_argument as periodic_argument, polar_lift as polar_lift, polarify as polarify, principal_branch as principal_branch, re as re, sign as sign, unbranched_argument as unbranched_argument, unpolarify as unpolarify
from sympy.functions.elementary.exponential import exp as exp, exp_polar as exp_polar, log as log
from sympy.functions.elementary.hyperbolic import HyperbolicFunction as HyperbolicFunction, _rewrite_hyperbolics_as_exp as _rewrite_hyperbolics_as_exp, cosh as cosh, sinh as sinh
from sympy.functions.elementary.integers import ceiling as ceiling
from sympy.functions.elementary.miscellaneous import sqrt as sqrt
from sympy.functions.elementary.piecewise import Piecewise as Piecewise, piecewise_fold as piecewise_fold
from sympy.functions.elementary.trigonometric import TrigonometricFunction as TrigonometricFunction, cos as cos, sin as sin, sinc as sinc
from sympy.functions.special.bessel import besseli as besseli, besselj as besselj, besselk as besselk, bessely as bessely
from sympy.functions.special.delta_functions import DiracDelta as DiracDelta, Heaviside as Heaviside
from sympy.functions.special.elliptic_integrals import elliptic_e as elliptic_e, elliptic_k as elliptic_k
from sympy.functions.special.error_functions import Chi as Chi, Ci as Ci, Ei as Ei, Shi as Shi, Si as Si, erf as erf, erfc as erfc, erfi as erfi, expint as expint, fresnelc as fresnelc, fresnels as fresnels
from sympy.functions.special.gamma_functions import gamma as gamma
from sympy.functions.special.hyper import hyper as hyper, meijerg as meijerg
from sympy.functions.special.singularity_functions import SingularityFunction as SingularityFunction
from sympy.logic.boolalg import And as And, BooleanAtom as BooleanAtom, BooleanFunction as BooleanFunction, Not as Not, Or as Or
from sympy.polys import cancel as cancel, factor as factor
from sympy.utilities.iterables import multiset_partitions as multiset_partitions
from sympy.utilities.timeutils import timethis as timethis

z: Incomplete

def _has(res, *f): ...
def _create_lookup_table(table):
    """ Add formulae for the function -> meijerg lookup table. """

timeit: Incomplete

def _mytype(f: Basic, x: Symbol) -> tuple[type[Basic], ...]:
    """ Create a hashable entity describing the type of f. """

class _CoeffExpValueError(ValueError):
    """
    Exception raised by _get_coeff_exp, for internal use only.
    """

def _get_coeff_exp(expr, x):
    """
    When expr is known to be of the form c*x**b, with c and/or b possibly 1,
    return c, b.

    Examples
    ========

    >>> from sympy.abc import x, a, b
    >>> from sympy.integrals.meijerint import _get_coeff_exp
    >>> _get_coeff_exp(a*x**b, x)
    (a, b)
    >>> _get_coeff_exp(x, x)
    (1, 1)
    >>> _get_coeff_exp(2*x, x)
    (2, 1)
    >>> _get_coeff_exp(x**3, x)
    (1, 3)
    """
def _exponents(expr, x):
    """
    Find the exponents of ``x`` (not including zero) in ``expr``.

    Examples
    ========

    >>> from sympy.integrals.meijerint import _exponents
    >>> from sympy.abc import x, y
    >>> from sympy import sin
    >>> _exponents(x, x)
    {1}
    >>> _exponents(x**2, x)
    {2}
    >>> _exponents(x**2 + x, x)
    {1, 2}
    >>> _exponents(x**3*sin(x + x**y) + 1/x, x)
    {-1, 1, 3, y}
    """
def _functions(expr, x):
    """ Find the types of functions in expr, to estimate the complexity. """
def _find_splitting_points(expr, x):
    """
    Find numbers a such that a linear substitution x -> x + a would
    (hopefully) simplify expr.

    Examples
    ========

    >>> from sympy.integrals.meijerint import _find_splitting_points as fsp
    >>> from sympy import sin
    >>> from sympy.abc import x
    >>> fsp(x, x)
    {0}
    >>> fsp((x-1)**3, x)
    {1}
    >>> fsp(sin(x+3)*x, x)
    {-3, 0}
    """
def _split_mul(f, x):
    '''
    Split expression ``f`` into fac, po, g, where fac is a constant factor,
    po = x**s for some s independent of s, and g is "the rest".

    Examples
    ========

    >>> from sympy.integrals.meijerint import _split_mul
    >>> from sympy import sin
    >>> from sympy.abc import s, x
    >>> _split_mul((3*x)**s*sin(x**2)*x, x)
    (3**s, x*x**s, sin(x**2))
    '''
def _mul_args(f):
    """
    Return a list ``L`` such that ``Mul(*L) == f``.

    If ``f`` is not a ``Mul`` or ``Pow``, ``L=[f]``.
    If ``f=g**n`` for an integer ``n``, ``L=[g]*n``.
    If ``f`` is a ``Mul``, ``L`` comes from applying ``_mul_args`` to all factors of ``f``.
    """
def _mul_as_two_parts(f):
    """
    Find all the ways to split ``f`` into a product of two terms.
    Return None on failure.

    Explanation
    ===========

    Although the order is canonical from multiset_partitions, this is
    not necessarily the best order to process the terms. For example,
    if the case of len(gs) == 2 is removed and multiset is allowed to
    sort the terms, some tests fail.

    Examples
    ========

    >>> from sympy.integrals.meijerint import _mul_as_two_parts
    >>> from sympy import sin, exp, ordered
    >>> from sympy.abc import x
    >>> list(ordered(_mul_as_two_parts(x*sin(x)*exp(x))))
    [(x, exp(x)*sin(x)), (x*exp(x), sin(x)), (x*sin(x), exp(x))]
    """
def _inflate_g(g, n):
    """ Return C, h such that h is a G function of argument z**n and
        g = C*h. """
def _flip_g(g):
    """ Turn the G function into one of inverse argument
        (i.e. G(1/x) -> G'(x)) """
def _inflate_fox_h(g, a):
    """
    Let d denote the integrand in the definition of the G function ``g``.
    Consider the function H which is defined in the same way, but with
    integrand d/Gamma(a*s) (contour conventions as usual).

    If ``a`` is rational, the function H can be written as C*G, for a constant C
    and a G-function G.

    This function returns C, G.
    """

_dummies: dict[tuple[str, str], Dummy]

def _dummy(name, token, expr, **kwargs):
    """
    Return a dummy. This will return the same dummy if the same token+name is
    requested more than once, and it is not already in expr.
    This is for being cache-friendly.
    """
def _dummy_(name, token, **kwargs):
    """
    Return a dummy associated to name and token. Same effect as declaring
    it globally.
    """
def _is_analytic(f, x):
    """ Check if f(x), when expressed using G functions on the positive reals,
        will in fact agree with the G functions almost everywhere """
def _condsimp(cond, first: bool = True):
    """
    Do naive simplifications on ``cond``.

    Explanation
    ===========

    Note that this routine is completely ad-hoc, simplification rules being
    added as need arises rather than following any logical pattern.

    Examples
    ========

    >>> from sympy.integrals.meijerint import _condsimp as simp
    >>> from sympy import Or, Eq
    >>> from sympy.abc import x, y
    >>> simp(Or(x < y, Eq(x, y)))
    x <= y
    """
def _eval_cond(cond):
    """ Re-evaluate the conditions. """
def _my_principal_branch(expr, period, full_pb: bool = False):
    """ Bring expr nearer to its principal branch by removing superfluous
        factors.
        This function does *not* guarantee to yield the principal branch,
        to avoid introducing opaque principal_branch() objects,
        unless full_pb=True. """
def _rewrite_saxena_1(fac, po, g, x):
    """
    Rewrite the integral fac*po*g dx, from zero to infinity, as
    integral fac*G, where G has argument a*x. Note po=x**s.
    Return fac, G.
    """
def _check_antecedents_1(g, x, helper: bool = False):
    """
    Return a condition under which the mellin transform of g exists.
    Any power of x has already been absorbed into the G function,
    so this is just $\\int_0^\\infty g\\, dx$.

    See [L, section 5.6.1]. (Note that s=1.)

    If ``helper`` is True, only check if the MT exists at infinity, i.e. if
    $\\int_1^\\infty g\\, dx$ exists.
    """
def _int0oo_1(g, x):
    """
    Evaluate $\\int_0^\\infty g\\, dx$ using G functions,
    assuming the necessary conditions are fulfilled.

    Examples
    ========

    >>> from sympy.abc import a, b, c, d, x, y
    >>> from sympy import meijerg
    >>> from sympy.integrals.meijerint import _int0oo_1
    >>> _int0oo_1(meijerg([a], [b], [c], [d], x*y), x)
    gamma(-a)*gamma(c + 1)/(y*gamma(-d)*gamma(b + 1))
    """
def _rewrite_saxena(fac, po, g1, g2, x, full_pb: bool = False):
    """
    Rewrite the integral ``fac*po*g1*g2`` from 0 to oo in terms of G
    functions with argument ``c*x``.

    Explanation
    ===========

    Return C, f1, f2 such that integral C f1 f2 from 0 to infinity equals
    integral fac ``po``, ``g1``, ``g2`` from 0 to infinity.

    Examples
    ========

    >>> from sympy.integrals.meijerint import _rewrite_saxena
    >>> from sympy.abc import s, t, m
    >>> from sympy import meijerg
    >>> g1 = meijerg([], [], [0], [], s*t)
    >>> g2 = meijerg([], [], [m/2], [-m/2], t**2/4)
    >>> r = _rewrite_saxena(1, t**0, g1, g2, t)
    >>> r[0]
    s/(4*sqrt(pi))
    >>> r[1]
    meijerg(((), ()), ((-1/2, 0), ()), s**2*t/4)
    >>> r[2]
    meijerg(((), ()), ((m/2,), (-m/2,)), t/4)
    """
def _check_antecedents(g1, g2, x):
    """ Return a condition under which the integral theorem applies. """
def _int0oo(g1, g2, x):
    """
    Express integral from zero to infinity g1*g2 using a G function,
    assuming the necessary conditions are fulfilled.

    Examples
    ========

    >>> from sympy.integrals.meijerint import _int0oo
    >>> from sympy.abc import s, t, m
    >>> from sympy import meijerg, S
    >>> g1 = meijerg([], [], [-S(1)/2, 0], [], s**2*t/4)
    >>> g2 = meijerg([], [], [m/2], [-m/2], t/4)
    >>> _int0oo(g1, g2, t)
    4*meijerg(((0, 1/2), ()), ((m/2,), (-m/2,)), s**(-2))/s**2
    """
def _rewrite_inversion(fac, po, g, x):
    """ Absorb ``po`` == x**s into g. """
def _check_antecedents_inversion(g, x):
    """ Check antecedents for the laplace inversion integral. """
def _int_inversion(g, x, t):
    """
    Compute the laplace inversion integral, assuming the formula applies.
    """

_lookup_table: Incomplete

@cacheit
@timeit
def _rewrite_single(f, x, recursive: bool = True):
    """
    Try to rewrite f as a sum of single G functions of the form
    C*x**s*G(a*x**b), where b is a rational number and C is independent of x.
    We guarantee that result.argument.as_coeff_mul(x) returns (a, (x**b,))
    or (a, ()).
    Returns a list of tuples (C, s, G) and a condition cond.
    Returns None on failure.
    """
def _rewrite1(f, x, recursive: bool = True):
    """
    Try to rewrite ``f`` using a (sum of) single G functions with argument a*x**b.
    Return fac, po, g such that f = fac*po*g, fac is independent of ``x``.
    and po = x**s.
    Here g is a result from _rewrite_single.
    Return None on failure.
    """
def _rewrite2(f, x):
    """
    Try to rewrite ``f`` as a product of two G functions of arguments a*x**b.
    Return fac, po, g1, g2 such that f = fac*po*g1*g2, where fac is
    independent of x and po is x**s.
    Here g1 and g2 are results of _rewrite_single.
    Returns None on failure.
    """
def meijerint_indefinite(f, x):
    """
    Compute an indefinite integral of ``f`` by rewriting it as a G function.

    Examples
    ========

    >>> from sympy.integrals.meijerint import meijerint_indefinite
    >>> from sympy import sin
    >>> from sympy.abc import x
    >>> meijerint_indefinite(sin(x), x)
    -cos(x)
    """
def _meijerint_indefinite_1(f, x):
    """ Helper that does not attempt any substitution. """
@timeit
def meijerint_definite(f, x, a, b):
    """
    Integrate ``f`` over the interval [``a``, ``b``], by rewriting it as a product
    of two G functions, or as a single G function.

    Return res, cond, where cond are convergence conditions.

    Examples
    ========

    >>> from sympy.integrals.meijerint import meijerint_definite
    >>> from sympy import exp, oo
    >>> from sympy.abc import x
    >>> meijerint_definite(exp(-x**2), x, -oo, oo)
    (sqrt(pi), True)

    This function is implemented as a succession of functions
    meijerint_definite, _meijerint_definite_2, _meijerint_definite_3,
    _meijerint_definite_4. Each function in the list calls the next one
    (presumably) several times. This means that calling meijerint_definite
    can be very costly.
    """
def _guess_expansion(f, x):
    """ Try to guess sensible rewritings for integrand f(x). """
def _meijerint_definite_2(f, x):
    """
    Try to integrate f dx from zero to infinity.

    The body of this function computes various 'simplifications'
    f1, f2, ... of f (e.g. by calling expand_mul(), trigexpand()
    - see _guess_expansion) and calls _meijerint_definite_3 with each of
    these in succession.
    If _meijerint_definite_3 succeeds with any of the simplified functions,
    returns this result.
    """
def _meijerint_definite_3(f, x):
    """
    Try to integrate f dx from zero to infinity.

    This function calls _meijerint_definite_4 to try to compute the
    integral. If this fails, it tries using linearity.
    """
def _my_unpolarify(f): ...
@timeit
def _meijerint_definite_4(f, x, only_double: bool = False):
    """
    Try to integrate f dx from zero to infinity.

    Explanation
    ===========

    This function tries to apply the integration theorems found in literature,
    i.e. it tries to rewrite f as either one or a product of two G-functions.

    The parameter ``only_double`` is used internally in the recursive algorithm
    to disable trying to rewrite f as a single G-function.
    """
def meijerint_inversion(f, x, t):
    """
    Compute the inverse laplace transform
    $\\int_{c+i\\infty}^{c-i\\infty} f(x) e^{tx}\\, dx$,
    for real c larger than the real part of all singularities of ``f``.

    Note that ``t`` is always assumed real and positive.

    Return None if the integral does not exist or could not be evaluated.

    Examples
    ========

    >>> from sympy.abc import x, t
    >>> from sympy.integrals.meijerint import meijerint_inversion
    >>> meijerint_inversion(1/x, x, t)
    Heaviside(t)
    """
