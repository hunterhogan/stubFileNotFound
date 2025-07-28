from _typeshed import Incomplete
from sympy.core.numbers import oo as oo
from sympy.core.sympify import CantSympify as CantSympify
from sympy.external.gmpy import GROUND_TYPES as GROUND_TYPES
from sympy.polys.densearith import dmp_abs as dmp_abs, dmp_add as dmp_add, dmp_add_ground as dmp_add_ground, dmp_add_mul as dmp_add_mul, dmp_div as dmp_div, dmp_exquo as dmp_exquo, dmp_exquo_ground as dmp_exquo_ground, dmp_l1_norm as dmp_l1_norm, dmp_l2_norm_squared as dmp_l2_norm_squared, dmp_max_norm as dmp_max_norm, dmp_mul as dmp_mul, dmp_mul_ground as dmp_mul_ground, dmp_neg as dmp_neg, dmp_pdiv as dmp_pdiv, dmp_pexquo as dmp_pexquo, dmp_pow as dmp_pow, dmp_pquo as dmp_pquo, dmp_prem as dmp_prem, dmp_quo as dmp_quo, dmp_quo_ground as dmp_quo_ground, dmp_rem as dmp_rem, dmp_sqr as dmp_sqr, dmp_sub as dmp_sub, dmp_sub_ground as dmp_sub_ground, dmp_sub_mul as dmp_sub_mul
from sympy.polys.densebasic import dmp_convert as dmp_convert, dmp_deflate as dmp_deflate, dmp_degree_in as dmp_degree_in, dmp_degree_list as dmp_degree_list, dmp_eject as dmp_eject, dmp_exclude as dmp_exclude, dmp_from_dict as dmp_from_dict, dmp_from_sympy as dmp_from_sympy, dmp_ground as dmp_ground, dmp_ground_LC as dmp_ground_LC, dmp_ground_TC as dmp_ground_TC, dmp_ground_nth as dmp_ground_nth, dmp_ground_p as dmp_ground_p, dmp_inject as dmp_inject, dmp_list_terms as dmp_list_terms, dmp_negative_p as dmp_negative_p, dmp_normal as dmp_normal, dmp_one as dmp_one, dmp_one_p as dmp_one_p, dmp_permute as dmp_permute, dmp_slice_in as dmp_slice_in, dmp_terms_gcd as dmp_terms_gcd, dmp_to_dict as dmp_to_dict, dmp_to_tuple as dmp_to_tuple, dmp_validate as dmp_validate, dmp_zero as dmp_zero, dmp_zero_p as dmp_zero_p, dup_convert as dup_convert, dup_from_dict as dup_from_dict, dup_normal as dup_normal, dup_slice as dup_slice, dup_strip as dup_strip, ninf as ninf
from sympy.polys.densetools import dmp_clear_denoms as dmp_clear_denoms, dmp_compose as dmp_compose, dmp_diff_in as dmp_diff_in, dmp_eval_in as dmp_eval_in, dmp_ground_content as dmp_ground_content, dmp_ground_monic as dmp_ground_monic, dmp_ground_primitive as dmp_ground_primitive, dmp_ground_trunc as dmp_ground_trunc, dmp_integrate_in as dmp_integrate_in, dmp_lift as dmp_lift, dmp_shift as dmp_shift, dup_decompose as dup_decompose, dup_revert as dup_revert, dup_shift as dup_shift, dup_transform as dup_transform
from sympy.polys.domains import Domain as Domain, QQ as QQ, ZZ as ZZ
from sympy.polys.euclidtools import dmp_cancel as dmp_cancel, dmp_discriminant as dmp_discriminant, dmp_gcd as dmp_gcd, dmp_inner_gcd as dmp_inner_gcd, dmp_lcm as dmp_lcm, dmp_resultant as dmp_resultant, dmp_subresultants as dmp_subresultants, dup_gcdex as dup_gcdex, dup_half_gcdex as dup_half_gcdex, dup_invert as dup_invert
from sympy.polys.factortools import dmp_factor_list as dmp_factor_list, dmp_factor_list_include as dmp_factor_list_include, dmp_irreducible_p as dmp_irreducible_p, dup_cyclotomic_p as dup_cyclotomic_p
from sympy.polys.polyerrors import CoercionFailed as CoercionFailed, DomainError as DomainError, ExactQuotientFailed as ExactQuotientFailed, NotInvertible as NotInvertible, PolynomialError as PolynomialError, UnificationFailed as UnificationFailed
from sympy.polys.polyutils import PicklableWithSlots as PicklableWithSlots, _sort_factors as _sort_factors
from sympy.polys.rootisolation import dup_cauchy_lower_bound as dup_cauchy_lower_bound, dup_cauchy_upper_bound as dup_cauchy_upper_bound, dup_count_complex_roots as dup_count_complex_roots, dup_count_real_roots as dup_count_real_roots, dup_isolate_all_roots as dup_isolate_all_roots, dup_isolate_all_roots_sqf as dup_isolate_all_roots_sqf, dup_isolate_real_roots as dup_isolate_real_roots, dup_isolate_real_roots_sqf as dup_isolate_real_roots_sqf, dup_mignotte_sep_bound_squared as dup_mignotte_sep_bound_squared, dup_refine_real_root as dup_refine_real_root, dup_sturm as dup_sturm
from sympy.polys.sqfreetools import dmp_norm as dmp_norm, dmp_sqf_list as dmp_sqf_list, dmp_sqf_list_include as dmp_sqf_list_include, dmp_sqf_norm as dmp_sqf_norm, dmp_sqf_p as dmp_sqf_p, dmp_sqf_part as dmp_sqf_part, dup_gff_list as dup_gff_list
from sympy.utilities.exceptions import sympy_deprecation_warning as sympy_deprecation_warning

def _supported_flint_domain(D): ...

class DMP(CantSympify):
    """Dense Multivariate Polynomials over `K`. """
    __slots__: Incomplete
    lev: int
    dom: Domain
    def __new__(cls, rep, dom, lev=None): ...
    @classmethod
    def new(cls, rep, dom, lev): ...
    @property
    def rep(f):
        """Get the representation of ``f``. """
    def to_best(f):
        """Convert to DUP_Flint if possible.

        This method should be used when the domain or level is changed and it
        potentially becomes possible to convert from DMP_Python to DUP_Flint.
        """
    @classmethod
    def _validate_args(cls, rep, dom, lev) -> None: ...
    @classmethod
    def from_dict(cls, rep, lev, dom): ...
    @classmethod
    def from_list(cls, rep, lev, dom):
        """Create an instance of ``cls`` given a list of native coefficients. """
    @classmethod
    def from_sympy_list(cls, rep, lev, dom):
        """Create an instance of ``cls`` given a list of SymPy coefficients. """
    @classmethod
    def from_monoms_coeffs(cls, monoms, coeffs, lev, dom): ...
    def convert(f, dom):
        """Convert ``f`` to a ``DMP`` over the new domain. """
    def _convert(f, dom) -> None: ...
    @classmethod
    def zero(cls, lev, dom): ...
    @classmethod
    def one(cls, lev, dom): ...
    def _one(f) -> None: ...
    def __repr__(f) -> str: ...
    def __hash__(f): ...
    def __getnewargs__(self): ...
    def ground_new(f, coeff) -> None:
        """Construct a new ground instance of ``f``. """
    def unify_DMP(f, g):
        """Unify and return ``DMP`` instances of ``f`` and ``g``. """
    def to_dict(f, zero: bool = False):
        """Convert ``f`` to a dict representation with native coefficients. """
    def to_sympy_dict(f, zero: bool = False):
        """Convert ``f`` to a dict representation with SymPy coefficients. """
    def to_sympy_list(f):
        """Convert ``f`` to a list representation with SymPy coefficients. """
    def to_list(f) -> None:
        """Convert ``f`` to a list representation with native coefficients. """
    def to_tuple(f) -> None:
        """
        Convert ``f`` to a tuple representation with native coefficients.

        This is needed for hashing.
        """
    def to_ring(f):
        """Make the ground domain a ring. """
    def to_field(f):
        """Make the ground domain a field. """
    def to_exact(f):
        """Make the ground domain exact. """
    def slice(f, m, n, j: int = 0):
        """Take a continuous subsequence of terms of ``f``. """
    def _slice(f, m, n) -> None: ...
    def _slice_lev(f, m, n, j) -> None: ...
    def coeffs(f, order=None):
        """Returns all non-zero coefficients from ``f`` in lex order. """
    def monoms(f, order=None):
        """Returns all non-zero monomials from ``f`` in lex order. """
    def terms(f, order=None):
        """Returns all non-zero terms from ``f`` in lex order. """
    def _terms(f, order=None) -> None: ...
    def all_coeffs(f):
        """Returns all coefficients from ``f``. """
    def all_monoms(f):
        """Returns all monomials from ``f``. """
    def all_terms(f):
        """Returns all terms from a ``f``. """
    def lift(f):
        """Convert algebraic coefficients to rationals. """
    def _lift(f) -> None: ...
    def deflate(f) -> None:
        """Reduce degree of `f` by mapping `x_i^m` to `y_i`. """
    def inject(f, front: bool = False) -> None:
        """Inject ground domain generators into ``f``. """
    def eject(f, dom, front: bool = False) -> None:
        """Eject selected generators into the ground domain. """
    def exclude(f):
        """
        Remove useless generators from ``f``.

        Returns the removed generators and the new excluded ``f``.

        Examples
        ========

        >>> from sympy.polys.polyclasses import DMP
        >>> from sympy.polys.domains import ZZ

        >>> DMP([[[ZZ(1)]], [[ZZ(1)], [ZZ(2)]]], ZZ).exclude()
        ([2], DMP_Python([[1], [1, 2]], ZZ))

        """
    def _exclude(f) -> None: ...
    def permute(f, P):
        """
        Returns a polynomial in `K[x_{P(1)}, ..., x_{P(n)}]`.

        Examples
        ========

        >>> from sympy.polys.polyclasses import DMP
        >>> from sympy.polys.domains import ZZ

        >>> DMP([[[ZZ(2)], [ZZ(1), ZZ(0)]], [[]]], ZZ).permute([1, 0, 2])
        DMP_Python([[[2], []], [[1, 0], []]], ZZ)

        >>> DMP([[[ZZ(2)], [ZZ(1), ZZ(0)]], [[]]], ZZ).permute([1, 2, 0])
        DMP_Python([[[1], []], [[2, 0], []]], ZZ)

        """
    def _permute(f, P) -> None: ...
    def terms_gcd(f) -> None:
        """Remove GCD of terms from the polynomial ``f``. """
    def abs(f) -> None:
        """Make all coefficients in ``f`` positive. """
    def neg(f) -> None:
        """Negate all coefficients in ``f``. """
    def add_ground(f, c):
        """Add an element of the ground domain to ``f``. """
    def sub_ground(f, c):
        """Subtract an element of the ground domain from ``f``. """
    def mul_ground(f, c):
        """Multiply ``f`` by a an element of the ground domain. """
    def quo_ground(f, c):
        """Quotient of ``f`` by a an element of the ground domain. """
    def exquo_ground(f, c):
        """Exact quotient of ``f`` by a an element of the ground domain. """
    def add(f, g):
        """Add two multivariate polynomials ``f`` and ``g``. """
    def sub(f, g):
        """Subtract two multivariate polynomials ``f`` and ``g``. """
    def mul(f, g):
        """Multiply two multivariate polynomials ``f`` and ``g``. """
    def sqr(f):
        """Square a multivariate polynomial ``f``. """
    def pow(f, n):
        """Raise ``f`` to a non-negative power ``n``. """
    def pdiv(f, g):
        """Polynomial pseudo-division of ``f`` and ``g``. """
    def prem(f, g):
        """Polynomial pseudo-remainder of ``f`` and ``g``. """
    def pquo(f, g):
        """Polynomial pseudo-quotient of ``f`` and ``g``. """
    def pexquo(f, g):
        """Polynomial exact pseudo-quotient of ``f`` and ``g``. """
    def div(f, g):
        """Polynomial division with remainder of ``f`` and ``g``. """
    def rem(f, g):
        """Computes polynomial remainder of ``f`` and ``g``. """
    def quo(f, g):
        """Computes polynomial quotient of ``f`` and ``g``. """
    def exquo(f, g):
        """Computes polynomial exact quotient of ``f`` and ``g``. """
    def _add_ground(f, c) -> None: ...
    def _sub_ground(f, c) -> None: ...
    def _mul_ground(f, c) -> None: ...
    def _quo_ground(f, c) -> None: ...
    def _exquo_ground(f, c) -> None: ...
    def _add(f, g) -> None: ...
    def _sub(f, g) -> None: ...
    def _mul(f, g) -> None: ...
    def _sqr(f) -> None: ...
    def _pow(f, n) -> None: ...
    def _pdiv(f, g) -> None: ...
    def _prem(f, g) -> None: ...
    def _pquo(f, g) -> None: ...
    def _pexquo(f, g) -> None: ...
    def _div(f, g) -> None: ...
    def _rem(f, g) -> None: ...
    def _quo(f, g) -> None: ...
    def _exquo(f, g) -> None: ...
    def degree(f, j: int = 0):
        """Returns the leading degree of ``f`` in ``x_j``. """
    def _degree(f, j) -> None: ...
    def degree_list(f) -> None:
        """Returns a list of degrees of ``f``. """
    def total_degree(f) -> None:
        """Returns the total degree of ``f``. """
    def homogenize(f, s):
        """Return homogeneous polynomial of ``f``"""
    def homogeneous_order(f):
        """Returns the homogeneous order of ``f``. """
    def LC(f) -> None:
        """Returns the leading coefficient of ``f``. """
    def TC(f) -> None:
        """Returns the trailing coefficient of ``f``. """
    def nth(f, *N):
        """Returns the ``n``-th coefficient of ``f``. """
    def _nth(f, N) -> None: ...
    def max_norm(f) -> None:
        """Returns maximum norm of ``f``. """
    def l1_norm(f) -> None:
        """Returns l1 norm of ``f``. """
    def l2_norm_squared(f) -> None:
        """Return squared l2 norm of ``f``. """
    def clear_denoms(f) -> None:
        """Clear denominators, but keep the ground domain. """
    def integrate(f, m: int = 1, j: int = 0):
        """Computes the ``m``-th order indefinite integral of ``f`` in ``x_j``. """
    def _integrate(f, m, j) -> None: ...
    def diff(f, m: int = 1, j: int = 0):
        """Computes the ``m``-th order derivative of ``f`` in ``x_j``. """
    def _diff(f, m, j) -> None: ...
    def eval(f, a, j: int = 0):
        """Evaluates ``f`` at the given point ``a`` in ``x_j``. """
    def _eval(f, a) -> None: ...
    def _eval_lev(f, a, j) -> None: ...
    def half_gcdex(f, g):
        """Half extended Euclidean algorithm, if univariate. """
    def _half_gcdex(f, g) -> None: ...
    def gcdex(f, g):
        """Extended Euclidean algorithm, if univariate. """
    def _gcdex(f, g) -> None: ...
    def invert(f, g):
        """Invert ``f`` modulo ``g``, if possible. """
    def _invert(f, g) -> None: ...
    def revert(f, n):
        """Compute ``f**(-1)`` mod ``x**n``. """
    def _revert(f, n) -> None: ...
    def subresultants(f, g):
        """Computes subresultant PRS sequence of ``f`` and ``g``. """
    def _subresultants(f, g) -> None: ...
    def resultant(f, g, includePRS: bool = False):
        """Computes resultant of ``f`` and ``g`` via PRS. """
    def _resultant(f, g, includePRS: bool = False) -> None: ...
    def discriminant(f) -> None:
        """Computes discriminant of ``f``. """
    def cofactors(f, g):
        """Returns GCD of ``f`` and ``g`` and their cofactors. """
    def _cofactors(f, g) -> None: ...
    def gcd(f, g):
        """Returns polynomial GCD of ``f`` and ``g``. """
    def _gcd(f, g) -> None: ...
    def lcm(f, g):
        """Returns polynomial LCM of ``f`` and ``g``. """
    def _lcm(f, g) -> None: ...
    def cancel(f, g, include: bool = True):
        """Cancel common factors in a rational function ``f/g``. """
    def _cancel(f, g) -> None: ...
    def _cancel_include(f, g) -> None: ...
    def trunc(f, p):
        """Reduce ``f`` modulo a constant ``p``. """
    def _trunc(f, p) -> None: ...
    def monic(f) -> None:
        """Divides all coefficients by ``LC(f)``. """
    def content(f) -> None:
        """Returns GCD of polynomial coefficients. """
    def primitive(f) -> None:
        """Returns content and a primitive form of ``f``. """
    def compose(f, g):
        """Computes functional composition of ``f`` and ``g``. """
    def _compose(f, g) -> None: ...
    def decompose(f):
        """Computes functional decomposition of ``f``. """
    def _decompose(f) -> None: ...
    def shift(f, a):
        """Efficiently compute Taylor shift ``f(x + a)``. """
    def shift_list(f, a):
        """Efficiently compute Taylor shift ``f(X + A)``. """
    def _shift(f, a) -> None: ...
    def transform(f, p, q):
        """Evaluate functional transformation ``q**n * f(p/q)``."""
    def _transform(f, p, q) -> None: ...
    def sturm(f):
        """Computes the Sturm sequence of ``f``. """
    def _sturm(f) -> None: ...
    def cauchy_upper_bound(f):
        """Computes the Cauchy upper bound on the roots of ``f``. """
    def _cauchy_upper_bound(f) -> None: ...
    def cauchy_lower_bound(f):
        """Computes the Cauchy lower bound on the nonzero roots of ``f``. """
    def _cauchy_lower_bound(f) -> None: ...
    def mignotte_sep_bound_squared(f):
        """Computes the squared Mignotte bound on root separations of ``f``. """
    def _mignotte_sep_bound_squared(f) -> None: ...
    def gff_list(f):
        """Computes greatest factorial factorization of ``f``. """
    def _gff_list(f) -> None: ...
    def norm(f) -> None:
        """Computes ``Norm(f)``."""
    def sqf_norm(f) -> None:
        """Computes square-free norm of ``f``. """
    def sqf_part(f) -> None:
        """Computes square-free part of ``f``. """
    def sqf_list(f, all: bool = False) -> None:
        """Returns a list of square-free factors of ``f``. """
    def sqf_list_include(f, all: bool = False) -> None:
        """Returns a list of square-free factors of ``f``. """
    def factor_list(f) -> None:
        """Returns a list of irreducible factors of ``f``. """
    def factor_list_include(f) -> None:
        """Returns a list of irreducible factors of ``f``. """
    def intervals(f, all: bool = False, eps=None, inf=None, sup=None, fast: bool = False, sqf: bool = False):
        """Compute isolating intervals for roots of ``f``. """
    def _isolate_all_roots(f, eps, inf, sup, fast) -> None: ...
    def _isolate_all_roots_sqf(f, eps, inf, sup, fast) -> None: ...
    def _isolate_real_roots(f, eps, inf, sup, fast) -> None: ...
    def _isolate_real_roots_sqf(f, eps, inf, sup, fast) -> None: ...
    def refine_root(f, s, t, eps=None, steps=None, fast: bool = False):
        """
        Refine an isolating interval to the given precision.

        ``eps`` should be a rational number.

        """
    def _refine_real_root(f, s, t, eps, steps, fast) -> None: ...
    def count_real_roots(f, inf=None, sup=None) -> None:
        """Return the number of real roots of ``f`` in ``[inf, sup]``. """
    def count_complex_roots(f, inf=None, sup=None) -> None:
        """Return the number of complex roots of ``f`` in ``[inf, sup]``. """
    @property
    def is_zero(f) -> None:
        """Returns ``True`` if ``f`` is a zero polynomial. """
    @property
    def is_one(f) -> None:
        """Returns ``True`` if ``f`` is a unit polynomial. """
    @property
    def is_ground(f) -> None:
        """Returns ``True`` if ``f`` is an element of the ground domain. """
    @property
    def is_sqf(f) -> None:
        """Returns ``True`` if ``f`` is a square-free polynomial. """
    @property
    def is_monic(f) -> None:
        """Returns ``True`` if the leading coefficient of ``f`` is one. """
    @property
    def is_primitive(f) -> None:
        """Returns ``True`` if the GCD of the coefficients of ``f`` is one. """
    @property
    def is_linear(f) -> None:
        """Returns ``True`` if ``f`` is linear in all its variables. """
    @property
    def is_quadratic(f) -> None:
        """Returns ``True`` if ``f`` is quadratic in all its variables. """
    @property
    def is_monomial(f) -> None:
        """Returns ``True`` if ``f`` is zero or has only one term. """
    @property
    def is_homogeneous(f) -> None:
        """Returns ``True`` if ``f`` is a homogeneous polynomial. """
    @property
    def is_irreducible(f) -> None:
        """Returns ``True`` if ``f`` has no factors over its domain. """
    @property
    def is_cyclotomic(f) -> None:
        """Returns ``True`` if ``f`` is a cyclotomic polynomial. """
    def __abs__(f): ...
    def __neg__(f): ...
    def __add__(f, g): ...
    def __radd__(f, g): ...
    def __sub__(f, g): ...
    def __rsub__(f, g): ...
    def __mul__(f, g): ...
    def __rmul__(f, g): ...
    def __truediv__(f, g): ...
    def __rtruediv__(f, g): ...
    def __pow__(f, n): ...
    def __divmod__(f, g): ...
    def __mod__(f, g): ...
    def __floordiv__(f, g): ...
    def __eq__(f, g): ...
    def _strict_eq(f, g) -> None: ...
    def eq(f, g, strict: bool = False): ...
    def ne(f, g, strict: bool = False): ...
    def __lt__(f, g): ...
    def __le__(f, g): ...
    def __gt__(f, g): ...
    def __ge__(f, g): ...
    def __bool__(f) -> bool: ...

class DMP_Python(DMP):
    """Dense Multivariate Polynomials over `K`. """
    __slots__: Incomplete
    @classmethod
    def _new(cls, rep, dom, lev): ...
    def _strict_eq(f, g): ...
    def per(f, rep):
        """Create a DMP out of the given representation. """
    def ground_new(f, coeff):
        """Construct a new ground instance of ``f``. """
    def _one(f): ...
    def unify(f, g):
        """Unify representations of two multivariate polynomials. """
    def to_DUP_Flint(f):
        """Convert ``f`` to a Flint representation. """
    def to_list(f):
        """Convert ``f`` to a list representation with native coefficients. """
    def to_tuple(f):
        """Convert ``f`` to a tuple representation with native coefficients. """
    def _convert(f, dom):
        """Convert the ground domain of ``f``. """
    def _slice(f, m, n):
        """Take a continuous subsequence of terms of ``f``. """
    def _slice_lev(f, m, n, j):
        """Take a continuous subsequence of terms of ``f``. """
    def _terms(f, order=None):
        """Returns all non-zero terms from ``f`` in lex order. """
    def _lift(f):
        """Convert algebraic coefficients to rationals. """
    def deflate(f):
        """Reduce degree of `f` by mapping `x_i^m` to `y_i`. """
    def inject(f, front: bool = False):
        """Inject ground domain generators into ``f``. """
    def eject(f, dom, front: bool = False):
        """Eject selected generators into the ground domain. """
    def _exclude(f):
        """Remove useless generators from ``f``. """
    def _permute(f, P):
        """Returns a polynomial in `K[x_{P(1)}, ..., x_{P(n)}]`. """
    def terms_gcd(f):
        """Remove GCD of terms from the polynomial ``f``. """
    def _add_ground(f, c):
        """Add an element of the ground domain to ``f``. """
    def _sub_ground(f, c):
        """Subtract an element of the ground domain from ``f``. """
    def _mul_ground(f, c):
        """Multiply ``f`` by a an element of the ground domain. """
    def _quo_ground(f, c):
        """Quotient of ``f`` by a an element of the ground domain. """
    def _exquo_ground(f, c):
        """Exact quotient of ``f`` by a an element of the ground domain. """
    def abs(f):
        """Make all coefficients in ``f`` positive. """
    def neg(f):
        """Negate all coefficients in ``f``. """
    def _add(f, g):
        """Add two multivariate polynomials ``f`` and ``g``. """
    def _sub(f, g):
        """Subtract two multivariate polynomials ``f`` and ``g``. """
    def _mul(f, g):
        """Multiply two multivariate polynomials ``f`` and ``g``. """
    def sqr(f):
        """Square a multivariate polynomial ``f``. """
    def _pow(f, n):
        """Raise ``f`` to a non-negative power ``n``. """
    def _pdiv(f, g):
        """Polynomial pseudo-division of ``f`` and ``g``. """
    def _prem(f, g):
        """Polynomial pseudo-remainder of ``f`` and ``g``. """
    def _pquo(f, g):
        """Polynomial pseudo-quotient of ``f`` and ``g``. """
    def _pexquo(f, g):
        """Polynomial exact pseudo-quotient of ``f`` and ``g``. """
    def _div(f, g):
        """Polynomial division with remainder of ``f`` and ``g``. """
    def _rem(f, g):
        """Computes polynomial remainder of ``f`` and ``g``. """
    def _quo(f, g):
        """Computes polynomial quotient of ``f`` and ``g``. """
    def _exquo(f, g):
        """Computes polynomial exact quotient of ``f`` and ``g``. """
    def _degree(f, j: int = 0):
        """Returns the leading degree of ``f`` in ``x_j``. """
    def degree_list(f):
        """Returns a list of degrees of ``f``. """
    def total_degree(f):
        """Returns the total degree of ``f``. """
    def LC(f):
        """Returns the leading coefficient of ``f``. """
    def TC(f):
        """Returns the trailing coefficient of ``f``. """
    def _nth(f, N):
        """Returns the ``n``-th coefficient of ``f``. """
    def max_norm(f):
        """Returns maximum norm of ``f``. """
    def l1_norm(f):
        """Returns l1 norm of ``f``. """
    def l2_norm_squared(f):
        """Return squared l2 norm of ``f``. """
    def clear_denoms(f):
        """Clear denominators, but keep the ground domain. """
    def _integrate(f, m: int = 1, j: int = 0):
        """Computes the ``m``-th order indefinite integral of ``f`` in ``x_j``. """
    def _diff(f, m: int = 1, j: int = 0):
        """Computes the ``m``-th order derivative of ``f`` in ``x_j``. """
    def _eval(f, a): ...
    def _eval_lev(f, a, j): ...
    def _half_gcdex(f, g):
        """Half extended Euclidean algorithm, if univariate. """
    def _gcdex(f, g):
        """Extended Euclidean algorithm, if univariate. """
    def _invert(f, g):
        """Invert ``f`` modulo ``g``, if possible. """
    def _revert(f, n):
        """Compute ``f**(-1)`` mod ``x**n``. """
    def _subresultants(f, g):
        """Computes subresultant PRS sequence of ``f`` and ``g``. """
    def _resultant_includePRS(f, g):
        """Computes resultant of ``f`` and ``g`` via PRS. """
    def _resultant(f, g): ...
    def discriminant(f):
        """Computes discriminant of ``f``. """
    def _cofactors(f, g):
        """Returns GCD of ``f`` and ``g`` and their cofactors. """
    def _gcd(f, g):
        """Returns polynomial GCD of ``f`` and ``g``. """
    def _lcm(f, g):
        """Returns polynomial LCM of ``f`` and ``g``. """
    def _cancel(f, g):
        """Cancel common factors in a rational function ``f/g``. """
    def _cancel_include(f, g):
        """Cancel common factors in a rational function ``f/g``. """
    def _trunc(f, p):
        """Reduce ``f`` modulo a constant ``p``. """
    def monic(f):
        """Divides all coefficients by ``LC(f)``. """
    def content(f):
        """Returns GCD of polynomial coefficients. """
    def primitive(f):
        """Returns content and a primitive form of ``f``. """
    def _compose(f, g):
        """Computes functional composition of ``f`` and ``g``. """
    def _decompose(f):
        """Computes functional decomposition of ``f``. """
    def _shift(f, a):
        """Efficiently compute Taylor shift ``f(x + a)``. """
    def _shift_list(f, a):
        """Efficiently compute Taylor shift ``f(X + A)``. """
    def _transform(f, p, q):
        """Evaluate functional transformation ``q**n * f(p/q)``."""
    def _sturm(f):
        """Computes the Sturm sequence of ``f``. """
    def _cauchy_upper_bound(f):
        """Computes the Cauchy upper bound on the roots of ``f``. """
    def _cauchy_lower_bound(f):
        """Computes the Cauchy lower bound on the nonzero roots of ``f``. """
    def _mignotte_sep_bound_squared(f):
        """Computes the squared Mignotte bound on root separations of ``f``. """
    def _gff_list(f):
        """Computes greatest factorial factorization of ``f``. """
    def norm(f):
        """Computes ``Norm(f)``."""
    def sqf_norm(f):
        """Computes square-free norm of ``f``. """
    def sqf_part(f):
        """Computes square-free part of ``f``. """
    def sqf_list(f, all: bool = False):
        """Returns a list of square-free factors of ``f``. """
    def sqf_list_include(f, all: bool = False):
        """Returns a list of square-free factors of ``f``. """
    def factor_list(f):
        """Returns a list of irreducible factors of ``f``. """
    def factor_list_include(f):
        """Returns a list of irreducible factors of ``f``. """
    def _isolate_real_roots(f, eps, inf, sup, fast): ...
    def _isolate_real_roots_sqf(f, eps, inf, sup, fast): ...
    def _isolate_all_roots(f, eps, inf, sup, fast): ...
    def _isolate_all_roots_sqf(f, eps, inf, sup, fast): ...
    def _refine_real_root(f, s, t, eps, steps, fast): ...
    def count_real_roots(f, inf=None, sup=None):
        """Return the number of real roots of ``f`` in ``[inf, sup]``. """
    def count_complex_roots(f, inf=None, sup=None):
        """Return the number of complex roots of ``f`` in ``[inf, sup]``. """
    @property
    def is_zero(f):
        """Returns ``True`` if ``f`` is a zero polynomial. """
    @property
    def is_one(f):
        """Returns ``True`` if ``f`` is a unit polynomial. """
    @property
    def is_ground(f):
        """Returns ``True`` if ``f`` is an element of the ground domain. """
    @property
    def is_sqf(f):
        """Returns ``True`` if ``f`` is a square-free polynomial. """
    @property
    def is_monic(f):
        """Returns ``True`` if the leading coefficient of ``f`` is one. """
    @property
    def is_primitive(f):
        """Returns ``True`` if the GCD of the coefficients of ``f`` is one. """
    @property
    def is_linear(f):
        """Returns ``True`` if ``f`` is linear in all its variables. """
    @property
    def is_quadratic(f):
        """Returns ``True`` if ``f`` is quadratic in all its variables. """
    @property
    def is_monomial(f):
        """Returns ``True`` if ``f`` is zero or has only one term. """
    @property
    def is_homogeneous(f):
        """Returns ``True`` if ``f`` is a homogeneous polynomial. """
    @property
    def is_irreducible(f):
        """Returns ``True`` if ``f`` has no factors over its domain. """
    @property
    def is_cyclotomic(f):
        """Returns ``True`` if ``f`` is a cyclotomic polynomial. """

class DUP_Flint(DMP):
    """Dense Multivariate Polynomials over `K`. """
    lev: int
    __slots__: Incomplete
    def __reduce__(self): ...
    @classmethod
    def _new(cls, rep, dom, lev): ...
    def to_list(f):
        """Convert ``f`` to a list representation with native coefficients. """
    @classmethod
    def _flint_poly(cls, rep, dom, lev): ...
    @classmethod
    def _get_flint_poly_cls(cls, dom): ...
    @classmethod
    def from_rep(cls, rep, dom):
        """Create a DMP from the given representation. """
    def _strict_eq(f, g): ...
    def ground_new(f, coeff):
        """Construct a new ground instance of ``f``. """
    def _one(f): ...
    def unify(f, g) -> None:
        """Unify representations of two polynomials. """
    def to_DMP_Python(f):
        """Convert ``f`` to a Python native representation. """
    def to_tuple(f):
        """Convert ``f`` to a tuple representation with native coefficients. """
    def _convert(f, dom):
        """Convert the ground domain of ``f``. """
    def _slice(f, m, n):
        """Take a continuous subsequence of terms of ``f``. """
    def _slice_lev(f, m, n, j) -> None:
        """Take a continuous subsequence of terms of ``f``. """
    def _terms(f, order=None):
        """Returns all non-zero terms from ``f`` in lex order. """
    def _lift(f) -> None:
        """Convert algebraic coefficients to rationals. """
    def deflate(f):
        """Reduce degree of `f` by mapping `x_i^m` to `y_i`. """
    def inject(f, front: bool = False) -> None:
        """Inject ground domain generators into ``f``. """
    def eject(f, dom, front: bool = False) -> None:
        """Eject selected generators into the ground domain. """
    def _exclude(f) -> None:
        """Remove useless generators from ``f``. """
    def _permute(f, P) -> None:
        """Returns a polynomial in `K[x_{P(1)}, ..., x_{P(n)}]`. """
    def terms_gcd(f):
        """Remove GCD of terms from the polynomial ``f``. """
    def _add_ground(f, c):
        """Add an element of the ground domain to ``f``. """
    def _sub_ground(f, c):
        """Subtract an element of the ground domain from ``f``. """
    def _mul_ground(f, c):
        """Multiply ``f`` by a an element of the ground domain. """
    def _quo_ground(f, c):
        """Quotient of ``f`` by a an element of the ground domain. """
    def _exquo_ground(f, c):
        """Exact quotient of ``f`` by an element of the ground domain. """
    def abs(f):
        """Make all coefficients in ``f`` positive. """
    def neg(f):
        """Negate all coefficients in ``f``. """
    def _add(f, g):
        """Add two multivariate polynomials ``f`` and ``g``. """
    def _sub(f, g):
        """Subtract two multivariate polynomials ``f`` and ``g``. """
    def _mul(f, g):
        """Multiply two multivariate polynomials ``f`` and ``g``. """
    def sqr(f):
        """Square a multivariate polynomial ``f``. """
    def _pow(f, n):
        """Raise ``f`` to a non-negative power ``n``. """
    def _pdiv(f, g):
        """Polynomial pseudo-division of ``f`` and ``g``. """
    def _prem(f, g):
        """Polynomial pseudo-remainder of ``f`` and ``g``. """
    def _pquo(f, g):
        """Polynomial pseudo-quotient of ``f`` and ``g``. """
    def _pexquo(f, g):
        """Polynomial exact pseudo-quotient of ``f`` and ``g``. """
    def _div(f, g):
        """Polynomial division with remainder of ``f`` and ``g``. """
    def _rem(f, g):
        """Computes polynomial remainder of ``f`` and ``g``. """
    def _quo(f, g):
        """Computes polynomial quotient of ``f`` and ``g``. """
    def _exquo(f, g):
        """Computes polynomial exact quotient of ``f`` and ``g``. """
    def _degree(f, j: int = 0):
        """Returns the leading degree of ``f`` in ``x_j``. """
    def degree_list(f):
        """Returns a list of degrees of ``f``. """
    def total_degree(f):
        """Returns the total degree of ``f``. """
    def LC(f):
        """Returns the leading coefficient of ``f``. """
    def TC(f):
        """Returns the trailing coefficient of ``f``. """
    def _nth(f, N):
        """Returns the ``n``-th coefficient of ``f``. """
    def max_norm(f):
        """Returns maximum norm of ``f``. """
    def l1_norm(f):
        """Returns l1 norm of ``f``. """
    def l2_norm_squared(f):
        """Return squared l2 norm of ``f``. """
    def clear_denoms(f):
        """Clear denominators, but keep the ground domain. """
    def _integrate(f, m: int = 1, j: int = 0):
        """Computes the ``m``-th order indefinite integral of ``f`` in ``x_j``. """
    def _diff(f, m: int = 1, j: int = 0):
        """Computes the ``m``-th order derivative of ``f``. """
    def _eval(f, a): ...
    def _eval_lev(f, a, j) -> None: ...
    def _half_gcdex(f, g):
        """Half extended Euclidean algorithm. """
    def _gcdex(f, g):
        """Extended Euclidean algorithm. """
    def _invert(f, g):
        """Invert ``f`` modulo ``g``, if possible. """
    def _revert(f, n):
        """Compute ``f**(-1)`` mod ``x**n``. """
    def _subresultants(f, g):
        """Computes subresultant PRS sequence of ``f`` and ``g``. """
    def _resultant_includePRS(f, g):
        """Computes resultant of ``f`` and ``g`` via PRS. """
    def _resultant(f, g):
        """Computes resultant of ``f`` and ``g``. """
    def discriminant(f):
        """Computes discriminant of ``f``. """
    def _cofactors(f, g):
        """Returns GCD of ``f`` and ``g`` and their cofactors. """
    def _gcd(f, g):
        """Returns polynomial GCD of ``f`` and ``g``. """
    def _lcm(f, g):
        """Returns polynomial LCM of ``f`` and ``g``. """
    def _cancel(f, g):
        """Cancel common factors in a rational function ``f/g``. """
    def _cancel_include(f, g):
        """Cancel common factors in a rational function ``f/g``. """
    def _trunc(f, p):
        """Reduce ``f`` modulo a constant ``p``. """
    def monic(f):
        """Divides all coefficients by ``LC(f)``. """
    def content(f):
        """Returns GCD of polynomial coefficients. """
    def primitive(f):
        """Returns content and a primitive form of ``f``. """
    def _compose(f, g):
        """Computes functional composition of ``f`` and ``g``. """
    def _decompose(f):
        """Computes functional decomposition of ``f``. """
    def _shift(f, a):
        """Efficiently compute Taylor shift ``f(x + a)``. """
    def _transform(f, p, q):
        """Evaluate functional transformation ``q**n * f(p/q)``."""
    def _sturm(f):
        """Computes the Sturm sequence of ``f``. """
    def _cauchy_upper_bound(f):
        """Computes the Cauchy upper bound on the roots of ``f``. """
    def _cauchy_lower_bound(f):
        """Computes the Cauchy lower bound on the nonzero roots of ``f``. """
    def _mignotte_sep_bound_squared(f):
        """Computes the squared Mignotte bound on root separations of ``f``. """
    def _gff_list(f):
        """Computes greatest factorial factorization of ``f``. """
    def norm(f) -> None:
        """Computes ``Norm(f)``."""
    def sqf_norm(f) -> None:
        """Computes square-free norm of ``f``. """
    def sqf_part(f):
        """Computes square-free part of ``f``. """
    def sqf_list(f, all: bool = False):
        """Returns a list of square-free factors of ``f``. """
    def sqf_list_include(f, all: bool = False):
        """Returns a list of square-free factors of ``f``. """
    def factor_list(f):
        """Returns a list of irreducible factors of ``f``. """
    def factor_list_include(f):
        """Returns a list of irreducible factors of ``f``. """
    def _sort_factors(f, factors):
        """Sort a list of factors to canonical order. """
    def _isolate_real_roots(f, eps, inf, sup, fast): ...
    def _isolate_real_roots_sqf(f, eps, inf, sup, fast): ...
    def _isolate_all_roots(f, eps, inf, sup, fast): ...
    def _isolate_all_roots_sqf(f, eps, inf, sup, fast): ...
    def _refine_real_root(f, s, t, eps, steps, fast): ...
    def count_real_roots(f, inf=None, sup=None):
        """Return the number of real roots of ``f`` in ``[inf, sup]``. """
    def count_complex_roots(f, inf=None, sup=None):
        """Return the number of complex roots of ``f`` in ``[inf, sup]``. """
    @property
    def is_zero(f):
        """Returns ``True`` if ``f`` is a zero polynomial. """
    @property
    def is_one(f):
        """Returns ``True`` if ``f`` is a unit polynomial. """
    @property
    def is_ground(f):
        """Returns ``True`` if ``f`` is an element of the ground domain. """
    @property
    def is_linear(f):
        """Returns ``True`` if ``f`` is linear in all its variables. """
    @property
    def is_quadratic(f):
        """Returns ``True`` if ``f`` is quadratic in all its variables. """
    @property
    def is_monomial(f):
        """Returns ``True`` if ``f`` is zero or has only one term. """
    @property
    def is_monic(f):
        """Returns ``True`` if the leading coefficient of ``f`` is one. """
    @property
    def is_primitive(f):
        """Returns ``True`` if the GCD of the coefficients of ``f`` is one. """
    @property
    def is_homogeneous(f):
        """Returns ``True`` if ``f`` is a homogeneous polynomial. """
    @property
    def is_sqf(f):
        """Returns ``True`` if ``f`` is a square-free polynomial. """
    @property
    def is_irreducible(f):
        """Returns ``True`` if ``f`` has no factors over its domain. """
    @property
    def is_cyclotomic(f):
        """Returns ``True`` if ``f`` is a cyclotomic polynomial. """

def init_normal_DMF(num, den, lev, dom): ...

class DMF(PicklableWithSlots, CantSympify):
    """Dense Multivariate Fractions over `K`. """
    __slots__: Incomplete
    num: Incomplete
    den: Incomplete
    lev: Incomplete
    dom: Incomplete
    def __init__(self, rep, dom, lev=None) -> None: ...
    @classmethod
    def new(cls, rep, dom, lev=None): ...
    def ground_new(self, rep): ...
    @classmethod
    def _parse(cls, rep, dom, lev=None): ...
    def __repr__(f) -> str: ...
    def __hash__(f): ...
    def poly_unify(f, g):
        """Unify a multivariate fraction and a polynomial. """
    def frac_unify(f, g):
        """Unify representations of two multivariate fractions. """
    def per(f, num, den, cancel: bool = True, kill: bool = False):
        """Create a DMF out of the given representation. """
    def half_per(f, rep, kill: bool = False):
        """Create a DMP out of the given representation. """
    @classmethod
    def zero(cls, lev, dom): ...
    @classmethod
    def one(cls, lev, dom): ...
    def numer(f):
        """Returns the numerator of ``f``. """
    def denom(f):
        """Returns the denominator of ``f``. """
    def cancel(f):
        """Remove common factors from ``f.num`` and ``f.den``. """
    def neg(f):
        """Negate all coefficients in ``f``. """
    def add_ground(f, c):
        """Add an element of the ground domain to ``f``. """
    def add(f, g):
        """Add two multivariate fractions ``f`` and ``g``. """
    def sub(f, g):
        """Subtract two multivariate fractions ``f`` and ``g``. """
    def mul(f, g):
        """Multiply two multivariate fractions ``f`` and ``g``. """
    def pow(f, n):
        """Raise ``f`` to a non-negative power ``n``. """
    def quo(f, g):
        """Computes quotient of fractions ``f`` and ``g``. """
    exquo = quo
    def invert(f, check: bool = True):
        """Computes inverse of a fraction ``f``. """
    @property
    def is_zero(f):
        """Returns ``True`` if ``f`` is a zero fraction. """
    @property
    def is_one(f):
        """Returns ``True`` if ``f`` is a unit fraction. """
    def __neg__(f): ...
    def __add__(f, g): ...
    def __radd__(f, g): ...
    def __sub__(f, g): ...
    def __rsub__(f, g): ...
    def __mul__(f, g): ...
    def __rmul__(f, g): ...
    def __pow__(f, n): ...
    def __truediv__(f, g): ...
    def __rtruediv__(self, g): ...
    def __eq__(f, g): ...
    def __ne__(f, g): ...
    def __lt__(f, g): ...
    def __le__(f, g): ...
    def __gt__(f, g): ...
    def __ge__(f, g): ...
    def __bool__(f) -> bool: ...

def init_normal_ANP(rep, mod, dom): ...

class ANP(CantSympify):
    """Dense Algebraic Number Polynomials over a field. """
    __slots__: Incomplete
    def __new__(cls, rep, mod, dom): ...
    @classmethod
    def new(cls, rep, mod, dom): ...
    def __reduce__(self): ...
    @property
    def rep(self): ...
    @property
    def mod(self): ...
    def to_DMP(self): ...
    def mod_to_DMP(self): ...
    def per(f, rep): ...
    def __repr__(f) -> str: ...
    def __hash__(f): ...
    def convert(f, dom):
        """Convert ``f`` to a ``ANP`` over a new domain. """
    def unify(f, g):
        """Unify representations of two algebraic numbers. """
    def unify_ANP(f, g):
        """Unify and return ``DMP`` instances of ``f`` and ``g``. """
    @classmethod
    def zero(cls, mod, dom): ...
    @classmethod
    def one(cls, mod, dom): ...
    def to_dict(f):
        """Convert ``f`` to a dict representation with native coefficients. """
    def to_sympy_dict(f):
        """Convert ``f`` to a dict representation with SymPy coefficients. """
    def to_list(f):
        """Convert ``f`` to a list representation with native coefficients. """
    def mod_to_list(f):
        """Return ``f.mod`` as a list with native coefficients. """
    def to_sympy_list(f):
        """Convert ``f`` to a list representation with SymPy coefficients. """
    def to_tuple(f):
        """
        Convert ``f`` to a tuple representation with native coefficients.

        This is needed for hashing.
        """
    @classmethod
    def from_list(cls, rep, mod, dom): ...
    def add_ground(f, c):
        """Add an element of the ground domain to ``f``. """
    def sub_ground(f, c):
        """Subtract an element of the ground domain from ``f``. """
    def mul_ground(f, c):
        """Multiply ``f`` by an element of the ground domain. """
    def quo_ground(f, c):
        """Quotient of ``f`` by an element of the ground domain. """
    def neg(f): ...
    def add(f, g): ...
    def sub(f, g): ...
    def mul(f, g): ...
    def pow(f, n):
        """Raise ``f`` to a non-negative power ``n``. """
    def exquo(f, g): ...
    def div(f, g): ...
    def quo(f, g): ...
    def rem(f, g): ...
    def LC(f):
        """Returns the leading coefficient of ``f``. """
    def TC(f):
        """Returns the trailing coefficient of ``f``. """
    @property
    def is_zero(f):
        """Returns ``True`` if ``f`` is a zero algebraic number. """
    @property
    def is_one(f):
        """Returns ``True`` if ``f`` is a unit algebraic number. """
    @property
    def is_ground(f):
        """Returns ``True`` if ``f`` is an element of the ground domain. """
    def __pos__(f): ...
    def __neg__(f): ...
    def __add__(f, g): ...
    def __radd__(f, g): ...
    def __sub__(f, g): ...
    def __rsub__(f, g): ...
    def __mul__(f, g): ...
    def __rmul__(f, g): ...
    def __pow__(f, n): ...
    def __divmod__(f, g): ...
    def __mod__(f, g): ...
    def __truediv__(f, g): ...
    def __eq__(f, g): ...
    def __ne__(f, g): ...
    def __lt__(f, g): ...
    def __le__(f, g): ...
    def __gt__(f, g): ...
    def __ge__(f, g): ...
    def __bool__(f) -> bool: ...
