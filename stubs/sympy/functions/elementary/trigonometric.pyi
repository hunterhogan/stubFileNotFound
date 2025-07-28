from _typeshed import Incomplete
from sympy.core.add import Add as Add
from sympy.core.cache import cacheit as cacheit
from sympy.core.expr import Expr as Expr
from sympy.core.function import ArgumentIndexError as ArgumentIndexError, DefinedFunction as DefinedFunction, PoleError as PoleError, expand_mul as expand_mul
from sympy.core.logic import FuzzyBool as FuzzyBool, fuzzy_and as fuzzy_and, fuzzy_not as fuzzy_not, fuzzy_or as fuzzy_or
from sympy.core.mod import Mod as Mod
from sympy.core.numbers import Float as Float, Integer as Integer, Rational as Rational, equal_valued as equal_valued, pi as pi
from sympy.core.relational import Eq as Eq, Ne as Ne
from sympy.core.singleton import S as S
from sympy.core.symbol import Dummy as Dummy, Symbol as Symbol
from sympy.core.sympify import sympify as sympify
from sympy.functions.combinatorial.factorials import RisingFactorial as RisingFactorial, factorial as factorial
from sympy.functions.combinatorial.numbers import bernoulli as bernoulli, euler as euler
from sympy.functions.elementary._trigonometric_special import cos_table as cos_table, fermat_coords as fermat_coords, ipartfrac as ipartfrac
from sympy.functions.elementary.complexes import im as im, re as re
from sympy.functions.elementary.exponential import exp as exp, log as log
from sympy.functions.elementary.integers import floor as floor
from sympy.functions.elementary.miscellaneous import Max as Max, Min as Min, sqrt as sqrt
from sympy.functions.elementary.piecewise import Piecewise as Piecewise
from sympy.logic.boolalg import And as And
from sympy.ntheory import factorint as factorint
from sympy.polys.specialpolys import symmetric_poly as symmetric_poly
from sympy.utilities.iterables import numbered_symbols as numbered_symbols

def _imaginary_unit_as_coefficient(arg):
    """ Helper to extract symbolic coefficient for imaginary unit """

class TrigonometricFunction(DefinedFunction):
    """Base class for trigonometric functions. """
    unbranched: bool
    _singularities: Incomplete
    def _eval_is_rational(self): ...
    def _eval_is_algebraic(self): ...
    def _eval_expand_complex(self, deep: bool = True, **hints): ...
    def _as_real_imag(self, deep: bool = True, **hints): ...
    def _period(self, general_period, symbol=None): ...

@cacheit
def _table2(): ...
def _peeloff_pi(arg):
    '''
    Split ARG into two parts, a "rest" and a multiple of $\\pi$.
    This assumes ARG to be an Add.
    The multiple of $\\pi$ returned in the second position is always a Rational.

    Examples
    ========

    >>> from sympy.functions.elementary.trigonometric import _peeloff_pi
    >>> from sympy import pi
    >>> from sympy.abc import x, y
    >>> _peeloff_pi(x + pi/2)
    (x, 1/2)
    >>> _peeloff_pi(x + 2*pi/3 + pi*y)
    (x + pi*y + pi/6, 1/2)

    '''
def _pi_coeff(arg: Expr, cycles: int = 1) -> Expr | None:
    """
    When arg is a Number times $\\pi$ (e.g. $3\\pi/2$) then return the Number
    normalized to be in the range $[0, 2]$, else `None`.

    When an even multiple of $\\pi$ is encountered, if it is multiplying
    something with known parity then the multiple is returned as 0 otherwise
    as 2.

    Examples
    ========

    >>> from sympy.functions.elementary.trigonometric import _pi_coeff
    >>> from sympy import pi, Dummy
    >>> from sympy.abc import x
    >>> _pi_coeff(3*x*pi)
    3*x
    >>> _pi_coeff(11*pi/7)
    11/7
    >>> _pi_coeff(-11*pi/7)
    3/7
    >>> _pi_coeff(4*pi)
    0
    >>> _pi_coeff(5*pi)
    1
    >>> _pi_coeff(5.0*pi)
    1
    >>> _pi_coeff(5.5*pi)
    3/2
    >>> _pi_coeff(2 + pi)

    >>> _pi_coeff(2*Dummy(integer=True)*pi)
    2
    >>> _pi_coeff(2*Dummy(even=True)*pi)
    0

    """

class sin(TrigonometricFunction):
    """
    The sine function.

    Returns the sine of x (measured in radians).

    Explanation
    ===========

    This function will evaluate automatically in the
    case $x/\\pi$ is some rational number [4]_.  For example,
    if $x$ is a multiple of $\\pi$, $\\pi/2$, $\\pi/3$, $\\pi/4$, and $\\pi/6$.

    Examples
    ========

    >>> from sympy import sin, pi
    >>> from sympy.abc import x
    >>> sin(x**2).diff(x)
    2*x*cos(x**2)
    >>> sin(1).diff(x)
    0
    >>> sin(pi)
    0
    >>> sin(pi/2)
    1
    >>> sin(pi/6)
    1/2
    >>> sin(pi/12)
    -sqrt(2)/4 + sqrt(6)/4


    See Also
    ========

    csc, cos, sec, tan, cot
    asin, acsc, acos, asec, atan, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.14
    .. [3] https://functions.wolfram.com/ElementaryFunctions/Sin
    .. [4] https://mathworld.wolfram.com/TrigonometryAngles.html

    """
    def period(self, symbol=None): ...
    def fdiff(self, argindex: int = 1): ...
    @classmethod
    def eval(cls, arg): ...
    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms): ...
    def _eval_nseries(self, x, n, logx, cdir: int = 0): ...
    def _eval_rewrite_as_exp(self, arg, **kwargs): ...
    def _eval_rewrite_as_Pow(self, arg, **kwargs): ...
    def _eval_rewrite_as_cos(self, arg, **kwargs): ...
    def _eval_rewrite_as_tan(self, arg, **kwargs): ...
    def _eval_rewrite_as_sincos(self, arg, **kwargs): ...
    def _eval_rewrite_as_cot(self, arg, **kwargs): ...
    def _eval_rewrite_as_pow(self, arg, **kwargs): ...
    def _eval_rewrite_as_sqrt(self, arg, **kwargs): ...
    def _eval_rewrite_as_csc(self, arg, **kwargs): ...
    def _eval_rewrite_as_sec(self, arg, **kwargs): ...
    def _eval_rewrite_as_sinc(self, arg, **kwargs): ...
    def _eval_rewrite_as_besselj(self, arg, **kwargs): ...
    def _eval_conjugate(self): ...
    def as_real_imag(self, deep: bool = True, **hints): ...
    def _eval_expand_trig(self, **hints): ...
    def _eval_as_leading_term(self, x, logx, cdir): ...
    def _eval_is_extended_real(self): ...
    def _eval_is_finite(self): ...
    def _eval_is_zero(self): ...
    def _eval_is_complex(self): ...

class cos(TrigonometricFunction):
    """
    The cosine function.

    Returns the cosine of x (measured in radians).

    Explanation
    ===========

    See :func:`sin` for notes about automatic evaluation.

    Examples
    ========

    >>> from sympy import cos, pi
    >>> from sympy.abc import x
    >>> cos(x**2).diff(x)
    -2*x*sin(x**2)
    >>> cos(1).diff(x)
    0
    >>> cos(pi)
    -1
    >>> cos(pi/2)
    0
    >>> cos(2*pi/3)
    -1/2
    >>> cos(pi/12)
    sqrt(2)/4 + sqrt(6)/4

    See Also
    ========

    sin, csc, sec, tan, cot
    asin, acsc, acos, asec, atan, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.14
    .. [3] https://functions.wolfram.com/ElementaryFunctions/Cos

    """
    def period(self, symbol=None): ...
    def fdiff(self, argindex: int = 1): ...
    @classmethod
    def eval(cls, arg): ...
    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms): ...
    def _eval_nseries(self, x, n, logx, cdir: int = 0): ...
    def _eval_rewrite_as_exp(self, arg, **kwargs): ...
    def _eval_rewrite_as_Pow(self, arg, **kwargs): ...
    def _eval_rewrite_as_sin(self, arg, **kwargs): ...
    def _eval_rewrite_as_tan(self, arg, **kwargs): ...
    def _eval_rewrite_as_sincos(self, arg, **kwargs): ...
    def _eval_rewrite_as_cot(self, arg, **kwargs): ...
    def _eval_rewrite_as_pow(self, arg, **kwargs): ...
    def _eval_rewrite_as_sqrt(self, arg: Expr, **kwargs): ...
    def _eval_rewrite_as_sec(self, arg, **kwargs): ...
    def _eval_rewrite_as_csc(self, arg, **kwargs): ...
    def _eval_rewrite_as_besselj(self, arg, **kwargs): ...
    def _eval_conjugate(self): ...
    def as_real_imag(self, deep: bool = True, **hints): ...
    def _eval_expand_trig(self, **hints): ...
    def _eval_as_leading_term(self, x, logx, cdir): ...
    def _eval_is_extended_real(self): ...
    def _eval_is_finite(self): ...
    def _eval_is_complex(self): ...
    def _eval_is_zero(self): ...

class tan(TrigonometricFunction):
    """
    The tangent function.

    Returns the tangent of x (measured in radians).

    Explanation
    ===========

    See :class:`sin` for notes about automatic evaluation.

    Examples
    ========

    >>> from sympy import tan, pi
    >>> from sympy.abc import x
    >>> tan(x**2).diff(x)
    2*x*(tan(x**2)**2 + 1)
    >>> tan(1).diff(x)
    0
    >>> tan(pi/8).expand()
    -1 + sqrt(2)

    See Also
    ========

    sin, csc, cos, sec, cot
    asin, acsc, acos, asec, atan, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.14
    .. [3] https://functions.wolfram.com/ElementaryFunctions/Tan

    """
    def period(self, symbol=None): ...
    def fdiff(self, argindex: int = 1): ...
    def inverse(self, argindex: int = 1):
        """
        Returns the inverse of this function.
        """
    @classmethod
    def eval(cls, arg): ...
    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms): ...
    def _eval_nseries(self, x, n, logx, cdir: int = 0): ...
    def _eval_rewrite_as_Pow(self, arg, **kwargs): ...
    def _eval_conjugate(self): ...
    def as_real_imag(self, deep: bool = True, **hints): ...
    def _eval_expand_trig(self, **hints): ...
    def _eval_rewrite_as_exp(self, arg, **kwargs): ...
    def _eval_rewrite_as_sin(self, x, **kwargs): ...
    def _eval_rewrite_as_cos(self, x, **kwargs): ...
    def _eval_rewrite_as_sincos(self, arg, **kwargs): ...
    def _eval_rewrite_as_cot(self, arg, **kwargs): ...
    def _eval_rewrite_as_sec(self, arg, **kwargs): ...
    def _eval_rewrite_as_csc(self, arg, **kwargs): ...
    def _eval_rewrite_as_pow(self, arg, **kwargs): ...
    def _eval_rewrite_as_sqrt(self, arg, **kwargs): ...
    def _eval_rewrite_as_besselj(self, arg, **kwargs): ...
    def _eval_as_leading_term(self, x, logx, cdir): ...
    def _eval_is_extended_real(self): ...
    def _eval_is_real(self): ...
    def _eval_is_finite(self): ...
    def _eval_is_zero(self): ...
    def _eval_is_complex(self): ...

class cot(TrigonometricFunction):
    """
    The cotangent function.

    Returns the cotangent of x (measured in radians).

    Explanation
    ===========

    See :class:`sin` for notes about automatic evaluation.

    Examples
    ========

    >>> from sympy import cot, pi
    >>> from sympy.abc import x
    >>> cot(x**2).diff(x)
    2*x*(-cot(x**2)**2 - 1)
    >>> cot(1).diff(x)
    0
    >>> cot(pi/12)
    sqrt(3) + 2

    See Also
    ========

    sin, csc, cos, sec, tan
    asin, acsc, acos, asec, atan, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.14
    .. [3] https://functions.wolfram.com/ElementaryFunctions/Cot

    """
    def period(self, symbol=None): ...
    def fdiff(self, argindex: int = 1): ...
    def inverse(self, argindex: int = 1):
        """
        Returns the inverse of this function.
        """
    @classmethod
    def eval(cls, arg): ...
    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms): ...
    def _eval_nseries(self, x, n, logx, cdir: int = 0): ...
    def _eval_conjugate(self): ...
    def as_real_imag(self, deep: bool = True, **hints): ...
    def _eval_rewrite_as_exp(self, arg, **kwargs): ...
    def _eval_rewrite_as_Pow(self, arg, **kwargs): ...
    def _eval_rewrite_as_sin(self, x, **kwargs): ...
    def _eval_rewrite_as_cos(self, x, **kwargs): ...
    def _eval_rewrite_as_sincos(self, arg, **kwargs): ...
    def _eval_rewrite_as_tan(self, arg, **kwargs): ...
    def _eval_rewrite_as_sec(self, arg, **kwargs): ...
    def _eval_rewrite_as_csc(self, arg, **kwargs): ...
    def _eval_rewrite_as_pow(self, arg, **kwargs): ...
    def _eval_rewrite_as_sqrt(self, arg, **kwargs): ...
    def _eval_rewrite_as_besselj(self, arg, **kwargs): ...
    def _eval_as_leading_term(self, x, logx, cdir): ...
    def _eval_is_extended_real(self): ...
    def _eval_expand_trig(self, **hints): ...
    def _eval_is_finite(self): ...
    def _eval_is_real(self): ...
    def _eval_is_complex(self): ...
    def _eval_is_zero(self): ...
    def _eval_subs(self, old, new): ...

class ReciprocalTrigonometricFunction(TrigonometricFunction):
    """Base class for reciprocal functions of trigonometric functions. """
    _reciprocal_of: Incomplete
    _singularities: Incomplete
    _is_even: FuzzyBool
    _is_odd: FuzzyBool
    @classmethod
    def eval(cls, arg): ...
    def _call_reciprocal(self, method_name, *args, **kwargs): ...
    def _calculate_reciprocal(self, method_name, *args, **kwargs): ...
    def _rewrite_reciprocal(self, method_name, arg): ...
    def _period(self, symbol): ...
    def fdiff(self, argindex: int = 1): ...
    def _eval_rewrite_as_exp(self, arg, **kwargs): ...
    def _eval_rewrite_as_Pow(self, arg, **kwargs): ...
    def _eval_rewrite_as_sin(self, arg, **kwargs): ...
    def _eval_rewrite_as_cos(self, arg, **kwargs): ...
    def _eval_rewrite_as_tan(self, arg, **kwargs): ...
    def _eval_rewrite_as_pow(self, arg, **kwargs): ...
    def _eval_rewrite_as_sqrt(self, arg, **kwargs): ...
    def _eval_conjugate(self): ...
    def as_real_imag(self, deep: bool = True, **hints): ...
    def _eval_expand_trig(self, **hints): ...
    def _eval_is_extended_real(self): ...
    def _eval_as_leading_term(self, x, logx, cdir): ...
    def _eval_is_finite(self): ...
    def _eval_nseries(self, x, n, logx, cdir: int = 0): ...

class sec(ReciprocalTrigonometricFunction):
    """
    The secant function.

    Returns the secant of x (measured in radians).

    Explanation
    ===========

    See :class:`sin` for notes about automatic evaluation.

    Examples
    ========

    >>> from sympy import sec
    >>> from sympy.abc import x
    >>> sec(x**2).diff(x)
    2*x*tan(x**2)*sec(x**2)
    >>> sec(1).diff(x)
    0

    See Also
    ========

    sin, csc, cos, tan, cot
    asin, acsc, acos, asec, atan, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.14
    .. [3] https://functions.wolfram.com/ElementaryFunctions/Sec

    """
    _reciprocal_of = cos
    _is_even: bool
    def period(self, symbol=None): ...
    def _eval_rewrite_as_cot(self, arg, **kwargs): ...
    def _eval_rewrite_as_cos(self, arg, **kwargs): ...
    def _eval_rewrite_as_sincos(self, arg, **kwargs): ...
    def _eval_rewrite_as_sin(self, arg, **kwargs): ...
    def _eval_rewrite_as_tan(self, arg, **kwargs): ...
    def _eval_rewrite_as_csc(self, arg, **kwargs): ...
    def fdiff(self, argindex: int = 1): ...
    def _eval_rewrite_as_besselj(self, arg, **kwargs): ...
    def _eval_is_complex(self): ...
    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms): ...
    def _eval_as_leading_term(self, x, logx, cdir): ...

class csc(ReciprocalTrigonometricFunction):
    """
    The cosecant function.

    Returns the cosecant of x (measured in radians).

    Explanation
    ===========

    See :func:`sin` for notes about automatic evaluation.

    Examples
    ========

    >>> from sympy import csc
    >>> from sympy.abc import x
    >>> csc(x**2).diff(x)
    -2*x*cot(x**2)*csc(x**2)
    >>> csc(1).diff(x)
    0

    See Also
    ========

    sin, cos, sec, tan, cot
    asin, acsc, acos, asec, atan, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.14
    .. [3] https://functions.wolfram.com/ElementaryFunctions/Csc

    """
    _reciprocal_of = sin
    _is_odd: bool
    def period(self, symbol=None): ...
    def _eval_rewrite_as_sin(self, arg, **kwargs): ...
    def _eval_rewrite_as_sincos(self, arg, **kwargs): ...
    def _eval_rewrite_as_cot(self, arg, **kwargs): ...
    def _eval_rewrite_as_cos(self, arg, **kwargs): ...
    def _eval_rewrite_as_sec(self, arg, **kwargs): ...
    def _eval_rewrite_as_tan(self, arg, **kwargs): ...
    def _eval_rewrite_as_besselj(self, arg, **kwargs): ...
    def fdiff(self, argindex: int = 1): ...
    def _eval_is_complex(self): ...
    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms): ...
    def _eval_as_leading_term(self, x, logx, cdir): ...

class sinc(DefinedFunction):
    """
    Represents an unnormalized sinc function:

    .. math::

        \\operatorname{sinc}(x) =
        \\begin{cases}
          \\frac{\\sin x}{x} & \\qquad x \\neq 0 \\\\\n          1 & \\qquad x = 0
        \\end{cases}

    Examples
    ========

    >>> from sympy import sinc, oo, jn
    >>> from sympy.abc import x
    >>> sinc(x)
    sinc(x)

    * Automated Evaluation

    >>> sinc(0)
    1
    >>> sinc(oo)
    0

    * Differentiation

    >>> sinc(x).diff()
    cos(x)/x - sin(x)/x**2

    * Series Expansion

    >>> sinc(x).series()
    1 - x**2/6 + x**4/120 + O(x**6)

    * As zero'th order spherical Bessel Function

    >>> sinc(x).rewrite(jn)
    jn(0, x)

    See also
    ========

    sin

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Sinc_function

    """
    _singularities: Incomplete
    def fdiff(self, argindex: int = 1): ...
    @classmethod
    def eval(cls, arg): ...
    def _eval_nseries(self, x, n, logx, cdir: int = 0): ...
    def _eval_rewrite_as_jn(self, arg, **kwargs): ...
    def _eval_rewrite_as_sin(self, arg, **kwargs): ...
    def _eval_is_zero(self): ...
    def _eval_is_real(self): ...
    _eval_is_finite = _eval_is_real

class InverseTrigonometricFunction(DefinedFunction):
    """Base class for inverse trigonometric functions."""
    _singularities: tuple[Expr, ...]
    @staticmethod
    @cacheit
    def _asin_table(): ...
    @staticmethod
    @cacheit
    def _atan_table(): ...
    @staticmethod
    @cacheit
    def _acsc_table(): ...

class asin(InverseTrigonometricFunction):
    """
    The inverse sine function.

    Returns the arcsine of x in radians.

    Explanation
    ===========

    ``asin(x)`` will evaluate automatically in the cases
    $x \\in \\{\\infty, -\\infty, 0, 1, -1\\}$ and for some instances when the
    result is a rational multiple of $\\pi$ (see the ``eval`` class method).

    A purely imaginary argument will lead to an asinh expression.

    Examples
    ========

    >>> from sympy import asin, oo
    >>> asin(1)
    pi/2
    >>> asin(-1)
    -pi/2
    >>> asin(-oo)
    oo*I
    >>> asin(oo)
    -oo*I

    See Also
    ========

    sin, csc, cos, sec, tan, cot
    acsc, acos, asec, atan, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Inverse_trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.23
    .. [3] https://functions.wolfram.com/ElementaryFunctions/ArcSin

    """
    def fdiff(self, argindex: int = 1): ...
    def _eval_is_rational(self): ...
    def _eval_is_positive(self): ...
    def _eval_is_negative(self): ...
    @classmethod
    def eval(cls, arg): ...
    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms): ...
    def _eval_as_leading_term(self, x, logx, cdir): ...
    def _eval_nseries(self, x, n, logx, cdir: int = 0): ...
    def _eval_rewrite_as_acos(self, x, **kwargs): ...
    def _eval_rewrite_as_atan(self, x, **kwargs): ...
    def _eval_rewrite_as_log(self, x, **kwargs): ...
    _eval_rewrite_as_tractable = _eval_rewrite_as_log
    def _eval_rewrite_as_acot(self, arg, **kwargs): ...
    def _eval_rewrite_as_asec(self, arg, **kwargs): ...
    def _eval_rewrite_as_acsc(self, arg, **kwargs): ...
    def _eval_is_extended_real(self): ...
    def inverse(self, argindex: int = 1):
        """
        Returns the inverse of this function.
        """

class acos(InverseTrigonometricFunction):
    """
    The inverse cosine function.

    Explanation
    ===========

    Returns the arc cosine of x (measured in radians).

    ``acos(x)`` will evaluate automatically in the cases
    $x \\in \\{\\infty, -\\infty, 0, 1, -1\\}$ and for some instances when
    the result is a rational multiple of $\\pi$ (see the eval class method).

    ``acos(zoo)`` evaluates to ``zoo``
    (see note in :class:`sympy.functions.elementary.trigonometric.asec`)

    A purely imaginary argument will be rewritten to asinh.

    Examples
    ========

    >>> from sympy import acos, oo
    >>> acos(1)
    0
    >>> acos(0)
    pi/2
    >>> acos(oo)
    oo*I

    See Also
    ========

    sin, csc, cos, sec, tan, cot
    asin, acsc, asec, atan, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Inverse_trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.23
    .. [3] https://functions.wolfram.com/ElementaryFunctions/ArcCos

    """
    def fdiff(self, argindex: int = 1): ...
    def _eval_is_rational(self): ...
    @classmethod
    def eval(cls, arg): ...
    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms): ...
    def _eval_as_leading_term(self, x, logx, cdir): ...
    def _eval_is_extended_real(self): ...
    def _eval_is_nonnegative(self): ...
    def _eval_nseries(self, x, n, logx, cdir: int = 0): ...
    def _eval_rewrite_as_log(self, x, **kwargs): ...
    _eval_rewrite_as_tractable = _eval_rewrite_as_log
    def _eval_rewrite_as_asin(self, x, **kwargs): ...
    def _eval_rewrite_as_atan(self, x, **kwargs): ...
    def inverse(self, argindex: int = 1):
        """
        Returns the inverse of this function.
        """
    def _eval_rewrite_as_acot(self, arg, **kwargs): ...
    def _eval_rewrite_as_asec(self, arg, **kwargs): ...
    def _eval_rewrite_as_acsc(self, arg, **kwargs): ...
    def _eval_conjugate(self): ...

class atan(InverseTrigonometricFunction):
    """
    The inverse tangent function.

    Returns the arc tangent of x (measured in radians).

    Explanation
    ===========

    ``atan(x)`` will evaluate automatically in the cases
    $x \\in \\{\\infty, -\\infty, 0, 1, -1\\}$ and for some instances when the
    result is a rational multiple of $\\pi$ (see the eval class method).

    Examples
    ========

    >>> from sympy import atan, oo
    >>> atan(0)
    0
    >>> atan(1)
    pi/4
    >>> atan(oo)
    pi/2

    See Also
    ========

    sin, csc, cos, sec, tan, cot
    asin, acsc, acos, asec, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Inverse_trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.23
    .. [3] https://functions.wolfram.com/ElementaryFunctions/ArcTan

    """
    args: tuple[Expr]
    _singularities: Incomplete
    def fdiff(self, argindex: int = 1): ...
    def _eval_is_rational(self): ...
    def _eval_is_positive(self): ...
    def _eval_is_nonnegative(self): ...
    def _eval_is_zero(self): ...
    def _eval_is_real(self): ...
    @classmethod
    def eval(cls, arg): ...
    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms): ...
    def _eval_as_leading_term(self, x, logx, cdir): ...
    def _eval_nseries(self, x, n, logx, cdir: int = 0): ...
    def _eval_rewrite_as_log(self, x, **kwargs): ...
    _eval_rewrite_as_tractable = _eval_rewrite_as_log
    def _eval_aseries(self, n, args0, x, logx): ...
    def inverse(self, argindex: int = 1):
        """
        Returns the inverse of this function.
        """
    def _eval_rewrite_as_asin(self, arg, **kwargs): ...
    def _eval_rewrite_as_acos(self, arg, **kwargs): ...
    def _eval_rewrite_as_acot(self, arg, **kwargs): ...
    def _eval_rewrite_as_asec(self, arg, **kwargs): ...
    def _eval_rewrite_as_acsc(self, arg, **kwargs): ...

class acot(InverseTrigonometricFunction):
    """
    The inverse cotangent function.

    Returns the arc cotangent of x (measured in radians).

    Explanation
    ===========

    ``acot(x)`` will evaluate automatically in the cases
    $x \\in \\{\\infty, -\\infty, \\tilde{\\infty}, 0, 1, -1\\}$
    and for some instances when the result is a rational multiple of $\\pi$
    (see the eval class method).

    A purely imaginary argument will lead to an ``acoth`` expression.

    ``acot(x)`` has a branch cut along $(-i, i)$, hence it is discontinuous
    at 0. Its range for real $x$ is $(-\\frac{\\pi}{2}, \\frac{\\pi}{2}]$.

    Examples
    ========

    >>> from sympy import acot, sqrt
    >>> acot(0)
    pi/2
    >>> acot(1)
    pi/4
    >>> acot(sqrt(3) - 2)
    -5*pi/12

    See Also
    ========

    sin, csc, cos, sec, tan, cot
    asin, acsc, acos, asec, atan, atan2

    References
    ==========

    .. [1] https://dlmf.nist.gov/4.23
    .. [2] https://functions.wolfram.com/ElementaryFunctions/ArcCot

    """
    _singularities: Incomplete
    def fdiff(self, argindex: int = 1): ...
    def _eval_is_rational(self): ...
    def _eval_is_positive(self): ...
    def _eval_is_negative(self): ...
    def _eval_is_extended_real(self): ...
    @classmethod
    def eval(cls, arg): ...
    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms): ...
    def _eval_as_leading_term(self, x, logx, cdir): ...
    def _eval_nseries(self, x, n, logx, cdir: int = 0): ...
    def _eval_aseries(self, n, args0, x, logx): ...
    def _eval_rewrite_as_log(self, x, **kwargs): ...
    _eval_rewrite_as_tractable = _eval_rewrite_as_log
    def inverse(self, argindex: int = 1):
        """
        Returns the inverse of this function.
        """
    def _eval_rewrite_as_asin(self, arg, **kwargs): ...
    def _eval_rewrite_as_acos(self, arg, **kwargs): ...
    def _eval_rewrite_as_atan(self, arg, **kwargs): ...
    def _eval_rewrite_as_asec(self, arg, **kwargs): ...
    def _eval_rewrite_as_acsc(self, arg, **kwargs): ...

class asec(InverseTrigonometricFunction):
    """
    The inverse secant function.

    Returns the arc secant of x (measured in radians).

    Explanation
    ===========

    ``asec(x)`` will evaluate automatically in the cases
    $x \\in \\{\\infty, -\\infty, 0, 1, -1\\}$ and for some instances when the
    result is a rational multiple of $\\pi$ (see the eval class method).

    ``asec(x)`` has branch cut in the interval $[-1, 1]$. For complex arguments,
    it can be defined [4]_ as

    .. math::
        \\operatorname{sec^{-1}}(z) = -i\\frac{\\log\\left(\\sqrt{1 - z^2} + 1\\right)}{z}

    At ``x = 0``, for positive branch cut, the limit evaluates to ``zoo``. For
    negative branch cut, the limit

    .. math::
        \\lim_{z \\to 0}-i\\frac{\\log\\left(-\\sqrt{1 - z^2} + 1\\right)}{z}

    simplifies to :math:`-i\\log\\left(z/2 + O\\left(z^3\\right)\\right)` which
    ultimately evaluates to ``zoo``.

    As ``acos(x) = asec(1/x)``, a similar argument can be given for
    ``acos(x)``.

    Examples
    ========

    >>> from sympy import asec, oo
    >>> asec(1)
    0
    >>> asec(-1)
    pi
    >>> asec(0)
    zoo
    >>> asec(-oo)
    pi/2

    See Also
    ========

    sin, csc, cos, sec, tan, cot
    asin, acsc, acos, atan, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Inverse_trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.23
    .. [3] https://functions.wolfram.com/ElementaryFunctions/ArcSec
    .. [4] https://reference.wolfram.com/language/ref/ArcSec.html

    """
    @classmethod
    def eval(cls, arg): ...
    def fdiff(self, argindex: int = 1): ...
    def inverse(self, argindex: int = 1):
        """
        Returns the inverse of this function.
        """
    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms): ...
    def _eval_as_leading_term(self, x, logx, cdir): ...
    def _eval_nseries(self, x, n, logx, cdir: int = 0): ...
    def _eval_is_extended_real(self): ...
    def _eval_rewrite_as_log(self, arg, **kwargs): ...
    _eval_rewrite_as_tractable = _eval_rewrite_as_log
    def _eval_rewrite_as_asin(self, arg, **kwargs): ...
    def _eval_rewrite_as_acos(self, arg, **kwargs): ...
    def _eval_rewrite_as_atan(self, x, **kwargs): ...
    def _eval_rewrite_as_acot(self, x, **kwargs): ...
    def _eval_rewrite_as_acsc(self, arg, **kwargs): ...

class acsc(InverseTrigonometricFunction):
    """
    The inverse cosecant function.

    Returns the arc cosecant of x (measured in radians).

    Explanation
    ===========

    ``acsc(x)`` will evaluate automatically in the cases
    $x \\in \\{\\infty, -\\infty, 0, 1, -1\\}$` and for some instances when the
    result is a rational multiple of $\\pi$ (see the ``eval`` class method).

    Examples
    ========

    >>> from sympy import acsc, oo
    >>> acsc(1)
    pi/2
    >>> acsc(-1)
    -pi/2
    >>> acsc(oo)
    0
    >>> acsc(-oo) == acsc(oo)
    True
    >>> acsc(0)
    zoo

    See Also
    ========

    sin, csc, cos, sec, tan, cot
    asin, acos, asec, atan, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Inverse_trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.23
    .. [3] https://functions.wolfram.com/ElementaryFunctions/ArcCsc

    """
    @classmethod
    def eval(cls, arg): ...
    def fdiff(self, argindex: int = 1): ...
    def inverse(self, argindex: int = 1):
        """
        Returns the inverse of this function.
        """
    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms): ...
    def _eval_as_leading_term(self, x, logx, cdir): ...
    def _eval_nseries(self, x, n, logx, cdir: int = 0): ...
    def _eval_rewrite_as_log(self, arg, **kwargs): ...
    _eval_rewrite_as_tractable = _eval_rewrite_as_log
    def _eval_rewrite_as_asin(self, arg, **kwargs): ...
    def _eval_rewrite_as_acos(self, arg, **kwargs): ...
    def _eval_rewrite_as_atan(self, x, **kwargs): ...
    def _eval_rewrite_as_acot(self, arg, **kwargs): ...
    def _eval_rewrite_as_asec(self, arg, **kwargs): ...

class atan2(InverseTrigonometricFunction):
    """
    The function ``atan2(y, x)`` computes `\\operatorname{atan}(y/x)` taking
    two arguments `y` and `x`.  Signs of both `y` and `x` are considered to
    determine the appropriate quadrant of `\\operatorname{atan}(y/x)`.
    The range is `(-\\pi, \\pi]`. The complete definition reads as follows:

    .. math::

        \\operatorname{atan2}(y, x) =
        \\begin{cases}
          \\arctan\\left(\\frac y x\\right) & \\qquad x > 0 \\\\\n          \\arctan\\left(\\frac y x\\right) + \\pi& \\qquad y \\ge 0, x < 0 \\\\\n          \\arctan\\left(\\frac y x\\right) - \\pi& \\qquad y < 0, x < 0 \\\\\n          +\\frac{\\pi}{2} & \\qquad y > 0, x = 0 \\\\\n          -\\frac{\\pi}{2} & \\qquad y < 0, x = 0 \\\\\n          \\text{undefined} & \\qquad y = 0, x = 0
        \\end{cases}

    Attention: Note the role reversal of both arguments. The `y`-coordinate
    is the first argument and the `x`-coordinate the second.

    If either `x` or `y` is complex:

    .. math::

        \\operatorname{atan2}(y, x) =
            -i\\log\\left(\\frac{x + iy}{\\sqrt{x^2 + y^2}}\\right)

    Examples
    ========

    Going counter-clock wise around the origin we find the
    following angles:

    >>> from sympy import atan2
    >>> atan2(0, 1)
    0
    >>> atan2(1, 1)
    pi/4
    >>> atan2(1, 0)
    pi/2
    >>> atan2(1, -1)
    3*pi/4
    >>> atan2(0, -1)
    pi
    >>> atan2(-1, -1)
    -3*pi/4
    >>> atan2(-1, 0)
    -pi/2
    >>> atan2(-1, 1)
    -pi/4

    which are all correct. Compare this to the results of the ordinary
    `\\operatorname{atan}` function for the point `(x, y) = (-1, 1)`

    >>> from sympy import atan, S
    >>> atan(S(1)/-1)
    -pi/4
    >>> atan2(1, -1)
    3*pi/4

    where only the `\\operatorname{atan2}` function returns what we expect.
    We can differentiate the function with respect to both arguments:

    >>> from sympy import diff
    >>> from sympy.abc import x, y
    >>> diff(atan2(y, x), x)
    -y/(x**2 + y**2)

    >>> diff(atan2(y, x), y)
    x/(x**2 + y**2)

    We can express the `\\operatorname{atan2}` function in terms of
    complex logarithms:

    >>> from sympy import log
    >>> atan2(y, x).rewrite(log)
    -I*log((x + I*y)/sqrt(x**2 + y**2))

    and in terms of `\\operatorname(atan)`:

    >>> from sympy import atan
    >>> atan2(y, x).rewrite(atan)
    Piecewise((2*atan(y/(x + sqrt(x**2 + y**2))), Ne(y, 0)), (pi, re(x) < 0), (0, Ne(x, 0)), (nan, True))

    but note that this form is undefined on the negative real axis.

    See Also
    ========

    sin, csc, cos, sec, tan, cot
    asin, acsc, acos, asec, atan, acot

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Inverse_trigonometric_functions
    .. [2] https://en.wikipedia.org/wiki/Atan2
    .. [3] https://functions.wolfram.com/ElementaryFunctions/ArcTan2

    """
    @classmethod
    def eval(cls, y, x): ...
    def _eval_rewrite_as_log(self, y, x, **kwargs): ...
    def _eval_rewrite_as_atan(self, y, x, **kwargs): ...
    def _eval_rewrite_as_arg(self, y, x, **kwargs): ...
    def _eval_is_extended_real(self): ...
    def _eval_conjugate(self): ...
    def fdiff(self, argindex): ...
    def _eval_evalf(self, prec): ...
