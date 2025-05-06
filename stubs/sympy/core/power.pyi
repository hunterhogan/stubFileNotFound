from .expr import Expr as Expr
from .function import PoleError as PoleError, _mexpand as _mexpand, expand_complex as expand_complex, expand_mul as expand_mul, expand_multinomial as expand_multinomial
from .kind import NumberKind as NumberKind, UndefinedKind as UndefinedKind
from .logic import fuzzy_and as fuzzy_and, fuzzy_bool as fuzzy_bool, fuzzy_not as fuzzy_not, fuzzy_or as fuzzy_or
from .mul import Mul as Mul, _keep_coeff as _keep_coeff
from .numbers import Integer as Integer, Rational as Rational
from .relational import is_gt as is_gt, is_lt as is_lt
from .symbol import Dummy as Dummy, Symbol as Symbol, symbols as symbols
from _typeshed import Incomplete

class Pow(Expr):
    '''
    Defines the expression x**y as "x raised to a power y"

    .. deprecated:: 1.7

       Using arguments that aren\'t subclasses of :class:`~.Expr` in core
       operators (:class:`~.Mul`, :class:`~.Add`, and :class:`~.Pow`) is
       deprecated. See :ref:`non-expr-args-deprecated` for details.

    Singleton definitions involving (0, 1, -1, oo, -oo, I, -I):

    +--------------+---------+-----------------------------------------------+
    | expr         | value   | reason                                        |
    +==============+=========+===============================================+
    | z**0         | 1       | Although arguments over 0**0 exist, see [2].  |
    +--------------+---------+-----------------------------------------------+
    | z**1         | z       |                                               |
    +--------------+---------+-----------------------------------------------+
    | (-oo)**(-1)  | 0       |                                               |
    +--------------+---------+-----------------------------------------------+
    | (-1)**-1     | -1      |                                               |
    +--------------+---------+-----------------------------------------------+
    | S.Zero**-1   | zoo     | This is not strictly true, as 0**-1 may be    |
    |              |         | undefined, but is convenient in some contexts |
    |              |         | where the base is assumed to be positive.     |
    +--------------+---------+-----------------------------------------------+
    | 1**-1        | 1       |                                               |
    +--------------+---------+-----------------------------------------------+
    | oo**-1       | 0       |                                               |
    +--------------+---------+-----------------------------------------------+
    | 0**oo        | 0       | Because for all complex numbers z near        |
    |              |         | 0, z**oo -> 0.                                |
    +--------------+---------+-----------------------------------------------+
    | 0**-oo       | zoo     | This is not strictly true, as 0**oo may be    |
    |              |         | oscillating between positive and negative     |
    |              |         | values or rotating in the complex plane.      |
    |              |         | It is convenient, however, when the base      |
    |              |         | is positive.                                  |
    +--------------+---------+-----------------------------------------------+
    | 1**oo        | nan     | Because there are various cases where         |
    | 1**-oo       |         | lim(x(t),t)=1, lim(y(t),t)=oo (or -oo),       |
    |              |         | but lim( x(t)**y(t), t) != 1.  See [3].       |
    +--------------+---------+-----------------------------------------------+
    | b**zoo       | nan     | Because b**z has no limit as z -> zoo         |
    +--------------+---------+-----------------------------------------------+
    | (-1)**oo     | nan     | Because of oscillations in the limit.         |
    | (-1)**(-oo)  |         |                                               |
    +--------------+---------+-----------------------------------------------+
    | oo**oo       | oo      |                                               |
    +--------------+---------+-----------------------------------------------+
    | oo**-oo      | 0       |                                               |
    +--------------+---------+-----------------------------------------------+
    | (-oo)**oo    | nan     |                                               |
    | (-oo)**-oo   |         |                                               |
    +--------------+---------+-----------------------------------------------+
    | oo**I        | nan     | oo**e could probably be best thought of as    |
    | (-oo)**I     |         | the limit of x**e for real x as x tends to    |
    |              |         | oo. If e is I, then the limit does not exist  |
    |              |         | and nan is used to indicate that.             |
    +--------------+---------+-----------------------------------------------+
    | oo**(1+I)    | zoo     | If the real part of e is positive, then the   |
    | (-oo)**(1+I) |         | limit of abs(x**e) is oo. So the limit value  |
    |              |         | is zoo.                                       |
    +--------------+---------+-----------------------------------------------+
    | oo**(-1+I)   | 0       | If the real part of e is negative, then the   |
    | -oo**(-1+I)  |         | limit is 0.                                   |
    +--------------+---------+-----------------------------------------------+

    Because symbolic computations are more flexible than floating point
    calculations and we prefer to never return an incorrect answer,
    we choose not to conform to all IEEE 754 conventions.  This helps
    us avoid extra test-case code in the calculation of limits.

    See Also
    ========

    sympy.core.numbers.Infinity
    sympy.core.numbers.NegativeInfinity
    sympy.core.numbers.NaN

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Exponentiation
    .. [2] https://en.wikipedia.org/wiki/Zero_to_the_power_of_zero
    .. [3] https://en.wikipedia.org/wiki/Indeterminate_forms

    '''
    is_Pow: bool
    __slots__: Incomplete
    args: tuple[Expr, Expr]
    _args: tuple[Expr, Expr]
    def __new__(cls, b, e, evaluate: Incomplete | None = None): ...
    def inverse(self, argindex: int = 1): ...
    @property
    def base(self) -> Expr: ...
    @property
    def exp(self) -> Expr: ...
    @property
    def kind(self): ...
    @classmethod
    def class_key(cls): ...
    def _eval_refine(self, assumptions): ...
    def _eval_power(self, other): ...
    def _eval_Mod(self, q):
        """A dispatched function to compute `b^e \\bmod q`, dispatched
        by ``Mod``.

        Notes
        =====

        Algorithms:

        1. For unevaluated integer power, use built-in ``pow`` function
        with 3 arguments, if powers are not too large wrt base.

        2. For very large powers, use totient reduction if $e \\ge \\log(m)$.
        Bound on m, is for safe factorization memory wise i.e. $m^{1/4}$.
        For pollard-rho to be faster than built-in pow $\\log(e) > m^{1/4}$
        check is added.

        3. For any unevaluated power found in `b` or `e`, the step 2
        will be recursed down to the base and the exponent
        such that the $b \\bmod q$ becomes the new base and
        $\\phi(q) + e \\bmod \\phi(q)$ becomes the new exponent, and then
        the computation for the reduced expression can be done.
        """
    def _eval_is_even(self): ...
    def _eval_is_negative(self): ...
    def _eval_is_extended_positive(self): ...
    def _eval_is_extended_negative(self): ...
    def _eval_is_zero(self): ...
    def _eval_is_integer(self): ...
    def _eval_is_extended_real(self): ...
    def _eval_is_complex(self): ...
    def _eval_is_imaginary(self): ...
    def _eval_is_odd(self): ...
    def _eval_is_finite(self): ...
    def _eval_is_prime(self):
        """
        An integer raised to the n(>=2)-th power cannot be a prime.
        """
    def _eval_is_composite(self):
        """
        A power is composite if both base and exponent are greater than 1
        """
    def _eval_is_polar(self): ...
    def _eval_subs(self, old, new): ...
    def as_base_exp(self):
        """Return base and exp of self.

        Explanation
        ===========

        If base a Rational less than 1, then return 1/Rational, -exp.
        If this extra processing is not needed, the base and exp
        properties will give the raw arguments.

        Examples
        ========

        >>> from sympy import Pow, S
        >>> p = Pow(S.Half, 2, evaluate=False)
        >>> p.as_base_exp()
        (2, -2)
        >>> p.args
        (1/2, 2)
        >>> p.base, p.exp
        (1/2, 2)

        """
    def _eval_adjoint(self): ...
    def _eval_conjugate(self): ...
    def _eval_transpose(self): ...
    def _eval_expand_power_exp(self, **hints):
        """a**(n + m) -> a**n*a**m"""
    def _eval_expand_power_base(self, **hints):
        """(a*b)**n -> a**n * b**n"""
    def _eval_expand_multinomial(self, **hints):
        """(a + b + ..)**n -> a**n + n*a**(n-1)*b + .., n is nonzero integer"""
    def as_real_imag(self, deep: bool = True, **hints): ...
    def _eval_derivative(self, s): ...
    def _eval_evalf(self, prec): ...
    def _eval_is_polynomial(self, syms): ...
    def _eval_is_rational(self): ...
    def _eval_is_algebraic(self): ...
    def _eval_is_rational_function(self, syms): ...
    def _eval_is_meromorphic(self, x, a): ...
    def _eval_is_algebraic_expr(self, syms): ...
    def _eval_rewrite_as_exp(self, base, expo, **kwargs): ...
    def as_numer_denom(self): ...
    def matches(self, expr, repl_dict: Incomplete | None = None, old: bool = False): ...
    def _eval_nseries(self, x, n, logx, cdir: int = 0): ...
    def _eval_as_leading_term(self, x, logx: Incomplete | None = None, cdir: int = 0): ...
    def _taylor_term(self, n, x, *previous_terms): ...
    def taylor_term(self, n, x, *previous_terms): ...
    def _eval_rewrite_as_sin(self, base, exp, **hints): ...
    def _eval_rewrite_as_cos(self, base, exp, **hints): ...
    def _eval_rewrite_as_tanh(self, base, exp, **hints): ...
    def _eval_rewrite_as_sqrt(self, base, exp, **kwargs): ...
    def as_content_primitive(self, radical: bool = False, clear: bool = True):
        """Return the tuple (R, self/R) where R is the positive Rational
        extracted from self.

        Examples
        ========

        >>> from sympy import sqrt
        >>> sqrt(4 + 4*sqrt(2)).as_content_primitive()
        (2, sqrt(1 + sqrt(2)))
        >>> sqrt(3 + 3*sqrt(2)).as_content_primitive()
        (1, sqrt(3)*sqrt(1 + sqrt(2)))

        >>> from sympy import expand_power_base, powsimp, Mul
        >>> from sympy.abc import x, y

        >>> ((2*x + 2)**2).as_content_primitive()
        (4, (x + 1)**2)
        >>> (4**((1 + y)/2)).as_content_primitive()
        (2, 4**(y/2))
        >>> (3**((1 + y)/2)).as_content_primitive()
        (1, 3**((y + 1)/2))
        >>> (3**((5 + y)/2)).as_content_primitive()
        (9, 3**((y + 1)/2))
        >>> eq = 3**(2 + 2*x)
        >>> powsimp(eq) == eq
        True
        >>> eq.as_content_primitive()
        (9, 3**(2*x))
        >>> powsimp(Mul(*_))
        3**(2*x + 2)

        >>> eq = (2 + 2*x)**y
        >>> s = expand_power_base(eq); s.is_Mul, s
        (False, (2*x + 2)**y)
        >>> eq.as_content_primitive()
        (1, (2*(x + 1))**y)
        >>> s = expand_power_base(_[1]); s.is_Mul, s
        (True, 2**y*(x + 1)**y)

        See docstring of Expr.as_content_primitive for more examples.
        """
    def is_constant(self, *wrt, **flags): ...
    def _eval_difference_delta(self, n, step): ...

power: Incomplete
