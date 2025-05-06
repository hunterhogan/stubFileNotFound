from _typeshed import Incomplete
from collections.abc import Generator
from sympy.core.function import Derivative as Derivative, Function as Function, expand as expand
from sympy.core.numbers import Rational as Rational, nan as nan, oo as oo, zoo as zoo
from sympy.core.symbol import Dummy as Dummy, Symbol as Symbol, Wild as Wild, symbols as symbols
from sympy.functions.combinatorial.factorials import binomial as binomial, factorial as factorial, rf as rf
from sympy.functions.elementary.integers import ceiling as ceiling, floor as floor, frac as frac
from sympy.functions.elementary.miscellaneous import Max as Max, Min as Min
from sympy.series.series_class import SeriesBase as SeriesBase

def rational_algorithm(f, x, k, order: int = 4, full: bool = False):
    """
    Rational algorithm for computing
    formula of coefficients of Formal Power Series
    of a function.

    Explanation
    ===========

    Applicable when f(x) or some derivative of f(x)
    is a rational function in x.

    :func:`rational_algorithm` uses :func:`~.apart` function for partial fraction
    decomposition. :func:`~.apart` by default uses 'undetermined coefficients
    method'. By setting ``full=True``, 'Bronstein's algorithm' can be used
    instead.

    Looks for derivative of a function up to 4'th order (by default).
    This can be overridden using order option.

    Parameters
    ==========

    x : Symbol
    order : int, optional
        Order of the derivative of ``f``, Default is 4.
    full : bool

    Returns
    =======

    formula : Expr
    ind : Expr
        Independent terms.
    order : int
    full : bool

    Examples
    ========

    >>> from sympy import log, atan
    >>> from sympy.series.formal import rational_algorithm as ra
    >>> from sympy.abc import x, k

    >>> ra(1 / (1 - x), x, k)
    (1, 0, 0)
    >>> ra(log(1 + x), x, k)
    (-1/((-1)**k*k), 0, 1)

    >>> ra(atan(x), x, k, full=True)
    ((-I/(2*(-I)**k) + I/(2*I**k))/k, 0, 1)

    Notes
    =====

    By setting ``full=True``, range of admissible functions to be solved using
    ``rational_algorithm`` can be increased. This option should be used
    carefully as it can significantly slow down the computation as ``doit`` is
    performed on the :class:`~.RootSum` object returned by the :func:`~.apart`
    function. Use ``full=False`` whenever possible.

    See Also
    ========

    sympy.polys.partfrac.apart

    References
    ==========

    .. [1] Formal Power Series - Dominik Gruntz, Wolfram Koepf
    .. [2] Power Series in Computer Algebra - Wolfram Koepf

    """
def rational_independent(terms, x):
    """
    Returns a list of all the rationally independent terms.

    Examples
    ========

    >>> from sympy import sin, cos
    >>> from sympy.series.formal import rational_independent
    >>> from sympy.abc import x

    >>> rational_independent([cos(x), sin(x)], x)
    [cos(x), sin(x)]
    >>> rational_independent([x**2, sin(x), x*sin(x), x**3], x)
    [x**3 + x**2, x*sin(x) + sin(x)]
    """
def simpleDE(f, x, g, order: int = 4) -> Generator[Incomplete, None, Incomplete]:
    """
    Generates simple DE.

    Explanation
    ===========

    DE is of the form

    .. math::
        f^k(x) + \\sum\\limits_{j=0}^{k-1} A_j f^j(x) = 0

    where :math:`A_j` should be rational function in x.

    Generates DE's upto order 4 (default). DE's can also have free parameters.

    By increasing order, higher order DE's can be found.

    Yields a tuple of (DE, order).
    """
def exp_re(DE, r, k):
    """Converts a DE with constant coefficients (explike) into a RE.

    Explanation
    ===========

    Performs the substitution:

    .. math::
        f^j(x) \\to r(k + j)

    Normalises the terms so that lowest order of a term is always r(k).

    Examples
    ========

    >>> from sympy import Function, Derivative
    >>> from sympy.series.formal import exp_re
    >>> from sympy.abc import x, k
    >>> f, r = Function('f'), Function('r')

    >>> exp_re(-f(x) + Derivative(f(x)), r, k)
    -r(k) + r(k + 1)
    >>> exp_re(Derivative(f(x), x) + Derivative(f(x), (x, 2)), r, k)
    r(k) + r(k + 1)

    See Also
    ========

    sympy.series.formal.hyper_re
    """
def hyper_re(DE, r, k):
    """
    Converts a DE into a RE.

    Explanation
    ===========

    Performs the substitution:

    .. math::
        x^l f^j(x) \\to (k + 1 - l)_j . a_{k + j - l}

    Normalises the terms so that lowest order of a term is always r(k).

    Examples
    ========

    >>> from sympy import Function, Derivative
    >>> from sympy.series.formal import hyper_re
    >>> from sympy.abc import x, k
    >>> f, r = Function('f'), Function('r')

    >>> hyper_re(-f(x) + Derivative(f(x)), r, k)
    (k + 1)*r(k + 1) - r(k)
    >>> hyper_re(-x*f(x) + Derivative(f(x), (x, 2)), r, k)
    (k + 2)*(k + 3)*r(k + 3) - r(k)

    See Also
    ========

    sympy.series.formal.exp_re
    """
def _transformation_a(f, x, P, Q, k, m, shift): ...
def _transformation_c(f, x, P, Q, k, m, scale): ...
def _transformation_e(f, x, P, Q, k, m): ...
def _apply_shift(sol, shift): ...
def _apply_scale(sol, scale): ...
def _apply_integrate(sol, x, k): ...
def _compute_formula(f, x, P, Q, k, m, k_max):
    """Computes the formula for f."""
def _rsolve_hypergeometric(f, x, P, Q, k, m):
    """
    Recursive wrapper to rsolve_hypergeometric.

    Explanation
    ===========

    Returns a Tuple of (formula, series independent terms,
    maximum power of x in independent terms) if successful
    otherwise ``None``.

    See :func:`rsolve_hypergeometric` for details.
    """
def rsolve_hypergeometric(f, x, P, Q, k, m):
    """
    Solves RE of hypergeometric type.

    Explanation
    ===========

    Attempts to solve RE of the form

    Q(k)*a(k + m) - P(k)*a(k)

    Transformations that preserve Hypergeometric type:

        a. x**n*f(x): b(k + m) = R(k - n)*b(k)
        b. f(A*x): b(k + m) = A**m*R(k)*b(k)
        c. f(x**n): b(k + n*m) = R(k/n)*b(k)
        d. f(x**(1/m)): b(k + 1) = R(k*m)*b(k)
        e. f'(x): b(k + m) = ((k + m + 1)/(k + 1))*R(k + 1)*b(k)

    Some of these transformations have been used to solve the RE.

    Returns
    =======

    formula : Expr
    ind : Expr
        Independent terms.
    order : int

    Examples
    ========

    >>> from sympy import exp, ln, S
    >>> from sympy.series.formal import rsolve_hypergeometric as rh
    >>> from sympy.abc import x, k

    >>> rh(exp(x), x, -S.One, (k + 1), k, 1)
    (Piecewise((1/factorial(k), Eq(Mod(k, 1), 0)), (0, True)), 1, 1)

    >>> rh(ln(1 + x), x, k**2, k*(k + 1), k, 1)
    (Piecewise(((-1)**(k - 1)*factorial(k - 1)/RisingFactorial(2, k - 1),
     Eq(Mod(k, 1), 0)), (0, True)), x, 2)

    References
    ==========

    .. [1] Formal Power Series - Dominik Gruntz, Wolfram Koepf
    .. [2] Power Series in Computer Algebra - Wolfram Koepf
    """
def _solve_hyper_RE(f, x, RE, g, k):
    """See docstring of :func:`rsolve_hypergeometric` for details."""
def _solve_explike_DE(f, x, DE, g, k):
    """Solves DE with constant coefficients."""
def _solve_simple(f, x, DE, g, k):
    """Converts DE into RE and solves using :func:`rsolve`."""
def _transform_explike_DE(DE, g, x, order, syms):
    """Converts DE with free parameters into DE with constant coefficients."""
def _transform_DE_RE(DE, g, k, order, syms):
    """Converts DE with free parameters into RE of hypergeometric type."""
def solve_de(f, x, DE, order, g, k):
    """
    Solves the DE.

    Explanation
    ===========

    Tries to solve DE by either converting into a RE containing two terms or
    converting into a DE having constant coefficients.

    Returns
    =======

    formula : Expr
    ind : Expr
        Independent terms.
    order : int

    Examples
    ========

    >>> from sympy import Derivative as D, Function
    >>> from sympy import exp, ln
    >>> from sympy.series.formal import solve_de
    >>> from sympy.abc import x, k
    >>> f = Function('f')

    >>> solve_de(exp(x), x, D(f(x), x) - f(x), 1, f, k)
    (Piecewise((1/factorial(k), Eq(Mod(k, 1), 0)), (0, True)), 1, 1)

    >>> solve_de(ln(1 + x), x, (x + 1)*D(f(x), x, 2) + D(f(x)), 2, f, k)
    (Piecewise(((-1)**(k - 1)*factorial(k - 1)/RisingFactorial(2, k - 1),
     Eq(Mod(k, 1), 0)), (0, True)), x, 2)
    """
def hyper_algorithm(f, x, k, order: int = 4):
    """
    Hypergeometric algorithm for computing Formal Power Series.

    Explanation
    ===========

    Steps:
        * Generates DE
        * Convert the DE into RE
        * Solves the RE

    Examples
    ========

    >>> from sympy import exp, ln
    >>> from sympy.series.formal import hyper_algorithm

    >>> from sympy.abc import x, k

    >>> hyper_algorithm(exp(x), x, k)
    (Piecewise((1/factorial(k), Eq(Mod(k, 1), 0)), (0, True)), 1, 1)

    >>> hyper_algorithm(ln(1 + x), x, k)
    (Piecewise(((-1)**(k - 1)*factorial(k - 1)/RisingFactorial(2, k - 1),
     Eq(Mod(k, 1), 0)), (0, True)), x, 2)

    See Also
    ========

    sympy.series.formal.simpleDE
    sympy.series.formal.solve_de
    """
def _compute_fps(f, x, x0, dir, hyper, order, rational, full):
    """Recursive wrapper to compute fps.

    See :func:`compute_fps` for details.
    """
def compute_fps(f, x, x0: int = 0, dir: int = 1, hyper: bool = True, order: int = 4, rational: bool = True, full: bool = False):
    """
    Computes the formula for Formal Power Series of a function.

    Explanation
    ===========

    Tries to compute the formula by applying the following techniques
    (in order):

    * rational_algorithm
    * Hypergeometric algorithm

    Parameters
    ==========

    x : Symbol
    x0 : number, optional
        Point to perform series expansion about. Default is 0.
    dir : {1, -1, '+', '-'}, optional
        If dir is 1 or '+' the series is calculated from the right and
        for -1 or '-' the series is calculated from the left. For smooth
        functions this flag will not alter the results. Default is 1.
    hyper : {True, False}, optional
        Set hyper to False to skip the hypergeometric algorithm.
        By default it is set to False.
    order : int, optional
        Order of the derivative of ``f``, Default is 4.
    rational : {True, False}, optional
        Set rational to False to skip rational algorithm. By default it is set
        to True.
    full : {True, False}, optional
        Set full to True to increase the range of rational algorithm.
        See :func:`rational_algorithm` for details. By default it is set to
        False.

    Returns
    =======

    ak : sequence
        Sequence of coefficients.
    xk : sequence
        Sequence of powers of x.
    ind : Expr
        Independent terms.
    mul : Pow
        Common terms.

    See Also
    ========

    sympy.series.formal.rational_algorithm
    sympy.series.formal.hyper_algorithm
    """

class Coeff(Function):
    """
    Coeff(p, x, n) represents the nth coefficient of the polynomial p in x
    """
    @classmethod
    def eval(cls, p, x, n): ...

class FormalPowerSeries(SeriesBase):
    """
    Represents Formal Power Series of a function.

    Explanation
    ===========

    No computation is performed. This class should only to be used to represent
    a series. No checks are performed.

    For computing a series use :func:`fps`.

    See Also
    ========

    sympy.series.formal.fps
    """
    def __new__(cls, *args): ...
    ak_seq: Incomplete
    fact_seq: Incomplete
    bell_coeff_seq: Incomplete
    sign_seq: Incomplete
    def __init__(self, *args) -> None: ...
    @property
    def function(self): ...
    @property
    def x(self): ...
    @property
    def x0(self): ...
    @property
    def dir(self): ...
    @property
    def ak(self): ...
    @property
    def xk(self): ...
    @property
    def ind(self): ...
    @property
    def interval(self): ...
    @property
    def start(self): ...
    @property
    def stop(self): ...
    @property
    def length(self): ...
    @property
    def infinite(self):
        """Returns an infinite representation of the series"""
    def _get_pow_x(self, term):
        """Returns the power of x in a term."""
    def polynomial(self, n: int = 6):
        """
        Truncated series as polynomial.

        Explanation
        ===========

        Returns series expansion of ``f`` upto order ``O(x**n)``
        as a polynomial(without ``O`` term).
        """
    def truncate(self, n: int = 6):
        """
        Truncated series.

        Explanation
        ===========

        Returns truncated series expansion of f upto
        order ``O(x**n)``.

        If n is ``None``, returns an infinite iterator.
        """
    def zero_coeff(self): ...
    def _eval_term(self, pt): ...
    def _eval_subs(self, old, new): ...
    def _eval_as_leading_term(self, x, logx: Incomplete | None = None, cdir: int = 0): ...
    def _eval_derivative(self, x): ...
    def integrate(self, x: Incomplete | None = None, **kwargs):
        """
        Integrate Formal Power Series.

        Examples
        ========

        >>> from sympy import fps, sin, integrate
        >>> from sympy.abc import x
        >>> f = fps(sin(x))
        >>> f.integrate(x).truncate()
        -1 + x**2/2 - x**4/24 + O(x**6)
        >>> integrate(f, (x, 0, 1))
        1 - cos(1)
        """
    def product(self, other, x: Incomplete | None = None, n: int = 6):
        """
        Multiplies two Formal Power Series, using discrete convolution and
        return the truncated terms upto specified order.

        Parameters
        ==========

        n : Number, optional
            Specifies the order of the term up to which the polynomial should
            be truncated.

        Examples
        ========

        >>> from sympy import fps, sin, exp
        >>> from sympy.abc import x
        >>> f1 = fps(sin(x))
        >>> f2 = fps(exp(x))

        >>> f1.product(f2, x).truncate(4)
        x + x**2 + x**3/3 + O(x**4)

        See Also
        ========

        sympy.discrete.convolutions
        sympy.series.formal.FormalPowerSeriesProduct

        """
    def coeff_bell(self, n):
        '''
        self.coeff_bell(n) returns a sequence of Bell polynomials of the second kind.
        Note that ``n`` should be a integer.

        The second kind of Bell polynomials (are sometimes called "partial" Bell
        polynomials or incomplete Bell polynomials) are defined as

        .. math::
            B_{n,k}(x_1, x_2,\\dotsc x_{n-k+1}) =
                \\sum_{j_1+j_2+j_2+\\dotsb=k \\atop j_1+2j_2+3j_2+\\dotsb=n}
                \\frac{n!}{j_1!j_2!\\dotsb j_{n-k+1}!}
                \\left(\\frac{x_1}{1!} \\right)^{j_1}
                \\left(\\frac{x_2}{2!} \\right)^{j_2} \\dotsb
                \\left(\\frac{x_{n-k+1}}{(n-k+1)!} \\right) ^{j_{n-k+1}}.

        * ``bell(n, k, (x1, x2, ...))`` gives Bell polynomials of the second kind,
          `B_{n,k}(x_1, x_2, \\dotsc, x_{n-k+1})`.

        See Also
        ========

        sympy.functions.combinatorial.numbers.bell

        '''
    def compose(self, other, x: Incomplete | None = None, n: int = 6):
        """
        Returns the truncated terms of the formal power series of the composed function,
        up to specified ``n``.

        Explanation
        ===========

        If ``f`` and ``g`` are two formal power series of two different functions,
        then the coefficient sequence ``ak`` of the composed formal power series `fp`
        will be as follows.

        .. math::
            \\sum\\limits_{k=0}^{n} b_k B_{n,k}(x_1, x_2, \\dotsc, x_{n-k+1})

        Parameters
        ==========

        n : Number, optional
            Specifies the order of the term up to which the polynomial should
            be truncated.

        Examples
        ========

        >>> from sympy import fps, sin, exp
        >>> from sympy.abc import x
        >>> f1 = fps(exp(x))
        >>> f2 = fps(sin(x))

        >>> f1.compose(f2, x).truncate()
        1 + x + x**2/2 - x**4/8 - x**5/15 + O(x**6)

        >>> f1.compose(f2, x).truncate(8)
        1 + x + x**2/2 - x**4/8 - x**5/15 - x**6/240 + x**7/90 + O(x**8)

        See Also
        ========

        sympy.functions.combinatorial.numbers.bell
        sympy.series.formal.FormalPowerSeriesCompose

        References
        ==========

        .. [1] Comtet, Louis: Advanced combinatorics; the art of finite and infinite expansions. Reidel, 1974.

        """
    def inverse(self, x: Incomplete | None = None, n: int = 6):
        """
        Returns the truncated terms of the inverse of the formal power series,
        up to specified ``n``.

        Explanation
        ===========

        If ``f`` and ``g`` are two formal power series of two different functions,
        then the coefficient sequence ``ak`` of the composed formal power series ``fp``
        will be as follows.

        .. math::
            \\sum\\limits_{k=0}^{n} (-1)^{k} x_0^{-k-1} B_{n,k}(x_1, x_2, \\dotsc, x_{n-k+1})

        Parameters
        ==========

        n : Number, optional
            Specifies the order of the term up to which the polynomial should
            be truncated.

        Examples
        ========

        >>> from sympy import fps, exp, cos
        >>> from sympy.abc import x
        >>> f1 = fps(exp(x))
        >>> f2 = fps(cos(x))

        >>> f1.inverse(x).truncate()
        1 - x + x**2/2 - x**3/6 + x**4/24 - x**5/120 + O(x**6)

        >>> f2.inverse(x).truncate(8)
        1 + x**2/2 + 5*x**4/24 + 61*x**6/720 + O(x**8)

        See Also
        ========

        sympy.functions.combinatorial.numbers.bell
        sympy.series.formal.FormalPowerSeriesInverse

        References
        ==========

        .. [1] Comtet, Louis: Advanced combinatorics; the art of finite and infinite expansions. Reidel, 1974.

        """
    def __add__(self, other): ...
    def __radd__(self, other): ...
    def __neg__(self): ...
    def __sub__(self, other): ...
    def __rsub__(self, other): ...
    def __mul__(self, other): ...
    def __rmul__(self, other): ...

class FiniteFormalPowerSeries(FormalPowerSeries):
    """Base Class for Product, Compose and Inverse classes"""
    def __init__(self, *args) -> None: ...
    @property
    def ffps(self): ...
    @property
    def gfps(self): ...
    @property
    def f(self): ...
    @property
    def g(self): ...
    @property
    def infinite(self) -> None: ...
    def _eval_terms(self, n) -> None: ...
    def _eval_term(self, pt) -> None: ...
    def polynomial(self, n): ...
    def truncate(self, n: int = 6): ...
    def _eval_derivative(self, x) -> None: ...
    def integrate(self, x) -> None: ...

class FormalPowerSeriesProduct(FiniteFormalPowerSeries):
    """Represents the product of two formal power series of two functions.

    Explanation
    ===========

    No computation is performed. Terms are calculated using a term by term logic,
    instead of a point by point logic.

    There are two differences between a :obj:`FormalPowerSeries` object and a
    :obj:`FormalPowerSeriesProduct` object. The first argument contains the two
    functions involved in the product. Also, the coefficient sequence contains
    both the coefficient sequence of the formal power series of the involved functions.

    See Also
    ========

    sympy.series.formal.FormalPowerSeries
    sympy.series.formal.FiniteFormalPowerSeries

    """
    coeff1: Incomplete
    coeff2: Incomplete
    def __init__(self, *args) -> None: ...
    @property
    def function(self):
        """Function of the product of two formal power series."""
    def _eval_terms(self, n):
        """
        Returns the first ``n`` terms of the product formal power series.
        Term by term logic is implemented here.

        Examples
        ========

        >>> from sympy import fps, sin, exp
        >>> from sympy.abc import x
        >>> f1 = fps(sin(x))
        >>> f2 = fps(exp(x))
        >>> fprod = f1.product(f2, x)

        >>> fprod._eval_terms(4)
        x**3/3 + x**2 + x

        See Also
        ========

        sympy.series.formal.FormalPowerSeries.product

        """

class FormalPowerSeriesCompose(FiniteFormalPowerSeries):
    """
    Represents the composed formal power series of two functions.

    Explanation
    ===========

    No computation is performed. Terms are calculated using a term by term logic,
    instead of a point by point logic.

    There are two differences between a :obj:`FormalPowerSeries` object and a
    :obj:`FormalPowerSeriesCompose` object. The first argument contains the outer
    function and the inner function involved in the omposition. Also, the
    coefficient sequence contains the generic sequence which is to be multiplied
    by a custom ``bell_seq`` finite sequence. The finite terms will then be added up to
    get the final terms.

    See Also
    ========

    sympy.series.formal.FormalPowerSeries
    sympy.series.formal.FiniteFormalPowerSeries

    """
    @property
    def function(self):
        """Function for the composed formal power series."""
    def _eval_terms(self, n):
        """
        Returns the first `n` terms of the composed formal power series.
        Term by term logic is implemented here.

        Explanation
        ===========

        The coefficient sequence of the :obj:`FormalPowerSeriesCompose` object is the generic sequence.
        It is multiplied by ``bell_seq`` to get a sequence, whose terms are added up to get
        the final terms for the polynomial.

        Examples
        ========

        >>> from sympy import fps, sin, exp
        >>> from sympy.abc import x
        >>> f1 = fps(exp(x))
        >>> f2 = fps(sin(x))
        >>> fcomp = f1.compose(f2, x)

        >>> fcomp._eval_terms(6)
        -x**5/15 - x**4/8 + x**2/2 + x + 1

        >>> fcomp._eval_terms(8)
        x**7/90 - x**6/240 - x**5/15 - x**4/8 + x**2/2 + x + 1

        See Also
        ========

        sympy.series.formal.FormalPowerSeries.compose
        sympy.series.formal.FormalPowerSeries.coeff_bell

        """

class FormalPowerSeriesInverse(FiniteFormalPowerSeries):
    """
    Represents the Inverse of a formal power series.

    Explanation
    ===========

    No computation is performed. Terms are calculated using a term by term logic,
    instead of a point by point logic.

    There is a single difference between a :obj:`FormalPowerSeries` object and a
    :obj:`FormalPowerSeriesInverse` object. The coefficient sequence contains the
    generic sequence which is to be multiplied by a custom ``bell_seq`` finite sequence.
    The finite terms will then be added up to get the final terms.

    See Also
    ========

    sympy.series.formal.FormalPowerSeries
    sympy.series.formal.FiniteFormalPowerSeries

    """
    aux_seq: Incomplete
    def __init__(self, *args) -> None: ...
    @property
    def function(self):
        """Function for the inverse of a formal power series."""
    @property
    def g(self) -> None: ...
    @property
    def gfps(self) -> None: ...
    def _eval_terms(self, n):
        """
        Returns the first ``n`` terms of the composed formal power series.
        Term by term logic is implemented here.

        Explanation
        ===========

        The coefficient sequence of the `FormalPowerSeriesInverse` object is the generic sequence.
        It is multiplied by ``bell_seq`` to get a sequence, whose terms are added up to get
        the final terms for the polynomial.

        Examples
        ========

        >>> from sympy import fps, exp, cos
        >>> from sympy.abc import x
        >>> f1 = fps(exp(x))
        >>> f2 = fps(cos(x))
        >>> finv1, finv2 = f1.inverse(), f2.inverse()

        >>> finv1._eval_terms(6)
        -x**5/120 + x**4/24 - x**3/6 + x**2/2 - x + 1

        >>> finv2._eval_terms(8)
        61*x**6/720 + 5*x**4/24 + x**2/2 + 1

        See Also
        ========

        sympy.series.formal.FormalPowerSeries.inverse
        sympy.series.formal.FormalPowerSeries.coeff_bell

        """

def fps(f, x: Incomplete | None = None, x0: int = 0, dir: int = 1, hyper: bool = True, order: int = 4, rational: bool = True, full: bool = False):
    """
    Generates Formal Power Series of ``f``.

    Explanation
    ===========

    Returns the formal series expansion of ``f`` around ``x = x0``
    with respect to ``x`` in the form of a ``FormalPowerSeries`` object.

    Formal Power Series is represented using an explicit formula
    computed using different algorithms.

    See :func:`compute_fps` for the more details regarding the computation
    of formula.

    Parameters
    ==========

    x : Symbol, optional
        If x is None and ``f`` is univariate, the univariate symbols will be
        supplied, otherwise an error will be raised.
    x0 : number, optional
        Point to perform series expansion about. Default is 0.
    dir : {1, -1, '+', '-'}, optional
        If dir is 1 or '+' the series is calculated from the right and
        for -1 or '-' the series is calculated from the left. For smooth
        functions this flag will not alter the results. Default is 1.
    hyper : {True, False}, optional
        Set hyper to False to skip the hypergeometric algorithm.
        By default it is set to False.
    order : int, optional
        Order of the derivative of ``f``, Default is 4.
    rational : {True, False}, optional
        Set rational to False to skip rational algorithm. By default it is set
        to True.
    full : {True, False}, optional
        Set full to True to increase the range of rational algorithm.
        See :func:`rational_algorithm` for details. By default it is set to
        False.

    Examples
    ========

    >>> from sympy import fps, ln, atan, sin
    >>> from sympy.abc import x, n

    Rational Functions

    >>> fps(ln(1 + x)).truncate()
    x - x**2/2 + x**3/3 - x**4/4 + x**5/5 + O(x**6)

    >>> fps(atan(x), full=True).truncate()
    x - x**3/3 + x**5/5 + O(x**6)

    Symbolic Functions

    >>> fps(x**n*sin(x**2), x).truncate(8)
    -x**(n + 6)/6 + x**(n + 2) + O(x**(n + 8))

    See Also
    ========

    sympy.series.formal.FormalPowerSeries
    sympy.series.formal.compute_fps
    """
