from _typeshed import Incomplete
from sympy.core import Dummy as Dummy, Mod as Mod, S as S, sympify as sympify
from sympy.core.cache import cacheit as cacheit
from sympy.core.function import ArgumentIndexError as ArgumentIndexError, Function as Function, PoleError as PoleError
from sympy.core.logic import fuzzy_and as fuzzy_and
from sympy.core.numbers import I as I, Integer as Integer, pi as pi
from sympy.core.relational import Eq as Eq
from sympy.ntheory import sieve as sieve
from sympy.ntheory.residue_ntheory import binomial_mod as binomial_mod
from sympy.polys.polytools import Poly as Poly

class CombinatorialFunction(Function):
    """Base class for combinatorial functions. """
    def _eval_simplify(self, **kwargs): ...

class factorial(CombinatorialFunction):
    """Implementation of factorial function over nonnegative integers.
       By convention (consistent with the gamma function and the binomial
       coefficients), factorial of a negative integer is complex infinity.

       The factorial is very important in combinatorics where it gives
       the number of ways in which `n` objects can be permuted. It also
       arises in calculus, probability, number theory, etc.

       There is strict relation of factorial with gamma function. In
       fact `n! = gamma(n+1)` for nonnegative integers. Rewrite of this
       kind is very useful in case of combinatorial simplification.

       Computation of the factorial is done using two algorithms. For
       small arguments a precomputed look up table is used. However for bigger
       input algorithm Prime-Swing is used. It is the fastest algorithm
       known and computes `n!` via prime factorization of special class
       of numbers, called here the 'Swing Numbers'.

       Examples
       ========

       >>> from sympy import Symbol, factorial, S
       >>> n = Symbol('n', integer=True)

       >>> factorial(0)
       1

       >>> factorial(7)
       5040

       >>> factorial(-2)
       zoo

       >>> factorial(n)
       factorial(n)

       >>> factorial(2*n)
       factorial(2*n)

       >>> factorial(S(1)/2)
       factorial(1/2)

       See Also
       ========

       factorial2, RisingFactorial, FallingFactorial
    """
    def fdiff(self, argindex: int = 1): ...
    _small_swing: Incomplete
    _small_factorials: list[int]
    @classmethod
    def _swing(cls, n): ...
    @classmethod
    def _recursive(cls, n): ...
    @classmethod
    def eval(cls, n): ...
    def _facmod(self, n, q): ...
    def _eval_Mod(self, q): ...
    def _eval_rewrite_as_gamma(self, n, piecewise: bool = True, **kwargs): ...
    def _eval_rewrite_as_Product(self, n, **kwargs): ...
    def _eval_is_integer(self): ...
    def _eval_is_positive(self): ...
    def _eval_is_even(self): ...
    def _eval_is_composite(self): ...
    def _eval_is_real(self): ...
    def _eval_as_leading_term(self, x, logx: Incomplete | None = None, cdir: int = 0): ...

class MultiFactorial(CombinatorialFunction): ...

class subfactorial(CombinatorialFunction):
    """The subfactorial counts the derangements of $n$ items and is
    defined for non-negative integers as:

    .. math:: !n = \\begin{cases} 1 & n = 0 \\\\ 0 & n = 1 \\\\\n                    (n-1)(!(n-1) + !(n-2)) & n > 1 \\end{cases}

    It can also be written as ``int(round(n!/exp(1)))`` but the
    recursive definition with caching is implemented for this function.

    An interesting analytic expression is the following [2]_

    .. math:: !x = \\Gamma(x + 1, -1)/e

    which is valid for non-negative integers `x`. The above formula
    is not very useful in case of non-integers. `\\Gamma(x + 1, -1)` is
    single-valued only for integral arguments `x`, elsewhere on the positive
    real axis it has an infinite number of branches none of which are real.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Subfactorial
    .. [2] https://mathworld.wolfram.com/Subfactorial.html

    Examples
    ========

    >>> from sympy import subfactorial
    >>> from sympy.abc import n
    >>> subfactorial(n + 1)
    subfactorial(n + 1)
    >>> subfactorial(5)
    44

    See Also
    ========

    factorial, uppergamma,
    sympy.utilities.iterables.generate_derangements
    """
    @classmethod
    def _eval(self, n): ...
    @classmethod
    def eval(cls, arg): ...
    def _eval_is_even(self): ...
    def _eval_is_integer(self): ...
    def _eval_rewrite_as_factorial(self, arg, **kwargs): ...
    def _eval_rewrite_as_gamma(self, arg, piecewise: bool = True, **kwargs): ...
    def _eval_rewrite_as_uppergamma(self, arg, **kwargs): ...
    def _eval_is_nonnegative(self): ...
    def _eval_is_odd(self): ...

class factorial2(CombinatorialFunction):
    """The double factorial `n!!`, not to be confused with `(n!)!`

    The double factorial is defined for nonnegative integers and for odd
    negative integers as:

    .. math:: n!! = \\begin{cases} 1 & n = 0 \\\\\n                    n(n-2)(n-4) \\cdots 1 & n\\ \\text{positive odd} \\\\\n                    n(n-2)(n-4) \\cdots 2 & n\\ \\text{positive even} \\\\\n                    (n+2)!!/(n+2) & n\\ \\text{negative odd} \\end{cases}

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Double_factorial

    Examples
    ========

    >>> from sympy import factorial2, var
    >>> n = var('n')
    >>> n
    n
    >>> factorial2(n + 1)
    factorial2(n + 1)
    >>> factorial2(5)
    15
    >>> factorial2(-1)
    1
    >>> factorial2(-5)
    1/3

    See Also
    ========

    factorial, RisingFactorial, FallingFactorial
    """
    @classmethod
    def eval(cls, arg): ...
    def _eval_is_even(self): ...
    def _eval_is_integer(self): ...
    def _eval_is_odd(self): ...
    def _eval_is_positive(self): ...
    def _eval_rewrite_as_gamma(self, n, piecewise: bool = True, **kwargs): ...

class RisingFactorial(CombinatorialFunction):
    '''
    Rising factorial (also called Pochhammer symbol [1]_) is a double valued
    function arising in concrete mathematics, hypergeometric functions
    and series expansions. It is defined by:

    .. math:: \\texttt{rf(y, k)} = (x)^k = x \\cdot (x+1) \\cdots (x+k-1)

    where `x` can be arbitrary expression and `k` is an integer. For
    more information check "Concrete mathematics" by Graham, pp. 66
    or visit https://mathworld.wolfram.com/RisingFactorial.html page.

    When `x` is a `~.Poly` instance of degree $\\ge 1$ with a single variable,
    `(x)^k = x(y) \\cdot x(y+1) \\cdots x(y+k-1)`, where `y` is the
    variable of `x`. This is as described in [2]_.

    Examples
    ========

    >>> from sympy import rf, Poly
    >>> from sympy.abc import x
    >>> rf(x, 0)
    1
    >>> rf(1, 5)
    120
    >>> rf(x, 5) == x*(1 + x)*(2 + x)*(3 + x)*(4 + x)
    True
    >>> rf(Poly(x**3, x), 2)
    Poly(x**6 + 3*x**5 + 3*x**4 + x**3, x, domain=\'ZZ\')

    Rewriting is complicated unless the relationship between
    the arguments is known, but rising factorial can
    be rewritten in terms of gamma, factorial, binomial,
    and falling factorial.

    >>> from sympy import Symbol, factorial, ff, binomial, gamma
    >>> n = Symbol(\'n\', integer=True, positive=True)
    >>> R = rf(n, n + 2)
    >>> for i in (rf, ff, factorial, binomial, gamma):
    ...  R.rewrite(i)
    ...
    RisingFactorial(n, n + 2)
    FallingFactorial(2*n + 1, n + 2)
    factorial(2*n + 1)/factorial(n - 1)
    binomial(2*n + 1, n + 2)*factorial(n + 2)
    gamma(2*n + 2)/gamma(n)

    See Also
    ========

    factorial, factorial2, FallingFactorial

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Pochhammer_symbol
    .. [2] Peter Paule, "Greatest Factorial Factorization and Symbolic
           Summation", Journal of Symbolic Computation, vol. 20, pp. 235-268,
           1995.

    '''
    @classmethod
    def eval(cls, x, k): ...
    def _eval_rewrite_as_gamma(self, x, k, piecewise: bool = True, **kwargs): ...
    def _eval_rewrite_as_FallingFactorial(self, x, k, **kwargs): ...
    def _eval_rewrite_as_factorial(self, x, k, **kwargs): ...
    def _eval_rewrite_as_binomial(self, x, k, **kwargs): ...
    def _eval_rewrite_as_tractable(self, x, k, limitvar: Incomplete | None = None, **kwargs): ...
    def _eval_is_integer(self): ...

class FallingFactorial(CombinatorialFunction):
    '''
    Falling factorial (related to rising factorial) is a double valued
    function arising in concrete mathematics, hypergeometric functions
    and series expansions. It is defined by

    .. math:: \\texttt{ff(x, k)} = (x)_k = x \\cdot (x-1) \\cdots (x-k+1)

    where `x` can be arbitrary expression and `k` is an integer. For
    more information check "Concrete mathematics" by Graham, pp. 66
    or [1]_.

    When `x` is a `~.Poly` instance of degree $\\ge 1$ with single variable,
    `(x)_k = x(y) \\cdot x(y-1) \\cdots x(y-k+1)`, where `y` is the
    variable of `x`. This is as described in

    >>> from sympy import ff, Poly, Symbol
    >>> from sympy.abc import x
    >>> n = Symbol(\'n\', integer=True)

    >>> ff(x, 0)
    1
    >>> ff(5, 5)
    120
    >>> ff(x, 5) == x*(x - 1)*(x - 2)*(x - 3)*(x - 4)
    True
    >>> ff(Poly(x**2, x), 2)
    Poly(x**4 - 2*x**3 + x**2, x, domain=\'ZZ\')
    >>> ff(n, n)
    factorial(n)

    Rewriting is complicated unless the relationship between
    the arguments is known, but falling factorial can
    be rewritten in terms of gamma, factorial and binomial
    and rising factorial.

    >>> from sympy import factorial, rf, gamma, binomial, Symbol
    >>> n = Symbol(\'n\', integer=True, positive=True)
    >>> F = ff(n, n - 2)
    >>> for i in (rf, ff, factorial, binomial, gamma):
    ...  F.rewrite(i)
    ...
    RisingFactorial(3, n - 2)
    FallingFactorial(n, n - 2)
    factorial(n)/2
    binomial(n, n - 2)*factorial(n - 2)
    gamma(n + 1)/2

    See Also
    ========

    factorial, factorial2, RisingFactorial

    References
    ==========

    .. [1] https://mathworld.wolfram.com/FallingFactorial.html
    .. [2] Peter Paule, "Greatest Factorial Factorization and Symbolic
           Summation", Journal of Symbolic Computation, vol. 20, pp. 235-268,
           1995.

    '''
    @classmethod
    def eval(cls, x, k): ...
    def _eval_rewrite_as_gamma(self, x, k, piecewise: bool = True, **kwargs): ...
    def _eval_rewrite_as_RisingFactorial(self, x, k, **kwargs): ...
    def _eval_rewrite_as_binomial(self, x, k, **kwargs): ...
    def _eval_rewrite_as_factorial(self, x, k, **kwargs): ...
    def _eval_rewrite_as_tractable(self, x, k, limitvar: Incomplete | None = None, **kwargs): ...
    def _eval_is_integer(self): ...
rf = RisingFactorial
ff = FallingFactorial

class binomial(CombinatorialFunction):
    """Implementation of the binomial coefficient. It can be defined
    in two ways depending on its desired interpretation:

    .. math:: \\binom{n}{k} = \\frac{n!}{k!(n-k)!}\\ \\text{or}\\\n                \\binom{n}{k} = \\frac{(n)_k}{k!}

    First, in a strict combinatorial sense it defines the
    number of ways we can choose `k` elements from a set of
    `n` elements. In this case both arguments are nonnegative
    integers and binomial is computed using an efficient
    algorithm based on prime factorization.

    The other definition is generalization for arbitrary `n`,
    however `k` must also be nonnegative. This case is very
    useful when evaluating summations.

    For the sake of convenience, for negative integer `k` this function
    will return zero no matter the other argument.

    To expand the binomial when `n` is a symbol, use either
    ``expand_func()`` or ``expand(func=True)``. The former will keep
    the polynomial in factored form while the latter will expand the
    polynomial itself. See examples for details.

    Examples
    ========

    >>> from sympy import Symbol, Rational, binomial, expand_func
    >>> n = Symbol('n', integer=True, positive=True)

    >>> binomial(15, 8)
    6435

    >>> binomial(n, -1)
    0

    Rows of Pascal's triangle can be generated with the binomial function:

    >>> for N in range(8):
    ...     print([binomial(N, i) for i in range(N + 1)])
    ...
    [1]
    [1, 1]
    [1, 2, 1]
    [1, 3, 3, 1]
    [1, 4, 6, 4, 1]
    [1, 5, 10, 10, 5, 1]
    [1, 6, 15, 20, 15, 6, 1]
    [1, 7, 21, 35, 35, 21, 7, 1]

    As can a given diagonal, e.g. the 4th diagonal:

    >>> N = -4
    >>> [binomial(N, i) for i in range(1 - N)]
    [1, -4, 10, -20, 35]

    >>> binomial(Rational(5, 4), 3)
    -5/128
    >>> binomial(Rational(-5, 4), 3)
    -195/128

    >>> binomial(n, 3)
    binomial(n, 3)

    >>> binomial(n, 3).expand(func=True)
    n**3/6 - n**2/2 + n/3

    >>> expand_func(binomial(n, 3))
    n*(n - 2)*(n - 1)/6

    In many cases, we can also compute binomial coefficients modulo a
    prime p quickly using Lucas' Theorem [2]_, though we need to include
    `evaluate=False` to postpone evaluation:

    >>> from sympy import Mod
    >>> Mod(binomial(156675, 4433, evaluate=False), 10**5 + 3)
    28625

    Using a generalisation of Lucas's Theorem given by Granville [3]_,
    we can extend this to arbitrary n:

    >>> Mod(binomial(10**18, 10**12, evaluate=False), (10**5 + 3)**2)
    3744312326

    References
    ==========

    .. [1] https://www.johndcook.com/blog/binomial_coefficients/
    .. [2] https://en.wikipedia.org/wiki/Lucas%27s_theorem
    .. [3] Binomial coefficients modulo prime powers, Andrew Granville,
        Available: https://web.archive.org/web/20170202003812/http://www.dms.umontreal.ca/~andrew/PDF/BinCoeff.pdf
    """
    def fdiff(self, argindex: int = 1): ...
    @classmethod
    def _eval(self, n, k): ...
    @classmethod
    def eval(cls, n, k): ...
    def _eval_Mod(self, q): ...
    def _eval_expand_func(self, **hints):
        """
        Function to expand binomial(n, k) when m is positive integer
        Also,
        n is self.args[0] and k is self.args[1] while using binomial(n, k)
        """
    def _eval_rewrite_as_factorial(self, n, k, **kwargs): ...
    def _eval_rewrite_as_gamma(self, n, k, piecewise: bool = True, **kwargs): ...
    def _eval_rewrite_as_tractable(self, n, k, limitvar: Incomplete | None = None, **kwargs): ...
    def _eval_rewrite_as_FallingFactorial(self, n, k, **kwargs): ...
    def _eval_is_integer(self): ...
    def _eval_is_nonnegative(self): ...
    def _eval_as_leading_term(self, x, logx: Incomplete | None = None, cdir: int = 0): ...
