from .add import Add as Add
from .basic import Basic as Basic
from .containers import Dict as Dict, Tuple as Tuple
from .coreerrors import NonCommutativeExpression as NonCommutativeExpression
from .expr import Expr as Expr
from .function import expand_power_exp as expand_power_exp
from .mul import Mul as Mul, _keep_coeff as _keep_coeff
from .numbers import I as I, Integer as Integer, Number as Number, Rational as Rational, equal_valued as equal_valued
from .power import Pow as Pow
from .singleton import S as S
from .sorting import default_sort_key as default_sort_key, ordered as ordered
from .symbol import Dummy as Dummy
from .sympify import sympify as sympify
from .traversal import preorder_traversal as preorder_traversal
from _typeshed import Incomplete
from sympy.external.gmpy import SYMPY_INTS as SYMPY_INTS
from sympy.utilities.iterables import common_prefix as common_prefix, common_suffix as common_suffix, is_sequence as is_sequence, iterable as iterable, variations as variations

_eps: Incomplete

def _isnumber(i): ...
def _monotonic_sign(self):
    """Return the value closest to 0 that ``self`` may have if all symbols
    are signed and the result is uniformly the same sign for all values of symbols.
    If a symbol is only signed but not known to be an
    integer or the result is 0 then a symbol representative of the sign of self
    will be returned. Otherwise, None is returned if a) the sign could be positive
    or negative or b) self is not in one of the following forms:

    - L(x, y, ...) + A: a function linear in all symbols x, y, ... with an
      additive constant; if A is zero then the function can be a monomial whose
      sign is monotonic over the range of the variables, e.g. (x + 1)**3 if x is
      nonnegative.
    - A/L(x, y, ...) + B: the inverse of a function linear in all symbols x, y, ...
      that does not have a sign change from positive to negative for any set
      of values for the variables.
    - M(x, y, ...) + A: a monomial M whose factors are all signed and a constant, A.
    - A/M(x, y, ...) + B: the inverse of a monomial and constants A and B.
    - P(x): a univariate polynomial

    Examples
    ========

    >>> from sympy.core.exprtools import _monotonic_sign as F
    >>> from sympy import Dummy
    >>> nn = Dummy(integer=True, nonnegative=True)
    >>> p = Dummy(integer=True, positive=True)
    >>> p2 = Dummy(integer=True, positive=True)
    >>> F(nn + 1)
    1
    >>> F(p - 1)
    _nneg
    >>> F(nn*p + 1)
    1
    >>> F(p2*p + 1)
    2
    >>> F(nn - 1)  # could be negative, zero or positive
    """
def decompose_power(expr: Expr) -> tuple[Expr, int]:
    """
    Decompose power into symbolic base and integer exponent.

    Examples
    ========

    >>> from sympy.core.exprtools import decompose_power
    >>> from sympy.abc import x, y
    >>> from sympy import exp

    >>> decompose_power(x)
    (x, 1)
    >>> decompose_power(x**2)
    (x, 2)
    >>> decompose_power(exp(2*y/3))
    (exp(y/3), 2)

    """
def decompose_power_rat(expr: Expr) -> tuple[Expr, Rational]:
    """
    Decompose power into symbolic base and rational exponent;
    if the exponent is not a Rational, then separate only the
    integer coefficient.

    Examples
    ========

    >>> from sympy.core.exprtools import decompose_power_rat
    >>> from sympy.abc import x
    >>> from sympy import sqrt, exp

    >>> decompose_power_rat(sqrt(x))
    (x, 1/2)
    >>> decompose_power_rat(exp(-3*x/2))
    (exp(x/2), -3)

    """

class Factors:
    """Efficient representation of ``f_1*f_2*...*f_n``."""
    __slots__: Incomplete
    factors: Incomplete
    gens: Incomplete
    def __init__(self, factors=None) -> None:
        """Initialize Factors from dict or expr.

        Examples
        ========

        >>> from sympy.core.exprtools import Factors
        >>> from sympy.abc import x
        >>> from sympy import I
        >>> e = 2*x**3
        >>> Factors(e)
        Factors({2: 1, x: 3})
        >>> Factors(e.as_powers_dict())
        Factors({2: 1, x: 3})
        >>> f = _
        >>> f.factors  # underlying dictionary
        {2: 1, x: 3}
        >>> f.gens  # base of each factor
        frozenset({2, x})
        >>> Factors(0)
        Factors({0: 1})
        >>> Factors(I)
        Factors({I: 1})

        Notes
        =====

        Although a dictionary can be passed, only minimal checking is
        performed: powers of -1 and I are made canonical.

        """
    def __hash__(self): ...
    def __repr__(self) -> str: ...
    @property
    def is_zero(self):
        """
        >>> from sympy.core.exprtools import Factors
        >>> Factors(0).is_zero
        True
        """
    @property
    def is_one(self):
        """
        >>> from sympy.core.exprtools import Factors
        >>> Factors(1).is_one
        True
        """
    def as_expr(self):
        """Return the underlying expression.

        Examples
        ========

        >>> from sympy.core.exprtools import Factors
        >>> from sympy.abc import x, y
        >>> Factors((x*y**2).as_powers_dict()).as_expr()
        x*y**2

        """
    def mul(self, other):
        """Return Factors of ``self * other``.

        Examples
        ========

        >>> from sympy.core.exprtools import Factors
        >>> from sympy.abc import x, y, z
        >>> a = Factors((x*y**2).as_powers_dict())
        >>> b = Factors((x*y/z).as_powers_dict())
        >>> a.mul(b)
        Factors({x: 2, y: 3, z: -1})
        >>> a*b
        Factors({x: 2, y: 3, z: -1})
        """
    def normal(self, other):
        """Return ``self`` and ``other`` with ``gcd`` removed from each.
        The only differences between this and method ``div`` is that this
        is 1) optimized for the case when there are few factors in common and
        2) this does not raise an error if ``other`` is zero.

        See Also
        ========
        div

        """
    def div(self, other):
        """Return ``self`` and ``other`` with ``gcd`` removed from each.
        This is optimized for the case when there are many factors in common.

        Examples
        ========

        >>> from sympy.core.exprtools import Factors
        >>> from sympy.abc import x, y, z
        >>> from sympy import S

        >>> a = Factors((x*y**2).as_powers_dict())
        >>> a.div(a)
        (Factors({}), Factors({}))
        >>> a.div(x*z)
        (Factors({y: 2}), Factors({z: 1}))

        The ``/`` operator only gives ``quo``:

        >>> a/x
        Factors({y: 2})

        Factors treats its factors as though they are all in the numerator, so
        if you violate this assumption the results will be correct but will
        not strictly correspond to the numerator and denominator of the ratio:

        >>> a.div(x/z)
        (Factors({y: 2}), Factors({z: -1}))

        Factors is also naive about bases: it does not attempt any denesting
        of Rational-base terms, for example the following does not become
        2**(2*x)/2.

        >>> Factors(2**(2*x + 2)).div(S(8))
        (Factors({2: 2*x + 2}), Factors({8: 1}))

        factor_terms can clean up such Rational-bases powers:

        >>> from sympy import factor_terms
        >>> n, d = Factors(2**(2*x + 2)).div(S(8))
        >>> n.as_expr()/d.as_expr()
        2**(2*x + 2)/8
        >>> factor_terms(_)
        2**(2*x)/2

        """
    def quo(self, other):
        """Return numerator Factor of ``self / other``.

        Examples
        ========

        >>> from sympy.core.exprtools import Factors
        >>> from sympy.abc import x, y, z
        >>> a = Factors((x*y**2).as_powers_dict())
        >>> b = Factors((x*y/z).as_powers_dict())
        >>> a.quo(b)  # same as a/b
        Factors({y: 1})
        """
    def rem(self, other):
        """Return denominator Factors of ``self / other``.

        Examples
        ========

        >>> from sympy.core.exprtools import Factors
        >>> from sympy.abc import x, y, z
        >>> a = Factors((x*y**2).as_powers_dict())
        >>> b = Factors((x*y/z).as_powers_dict())
        >>> a.rem(b)
        Factors({z: -1})
        >>> a.rem(a)
        Factors({})
        """
    def pow(self, other):
        """Return self raised to a non-negative integer power.

        Examples
        ========

        >>> from sympy.core.exprtools import Factors
        >>> from sympy.abc import x, y
        >>> a = Factors((x*y**2).as_powers_dict())
        >>> a**2
        Factors({x: 2, y: 4})

        """
    def gcd(self, other):
        """Return Factors of ``gcd(self, other)``. The keys are
        the intersection of factors with the minimum exponent for
        each factor.

        Examples
        ========

        >>> from sympy.core.exprtools import Factors
        >>> from sympy.abc import x, y, z
        >>> a = Factors((x*y**2).as_powers_dict())
        >>> b = Factors((x*y/z).as_powers_dict())
        >>> a.gcd(b)
        Factors({x: 1, y: 1})
        """
    def lcm(self, other):
        """Return Factors of ``lcm(self, other)`` which are
        the union of factors with the maximum exponent for
        each factor.

        Examples
        ========

        >>> from sympy.core.exprtools import Factors
        >>> from sympy.abc import x, y, z
        >>> a = Factors((x*y**2).as_powers_dict())
        >>> b = Factors((x*y/z).as_powers_dict())
        >>> a.lcm(b)
        Factors({x: 1, y: 2, z: -1})
        """
    def __mul__(self, other): ...
    def __divmod__(self, other): ...
    def __truediv__(self, other): ...
    def __mod__(self, other): ...
    def __pow__(self, other): ...
    def __eq__(self, other): ...
    def __ne__(self, other): ...

class Term:
    """Efficient representation of ``coeff*(numer/denom)``. """
    __slots__: Incomplete
    coeff: Incomplete
    numer: Incomplete
    denom: Incomplete
    def __init__(self, term, numer=None, denom=None) -> None: ...
    def __hash__(self): ...
    def __repr__(self) -> str: ...
    def as_expr(self): ...
    def mul(self, other): ...
    def inv(self): ...
    def quo(self, other): ...
    def pow(self, other): ...
    def gcd(self, other): ...
    def lcm(self, other): ...
    def __mul__(self, other): ...
    def __truediv__(self, other): ...
    def __pow__(self, other): ...
    def __eq__(self, other): ...
    def __ne__(self, other): ...

def _gcd_terms(terms, isprimitive: bool = False, fraction: bool = True):
    """Helper function for :func:`gcd_terms`.

    Parameters
    ==========

    isprimitive : boolean, optional
        If ``isprimitive`` is True then the call to primitive
        for an Add will be skipped. This is useful when the
        content has already been extracted.

    fraction : boolean, optional
        If ``fraction`` is True then the expression will appear over a common
        denominator, the lcm of all term denominators.
    """
def gcd_terms(terms, isprimitive: bool = False, clear: bool = True, fraction: bool = True):
    """Compute the GCD of ``terms`` and put them together.

    Parameters
    ==========

    terms : Expr
        Can be an expression or a non-Basic sequence of expressions
        which will be handled as though they are terms from a sum.

    isprimitive : bool, optional
        If ``isprimitive`` is True the _gcd_terms will not run the primitive
        method on the terms.

    clear : bool, optional
        It controls the removal of integers from the denominator of an Add
        expression. When True (default), all numerical denominator will be cleared;
        when False the denominators will be cleared only if all terms had numerical
        denominators other than 1.

    fraction : bool, optional
        When True (default), will put the expression over a common
        denominator.

    Examples
    ========

    >>> from sympy import gcd_terms
    >>> from sympy.abc import x, y

    >>> gcd_terms((x + 1)**2*y + (x + 1)*y**2)
    y*(x + 1)*(x + y + 1)
    >>> gcd_terms(x/2 + 1)
    (x + 2)/2
    >>> gcd_terms(x/2 + 1, clear=False)
    x/2 + 1
    >>> gcd_terms(x/2 + y/2, clear=False)
    (x + y)/2
    >>> gcd_terms(x/2 + 1/x)
    (x**2 + 2)/(2*x)
    >>> gcd_terms(x/2 + 1/x, fraction=False)
    (x + 2/x)/2
    >>> gcd_terms(x/2 + 1/x, fraction=False, clear=False)
    x/2 + 1/x

    >>> gcd_terms(x/2/y + 1/x/y)
    (x**2 + 2)/(2*x*y)
    >>> gcd_terms(x/2/y + 1/x/y, clear=False)
    (x**2/2 + 1)/(x*y)
    >>> gcd_terms(x/2/y + 1/x/y, clear=False, fraction=False)
    (x/2 + 1/x)/y

    The ``clear`` flag was ignored in this case because the returned
    expression was a rational expression, not a simple sum.

    See Also
    ========

    factor_terms, sympy.polys.polytools.terms_gcd

    """
def _factor_sum_int(expr, **kwargs):
    """Return Sum or Integral object with factors that are not
    in the wrt variables removed. In cases where there are additive
    terms in the function of the object that are independent, the
    object will be separated into two objects.

    Examples
    ========

    >>> from sympy import Sum, factor_terms
    >>> from sympy.abc import x, y
    >>> factor_terms(Sum(x + y, (x, 1, 3)))
    y*Sum(1, (x, 1, 3)) + Sum(x, (x, 1, 3))
    >>> factor_terms(Sum(x*y, (x, 1, 3)))
    y*Sum(x, (x, 1, 3))

    Notes
    =====

    If a function in the summand or integrand is replaced
    with a symbol, then this simplification should not be
    done or else an incorrect result will be obtained when
    the symbol is replaced with an expression that depends
    on the variables of summation/integration:

    >>> eq = Sum(y, (x, 1, 3))
    >>> factor_terms(eq).subs(y, x).doit()
    3*x
    >>> eq.subs(y, x).doit()
    6
    """
def factor_terms(expr: Expr | complex, radical: bool = False, clear: bool = False, fraction: bool = False, sign: bool = True) -> Expr:
    """Remove common factors from terms in all arguments without
    changing the underlying structure of the expr. No expansion or
    simplification (and no processing of non-commutatives) is performed.

    Parameters
    ==========

    radical: bool, optional
        If radical=True then a radical common to all terms will be factored
        out of any Add sub-expressions of the expr.

    clear : bool, optional
        If clear=False (default) then coefficients will not be separated
        from a single Add if they can be distributed to leave one or more
        terms with integer coefficients.

    fraction : bool, optional
        If fraction=True (default is False) then a common denominator will be
        constructed for the expression.

    sign : bool, optional
        If sign=True (default) then even if the only factor in common is a -1,
        it will be factored out of the expression.

    Examples
    ========

    >>> from sympy import factor_terms, Symbol
    >>> from sympy.abc import x, y
    >>> factor_terms(x + x*(2 + 4*y)**3)
    x*(8*(2*y + 1)**3 + 1)
    >>> A = Symbol('A', commutative=False)
    >>> factor_terms(x*A + x*A + x*y*A)
    x*(y*A + 2*A)

    When ``clear`` is False, a rational will only be factored out of an
    Add expression if all terms of the Add have coefficients that are
    fractions:

    >>> factor_terms(x/2 + 1, clear=False)
    x/2 + 1
    >>> factor_terms(x/2 + 1, clear=True)
    (x + 2)/2

    If a -1 is all that can be factored out, to *not* factor it out, the
    flag ``sign`` must be False:

    >>> factor_terms(-x - y)
    -(x + y)
    >>> factor_terms(-x - y, sign=False)
    -x - y
    >>> factor_terms(-2*x - 2*y, sign=False)
    -2*(x + y)

    See Also
    ========

    gcd_terms, sympy.polys.polytools.terms_gcd

    """
def _mask_nc(eq, name=None):
    """
    Return ``eq`` with non-commutative objects replaced with Dummy
    symbols. A dictionary that can be used to restore the original
    values is returned: if it is None, the expression is noncommutative
    and cannot be made commutative. The third value returned is a list
    of any non-commutative symbols that appear in the returned equation.

    Explanation
    ===========

    All non-commutative objects other than Symbols are replaced with
    a non-commutative Symbol. Identical objects will be identified
    by identical symbols.

    If there is only 1 non-commutative object in an expression it will
    be replaced with a commutative symbol. Otherwise, the non-commutative
    entities are retained and the calling routine should handle
    replacements in this case since some care must be taken to keep
    track of the ordering of symbols when they occur within Muls.

    Parameters
    ==========

    name : str
        ``name``, if given, is the name that will be used with numbered Dummy
        variables that will replace the non-commutative objects and is mainly
        used for doctesting purposes.

    Examples
    ========

    >>> from sympy.physics.secondquant import Commutator, NO, F, Fd
    >>> from sympy import symbols
    >>> from sympy.core.exprtools import _mask_nc
    >>> from sympy.abc import x, y
    >>> A, B, C = symbols('A,B,C', commutative=False)

    One nc-symbol:

    >>> _mask_nc(A**2 - x**2, 'd')
    (_d0**2 - x**2, {_d0: A}, [])

    Multiple nc-symbols:

    >>> _mask_nc(A**2 - B**2, 'd')
    (A**2 - B**2, {}, [A, B])

    An nc-object with nc-symbols but no others outside of it:

    >>> _mask_nc(1 + x*Commutator(A, B), 'd')
    (_d0*x + 1, {_d0: Commutator(A, B)}, [])
    >>> _mask_nc(NO(Fd(x)*F(y)), 'd')
    (_d0, {_d0: NO(CreateFermion(x)*AnnihilateFermion(y))}, [])

    Multiple nc-objects:

    >>> eq = x*Commutator(A, B) + x*Commutator(A, C)*Commutator(A, B)
    >>> _mask_nc(eq, 'd')
    (x*_d0 + x*_d1*_d0, {_d0: Commutator(A, B), _d1: Commutator(A, C)}, [_d0, _d1])

    Multiple nc-objects and nc-symbols:

    >>> eq = A*Commutator(A, B) + B*Commutator(A, C)
    >>> _mask_nc(eq, 'd')
    (A*_d0 + B*_d1, {_d0: Commutator(A, B), _d1: Commutator(A, C)}, [_d0, _d1, A, B])

    """
def factor_nc(expr):
    """Return the factored form of ``expr`` while handling non-commutative
    expressions.

    Examples
    ========

    >>> from sympy import factor_nc, Symbol
    >>> from sympy.abc import x
    >>> A = Symbol('A', commutative=False)
    >>> B = Symbol('B', commutative=False)
    >>> factor_nc((x**2 + 2*A*x + A**2).expand())
    (x + A)**2
    >>> factor_nc(((x + A)*(x + B)).expand())
    (x + A)*(x + B)
    """
