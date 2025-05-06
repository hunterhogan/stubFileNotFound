from _typeshed import Incomplete
from sympy.core.expr import Expr as Expr
from sympy.core.function import AppliedUndef as AppliedUndef, UndefinedFunction as UndefinedFunction
from sympy.core.relational import Equality as Equality, Relational as Relational
from sympy.core.symbol import Dummy as Dummy, Symbol as Symbol
from sympy.functions.elementary.piecewise import Piecewise as Piecewise, piecewise_fold as piecewise_fold
from sympy.sets.sets import Interval as Interval, Set as Set
from sympy.utilities.iterables import is_sequence as is_sequence, sift as sift

def _common_new(cls, function, *symbols, discrete, **assumptions):
    """Return either a special return value or the tuple,
    (function, limits, orientation). This code is common to
    both ExprWithLimits and AddWithLimits."""
def _process_limits(*symbols, discrete: Incomplete | None = None):
    """Process the list of symbols and convert them to canonical limits,
    storing them as Tuple(symbol, lower, upper). The orientation of
    the function is also returned when the upper limit is missing
    so (x, 1, None) becomes (x, None, 1) and the orientation is changed.
    In the case that a limit is specified as (symbol, Range), a list of
    length 4 may be returned if a change of variables is needed; the
    expression that should replace the symbol in the expression is
    the fourth element in the list.
    """

class ExprWithLimits(Expr):
    __slots__: Incomplete
    def __new__(cls, function, *symbols, **assumptions): ...
    @property
    def function(self):
        """Return the function applied across limits.

        Examples
        ========

        >>> from sympy import Integral
        >>> from sympy.abc import x
        >>> Integral(x**2, (x,)).function
        x**2

        See Also
        ========

        limits, variables, free_symbols
        """
    @property
    def kind(self): ...
    @property
    def limits(self):
        """Return the limits of expression.

        Examples
        ========

        >>> from sympy import Integral
        >>> from sympy.abc import x, i
        >>> Integral(x**i, (i, 1, 3)).limits
        ((i, 1, 3),)

        See Also
        ========

        function, variables, free_symbols
        """
    @property
    def variables(self):
        """Return a list of the limit variables.

        >>> from sympy import Sum
        >>> from sympy.abc import x, i
        >>> Sum(x**i, (i, 1, 3)).variables
        [i]

        See Also
        ========

        function, limits, free_symbols
        as_dummy : Rename dummy variables
        sympy.integrals.integrals.Integral.transform : Perform mapping on the dummy variable
        """
    @property
    def bound_symbols(self):
        """Return only variables that are dummy variables.

        Examples
        ========

        >>> from sympy import Integral
        >>> from sympy.abc import x, i, j, k
        >>> Integral(x**i, (i, 1, 3), (j, 2), k).bound_symbols
        [i, j]

        See Also
        ========

        function, limits, free_symbols
        as_dummy : Rename dummy variables
        sympy.integrals.integrals.Integral.transform : Perform mapping on the dummy variable
        """
    @property
    def free_symbols(self):
        """
        This method returns the symbols in the object, excluding those
        that take on a specific value (i.e. the dummy symbols).

        Examples
        ========

        >>> from sympy import Sum
        >>> from sympy.abc import x, y
        >>> Sum(x, (x, y, 1)).free_symbols
        {y}
        """
    @property
    def is_number(self):
        """Return True if the Sum has no free symbols, else False."""
    def _eval_interval(self, x, a, b): ...
    def _eval_subs(self, old, new):
        """
        Perform substitutions over non-dummy variables
        of an expression with limits.  Also, can be used
        to specify point-evaluation of an abstract antiderivative.

        Examples
        ========

        >>> from sympy import Sum, oo
        >>> from sympy.abc import s, n
        >>> Sum(1/n**s, (n, 1, oo)).subs(s, 2)
        Sum(n**(-2), (n, 1, oo))

        >>> from sympy import Integral
        >>> from sympy.abc import x, a
        >>> Integral(a*x**2, x).subs(x, 4)
        Integral(a*x**2, (x, 4))

        See Also
        ========

        variables : Lists the integration variables
        transform : Perform mapping on the dummy variable for integrals
        change_index : Perform mapping on the sum and product dummy variables

        """
    @property
    def has_finite_limits(self):
        """
        Returns True if the limits are known to be finite, either by the
        explicit bounds, assumptions on the bounds, or assumptions on the
        variables.  False if known to be infinite, based on the bounds.
        None if not enough information is available to determine.

        Examples
        ========

        >>> from sympy import Sum, Integral, Product, oo, Symbol
        >>> x = Symbol('x')
        >>> Sum(x, (x, 1, 8)).has_finite_limits
        True

        >>> Integral(x, (x, 1, oo)).has_finite_limits
        False

        >>> M = Symbol('M')
        >>> Sum(x, (x, 1, M)).has_finite_limits

        >>> N = Symbol('N', integer=True)
        >>> Product(x, (x, 1, N)).has_finite_limits
        True

        See Also
        ========

        has_reversed_limits

        """
    @property
    def has_reversed_limits(self):
        """
        Returns True if the limits are known to be in reversed order, either
        by the explicit bounds, assumptions on the bounds, or assumptions on the
        variables.  False if known to be in normal order, based on the bounds.
        None if not enough information is available to determine.

        Examples
        ========

        >>> from sympy import Sum, Integral, Product, oo, Symbol
        >>> x = Symbol('x')
        >>> Sum(x, (x, 8, 1)).has_reversed_limits
        True

        >>> Sum(x, (x, 1, oo)).has_reversed_limits
        False

        >>> M = Symbol('M')
        >>> Integral(x, (x, 1, M)).has_reversed_limits

        >>> N = Symbol('N', integer=True, positive=True)
        >>> Sum(x, (x, 1, N)).has_reversed_limits
        False

        >>> Product(x, (x, 2, N)).has_reversed_limits

        >>> Product(x, (x, 2, N)).subs(N, N + 2).has_reversed_limits
        False

        See Also
        ========

        sympy.concrete.expr_with_intlimits.ExprWithIntLimits.has_empty_sequence

        """

class AddWithLimits(ExprWithLimits):
    """Represents unevaluated oriented additions.
        Parent class for Integral and Sum.
    """
    __slots__: Incomplete
    def __new__(cls, function, *symbols, **assumptions): ...
    def _eval_adjoint(self): ...
    def _eval_conjugate(self): ...
    def _eval_transpose(self): ...
    def _eval_factor(self, **hints): ...
    def _eval_expand_basic(self, **hints): ...
