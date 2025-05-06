from _typeshed import Incomplete
from sympy.stats.symbolic_probability import Covariance as Covariance, RandomSymbol as RandomSymbol, Variance as Variance

_arg0_or_var: Incomplete

def variance_prop(expr, consts=(), include_covar: bool = False):
    """Symbolically propagates variance (`\\sigma^2`) for expressions.
    This is computed as as seen in [1]_.

    Parameters
    ==========

    expr : Expr
        A SymPy expression to compute the variance for.
    consts : sequence of Symbols, optional
        Represents symbols that are known constants in the expr,
        and thus have zero variance. All symbols not in consts are
        assumed to be variant.
    include_covar : bool, optional
        Flag for whether or not to include covariances, default=False.

    Returns
    =======

    var_expr : Expr
        An expression for the total variance of the expr.
        The variance for the original symbols (e.g. x) are represented
        via instance of the Variance symbol (e.g. Variance(x)).

    Examples
    ========

    >>> from sympy import symbols, exp
    >>> from sympy.stats.error_prop import variance_prop
    >>> x, y = symbols('x y')

    >>> variance_prop(x + y)
    Variance(x) + Variance(y)

    >>> variance_prop(x * y)
    x**2*Variance(y) + y**2*Variance(x)

    >>> variance_prop(exp(2*x))
    4*exp(4*x)*Variance(x)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Propagation_of_uncertainty

    """
