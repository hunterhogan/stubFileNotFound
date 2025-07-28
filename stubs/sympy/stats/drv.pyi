from _typeshed import Incomplete
from sympy.concrete.summations import Sum as Sum, summation as summation
from sympy.core.basic import Basic as Basic
from sympy.core.cache import cacheit as cacheit
from sympy.core.function import Lambda as Lambda
from sympy.core.numbers import I as I
from sympy.core.relational import Eq as Eq, Ne as Ne
from sympy.core.singleton import S as S
from sympy.core.symbol import Dummy as Dummy, symbols as symbols
from sympy.core.sympify import _sympify as _sympify, sympify as sympify
from sympy.functions.combinatorial.factorials import factorial as factorial
from sympy.functions.elementary.exponential import exp as exp
from sympy.functions.elementary.integers import floor as floor
from sympy.functions.elementary.piecewise import Piecewise as Piecewise
from sympy.logic.boolalg import And as And
from sympy.polys.polyerrors import PolynomialError as PolynomialError
from sympy.polys.polytools import poly as poly
from sympy.series.series import series as series
from sympy.sets.contains import Contains as Contains
from sympy.sets.fancysets import FiniteSet as FiniteSet, Range as Range
from sympy.sets.sets import Union as Union
from sympy.stats.crv import reduce_rational_inequalities_wrap as reduce_rational_inequalities_wrap
from sympy.stats.rv import ConditionalDomain as ConditionalDomain, Distribution as Distribution, NamedArgsMixin as NamedArgsMixin, PSpace as PSpace, ProductDomain as ProductDomain, RandomDomain as RandomDomain, SingleDomain as SingleDomain, SinglePSpace as SinglePSpace, random_symbols as random_symbols
from sympy.stats.symbolic_probability import Probability as Probability
from sympy.utilities import filldedent as filldedent

class DiscreteDistribution(Distribution):
    def __call__(self, *args): ...

class SingleDiscreteDistribution(DiscreteDistribution, NamedArgsMixin):
    """ Discrete distribution of a single variable.

    Serves as superclass for PoissonDistribution etc....

    Provides methods for pdf, cdf, and sampling

    See Also:
        sympy.stats.crv_types.*
    """
    set: Incomplete
    def __new__(cls, *args): ...
    @staticmethod
    def check(*args) -> None: ...
    @cacheit
    def compute_cdf(self, **kwargs):
        """ Compute the CDF from the PDF.

        Returns a Lambda.
        """
    def _cdf(self, x) -> None: ...
    def cdf(self, x, **kwargs):
        """ Cumulative density function """
    @cacheit
    def compute_characteristic_function(self, **kwargs):
        """ Compute the characteristic function from the PDF.

        Returns a Lambda.
        """
    def _characteristic_function(self, t) -> None: ...
    def characteristic_function(self, t, **kwargs):
        """ Characteristic function """
    @cacheit
    def compute_moment_generating_function(self, **kwargs): ...
    def _moment_generating_function(self, t) -> None: ...
    def moment_generating_function(self, t, **kwargs): ...
    @cacheit
    def compute_quantile(self, **kwargs):
        """ Compute the Quantile from the PDF.

        Returns a Lambda.
        """
    def _quantile(self, x) -> None: ...
    def quantile(self, x, **kwargs):
        """ Cumulative density function """
    def expectation(self, expr, var, evaluate: bool = True, **kwargs):
        """ Expectation of expression over distribution """
    def __call__(self, *args): ...

class DiscreteDomain(RandomDomain):
    """
    A domain with discrete support with step size one.
    Represented using symbols and Range.
    """
    is_Discrete: bool

class SingleDiscreteDomain(DiscreteDomain, SingleDomain):
    def as_boolean(self): ...

class ConditionalDiscreteDomain(DiscreteDomain, ConditionalDomain):
    """
    Domain with discrete support of step size one, that is restricted by
    some condition.
    """
    @property
    def set(self): ...

class DiscretePSpace(PSpace):
    is_real: bool
    is_Discrete: bool
    @property
    def pdf(self): ...
    def where(self, condition): ...
    def probability(self, condition): ...
    def eval_prob(self, _domain): ...
    def conditional_space(self, condition): ...

class ProductDiscreteDomain(ProductDomain, DiscreteDomain):
    def as_boolean(self): ...

class SingleDiscretePSpace(DiscretePSpace, SinglePSpace):
    """ Discrete probability space over a single univariate variable """
    is_real: bool
    @property
    def set(self): ...
    @property
    def domain(self): ...
    def sample(self, size=(), library: str = 'scipy', seed=None):
        """
        Internal sample method.

        Returns dictionary mapping RandomSymbol to realization value.
        """
    def compute_expectation(self, expr, rvs=None, evaluate: bool = True, **kwargs): ...
    def compute_cdf(self, expr, **kwargs): ...
    def compute_density(self, expr, **kwargs): ...
    def compute_characteristic_function(self, expr, **kwargs): ...
    def compute_moment_generating_function(self, expr, **kwargs): ...
    def compute_quantile(self, expr, **kwargs): ...
