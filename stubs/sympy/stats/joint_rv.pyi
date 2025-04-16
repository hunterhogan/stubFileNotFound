from _typeshed import Incomplete
from sympy.concrete.products import Product as Product
from sympy.concrete.summations import Sum as Sum, summation as summation
from sympy.core.basic import Basic as Basic
from sympy.core.containers import Tuple as Tuple
from sympy.core.function import Lambda as Lambda
from sympy.core.singleton import S as S
from sympy.core.symbol import Dummy as Dummy, Symbol as Symbol
from sympy.core.sympify import sympify as sympify
from sympy.external import import_module as import_module
from sympy.integrals.integrals import Integral as Integral, integrate as integrate
from sympy.matrices import ImmutableMatrix as ImmutableMatrix, list2numpy as list2numpy, matrix2numpy as matrix2numpy
from sympy.sets.sets import ProductSet as ProductSet
from sympy.stats.crv import SingleContinuousDistribution as SingleContinuousDistribution, SingleContinuousPSpace as SingleContinuousPSpace
from sympy.stats.drv import SingleDiscreteDistribution as SingleDiscreteDistribution, SingleDiscretePSpace as SingleDiscretePSpace
from sympy.stats.rv import Distribution as Distribution, NamedArgsMixin as NamedArgsMixin, ProductDomain as ProductDomain, ProductPSpace as ProductPSpace, RandomSymbol as RandomSymbol, SingleDomain as SingleDomain, _symbol_converter as _symbol_converter, random_symbols as random_symbols
from sympy.tensor.indexed import Indexed as Indexed
from sympy.utilities.iterables import iterable as iterable
from sympy.utilities.misc import filldedent as filldedent

class JointPSpace(ProductPSpace):
    """
    Represents a joint probability space. Represented using symbols for
    each component and a distribution.
    """
    def __new__(cls, sym, dist): ...
    @property
    def set(self): ...
    @property
    def symbol(self): ...
    @property
    def distribution(self): ...
    @property
    def value(self): ...
    @property
    def component_count(self): ...
    @property
    def pdf(self): ...
    @property
    def domain(self): ...
    def component_domain(self, index): ...
    def marginal_distribution(self, *indices): ...
    def compute_expectation(self, expr, rvs: Incomplete | None = None, evaluate: bool = False, **kwargs): ...
    def where(self, condition) -> None: ...
    def compute_density(self, expr) -> None: ...
    def sample(self, size=(), library: str = 'scipy', seed: Incomplete | None = None):
        """
        Internal sample method

        Returns dictionary mapping RandomSymbol to realization value.
        """
    def probability(self, condition) -> None: ...

class SampleJointScipy:
    """Returns the sample from scipy of the given distribution"""
    def __new__(cls, dist, size, seed: Incomplete | None = None): ...
    @classmethod
    def _sample_scipy(cls, dist, size, seed):
        """Sample from SciPy."""

class SampleJointNumpy:
    """Returns the sample from numpy of the given distribution"""
    def __new__(cls, dist, size, seed: Incomplete | None = None): ...
    @classmethod
    def _sample_numpy(cls, dist, size, seed):
        """Sample from NumPy."""

class SampleJointPymc:
    """Returns the sample from pymc of the given distribution"""
    def __new__(cls, dist, size, seed: Incomplete | None = None): ...
    @classmethod
    def _sample_pymc(cls, dist, size, seed):
        """Sample from PyMC."""

_get_sample_class_jrv: Incomplete

class JointDistribution(Distribution, NamedArgsMixin):
    """
    Represented by the random variables part of the joint distribution.
    Contains methods for PDF, CDF, sampling, marginal densities, etc.
    """
    _argnames: Incomplete
    def __new__(cls, *args): ...
    @property
    def domain(self): ...
    @property
    def pdf(self): ...
    def cdf(self, other): ...
    def sample(self, size=(), library: str = 'scipy', seed: Incomplete | None = None):
        """ A random realization from the distribution """
    def __call__(self, *args): ...

class JointRandomSymbol(RandomSymbol):
    '''
    Representation of random symbols with joint probability distributions
    to allow indexing."
    '''
    def __getitem__(self, key): ...

class MarginalDistribution(Distribution):
    """
    Represents the marginal distribution of a joint probability space.

    Initialised using a probability distribution and random variables(or
    their indexed components) which should be a part of the resultant
    distribution.
    """
    def __new__(cls, dist, *rvs): ...
    def check(self) -> None: ...
    @property
    def set(self): ...
    @property
    def symbols(self): ...
    def pdf(self, *x): ...
    def compute_pdf(self, expr, rvs): ...
    def marginalise_out(self, expr, rv): ...
    def __call__(self, *args): ...
