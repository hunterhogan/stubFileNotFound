from _typeshed import Incomplete
from sympy.stats.crv import SingleContinuousDistribution as SingleContinuousDistribution
from sympy.stats.crv_types import BetaDistribution as BetaDistribution, CauchyDistribution as CauchyDistribution, ChiSquaredDistribution as ChiSquaredDistribution, ExponentialDistribution as ExponentialDistribution, GammaDistribution as GammaDistribution, LogNormalDistribution as LogNormalDistribution, NormalDistribution as NormalDistribution, ParetoDistribution as ParetoDistribution, StudentTDistribution as StudentTDistribution, UniformDistribution as UniformDistribution
from sympy.stats.drv_types import GeometricDistribution as GeometricDistribution, LogarithmicDistribution as LogarithmicDistribution, NegativeBinomialDistribution as NegativeBinomialDistribution, PoissonDistribution as PoissonDistribution, SkellamDistribution as SkellamDistribution, YuleSimonDistribution as YuleSimonDistribution, ZetaDistribution as ZetaDistribution

scipy: Incomplete

def do_sample_scipy(dist, size, seed) -> None: ...
def _(dist: SingleContinuousDistribution, size, seed): ...
