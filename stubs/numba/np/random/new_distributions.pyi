from numba.core.extending import register_jitable as register_jitable
from numba.np.random._constants import INT64_MAX as INT64_MAX, fe_double as fe_double, fe_float as fe_float, fi_double as fi_double, fi_float as fi_float, ke_double as ke_double, ke_float as ke_float, ki_double as ki_double, ki_float as ki_float, we_double as we_double, we_float as we_float, wi_double as wi_double, wi_float as wi_float, ziggurat_exp_r as ziggurat_exp_r, ziggurat_exp_r_f as ziggurat_exp_r_f, ziggurat_nor_inv_r as ziggurat_nor_inv_r, ziggurat_nor_inv_r_f as ziggurat_nor_inv_r_f, ziggurat_nor_r as ziggurat_nor_r, ziggurat_nor_r_f as ziggurat_nor_r_f
from numba.np.random.generator_core import next_double as next_double, next_float as next_float, next_uint32 as next_uint32, next_uint64 as next_uint64

def np_log1p(x): ...
def np_log1pf(x): ...
def random_rayleigh(bitgen, mode): ...
def np_expm1(x): ...
def random_standard_normal(bitgen): ...
def random_standard_normal_f(bitgen): ...
def random_standard_exponential(bitgen): ...
def random_standard_exponential_f(bitgen): ...
def random_standard_exponential_inv(bitgen): ...
def random_standard_exponential_inv_f(bitgen): ...
def random_standard_gamma(bitgen, shape): ...
def random_standard_gamma_f(bitgen, shape): ...
def random_normal(bitgen, loc, scale): ...
def random_normal_f(bitgen, loc, scale): ...
def random_exponential(bitgen, scale): ...
def random_uniform(bitgen, lower, range): ...
def random_gamma(bitgen, shape, scale): ...
def random_gamma_f(bitgen, shape, scale): ...
def random_beta(bitgen, a, b): ...
def random_chisquare(bitgen, df): ...
def random_f(bitgen, dfnum, dfden): ...
def random_standard_cauchy(bitgen): ...
def random_pareto(bitgen, a): ...
def random_weibull(bitgen, a): ...
def random_power(bitgen, a): ...
def random_laplace(bitgen, loc, scale): ...
def random_logistic(bitgen, loc, scale): ...
def random_lognormal(bitgen, mean, sigma): ...
def random_standard_t(bitgen, df): ...
def random_wald(bitgen, mean, scale): ...
def random_geometric_search(bitgen, p): ...
def random_geometric_inversion(bitgen, p): ...
def random_geometric(bitgen, p): ...
def random_zipf(bitgen, a): ...
def random_triangular(bitgen, left, mode, right): ...
def random_loggam(x): ...
def random_poisson_mult(bitgen, lam): ...
def random_poisson_ptrs(bitgen, lam): ...
def random_poisson(bitgen, lam): ...
def random_negative_binomial(bitgen, n, p): ...
def random_noncentral_chisquare(bitgen, df, nonc): ...
def random_noncentral_f(bitgen, dfnum, dfden, nonc): ...
def random_logseries(bitgen, p): ...
def random_binomial_btpe(bitgen, n, p): ...
def random_binomial_inversion(bitgen, n, p): ...
def random_binomial(bitgen, n, p): ...
