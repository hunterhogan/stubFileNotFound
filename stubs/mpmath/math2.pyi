import math
from _typeshed import Incomplete

pi: float
e: float
sqrt2: float
sqrt5: float
phi: float
ln2: float
ln10: float
euler: float
catalan: float
khinchin: float
apery: float
logpi: float

def _mathfun_real(f_real, f_complex): ...
def _mathfun(f_real, f_complex): ...
def _mathfun_n(f_real, f_complex): ...
def math_log(x): ...
def math_sqrt(x): ...
math_log = math.log
math_sqrt = math.sqrt
pow: Incomplete
log: Incomplete
sqrt: Incomplete
exp: Incomplete
cos: Incomplete
sin: Incomplete
tan: Incomplete
acos: Incomplete
asin: Incomplete
atan: Incomplete
cosh: Incomplete
sinh: Incomplete
tanh: Incomplete
floor: Incomplete
ceil: Incomplete
cos_sin: Incomplete
cbrt: Incomplete

def nthroot(x, n): ...
def _sinpi_real(x): ...
def _cospi_real(x): ...
def _sinpi_complex(z): ...
def _cospi_complex(z): ...

cospi: Incomplete
sinpi: Incomplete

def tanpi(x): ...
def cotpi(x): ...

INF: Incomplete
NINF: Incomplete
NAN: Incomplete
EPS: float
_exact_gamma: Incomplete
_max_exact_gamma: Incomplete
_lanczos_g: int
_lanczos_p: Incomplete

def _gamma_real(x): ...
def _gamma_complex(x): ...

gamma: Incomplete

def rgamma(x): ...
def factorial(x): ...
def arg(x): ...
def loggamma(x): ...

_psi_coeff: Incomplete

def _digamma_real(x): ...
def _digamma_complex(x): ...

digamma: Incomplete
_erfc_coeff_P: Incomplete
_erfc_coeff_Q: Incomplete

def _polyval(coeffs, x): ...
def _erf_taylor(x): ...
def _erfc_mid(x): ...
def _erfc_asymp(x): ...
def erf(x):
    """
    erf of a real number.
    """
def erfc(x):
    """
    erfc of a real number.
    """

gauss42: Incomplete
EI_ASYMP_CONVERGENCE_RADIUS: float

def ei_asymp(z, _e1: bool = False): ...
def ei_taylor(z, _e1: bool = False): ...
def ei(z, _e1: bool = False): ...
def e1(z): ...

_zeta_int: Incomplete
_zeta_P: Incomplete
_zeta_Q: Incomplete
_zeta_1: Incomplete
_zeta_0: Incomplete

def zeta(s):
    """
    Riemann zeta function, real argument
    """
