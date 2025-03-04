from .functions import defun as defun, defun_wrapped as defun_wrapped
from _typeshed import Incomplete

def _erf_complex(ctx, z): ...
def _erfc_complex(ctx, z): ...
def erf(ctx, z): ...
def erfc(ctx, z): ...
def square_exp_arg(ctx, z, mult: int = 1, reciprocal: bool = False): ...
def erfi(ctx, z): ...
def erfinv(ctx, x): ...
def npdf(ctx, x, mu: int = 0, sigma: int = 1): ...
def ncdf(ctx, x, mu: int = 0, sigma: int = 1): ...
def betainc(ctx, a, b, x1: int = 0, x2: int = 1, regularized: bool = False): ...
def gammainc(ctx, z, a: int = 0, b: Incomplete | None = None, regularized: bool = False): ...
def _lower_gamma(ctx, z, b, regularized: bool = False): ...
def _upper_gamma(ctx, z, a, regularized: bool = False): ...
def _gamma3(ctx, z, a, b, regularized: bool = False): ...
def expint(ctx, n, z): ...
def li(ctx, z, offset: bool = False): ...
def ei(ctx, z): ...
def _ei_generic(ctx, z): ...
def e1(ctx, z): ...
def ci(ctx, z): ...
def _ci_generic(ctx, z): ...
def si(ctx, z): ...
def _si_generic(ctx, z): ...
def chi(ctx, z): ...
def shi(ctx, z): ...
def fresnels(ctx, z): ...
def fresnelc(ctx, z): ...
