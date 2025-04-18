from .backend import xrange as xrange
from .gammazeta import mpc_loggamma as mpc_loggamma, mpf_gamma as mpf_gamma, mpf_loggamma as mpf_loggamma, mpf_rgamma as mpf_rgamma
from .libelefun import mod_pi2 as mod_pi2, mpf_atan as mpf_atan, mpf_atan2 as mpf_atan2, mpf_cos_sin as mpf_cos_sin, mpf_exp as mpf_exp, mpf_log as mpf_log, mpf_pi as mpf_pi, mpf_sqrt as mpf_sqrt
from .libmpf import ComplexResult as ComplexResult, MPZ_ONE as MPZ_ONE, bitcount as bitcount, dps_to_prec as dps_to_prec, fhalf as fhalf, finf as finf, fnan as fnan, fninf as fninf, fnone as fnone, fone as fone, from_float as from_float, from_int as from_int, from_man_exp as from_man_exp, from_str as from_str, fzero as fzero, mpf_abs as mpf_abs, mpf_add as mpf_add, mpf_cmp as mpf_cmp, mpf_div as mpf_div, mpf_eq as mpf_eq, mpf_floor as mpf_floor, mpf_ge as mpf_ge, mpf_gt as mpf_gt, mpf_le as mpf_le, mpf_lt as mpf_lt, mpf_min_max as mpf_min_max, mpf_mul as mpf_mul, mpf_mul_int as mpf_mul_int, mpf_neg as mpf_neg, mpf_pos as mpf_pos, mpf_pow_int as mpf_pow_int, mpf_shift as mpf_shift, mpf_sign as mpf_sign, mpf_sub as mpf_sub, prec_to_dps as prec_to_dps, repr_dps as repr_dps, round_ceiling as round_ceiling, round_down as round_down, round_floor as round_floor, round_nearest as round_nearest, round_up as round_up, to_int as to_int, to_str as to_str
from _typeshed import Incomplete

def mpi_str(s, prec): ...

mpi_zero: Incomplete
mpi_one: Incomplete

def mpi_eq(s, t): ...
def mpi_ne(s, t): ...
def mpi_lt(s, t): ...
def mpi_le(s, t): ...
def mpi_gt(s, t): ...
def mpi_ge(s, t): ...
def mpi_add(s, t, prec: int = 0): ...
def mpi_sub(s, t, prec: int = 0): ...
def mpi_delta(s, prec): ...
def mpi_mid(s, prec): ...
def mpi_pos(s, prec): ...
def mpi_neg(s, prec: int = 0): ...
def mpi_abs(s, prec: int = 0): ...
def mpi_mul_mpf(s, t, prec): ...
def mpi_div_mpf(s, t, prec): ...
def mpi_mul(s, t, prec: int = 0): ...
def mpi_square(s, prec: int = 0): ...
def mpi_div(s, t, prec): ...
def mpi_pi(prec): ...
def mpi_exp(s, prec): ...
def mpi_log(s, prec): ...
def mpi_sqrt(s, prec): ...
def mpi_atan(s, prec): ...
def mpi_pow_int(s, n, prec): ...
def mpi_pow(s, t, prec): ...
def MIN(x, y): ...
def MAX(x, y): ...
def cos_sin_quadrant(x, wp): ...
def mpi_cos_sin(x, prec): ...
def mpi_cos(x, prec): ...
def mpi_sin(x, prec): ...
def mpi_tan(x, prec): ...
def mpi_cot(x, prec): ...
def mpi_from_str_a_b(x, y, percent, prec): ...
def mpi_from_str(s, prec):
    '''
    Parse an interval number given as a string.

    Allowed forms are

    "-1.23e-27"
        Any single decimal floating-point literal.
    "a +- b"  or  "a (b)"
        a is the midpoint of the interval and b is the half-width
    "a +- b%"  or  "a (b%)"
        a is the midpoint of the interval and the half-width
        is b percent of a (`a \times b / 100`).
    "[a, b]"
        The interval indicated directly.
    "x[y,z]e"
        x are shared digits, y and z are unequal digits, e is the exponent.

    '''
def mpi_to_str(x, dps, use_spaces: bool = True, brackets: str = '[]', mode: str = 'brackets', error_dps: int = 4, **kwargs):
    """
    Convert a mpi interval to a string.

    **Arguments**

    *dps*
        decimal places to use for printing
    *use_spaces*
        use spaces for more readable output, defaults to true
    *brackets*
        pair of strings (or two-character string) giving left and right brackets
    *mode*
        mode of display: 'plusminus', 'percent', 'brackets' (default) or 'diff'
    *error_dps*
        limit the error to *error_dps* digits (mode 'plusminus and 'percent')

    Additional keyword arguments are forwarded to the mpf-to-string conversion
    for the components of the output.

    **Examples**

        >>> from mpmath import mpi, mp
        >>> mp.dps = 30
        >>> x = mpi(1, 2)._mpi_
        >>> mpi_to_str(x, 2, mode='plusminus')
        '1.5 +- 0.5'
        >>> mpi_to_str(x, 2, mode='percent')
        '1.5 (33.33%)'
        >>> mpi_to_str(x, 2, mode='brackets')
        '[1.0, 2.0]'
        >>> mpi_to_str(x, 2, mode='brackets' , brackets=('<', '>'))
        '<1.0, 2.0>'
        >>> x = mpi('5.2582327113062393041', '5.2582327113062749951')._mpi_
        >>> mpi_to_str(x, 15, mode='diff')
        '5.2582327113062[4, 7]'
        >>> mpi_to_str(mpi(0)._mpi_, 2, mode='percent')
        '0.0 (0.0%)'

    """
def mpci_add(x, y, prec): ...
def mpci_sub(x, y, prec): ...
def mpci_neg(x, prec: int = 0): ...
def mpci_pos(x, prec): ...
def mpci_mul(x, y, prec): ...
def mpci_div(x, y, prec): ...
def mpci_exp(x, prec): ...
def mpi_shift(x, n): ...
def mpi_cosh_sinh(x, prec): ...
def mpci_cos(x, prec): ...
def mpci_sin(x, prec): ...
def mpci_abs(x, prec): ...
def mpi_atan2(y, x, prec): ...
def mpci_arg(z, prec): ...
def mpci_log(z, prec): ...
def mpci_pow(x, y, prec): ...
def mpci_square(x, prec): ...
def mpci_pow_int(x, n, prec): ...

gamma_min_a: Incomplete
gamma_min_b: Incomplete
gamma_min: Incomplete
gamma_mono_imag_a: Incomplete
gamma_mono_imag_b: Incomplete

def mpi_overlap(x, y): ...
def mpi_gamma(z, prec, type: int = 0): ...
def mpci_gamma(z, prec, type: int = 0): ...
def mpi_loggamma(z, prec): ...
def mpci_loggamma(z, prec): ...
def mpi_rgamma(z, prec): ...
def mpci_rgamma(z, prec): ...
def mpi_factorial(z, prec): ...
def mpci_factorial(z, prec): ...
