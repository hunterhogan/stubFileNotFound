from .residue_ntheory import _sqrt_mod_prime_power as _sqrt_mod_prime_power, is_quad_residue as is_quad_residue
from sympy.external.gmpy import gcd as gcd, jacobi as jacobi, legendre as legendre

def _pre() -> None: ...
def _a(n, k, prec):
    """ Compute the inner sum in HRR formula [1]_

    References
    ==========

    .. [1] https://msp.org/pjm/1956/6-1/pjm-v6-n1-p18-p.pdf

    """
def _d(n, j, prec, sq23pi, sqrt8):
    """
    Compute the sinh term in the outer sum of the HRR formula.
    The constants sqrt(2/3*pi) and sqrt(8) must be precomputed.
    """
def _partition_rec(n: int, prev) -> int:
    """ Calculate the partition function P(n)

    Parameters
    ==========

    n : int
        nonnegative integer

    """
def _partition(n: int) -> int:
    """ Calculate the partition function P(n)

    Parameters
    ==========

    n : int

    """
def npartitions(n, verbose: bool = False):
    """
    Calculate the partition function P(n), i.e. the number of ways that
    n can be written as a sum of positive integers.

    .. deprecated:: 1.13

        The ``npartitions`` function is deprecated. Use :class:`sympy.functions.combinatorial.numbers.partition`
        instead. See its documentation for more information. See
        :ref:`deprecated-ntheory-symbolic-functions` for details.

    P(n) is computed using the Hardy-Ramanujan-Rademacher formula [1]_.


    The correctness of this implementation has been tested through $10^{10}$.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import partition
    >>> partition(25)
    1958

    References
    ==========

    .. [1] https://mathworld.wolfram.com/PartitionFunctionP.html

    """
