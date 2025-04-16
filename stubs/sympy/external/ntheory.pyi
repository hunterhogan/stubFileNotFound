import math
from _typeshed import Incomplete

_small_trailing: Incomplete

def bit_scan1(x, n: int = 0): ...
def bit_scan0(x, n: int = 0): ...
def remove(x, f): ...
def factorial(x):
    """Return x!."""
def sqrt(x):
    """Integer square root of x."""
def sqrtrem(x):
    """Integer square root of x and remainder."""
gcd = math.gcd
lcm = math.lcm

def _sign(n): ...
def gcdext(a, b): ...
def is_square(x):
    """Return True if x is a square number."""
def invert(x, m):
    """Modular inverse of x modulo m.

    Returns y such that x*y == 1 mod m.

    Uses ``math.pow`` but reproduces the behaviour of ``gmpy2.invert``
    which raises ZeroDivisionError if no inverse exists.
    """
def legendre(x, y):
    """Legendre symbol (x / y).

    Following the implementation of gmpy2,
    the error is raised only when y is an even number.
    """
def jacobi(x, y):
    """Jacobi symbol (x / y)."""
def kronecker(x, y):
    """Kronecker symbol (x / y)."""
def iroot(y, n): ...
def is_fermat_prp(n, a): ...
def is_euler_prp(n, a): ...
def _is_strong_prp(n, a): ...
def is_strong_prp(n, a): ...
def _lucas_sequence(n, P, Q, k):
    """Return the modular Lucas sequence (U_k, V_k, Q_k).

    Explanation
    ===========

    Given a Lucas sequence defined by P, Q, returns the kth values for
    U and V, along with Q^k, all modulo n. This is intended for use with
    possibly very large values of n and k, where the combinatorial functions
    would be completely unusable.

    .. math ::
        U_k = \\begin{cases}
             0 & \\text{if } k = 0\\\\\n             1 & \\text{if } k = 1\\\\\n             PU_{k-1} - QU_{k-2} & \\text{if } k > 1
        \\end{cases}\\\\\n        V_k = \\begin{cases}
             2 & \\text{if } k = 0\\\\\n             P & \\text{if } k = 1\\\\\n             PV_{k-1} - QV_{k-2} & \\text{if } k > 1
        \\end{cases}

    The modular Lucas sequences are used in numerous places in number theory,
    especially in the Lucas compositeness tests and the various n + 1 proofs.

    Parameters
    ==========

    n : int
        n is an odd number greater than or equal to 3
    P : int
    Q : int
        D determined by D = P**2 - 4*Q is non-zero
    k : int
        k is a nonnegative integer

    Returns
    =======

    U, V, Qk : (int, int, int)
        `(U_k \\bmod{n}, V_k \\bmod{n}, Q^k \\bmod{n})`

    Examples
    ========

    >>> from sympy.external.ntheory import _lucas_sequence
    >>> N = 10**2000 + 4561
    >>> sol = U, V, Qk = _lucas_sequence(N, 3, 1, N//2); sol
    (0, 2, 1)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Lucas_sequence

    """
def is_fibonacci_prp(n, p, q): ...
def is_lucas_prp(n, p, q): ...
def _is_selfridge_prp(n):
    '''Lucas compositeness test with the Selfridge parameters for n.

    Explanation
    ===========

    The Lucas compositeness test checks whether n is a prime number.
    The test can be run with arbitrary parameters ``P`` and ``Q``, which also change the performance of the test.
    So, which parameters are most effective for running the Lucas compositeness test?
    As an algorithm for determining ``P`` and ``Q``, Selfridge proposed method A [1]_ page 1401
    (Since two methods were proposed, referred to simply as A and B in the paper,
    we will refer to one of them as "method A").

    method A fixes ``P = 1``. Then, ``D`` defined by ``D = P**2 - 4Q`` is varied from 5, -7, 9, -11, 13, and so on,
    with the first ``D`` being ``jacobi(D, n) == -1``. Once ``D`` is determined,
    ``Q`` is determined to be ``(P**2 - D)//4``.

    References
    ==========

    .. [1] Robert Baillie, Samuel S. Wagstaff, Lucas Pseudoprimes,
           Math. Comp. Vol 35, Number 152 (1980), pp. 1391-1417,
           https://doi.org/10.1090%2FS0025-5718-1980-0583518-6
           http://mpqs.free.fr/LucasPseudoprimes.pdf

    '''
def is_selfridge_prp(n): ...
def is_strong_lucas_prp(n, p, q): ...
def _is_strong_selfridge_prp(n): ...
def is_strong_selfridge_prp(n): ...
def is_bpsw_prp(n): ...
def is_strong_bpsw_prp(n): ...
