from _typeshed import Incomplete
from sympy.core import Rational as Rational, S as S, sympify as sympify
from sympy.core.function import expand_mul as expand_mul
from sympy.discrete.transforms import fft as fft, fwht as fwht, ifft as ifft, ifwht as ifwht, intt as intt, inverse_mobius_transform as inverse_mobius_transform, mobius_transform as mobius_transform, ntt as ntt
from sympy.external.gmpy import MPZ as MPZ, lcm as lcm
from sympy.utilities.iterables import iterable as iterable
from sympy.utilities.misc import as_int as as_int

def convolution(a, b, cycle: int = 0, dps: Incomplete | None = None, prime: Incomplete | None = None, dyadic: Incomplete | None = None, subset: Incomplete | None = None):
    """
    Performs convolution by determining the type of desired
    convolution using hints.

    Exactly one of ``dps``, ``prime``, ``dyadic``, ``subset`` arguments
    should be specified explicitly for identifying the type of convolution,
    and the argument ``cycle`` can be specified optionally.

    For the default arguments, linear convolution is performed using **FFT**.

    Parameters
    ==========

    a, b : iterables
        The sequences for which convolution is performed.
    cycle : Integer
        Specifies the length for doing cyclic convolution.
    dps : Integer
        Specifies the number of decimal digits for precision for
        performing **FFT** on the sequence.
    prime : Integer
        Prime modulus of the form `(m 2^k + 1)` to be used for
        performing **NTT** on the sequence.
    dyadic : bool
        Identifies the convolution type as dyadic (*bitwise-XOR*)
        convolution, which is performed using **FWHT**.
    subset : bool
        Identifies the convolution type as subset convolution.

    Examples
    ========

    >>> from sympy import convolution, symbols, S, I
    >>> u, v, w, x, y, z = symbols('u v w x y z')

    >>> convolution([1 + 2*I, 4 + 3*I], [S(5)/4, 6], dps=3)
    [1.25 + 2.5*I, 11.0 + 15.8*I, 24.0 + 18.0*I]
    >>> convolution([1, 2, 3], [4, 5, 6], cycle=3)
    [31, 31, 28]

    >>> convolution([111, 777], [888, 444], prime=19*2**10 + 1)
    [1283, 19351, 14219]
    >>> convolution([111, 777], [888, 444], prime=19*2**10 + 1, cycle=2)
    [15502, 19351]

    >>> convolution([u, v], [x, y, z], dyadic=True)
    [u*x + v*y, u*y + v*x, u*z, v*z]
    >>> convolution([u, v], [x, y, z], dyadic=True, cycle=2)
    [u*x + u*z + v*y, u*y + v*x + v*z]

    >>> convolution([u, v, w], [x, y, z], subset=True)
    [u*x, u*y + v*x, u*z + w*x, v*z + w*y]
    >>> convolution([u, v, w], [x, y, z], subset=True, cycle=3)
    [u*x + v*z + w*y, u*y + v*x, u*z + w*x]

    """
def convolution_fft(a, b, dps: Incomplete | None = None):
    """
    Performs linear convolution using Fast Fourier Transform.

    Parameters
    ==========

    a, b : iterables
        The sequences for which convolution is performed.
    dps : Integer
        Specifies the number of decimal digits for precision.

    Examples
    ========

    >>> from sympy import S, I
    >>> from sympy.discrete.convolutions import convolution_fft

    >>> convolution_fft([2, 3], [4, 5])
    [8, 22, 15]
    >>> convolution_fft([2, 5], [6, 7, 3])
    [12, 44, 41, 15]
    >>> convolution_fft([1 + 2*I, 4 + 3*I], [S(5)/4, 6])
    [5/4 + 5*I/2, 11 + 63*I/4, 24 + 18*I]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Convolution_theorem
    .. [2] https://en.wikipedia.org/wiki/Discrete_Fourier_transform_(general%29

    """
def convolution_ntt(a, b, prime):
    """
    Performs linear convolution using Number Theoretic Transform.

    Parameters
    ==========

    a, b : iterables
        The sequences for which convolution is performed.
    prime : Integer
        Prime modulus of the form `(m 2^k + 1)` to be used for performing
        **NTT** on the sequence.

    Examples
    ========

    >>> from sympy.discrete.convolutions import convolution_ntt
    >>> convolution_ntt([2, 3], [4, 5], prime=19*2**10 + 1)
    [8, 22, 15]
    >>> convolution_ntt([2, 5], [6, 7, 3], prime=19*2**10 + 1)
    [12, 44, 41, 15]
    >>> convolution_ntt([333, 555], [222, 666], prime=19*2**10 + 1)
    [15555, 14219, 19404]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Convolution_theorem
    .. [2] https://en.wikipedia.org/wiki/Discrete_Fourier_transform_(general%29

    """
def convolution_fwht(a, b):
    """
    Performs dyadic (*bitwise-XOR*) convolution using Fast Walsh Hadamard
    Transform.

    The convolution is automatically padded to the right with zeros, as the
    *radix-2 FWHT* requires the number of sample points to be a power of 2.

    Parameters
    ==========

    a, b : iterables
        The sequences for which convolution is performed.

    Examples
    ========

    >>> from sympy import symbols, S, I
    >>> from sympy.discrete.convolutions import convolution_fwht

    >>> u, v, x, y = symbols('u v x y')
    >>> convolution_fwht([u, v], [x, y])
    [u*x + v*y, u*y + v*x]

    >>> convolution_fwht([2, 3], [4, 5])
    [23, 22]
    >>> convolution_fwht([2, 5 + 4*I, 7], [6*I, 7, 3 + 4*I])
    [56 + 68*I, -10 + 30*I, 6 + 50*I, 48 + 32*I]

    >>> convolution_fwht([S(33)/7, S(55)/6, S(7)/4], [S(2)/3, 5])
    [2057/42, 1870/63, 7/6, 35/4]

    References
    ==========

    .. [1] https://www.radioeng.cz/fulltexts/2002/02_03_40_42.pdf
    .. [2] https://en.wikipedia.org/wiki/Hadamard_transform

    """
def convolution_subset(a, b):
    """
    Performs Subset Convolution of given sequences.

    The indices of each argument, considered as bit strings, correspond to
    subsets of a finite set.

    The sequence is automatically padded to the right with zeros, as the
    definition of subset based on bitmasks (indices) requires the size of
    sequence to be a power of 2.

    Parameters
    ==========

    a, b : iterables
        The sequences for which convolution is performed.

    Examples
    ========

    >>> from sympy import symbols, S
    >>> from sympy.discrete.convolutions import convolution_subset
    >>> u, v, x, y, z = symbols('u v x y z')

    >>> convolution_subset([u, v], [x, y])
    [u*x, u*y + v*x]
    >>> convolution_subset([u, v, x], [y, z])
    [u*y, u*z + v*y, x*y, x*z]

    >>> convolution_subset([1, S(2)/3], [3, 4])
    [3, 6]
    >>> convolution_subset([1, 3, S(5)/7], [7])
    [7, 21, 5, 0]

    References
    ==========

    .. [1] https://people.csail.mit.edu/rrw/presentations/subset-conv.pdf

    """
def covering_product(a, b):
    """
    Returns the covering product of given sequences.

    The indices of each argument, considered as bit strings, correspond to
    subsets of a finite set.

    The covering product of given sequences is a sequence which contains
    the sum of products of the elements of the given sequences grouped by
    the *bitwise-OR* of the corresponding indices.

    The sequence is automatically padded to the right with zeros, as the
    definition of subset based on bitmasks (indices) requires the size of
    sequence to be a power of 2.

    Parameters
    ==========

    a, b : iterables
        The sequences for which covering product is to be obtained.

    Examples
    ========

    >>> from sympy import symbols, S, I, covering_product
    >>> u, v, x, y, z = symbols('u v x y z')

    >>> covering_product([u, v], [x, y])
    [u*x, u*y + v*x + v*y]
    >>> covering_product([u, v, x], [y, z])
    [u*y, u*z + v*y + v*z, x*y, x*z]

    >>> covering_product([1, S(2)/3], [3, 4 + 5*I])
    [3, 26/3 + 25*I/3]
    >>> covering_product([1, 3, S(5)/7], [7, 8])
    [7, 53, 5, 40/7]

    References
    ==========

    .. [1] https://people.csail.mit.edu/rrw/presentations/subset-conv.pdf

    """
def intersecting_product(a, b):
    """
    Returns the intersecting product of given sequences.

    The indices of each argument, considered as bit strings, correspond to
    subsets of a finite set.

    The intersecting product of given sequences is the sequence which
    contains the sum of products of the elements of the given sequences
    grouped by the *bitwise-AND* of the corresponding indices.

    The sequence is automatically padded to the right with zeros, as the
    definition of subset based on bitmasks (indices) requires the size of
    sequence to be a power of 2.

    Parameters
    ==========

    a, b : iterables
        The sequences for which intersecting product is to be obtained.

    Examples
    ========

    >>> from sympy import symbols, S, I, intersecting_product
    >>> u, v, x, y, z = symbols('u v x y z')

    >>> intersecting_product([u, v], [x, y])
    [u*x + u*y + v*x, v*y]
    >>> intersecting_product([u, v, x], [y, z])
    [u*y + u*z + v*y + x*y + x*z, v*z, 0, 0]

    >>> intersecting_product([1, S(2)/3], [3, 4 + 5*I])
    [9 + 5*I, 8/3 + 10*I/3]
    >>> intersecting_product([1, 3, S(5)/7], [7, 8])
    [327/7, 24, 0, 0]

    References
    ==========

    .. [1] https://people.csail.mit.edu/rrw/presentations/subset-conv.pdf

    """
def convolution_int(a, b):
    """Return the convolution of two sequences as a list.

    The iterables must consist solely of integers.

    Parameters
    ==========

    a, b : Sequence
        The sequences for which convolution is performed.

    Explanation
    ===========

    This function performs the convolution of ``a`` and ``b`` by packing
    each into a single integer, multiplying them together, and then
    unpacking the result from the product.  The intuition behind this is
    that if we evaluate some polynomial [1]:

    .. math ::
        1156x^6 + 3808x^5 + 8440x^4 + 14856x^3 + 16164x^2 + 14040x + 8100

    at say $x = 10^5$ we obtain $1156038080844014856161641404008100$.
    Note we can read of the coefficients for each term every five digits.
    If the $x$ we chose to evaluate at is large enough, the same will hold
    for the product.

    The idea now is since big integer multiplication in libraries such
    as GMP is highly optimised, this will be reasonably fast.

    Examples
    ========

    >>> from sympy.discrete.convolutions import convolution_int

    >>> convolution_int([2, 3], [4, 5])
    [8, 22, 15]
    >>> convolution_int([1, 1, -1], [1, 1])
    [1, 2, 0, -1]

    References
    ==========

    .. [1] Fateman, Richard J.
           Can you save time in multiplying polynomials by encoding them as integers?
           University of California, Berkeley, California (2004).
           https://people.eecs.berkeley.edu/~fateman/papers/polysbyGMP.pdf
    """
