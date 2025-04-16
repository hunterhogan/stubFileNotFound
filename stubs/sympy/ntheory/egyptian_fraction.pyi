from sympy.core.containers import Tuple as Tuple
from sympy.core.numbers import Integer as Integer, Rational as Rational
from sympy.core.singleton import S as S

def egyptian_fraction(r, algorithm: str = 'Greedy'):
    '''
    Return the list of denominators of an Egyptian fraction
    expansion [1]_ of the said rational `r`.

    Parameters
    ==========

    r : Rational or (p, q)
        a positive rational number, ``p/q``.
    algorithm : { "Greedy", "Graham Jewett", "Takenouchi", "Golomb" }, optional
        Denotes the algorithm to be used (the default is "Greedy").

    Examples
    ========

    >>> from sympy import Rational
    >>> from sympy.ntheory.egyptian_fraction import egyptian_fraction
    >>> egyptian_fraction(Rational(3, 7))
    [3, 11, 231]
    >>> egyptian_fraction((3, 7), "Graham Jewett")
    [7, 8, 9, 56, 57, 72, 3192]
    >>> egyptian_fraction((3, 7), "Takenouchi")
    [4, 7, 28]
    >>> egyptian_fraction((3, 7), "Golomb")
    [3, 15, 35]
    >>> egyptian_fraction((11, 5), "Golomb")
    [1, 2, 3, 4, 9, 234, 1118, 2580]

    See Also
    ========

    sympy.core.numbers.Rational

    Notes
    =====

    Currently the following algorithms are supported:

    1) Greedy Algorithm

       Also called the Fibonacci-Sylvester algorithm [2]_.
       At each step, extract the largest unit fraction less
       than the target and replace the target with the remainder.

       It has some distinct properties:

       a) Given `p/q` in lowest terms, generates an expansion of maximum
          length `p`. Even as the numerators get large, the number of
          terms is seldom more than a handful.

       b) Uses minimal memory.

       c) The terms can blow up (standard examples of this are 5/121 and
          31/311).  The denominator is at most squared at each step
          (doubly-exponential growth) and typically exhibits
          singly-exponential growth.

    2) Graham Jewett Algorithm

       The algorithm suggested by the result of Graham and Jewett.
       Note that this has a tendency to blow up: the length of the
       resulting expansion is always ``2**(x/gcd(x, y)) - 1``.  See [3]_.

    3) Takenouchi Algorithm

       The algorithm suggested by Takenouchi (1921).
       Differs from the Graham-Jewett algorithm only in the handling
       of duplicates.  See [3]_.

    4) Golomb\'s Algorithm

       A method given by Golumb (1962), using modular arithmetic and
       inverses.  It yields the same results as a method using continued
       fractions proposed by Bleicher (1972).  See [4]_.

    If the given rational is greater than or equal to 1, a greedy algorithm
    of summing the harmonic sequence 1/1 + 1/2 + 1/3 + ... is used, taking
    all the unit fractions of this sequence until adding one more would be
    greater than the given number.  This list of denominators is prefixed
    to the result from the requested algorithm used on the remainder.  For
    example, if r is 8/3, using the Greedy algorithm, we get [1, 2, 3, 4,
    5, 6, 7, 14, 420], where the beginning of the sequence, [1, 2, 3, 4, 5,
    6, 7] is part of the harmonic sequence summing to 363/140, leaving a
    remainder of 31/420, which yields [14, 420] by the Greedy algorithm.
    The result of egyptian_fraction(Rational(8, 3), "Golomb") is [1, 2, 3,
    4, 5, 6, 7, 14, 574, 2788, 6460, 11590, 33062, 113820], and so on.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Egyptian_fraction
    .. [2] https://en.wikipedia.org/wiki/Greedy_algorithm_for_Egyptian_fractions
    .. [3] https://www.ics.uci.edu/~eppstein/numth/egypt/conflict.html
    .. [4] https://web.archive.org/web/20180413004012/https://ami.ektf.hu/uploads/papers/finalpdf/AMI_42_from129to134.pdf

    '''
def egypt_greedy(x, y): ...
def egypt_graham_jewett(x, y): ...
def egypt_takenouchi(x, y): ...
def egypt_golomb(x, y): ...
def egypt_harmonic(r): ...
