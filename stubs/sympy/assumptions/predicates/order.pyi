from _typeshed import Incomplete
from sympy.assumptions import Predicate as Predicate
from sympy.multipledispatch import Dispatcher as Dispatcher

class NegativePredicate(Predicate):
    """
    Negative number predicate.

    Explanation
    ===========

    ``Q.negative(x)`` is true iff ``x`` is a real number and :math:`x < 0`, that is,
    it is in the interval :math:`(-\\infty, 0)`.  Note in particular that negative
    infinity is not negative.

    A few important facts about negative numbers:

    - Note that ``Q.nonnegative`` and ``~Q.negative`` are *not* the same
        thing. ``~Q.negative(x)`` simply means that ``x`` is not negative,
        whereas ``Q.nonnegative(x)`` means that ``x`` is real and not
        negative, i.e., ``Q.nonnegative(x)`` is logically equivalent to
        ``Q.zero(x) | Q.positive(x)``.  So for example, ``~Q.negative(I)`` is
        true, whereas ``Q.nonnegative(I)`` is false.

    - See the documentation of ``Q.real`` for more information about
        related facts.

    Examples
    ========

    >>> from sympy import Q, ask, symbols, I
    >>> x = symbols('x')
    >>> ask(Q.negative(x), Q.real(x) & ~Q.positive(x) & ~Q.zero(x))
    True
    >>> ask(Q.negative(-1))
    True
    >>> ask(Q.nonnegative(I))
    False
    >>> ask(~Q.negative(I))
    True

    """
    name: str
    handler: Incomplete

class NonNegativePredicate(Predicate):
    """
    Nonnegative real number predicate.

    Explanation
    ===========

    ``ask(Q.nonnegative(x))`` is true iff ``x`` belongs to the set of
    positive numbers including zero.

    - Note that ``Q.nonnegative`` and ``~Q.negative`` are *not* the same
        thing. ``~Q.negative(x)`` simply means that ``x`` is not negative,
        whereas ``Q.nonnegative(x)`` means that ``x`` is real and not
        negative, i.e., ``Q.nonnegative(x)`` is logically equivalent to
        ``Q.zero(x) | Q.positive(x)``.  So for example, ``~Q.negative(I)`` is
        true, whereas ``Q.nonnegative(I)`` is false.

    Examples
    ========

    >>> from sympy import Q, ask, I
    >>> ask(Q.nonnegative(1))
    True
    >>> ask(Q.nonnegative(0))
    True
    >>> ask(Q.nonnegative(-1))
    False
    >>> ask(Q.nonnegative(I))
    False
    >>> ask(Q.nonnegative(-I))
    False

    """
    name: str
    handler: Incomplete

class NonZeroPredicate(Predicate):
    """
    Nonzero real number predicate.

    Explanation
    ===========

    ``ask(Q.nonzero(x))`` is true iff ``x`` is real and ``x`` is not zero.  Note in
    particular that ``Q.nonzero(x)`` is false if ``x`` is not real.  Use
    ``~Q.zero(x)`` if you want the negation of being zero without any real
    assumptions.

    A few important facts about nonzero numbers:

    - ``Q.nonzero`` is logically equivalent to ``Q.positive | Q.negative``.

    - See the documentation of ``Q.real`` for more information about
        related facts.

    Examples
    ========

    >>> from sympy import Q, ask, symbols, I, oo
    >>> x = symbols('x')
    >>> print(ask(Q.nonzero(x), ~Q.zero(x)))
    None
    >>> ask(Q.nonzero(x), Q.positive(x))
    True
    >>> ask(Q.nonzero(x), Q.zero(x))
    False
    >>> ask(Q.nonzero(0))
    False
    >>> ask(Q.nonzero(I))
    False
    >>> ask(~Q.zero(I))
    True
    >>> ask(Q.nonzero(oo))
    False

    """
    name: str
    handler: Incomplete

class ZeroPredicate(Predicate):
    """
    Zero number predicate.

    Explanation
    ===========

    ``ask(Q.zero(x))`` is true iff the value of ``x`` is zero.

    Examples
    ========

    >>> from sympy import ask, Q, oo, symbols
    >>> x, y = symbols('x, y')
    >>> ask(Q.zero(0))
    True
    >>> ask(Q.zero(1/oo))
    True
    >>> print(ask(Q.zero(0*oo)))
    None
    >>> ask(Q.zero(1))
    False
    >>> ask(Q.zero(x*y), Q.zero(x) | Q.zero(y))
    True

    """
    name: str
    handler: Incomplete

class NonPositivePredicate(Predicate):
    """
    Nonpositive real number predicate.

    Explanation
    ===========

    ``ask(Q.nonpositive(x))`` is true iff ``x`` belongs to the set of
    negative numbers including zero.

    - Note that ``Q.nonpositive`` and ``~Q.positive`` are *not* the same
        thing. ``~Q.positive(x)`` simply means that ``x`` is not positive,
        whereas ``Q.nonpositive(x)`` means that ``x`` is real and not
        positive, i.e., ``Q.nonpositive(x)`` is logically equivalent to
        `Q.negative(x) | Q.zero(x)``.  So for example, ``~Q.positive(I)`` is
        true, whereas ``Q.nonpositive(I)`` is false.

    Examples
    ========

    >>> from sympy import Q, ask, I

    >>> ask(Q.nonpositive(-1))
    True
    >>> ask(Q.nonpositive(0))
    True
    >>> ask(Q.nonpositive(1))
    False
    >>> ask(Q.nonpositive(I))
    False
    >>> ask(Q.nonpositive(-I))
    False

    """
    name: str
    handler: Incomplete

class PositivePredicate(Predicate):
    """
    Positive real number predicate.

    Explanation
    ===========

    ``Q.positive(x)`` is true iff ``x`` is real and `x > 0`, that is if ``x``
    is in the interval `(0, \\infty)`.  In particular, infinity is not
    positive.

    A few important facts about positive numbers:

    - Note that ``Q.nonpositive`` and ``~Q.positive`` are *not* the same
        thing. ``~Q.positive(x)`` simply means that ``x`` is not positive,
        whereas ``Q.nonpositive(x)`` means that ``x`` is real and not
        positive, i.e., ``Q.nonpositive(x)`` is logically equivalent to
        `Q.negative(x) | Q.zero(x)``.  So for example, ``~Q.positive(I)`` is
        true, whereas ``Q.nonpositive(I)`` is false.

    - See the documentation of ``Q.real`` for more information about
        related facts.

    Examples
    ========

    >>> from sympy import Q, ask, symbols, I
    >>> x = symbols('x')
    >>> ask(Q.positive(x), Q.real(x) & ~Q.negative(x) & ~Q.zero(x))
    True
    >>> ask(Q.positive(1))
    True
    >>> ask(Q.nonpositive(I))
    False
    >>> ask(~Q.positive(I))
    True

    """
    name: str
    handler: Incomplete

class ExtendedPositivePredicate(Predicate):
    """
    Positive extended real number predicate.

    Explanation
    ===========

    ``Q.extended_positive(x)`` is true iff ``x`` is extended real and
    `x > 0`, that is if ``x`` is in the interval `(0, \\infty]`.

    Examples
    ========

    >>> from sympy import ask, I, oo, Q
    >>> ask(Q.extended_positive(1))
    True
    >>> ask(Q.extended_positive(oo))
    True
    >>> ask(Q.extended_positive(I))
    False

    """
    name: str
    handler: Incomplete

class ExtendedNegativePredicate(Predicate):
    """
    Negative extended real number predicate.

    Explanation
    ===========

    ``Q.extended_negative(x)`` is true iff ``x`` is extended real and
    `x < 0`, that is if ``x`` is in the interval `[-\\infty, 0)`.

    Examples
    ========

    >>> from sympy import ask, I, oo, Q
    >>> ask(Q.extended_negative(-1))
    True
    >>> ask(Q.extended_negative(-oo))
    True
    >>> ask(Q.extended_negative(-I))
    False

    """
    name: str
    handler: Incomplete

class ExtendedNonZeroPredicate(Predicate):
    """
    Nonzero extended real number predicate.

    Explanation
    ===========

    ``ask(Q.extended_nonzero(x))`` is true iff ``x`` is extended real and
    ``x`` is not zero.

    Examples
    ========

    >>> from sympy import ask, I, oo, Q
    >>> ask(Q.extended_nonzero(-1))
    True
    >>> ask(Q.extended_nonzero(oo))
    True
    >>> ask(Q.extended_nonzero(I))
    False

    """
    name: str
    handler: Incomplete

class ExtendedNonPositivePredicate(Predicate):
    """
    Nonpositive extended real number predicate.

    Explanation
    ===========

    ``ask(Q.extended_nonpositive(x))`` is true iff ``x`` is extended real and
    ``x`` is not positive.

    Examples
    ========

    >>> from sympy import ask, I, oo, Q
    >>> ask(Q.extended_nonpositive(-1))
    True
    >>> ask(Q.extended_nonpositive(oo))
    False
    >>> ask(Q.extended_nonpositive(0))
    True
    >>> ask(Q.extended_nonpositive(I))
    False

    """
    name: str
    handler: Incomplete

class ExtendedNonNegativePredicate(Predicate):
    """
    Nonnegative extended real number predicate.

    Explanation
    ===========

    ``ask(Q.extended_nonnegative(x))`` is true iff ``x`` is extended real and
    ``x`` is not negative.

    Examples
    ========

    >>> from sympy import ask, I, oo, Q
    >>> ask(Q.extended_nonnegative(-1))
    False
    >>> ask(Q.extended_nonnegative(oo))
    True
    >>> ask(Q.extended_nonnegative(0))
    True
    >>> ask(Q.extended_nonnegative(I))
    False

    """
    name: str
    handler: Incomplete
