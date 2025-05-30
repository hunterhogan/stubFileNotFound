from _typeshed import Incomplete
from toolz.itertoolz import cons as cons, getter as getter, pluck as pluck

class EqualityHashKey:
    """ Create a hash key that uses equality comparisons between items.

    This may be used to create hash keys for otherwise unhashable types:

    >>> from toolz import curry
    >>> EqualityHashDefault = curry(EqualityHashKey, None)
    >>> set(map(EqualityHashDefault, [[], (), [1], [1]]))  # doctest: +SKIP
    {=[]=, =()=, =[1]=}

    **Caution:** adding N ``EqualityHashKey`` items to a hash container
    may require O(N**2) operations, not O(N) as for typical hashable types.
    Therefore, a suitable key function such as ``tuple`` or ``frozenset``
    is usually preferred over using ``EqualityHashKey`` if possible.

    The ``key`` argument to ``EqualityHashKey`` should be a function or
    index that returns a hashable object that effectively distinguishes
    unequal items.  This helps avoid the poor scaling that occurs when
    using the default key.  For example, the above example can be improved
    by using a key function that distinguishes items by length or type:

    >>> EqualityHashLen = curry(EqualityHashKey, len)
    >>> EqualityHashType = curry(EqualityHashKey, type)  # this works too
    >>> set(map(EqualityHashLen, [[], (), [1], [1]]))  # doctest: +SKIP
    {=[]=, =()=, =[1]=}

    ``EqualityHashKey`` is convenient to use when a suitable key function
    is complicated or unavailable.  For example, the following returns all
    unique values based on equality:

    >>> from toolz import unique
    >>> vals = [[], [], (), [1], [1], [2], {}, {}, {}]
    >>> list(unique(vals, key=EqualityHashDefault))
    [[], (), [1], [2], {}]

    **Warning:** don't change the equality value of an item already in a hash
    container.  Unhashable types are unhashable for a reason.  For example:

    >>> L1 = [1] ; L2 = [2]
    >>> s = set(map(EqualityHashDefault, [L1, L2]))
    >>> s  # doctest: +SKIP
    {=[1]=, =[2]=}

    >>> L1[0] = 2  # Don't do this!  ``s`` now has duplicate items!
    >>> s  # doctest: +SKIP
    {=[2]=, =[2]=}

    Although this may appear problematic, immutable data types is a common
    idiom in functional programming, and``EqualityHashKey`` easily allows
    the same idiom to be used by convention rather than strict requirement.

    See Also:
        identity
    """
    __slots__: Incomplete
    _default_hashkey: str
    key: Incomplete
    item: Incomplete
    def __init__(self, key, item) -> None: ...
    def __hash__(self): ...
    def __eq__(self, other): ...
    def __ne__(self, other): ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

def unzip(seq):
    """Inverse of ``zip``

    >>> a, b = unzip([('a', 1), ('b', 2)])
    >>> list(a)
    ['a', 'b']
    >>> list(b)
    [1, 2]

    Unlike the naive implementation ``def unzip(seq): zip(*seq)`` this
    implementation can handle an infinite sequence ``seq``.

    Caveats:

    * The implementation uses ``tee``, and so can use a significant amount
      of auxiliary storage if the resulting iterators are consumed at
      different times.

    * The inner sequence cannot be infinite. In Python 3 ``zip(*seq)`` can be
      used if ``seq`` is a finite sequence of infinite sequences.

    """
