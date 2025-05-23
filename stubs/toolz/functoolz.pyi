from _typeshed import Incomplete

__all__ = ['identity', 'apply', 'thread_first', 'thread_last', 'memoize', 'compose', 'compose_left', 'pipe', 'complement', 'juxt', 'do', 'curry', 'flip', 'excepts']

def identity(x):
    """ Identity function. Return x

    >>> identity(3)
    3
    """
def apply(*func_and_args, **kwargs):
    """ Applies a function and returns the results

    >>> def double(x): return 2*x
    >>> def inc(x):    return x + 1
    >>> apply(double, 5)
    10

    >>> tuple(map(apply, [double, inc, double], [10, 500, 8000]))
    (20, 501, 16000)
    """
def thread_first(val, *forms):
    """ Thread value through a sequence of functions/forms

    >>> def double(x): return 2*x
    >>> def inc(x):    return x + 1
    >>> thread_first(1, inc, double)
    4

    If the function expects more than one input you can specify those inputs
    in a tuple.  The value is used as the first input.

    >>> def add(x, y): return x + y
    >>> def pow(x, y): return x**y
    >>> thread_first(1, (add, 4), (pow, 2))  # pow(add(1, 4), 2)
    25

    So in general
        thread_first(x, f, (g, y, z))
    expands to
        g(f(x), y, z)

    See Also:
        thread_last
    """
def thread_last(val, *forms):
    """ Thread value through a sequence of functions/forms

    >>> def double(x): return 2*x
    >>> def inc(x):    return x + 1
    >>> thread_last(1, inc, double)
    4

    If the function expects more than one input you can specify those inputs
    in a tuple.  The value is used as the last input.

    >>> def add(x, y): return x + y
    >>> def pow(x, y): return x**y
    >>> thread_last(1, (add, 4), (pow, 2))  # pow(2, add(4, 1))
    32

    So in general
        thread_last(x, f, (g, y, z))
    expands to
        g(y, z, f(x))

    >>> def iseven(x):
    ...     return x % 2 == 0
    >>> list(thread_last([1, 2, 3], (map, inc), (filter, iseven)))
    [2, 4]

    See Also:
        thread_first
    """

class InstanceProperty(property):
    """ Like @property, but returns ``classval`` when used as a class attribute

    Should not be used directly.  Use ``instanceproperty`` instead.
    """
    classval: Incomplete
    def __init__(self, fget: Incomplete | None = None, fset: Incomplete | None = None, fdel: Incomplete | None = None, doc: Incomplete | None = None, classval: Incomplete | None = None) -> None: ...
    def __get__(self, obj, type: Incomplete | None = None): ...
    def __reduce__(self): ...

class curry:
    """ Curry a callable function

    Enables partial application of arguments through calling a function with an
    incomplete set of arguments.

    >>> def mul(x, y):
    ...     return x * y
    >>> mul = curry(mul)

    >>> double = mul(2)
    >>> double(10)
    20

    Also supports keyword arguments

    >>> @curry                  # Can use curry as a decorator
    ... def f(x, y, a=10):
    ...     return a * (x + y)

    >>> add = f(a=1)
    >>> add(2, 3)
    5

    See Also:
        toolz.curried - namespace of curried functions
                        https://toolz.readthedocs.io/en/latest/curry.html
    """
    _partial: Incomplete
    __doc__: Incomplete
    __name__: Incomplete
    __module__: Incomplete
    __qualname__: Incomplete
    _sigspec: Incomplete
    _has_unknown_args: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def func(self): ...
    def __signature__(self): ...
    def args(self): ...
    def keywords(self): ...
    def func_name(self): ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __hash__(self): ...
    def __eq__(self, other): ...
    def __ne__(self, other): ...
    def __call__(self, *args, **kwargs): ...
    def _should_curry(self, args, kwargs, exc: Incomplete | None = None): ...
    def bind(self, *args, **kwargs): ...
    def call(self, *args, **kwargs): ...
    def __get__(self, instance, owner): ...
    def __reduce__(self): ...

def memoize(func, cache: Incomplete | None = None, key: Incomplete | None = None):
    """ Cache a function's result for speedy future evaluation

    Considerations:
        Trades memory for speed.
        Only use on pure functions.

    >>> def add(x, y):  return x + y
    >>> add = memoize(add)

    Or use as a decorator

    >>> @memoize
    ... def add(x, y):
    ...     return x + y

    Use the ``cache`` keyword to provide a dict-like object as an initial cache

    >>> @memoize(cache={(1, 2): 3})
    ... def add(x, y):
    ...     return x + y

    Note that the above works as a decorator because ``memoize`` is curried.

    It is also possible to provide a ``key(args, kwargs)`` function that
    calculates keys used for the cache, which receives an ``args`` tuple and
    ``kwargs`` dict as input, and must return a hashable value.  However,
    the default key function should be sufficient most of the time.

    >>> # Use key function that ignores extraneous keyword arguments
    >>> @memoize(key=lambda args, kwargs: args)
    ... def add(x, y, verbose=False):
    ...     if verbose:
    ...         print('Calculating %s + %s' % (x, y))
    ...     return x + y
    """

class Compose:
    """ A composition of functions

    See Also:
        compose
    """
    __slots__: Incomplete
    first: Incomplete
    funcs: Incomplete
    def __init__(self, funcs) -> None: ...
    def __call__(self, *args, **kwargs): ...
    def __getstate__(self): ...
    def __setstate__(self, state) -> None: ...
    def __doc__(self): ...
    @property
    def __name__(self): ...
    def __repr__(self) -> str: ...
    def __eq__(self, other): ...
    def __ne__(self, other): ...
    def __hash__(self): ...
    def __get__(self, obj, objtype: Incomplete | None = None): ...
    def __signature__(self): ...
    __wrapped__: Incomplete

def compose(*funcs):
    """ Compose functions to operate in series.

    Returns a function that applies other functions in sequence.

    Functions are applied from right to left so that
    ``compose(f, g, h)(x, y)`` is the same as ``f(g(h(x, y)))``.

    If no arguments are provided, the identity function (f(x) = x) is returned.

    >>> inc = lambda i: i + 1
    >>> compose(str, inc)(3)
    '4'

    See Also:
        compose_left
        pipe
    """
def compose_left(*funcs):
    """ Compose functions to operate in series.

    Returns a function that applies other functions in sequence.

    Functions are applied from left to right so that
    ``compose_left(f, g, h)(x, y)`` is the same as ``h(g(f(x, y)))``.

    If no arguments are provided, the identity function (f(x) = x) is returned.

    >>> inc = lambda i: i + 1
    >>> compose_left(inc, str)(3)
    '4'

    See Also:
        compose
        pipe
    """
def pipe(data, *funcs):
    """ Pipe a value through a sequence of functions

    I.e. ``pipe(data, f, g, h)`` is equivalent to ``h(g(f(data)))``

    We think of the value as progressing through a pipe of several
    transformations, much like pipes in UNIX

    ``$ cat data | f | g | h``

    >>> double = lambda i: 2 * i
    >>> pipe(3, double, str)
    '6'

    See Also:
        compose
        compose_left
        thread_first
        thread_last
    """
def complement(func):
    """ Convert a predicate function to its logical complement.

    In other words, return a function that, for inputs that normally
    yield True, yields False, and vice-versa.

    >>> def iseven(n): return n % 2 == 0
    >>> isodd = complement(iseven)
    >>> iseven(2)
    True
    >>> isodd(2)
    False
    """

class juxt:
    """ Creates a function that calls several functions with the same arguments

    Takes several functions and returns a function that applies its arguments
    to each of those functions then returns a tuple of the results.

    Name comes from juxtaposition: the fact of two things being seen or placed
    close together with contrasting effect.

    >>> inc = lambda x: x + 1
    >>> double = lambda x: x * 2
    >>> juxt(inc, double)(10)
    (11, 20)
    >>> juxt([inc, double])(10)
    (11, 20)
    """
    __slots__: Incomplete
    funcs: Incomplete
    def __init__(self, *funcs) -> None: ...
    def __call__(self, *args, **kwargs): ...
    def __getstate__(self): ...
    def __setstate__(self, state) -> None: ...

def do(func, x):
    """ Runs ``func`` on ``x``, returns ``x``

    Because the results of ``func`` are not returned, only the side
    effects of ``func`` are relevant.

    Logging functions can be made by composing ``do`` with a storage function
    like ``list.append`` or ``file.write``

    >>> from toolz import compose
    >>> from toolz.curried import do

    >>> log = []
    >>> inc = lambda x: x + 1
    >>> inc = compose(inc, do(log.append))
    >>> inc(1)
    2
    >>> inc(11)
    12
    >>> log
    [1, 11]
    """
def flip(func, a, b):
    """ Call the function call with the arguments flipped

    This function is curried.

    >>> def div(a, b):
    ...     return a // b
    ...
    >>> flip(div, 2, 6)
    3
    >>> div_by_two = flip(div, 2)
    >>> div_by_two(4)
    2

    This is particularly useful for built in functions and functions defined
    in C extensions that accept positional only arguments. For example:
    isinstance, issubclass.

    >>> data = [1, 'a', 'b', 2, 1.5, object(), 3]
    >>> only_ints = list(filter(flip(isinstance, int), data))
    >>> only_ints
    [1, 2, 3]
    """

class excepts:
    """A wrapper around a function to catch exceptions and
    dispatch to a handler.

    This is like a functional try/except block, in the same way that
    ifexprs are functional if/else blocks.

    Examples
    --------
    >>> excepting = excepts(
    ...     ValueError,
    ...     lambda a: [1, 2].index(a),
    ...     lambda _: -1,
    ... )
    >>> excepting(1)
    0
    >>> excepting(3)
    -1

    Multiple exceptions and default except clause.

    >>> excepting = excepts((IndexError, KeyError), lambda a: a[0])
    >>> excepting([])
    >>> excepting([1])
    1
    >>> excepting({})
    >>> excepting({0: 1})
    1
    """
    exc: Incomplete
    func: Incomplete
    handler: Incomplete
    def __init__(self, exc, func, handler=...) -> None: ...
    def __call__(self, *args, **kwargs): ...
    def __doc__(self): ...
    @property
    def __name__(self): ...
_check_sigspec_orig = _check_sigspec
