import collections.abc
from _typeshed import Incomplete
from collections.abc import Generator
from matplotlib import _api as _api, _c_internal_utils as _c_internal_utils

class _ExceptionInfo:
    """
    A class to carry exception information around.

    This is used to store and later raise exceptions. It's an alternative to
    directly storing Exception instances that circumvents traceback-related
    issues: caching tracebacks can keep user's objects in local namespaces
    alive indefinitely, which can lead to very surprising memory issues for
    users and result in incorrect tracebacks.
    """
    _cls: Incomplete
    _args: Incomplete
    def __init__(self, cls, *args) -> None: ...
    @classmethod
    def from_exception(cls, exc): ...
    def to_exception(self): ...

def _get_running_interactive_framework():
    '''
    Return the interactive framework whose event loop is currently running, if
    any, or "headless" if no event loop can be started, or None.

    Returns
    -------
    Optional[str]
        One of the following values: "qt", "gtk3", "gtk4", "wx", "tk",
        "macosx", "headless", ``None``.
    '''
def _exception_printer(exc) -> None: ...

class _StrongRef:
    """
    Wrapper similar to a weakref, but keeping a strong reference to the object.
    """
    _obj: Incomplete
    def __init__(self, obj) -> None: ...
    def __call__(self): ...
    def __eq__(self, other): ...
    def __hash__(self): ...

def _weak_or_strong_ref(func, callback):
    """
    Return a `WeakMethod` wrapping *func* if possible, else a `_StrongRef`.
    """

class _UnhashDict:
    """
    A minimal dict-like class that also supports unhashable keys, storing them
    in a list of key-value pairs.

    This class only implements the interface needed for `CallbackRegistry`, and
    tries to minimize the overhead for the hashable case.
    """
    _dict: Incomplete
    _pairs: Incomplete
    def __init__(self, pairs) -> None: ...
    def __setitem__(self, key, value) -> None: ...
    def __getitem__(self, key): ...
    def pop(self, key, *args): ...
    def __iter__(self): ...

class CallbackRegistry:
    """
    Handle registering, processing, blocking, and disconnecting
    for a set of signals and callbacks:

        >>> def oneat(x):
        ...     print('eat', x)
        >>> def ondrink(x):
        ...     print('drink', x)

        >>> from matplotlib.cbook import CallbackRegistry
        >>> callbacks = CallbackRegistry()

        >>> id_eat = callbacks.connect('eat', oneat)
        >>> id_drink = callbacks.connect('drink', ondrink)

        >>> callbacks.process('drink', 123)
        drink 123
        >>> callbacks.process('eat', 456)
        eat 456
        >>> callbacks.process('be merry', 456)   # nothing will be called

        >>> callbacks.disconnect(id_eat)
        >>> callbacks.process('eat', 456)        # nothing will be called

        >>> with callbacks.blocked(signal='drink'):
        ...     callbacks.process('drink', 123)  # nothing will be called
        >>> callbacks.process('drink', 123)
        drink 123

    In practice, one should always disconnect all callbacks when they are
    no longer needed to avoid dangling references (and thus memory leaks).
    However, real code in Matplotlib rarely does so, and due to its design,
    it is rather difficult to place this kind of code.  To get around this,
    and prevent this class of memory leaks, we instead store weak references
    to bound methods only, so when the destination object needs to die, the
    CallbackRegistry won't keep it alive.

    Parameters
    ----------
    exception_handler : callable, optional
       If not None, *exception_handler* must be a function that takes an
       `Exception` as single parameter.  It gets called with any `Exception`
       raised by the callbacks during `CallbackRegistry.process`, and may
       either re-raise the exception or handle it in another manner.

       The default handler prints the exception (with `traceback.print_exc`) if
       an interactive event loop is running; it re-raises the exception if no
       interactive event loop is running.

    signals : list, optional
        If not None, *signals* is a list of signals that this registry handles:
        attempting to `process` or to `connect` to a signal not in the list
        throws a `ValueError`.  The default, None, does not restrict the
        handled signals.
    """
    _signals: Incomplete
    exception_handler: Incomplete
    callbacks: Incomplete
    _cid_gen: Incomplete
    _func_cid_map: Incomplete
    _pickled_cids: Incomplete
    def __init__(self, exception_handler=..., *, signals: Incomplete | None = None) -> None: ...
    def __getstate__(self): ...
    def __setstate__(self, state) -> None: ...
    def connect(self, signal, func):
        """Register *func* to be called when signal *signal* is generated."""
    def _connect_picklable(self, signal, func):
        """
        Like `.connect`, but the callback is kept when pickling/unpickling.

        Currently internal-use only.
        """
    def _remove_proxy(self, signal, proxy, *, _is_finalizing=...) -> None: ...
    def disconnect(self, cid) -> None:
        """
        Disconnect the callback registered with callback id *cid*.

        No error is raised if such a callback does not exist.
        """
    def process(self, s, *args, **kwargs) -> None:
        """
        Process signal *s*.

        All of the functions registered to receive callbacks on *s* will be
        called with ``*args`` and ``**kwargs``.
        """
    def blocked(self, *, signal: Incomplete | None = None) -> Generator[None]:
        """
        Block callback signals from being processed.

        A context manager to temporarily block/disable callback signals
        from being processed by the registered listeners.

        Parameters
        ----------
        signal : str, optional
            The callback signal to block. The default is to block all signals.
        """

class silent_list(list):
    """
    A list with a short ``repr()``.

    This is meant to be used for a homogeneous list of artists, so that they
    don't cause long, meaningless output.

    Instead of ::

        [<matplotlib.lines.Line2D object at 0x7f5749fed3c8>,
         <matplotlib.lines.Line2D object at 0x7f5749fed4e0>,
         <matplotlib.lines.Line2D object at 0x7f5758016550>]

    one will get ::

        <a list of 3 Line2D objects>

    If ``self.type`` is None, the type name is obtained from the first item in
    the list (if any).
    """
    type: Incomplete
    def __init__(self, type, seq: Incomplete | None = None) -> None: ...
    def __repr__(self) -> str: ...

def _local_over_kwdict(local_var, kwargs, *keys, warning_cls=...): ...
def strip_math(s):
    """
    Remove latex formatting from mathtext.

    Only handles fully math and fully non-math strings.
    """
def _strip_comment(s):
    """Strip everything from the first unquoted #."""
def is_writable_file_like(obj):
    """Return whether *obj* looks like a file object with a *write* method."""
def file_requires_unicode(x):
    """
    Return whether the given writable file-like object requires Unicode to be
    written to it.
    """
def to_filehandle(fname, flag: str = 'r', return_opened: bool = False, encoding: Incomplete | None = None):
    """
    Convert a path to an open file handle or pass-through a file-like object.

    Consider using `open_file_cm` instead, as it allows one to properly close
    newly created file objects more easily.

    Parameters
    ----------
    fname : str or path-like or file-like
        If `str` or `os.PathLike`, the file is opened using the flags specified
        by *flag* and *encoding*.  If a file-like object, it is passed through.
    flag : str, default: 'r'
        Passed as the *mode* argument to `open` when *fname* is `str` or
        `os.PathLike`; ignored if *fname* is file-like.
    return_opened : bool, default: False
        If True, return both the file object and a boolean indicating whether
        this was a new file (that the caller needs to close).  If False, return
        only the new file.
    encoding : str or None, default: None
        Passed as the *mode* argument to `open` when *fname* is `str` or
        `os.PathLike`; ignored if *fname* is file-like.

    Returns
    -------
    fh : file-like
    opened : bool
        *opened* is only returned if *return_opened* is True.
    """
def open_file_cm(path_or_file, mode: str = 'r', encoding: Incomplete | None = None):
    """Pass through file objects and context-manage path-likes."""
def is_scalar_or_string(val):
    """Return whether the given object is a scalar or string like."""
def get_sample_data(fname, asfileobj: bool = True):
    """
    Return a sample data file.  *fname* is a path relative to the
    :file:`mpl-data/sample_data` directory.  If *asfileobj* is `True`
    return a file object, otherwise just a file path.

    Sample data files are stored in the 'mpl-data/sample_data' directory within
    the Matplotlib package.

    If the filename ends in .gz, the file is implicitly ungzipped.  If the
    filename ends with .npy or .npz, and *asfileobj* is `True`, the file is
    loaded with `numpy.load`.
    """
def _get_data_path(*args):
    """
    Return the `pathlib.Path` to a resource file provided by Matplotlib.

    ``*args`` specify a path relative to the base data path.
    """
def flatten(seq, scalarp=...) -> Generator[Incomplete, Incomplete]:
    """
    Return a generator of flattened nested containers.

    For example:

        >>> from matplotlib.cbook import flatten
        >>> l = (('John', ['Hunter']), (1, 23), [[([42, (5, 23)], )]])
        >>> print(list(flatten(l)))
        ['John', 'Hunter', 1, 23, 42, 5, 23]

    By: Composite of Holger Krekel and Luther Blissett
    From: https://code.activestate.com/recipes/121294-simple-generator-for-flattening-nested-containers/
    and Recipe 1.12 in cookbook
    """

class _Stack:
    """
    Stack of elements with a movable cursor.

    Mimics home/back/forward in a web browser.
    """
    _pos: int
    _elements: Incomplete
    def __init__(self) -> None: ...
    def clear(self) -> None:
        """Empty the stack."""
    def __call__(self):
        """Return the current element, or None."""
    def __len__(self) -> int: ...
    def __getitem__(self, ind): ...
    def forward(self):
        """Move the position forward and return the current element."""
    def back(self):
        """Move the position back and return the current element."""
    def push(self, o):
        """
        Push *o* to the stack after the current position, and return *o*.

        Discard all later elements.
        """
    def home(self):
        """
        Push the first element onto the top of the stack.

        The first element is returned.
        """

def safe_masked_invalid(x, copy: bool = False): ...
def print_cycles(objects, outstream=..., show_progress: bool = False) -> None:
    """
    Print loops of cyclic references in the given *objects*.

    It is often useful to pass in ``gc.garbage`` to find the cycles that are
    preventing some objects from being garbage collected.

    Parameters
    ----------
    objects
        A list of objects to find cycles in.
    outstream
        The stream for output.
    show_progress : bool
        If True, print the number of objects reached as they are found.
    """

class Grouper:
    """
    A disjoint-set data structure.

    Objects can be joined using :meth:`join`, tested for connectedness
    using :meth:`joined`, and all disjoint sets can be retrieved by
    using the object as an iterator.

    The objects being joined must be hashable and weak-referenceable.

    Examples
    --------
    >>> from matplotlib.cbook import Grouper
    >>> class Foo:
    ...     def __init__(self, s):
    ...         self.s = s
    ...     def __repr__(self):
    ...         return self.s
    ...
    >>> a, b, c, d, e, f = [Foo(x) for x in 'abcdef']
    >>> grp = Grouper()
    >>> grp.join(a, b)
    >>> grp.join(b, c)
    >>> grp.join(d, e)
    >>> list(grp)
    [[a, b, c], [d, e]]
    >>> grp.joined(a, b)
    True
    >>> grp.joined(a, c)
    True
    >>> grp.joined(a, d)
    False
    """
    _mapping: Incomplete
    _ordering: Incomplete
    _next_order: Incomplete
    def __init__(self, init=()) -> None: ...
    def __getstate__(self): ...
    def __setstate__(self, state) -> None: ...
    def __contains__(self, item) -> bool: ...
    def join(self, a, *args) -> None:
        """
        Join given arguments into the same set.  Accepts one or more arguments.
        """
    def joined(self, a, b):
        """Return whether *a* and *b* are members of the same set."""
    def remove(self, a) -> None:
        """Remove *a* from the grouper, doing nothing if it is not there."""
    def __iter__(self):
        """
        Iterate over each of the disjoint sets as a list.

        The iterator is invalid if interleaved with calls to join().
        """
    def get_siblings(self, a):
        """Return all of the items joined with *a*, including itself."""

class GrouperView:
    """Immutable view over a `.Grouper`."""
    _grouper: Incomplete
    def __init__(self, grouper) -> None: ...
    def __contains__(self, item) -> bool: ...
    def __iter__(self): ...
    def joined(self, a, b):
        """
        Return whether *a* and *b* are members of the same set.
        """
    def get_siblings(self, a):
        """
        Return all of the items joined with *a*, including itself.
        """

def simple_linear_interpolation(a, steps):
    """
    Resample an array with ``steps - 1`` points between original point pairs.

    Along each column of *a*, ``(steps - 1)`` points are introduced between
    each original values; the values are linearly interpolated.

    Parameters
    ----------
    a : array, shape (n, ...)
    steps : int

    Returns
    -------
    array
        shape ``((n - 1) * steps + 1, ...)``
    """
def delete_masked_points(*args):
    """
    Find all masked and/or non-finite points in a set of arguments,
    and return the arguments with only the unmasked points remaining.

    Arguments can be in any of 5 categories:

    1) 1-D masked arrays
    2) 1-D ndarrays
    3) ndarrays with more than one dimension
    4) other non-string iterables
    5) anything else

    The first argument must be in one of the first four categories;
    any argument with a length differing from that of the first
    argument (and hence anything in category 5) then will be
    passed through unchanged.

    Masks are obtained from all arguments of the correct length
    in categories 1, 2, and 4; a point is bad if masked in a masked
    array or if it is a nan or inf.  No attempt is made to
    extract a mask from categories 2, 3, and 4 if `numpy.isfinite`
    does not yield a Boolean array.

    All input arguments that are not passed unchanged are returned
    as ndarrays after removing the points or rows corresponding to
    masks in any of the arguments.

    A vastly simpler version of this function was originally
    written as a helper for Axes.scatter().

    """
def _combine_masks(*args):
    """
    Find all masked and/or non-finite points in a set of arguments,
    and return the arguments as masked arrays with a common mask.

    Arguments can be in any of 5 categories:

    1) 1-D masked arrays
    2) 1-D ndarrays
    3) ndarrays with more than one dimension
    4) other non-string iterables
    5) anything else

    The first argument must be in one of the first four categories;
    any argument with a length differing from that of the first
    argument (and hence anything in category 5) then will be
    passed through unchanged.

    Masks are obtained from all arguments of the correct length
    in categories 1, 2, and 4; a point is bad if masked in a masked
    array or if it is a nan or inf.  No attempt is made to
    extract a mask from categories 2 and 4 if `numpy.isfinite`
    does not yield a Boolean array.  Category 3 is included to
    support RGB or RGBA ndarrays, which are assumed to have only
    valid values and which are passed through unchanged.

    All input arguments that are not passed unchanged are returned
    as masked arrays if any masked points are found, otherwise as
    ndarrays.

    """
def _broadcast_with_masks(*args, compress: bool = False):
    """
    Broadcast inputs, combining all masked arrays.

    Parameters
    ----------
    *args : array-like
        The inputs to broadcast.
    compress : bool, default: False
        Whether to compress the masked arrays. If False, the masked values
        are replaced by NaNs.

    Returns
    -------
    list of array-like
        The broadcasted and masked inputs.
    """
def boxplot_stats(X, whis: float = 1.5, bootstrap: Incomplete | None = None, labels: Incomplete | None = None, autorange: bool = False):
    '''
    Return a list of dictionaries of statistics used to draw a series of box
    and whisker plots using `~.Axes.bxp`.

    Parameters
    ----------
    X : array-like
        Data that will be represented in the boxplots. Should have 2 or
        fewer dimensions.

    whis : float or (float, float), default: 1.5
        The position of the whiskers.

        If a float, the lower whisker is at the lowest datum above
        ``Q1 - whis*(Q3-Q1)``, and the upper whisker at the highest datum below
        ``Q3 + whis*(Q3-Q1)``, where Q1 and Q3 are the first and third
        quartiles.  The default value of ``whis = 1.5`` corresponds to Tukey\'s
        original definition of boxplots.

        If a pair of floats, they indicate the percentiles at which to draw the
        whiskers (e.g., (5, 95)).  In particular, setting this to (0, 100)
        results in whiskers covering the whole range of the data.

        In the edge case where ``Q1 == Q3``, *whis* is automatically set to
        (0, 100) (cover the whole range of the data) if *autorange* is True.

        Beyond the whiskers, data are considered outliers and are plotted as
        individual points.

    bootstrap : int, optional
        Number of times the confidence intervals around the median
        should be bootstrapped (percentile method).

    labels : list of str, optional
        Labels for each dataset. Length must be compatible with
        dimensions of *X*.

    autorange : bool, optional (False)
        When `True` and the data are distributed such that the 25th and 75th
        percentiles are equal, ``whis`` is set to (0, 100) such that the
        whisker ends are at the minimum and maximum of the data.

    Returns
    -------
    list of dict
        A list of dictionaries containing the results for each column
        of data. Keys of each dictionary are the following:

        ========   ===================================
        Key        Value Description
        ========   ===================================
        label      tick label for the boxplot
        mean       arithmetic mean value
        med        50th percentile
        q1         first quartile (25th percentile)
        q3         third quartile (75th percentile)
        iqr        interquartile range
        cilo       lower notch around the median
        cihi       upper notch around the median
        whislo     end of the lower whisker
        whishi     end of the upper whisker
        fliers     outliers
        ========   ===================================

    Notes
    -----
    Non-bootstrapping approach to confidence interval uses Gaussian-based
    asymptotic approximation:

    .. math::

        \\mathrm{med} \\pm 1.57 \\times \\frac{\\mathrm{iqr}}{\\sqrt{N}}

    General approach from:
    McGill, R., Tukey, J.W., and Larsen, W.A. (1978) "Variations of
    Boxplots", The American Statistician, 32:12-16.
    '''

ls_mapper: Incomplete
ls_mapper_r: Incomplete

def contiguous_regions(mask):
    """
    Return a list of (ind0, ind1) such that ``mask[ind0:ind1].all()`` is
    True and we cover all such regions.
    """
def is_math_text(s):
    """
    Return whether the string *s* contains math expressions.

    This is done by checking whether *s* contains an even number of
    non-escaped dollar signs.
    """
def _to_unmasked_float_array(x):
    """
    Convert a sequence to a float array; if input was a masked array, masked
    values are converted to nans.
    """
def _check_1d(x):
    """Convert scalars to 1D arrays; pass-through arrays as is."""
def _reshape_2D(X, name):
    """
    Use Fortran ordering to convert ndarrays and lists of iterables to lists of
    1D arrays.

    Lists of iterables are converted by applying `numpy.asanyarray` to each of
    their elements.  1D ndarrays are returned in a singleton list containing
    them.  2D ndarrays are converted to the list of their *columns*.

    *name* is used to generate the error message for invalid inputs.
    """
def violin_stats(X, method, points: int = 100, quantiles: Incomplete | None = None):
    """
    Return a list of dictionaries of data which can be used to draw a series
    of violin plots.

    See the ``Returns`` section below to view the required keys of the
    dictionary.

    Users can skip this function and pass a user-defined set of dictionaries
    with the same keys to `~.axes.Axes.violinplot` instead of using Matplotlib
    to do the calculations. See the *Returns* section below for the keys
    that must be present in the dictionaries.

    Parameters
    ----------
    X : array-like
        Sample data that will be used to produce the gaussian kernel density
        estimates. Must have 2 or fewer dimensions.

    method : callable
        The method used to calculate the kernel density estimate for each
        column of data. When called via ``method(v, coords)``, it should
        return a vector of the values of the KDE evaluated at the values
        specified in coords.

    points : int, default: 100
        Defines the number of points to evaluate each of the gaussian kernel
        density estimates at.

    quantiles : array-like, default: None
        Defines (if not None) a list of floats in interval [0, 1] for each
        column of data, which represents the quantiles that will be rendered
        for that column of data. Must have 2 or fewer dimensions. 1D array will
        be treated as a singleton list containing them.

    Returns
    -------
    list of dict
        A list of dictionaries containing the results for each column of data.
        The dictionaries contain at least the following:

        - coords: A list of scalars containing the coordinates this particular
          kernel density estimate was evaluated at.
        - vals: A list of scalars containing the values of the kernel density
          estimate at each of the coordinates given in *coords*.
        - mean: The mean value for this column of data.
        - median: The median value for this column of data.
        - min: The minimum value for this column of data.
        - max: The maximum value for this column of data.
        - quantiles: The quantile values for this column of data.
    """
def pts_to_prestep(x, *args):
    """
    Convert continuous line to pre-steps.

    Given a set of ``N`` points, convert to ``2N - 1`` points, which when
    connected linearly give a step function which changes values at the
    beginning of the intervals.

    Parameters
    ----------
    x : array
        The x location of the steps. May be empty.

    y1, ..., yp : array
        y arrays to be turned into steps; all must be the same length as ``x``.

    Returns
    -------
    array
        The x and y values converted to steps in the same order as the input;
        can be unpacked as ``x_out, y1_out, ..., yp_out``.  If the input is
        length ``N``, each of these arrays will be length ``2N + 1``. For
        ``N=0``, the length will be 0.

    Examples
    --------
    >>> x_s, y1_s, y2_s = pts_to_prestep(x, y1, y2)
    """
def pts_to_poststep(x, *args):
    """
    Convert continuous line to post-steps.

    Given a set of ``N`` points convert to ``2N + 1`` points, which when
    connected linearly give a step function which changes values at the end of
    the intervals.

    Parameters
    ----------
    x : array
        The x location of the steps. May be empty.

    y1, ..., yp : array
        y arrays to be turned into steps; all must be the same length as ``x``.

    Returns
    -------
    array
        The x and y values converted to steps in the same order as the input;
        can be unpacked as ``x_out, y1_out, ..., yp_out``.  If the input is
        length ``N``, each of these arrays will be length ``2N + 1``. For
        ``N=0``, the length will be 0.

    Examples
    --------
    >>> x_s, y1_s, y2_s = pts_to_poststep(x, y1, y2)
    """
def pts_to_midstep(x, *args):
    """
    Convert continuous line to mid-steps.

    Given a set of ``N`` points convert to ``2N`` points which when connected
    linearly give a step function which changes values at the middle of the
    intervals.

    Parameters
    ----------
    x : array
        The x location of the steps. May be empty.

    y1, ..., yp : array
        y arrays to be turned into steps; all must be the same length as
        ``x``.

    Returns
    -------
    array
        The x and y values converted to steps in the same order as the input;
        can be unpacked as ``x_out, y1_out, ..., yp_out``.  If the input is
        length ``N``, each of these arrays will be length ``2N``.

    Examples
    --------
    >>> x_s, y1_s, y2_s = pts_to_midstep(x, y1, y2)
    """

STEP_LOOKUP_MAP: Incomplete

def index_of(y):
    """
    A helper function to create reasonable x values for the given *y*.

    This is used for plotting (x, y) if x values are not explicitly given.

    First try ``y.index`` (assuming *y* is a `pandas.Series`), if that
    fails, use ``range(len(y))``.

    This will be extended in the future to deal with more types of
    labeled data.

    Parameters
    ----------
    y : float or array-like

    Returns
    -------
    x, y : ndarray
       The x and y values to plot.
    """
def safe_first_element(obj):
    """
    Return the first element in *obj*.

    This is a type-independent way of obtaining the first element,
    supporting both index access and the iterator protocol.
    """
def _safe_first_finite(obj):
    """
    Return the first finite element in *obj* if one is available and skip_nonfinite is
    True. Otherwise, return the first element.

    This is a method for internal use.

    This is a type-independent way of obtaining the first finite element, supporting
    both index access and the iterator protocol.
    """
def sanitize_sequence(data):
    """
    Convert dictview objects to list. Other inputs are returned unchanged.
    """
def normalize_kwargs(kw, alias_mapping: Incomplete | None = None):
    """
    Helper function to normalize kwarg inputs.

    Parameters
    ----------
    kw : dict or None
        A dict of keyword arguments.  None is explicitly supported and treated
        as an empty dict, to support functions with an optional parameter of
        the form ``props=None``.

    alias_mapping : dict or Artist subclass or Artist instance, optional
        A mapping between a canonical name to a list of aliases, in order of
        precedence from lowest to highest.

        If the canonical value is not in the list it is assumed to have the
        highest priority.

        If an Artist subclass or instance is passed, use its properties alias
        mapping.

    Raises
    ------
    TypeError
        To match what Python raises if invalid arguments/keyword arguments are
        passed to a callable.
    """
def _lock_path(path) -> Generator[None]:
    """
    Context manager for locking a path.

    Usage::

        with _lock_path(path):
            ...

    Another thread or process that attempts to lock the same path will wait
    until this context manager is exited.

    The lock is implemented by creating a temporary file in the parent
    directory, so that directory must exist and be writable.
    """
def _topmost_artist(artists, _cached_max=...):
    """
    Get the topmost artist of a list.

    In case of a tie, return the *last* of the tied artists, as it will be
    drawn on top of the others. `max` returns the first maximum in case of
    ties, so we need to iterate over the list in reverse order.
    """
def _str_equal(obj, s):
    """
    Return whether *obj* is a string equal to string *s*.

    This helper solely exists to handle the case where *obj* is a numpy array,
    because in such cases, a naive ``obj == s`` would yield an array, which
    cannot be used in a boolean context.
    """
def _str_lower_equal(obj, s):
    """
    Return whether *obj* is a string equal, when lowercased, to string *s*.

    This helper solely exists to handle the case where *obj* is a numpy array,
    because in such cases, a naive ``obj == s`` would yield an array, which
    cannot be used in a boolean context.
    """
def _array_perimeter(arr):
    """
    Get the elements on the perimeter of *arr*.

    Parameters
    ----------
    arr : ndarray, shape (M, N)
        The input array.

    Returns
    -------
    ndarray, shape (2*(M - 1) + 2*(N - 1),)
        The elements on the perimeter of the array::

           [arr[0, 0], ..., arr[0, -1], ..., arr[-1, -1], ..., arr[-1, 0], ...]

    Examples
    --------
    >>> i, j = np.ogrid[:3, :4]
    >>> a = i*10 + j
    >>> a
    array([[ 0,  1,  2,  3],
           [10, 11, 12, 13],
           [20, 21, 22, 23]])
    >>> _array_perimeter(a)
    array([ 0,  1,  2,  3, 13, 23, 22, 21, 20, 10])
    """
def _unfold(arr, axis, size, step):
    """
    Append an extra dimension containing sliding windows along *axis*.

    All windows are of size *size* and begin with every *step* elements.

    Parameters
    ----------
    arr : ndarray, shape (N_1, ..., N_k)
        The input array
    axis : int
        Axis along which the windows are extracted
    size : int
        Size of the windows
    step : int
        Stride between first elements of subsequent windows.

    Returns
    -------
    ndarray, shape (N_1, ..., 1 + (N_axis-size)/step, ..., N_k, size)

    Examples
    --------
    >>> i, j = np.ogrid[:3, :7]
    >>> a = i*10 + j
    >>> a
    array([[ 0,  1,  2,  3,  4,  5,  6],
           [10, 11, 12, 13, 14, 15, 16],
           [20, 21, 22, 23, 24, 25, 26]])
    >>> _unfold(a, axis=1, size=3, step=2)
    array([[[ 0,  1,  2],
            [ 2,  3,  4],
            [ 4,  5,  6]],
           [[10, 11, 12],
            [12, 13, 14],
            [14, 15, 16]],
           [[20, 21, 22],
            [22, 23, 24],
            [24, 25, 26]]])
    """
def _array_patch_perimeters(x, rstride, cstride):
    """
    Extract perimeters of patches from *arr*.

    Extracted patches are of size (*rstride* + 1) x (*cstride* + 1) and
    share perimeters with their neighbors. The ordering of the vertices matches
    that returned by ``_array_perimeter``.

    Parameters
    ----------
    x : ndarray, shape (N, M)
        Input array
    rstride : int
        Vertical (row) stride between corresponding elements of each patch
    cstride : int
        Horizontal (column) stride between corresponding elements of each patch

    Returns
    -------
    ndarray, shape (N/rstride * M/cstride, 2 * (rstride + cstride))
    """
def _setattr_cm(obj, **kwargs) -> Generator[None]:
    """
    Temporarily set some attributes; restore original state at context exit.
    """

class _OrderedSet(collections.abc.MutableSet):
    _od: Incomplete
    def __init__(self) -> None: ...
    def __contains__(self, key) -> bool: ...
    def __iter__(self): ...
    def __len__(self) -> int: ...
    def add(self, key) -> None: ...
    def discard(self, key) -> None: ...

def _premultiplied_argb32_to_unmultiplied_rgba8888(buf):
    """
    Convert a premultiplied ARGB32 buffer to an unmultiplied RGBA8888 buffer.
    """
def _unmultiplied_rgba8888_to_premultiplied_argb32(rgba8888):
    """
    Convert an unmultiplied RGBA8888 buffer to a premultiplied ARGB32 buffer.
    """
def _get_nonzero_slices(buf):
    """
    Return the bounds of the nonzero region of a 2D array as a pair of slices.

    ``buf[_get_nonzero_slices(buf)]`` is the smallest sub-rectangle in *buf*
    that encloses all non-zero entries in *buf*.  If *buf* is fully zero, then
    ``(slice(0, 0), slice(0, 0))`` is returned.
    """
def _pformat_subprocess(command):
    """Pretty-format a subprocess command for printing/logging purposes."""
def _check_and_log_subprocess(command, logger, **kwargs):
    """
    Run *command*, returning its stdout output if it succeeds.

    If it fails (exits with nonzero return code), raise an exception whose text
    includes the failed command and captured stdout and stderr output.

    Regardless of the return code, the command is logged at DEBUG level on
    *logger*.  In case of success, the output is likewise logged.
    """
def _setup_new_guiapp() -> None:
    """
    Perform OS-dependent setup when Matplotlib creates a new GUI application.
    """
def _format_approx(number, precision):
    """
    Format the number with at most the number of decimals given as precision.
    Remove trailing zeros and possibly the decimal point.
    """
def _g_sig_digits(value, delta):
    """
    Return the number of significant digits to %g-format *value*, assuming that
    it is known with an error of *delta*.
    """
def _unikey_or_keysym_to_mplkey(unikey, keysym):
    """
    Convert a Unicode key or X keysym to a Matplotlib key name.

    The Unicode key is checked first; this avoids having to list most printable
    keysyms such as ``EuroSign``.
    """
def _make_class_factory(mixin_class, fmt, attr_name: Incomplete | None = None):
    """
    Return a function that creates picklable classes inheriting from a mixin.

    After ::

        factory = _make_class_factory(FooMixin, fmt, attr_name)
        FooAxes = factory(Axes)

    ``Foo`` is a class that inherits from ``FooMixin`` and ``Axes`` and **is
    picklable** (picklability is what differentiates this from a plain call to
    `type`).  Its ``__name__`` is set to ``fmt.format(Axes.__name__)`` and the
    base class is stored in the ``attr_name`` attribute, if not None.

    Moreover, the return value of ``factory`` is memoized: calls with the same
    ``Axes`` class always return the same subclass.
    """
def _picklable_class_constructor(mixin_class, fmt, attr_name, base_class):
    """Internal helper for _make_class_factory."""
def _is_torch_array(x):
    """Check if 'x' is a PyTorch Tensor."""
def _is_jax_array(x):
    """Check if 'x' is a JAX Array."""
def _is_tensorflow_array(x):
    """Check if 'x' is a TensorFlow Tensor or Variable."""
def _unpack_to_numpy(x):
    """Internal helper to extract data from e.g. pandas and xarray objects."""
def _auto_format_str(fmt, value):
    """
    Apply *value* to the format string *fmt*.

    This works both with unnamed %-style formatting and
    unnamed {}-style formatting. %-style formatting has priority.
    If *fmt* is %-style formattable that will be used. Otherwise,
    {}-formatting is applied. Strings without formatting placeholders
    are passed through as is.

    Examples
    --------
    >>> _auto_format_str('%.2f m', 0.2)
    '0.20 m'
    >>> _auto_format_str('{} m', 0.2)
    '0.2 m'
    >>> _auto_format_str('const', 0.2)
    'const'
    >>> _auto_format_str('%d or {}', 0.2)
    '0 or {}'
    """
def _is_pandas_dataframe(x):
    """Check if 'x' is a Pandas DataFrame."""
