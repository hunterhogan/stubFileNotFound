import pickle
from _typeshed import Incomplete

DEFAULT_PROTOCOL: Incomplete
PYPY: Incomplete
builtin_code_type: Incomplete

def register_pickle_by_value(module) -> None:
    """Register a module to make its functions and classes picklable by value.

    By default, functions and classes that are attributes of an importable
    module are to be pickled by reference, that is relying on re-importing
    the attribute from the module at load time.

    If `register_pickle_by_value(module)` is called, all its functions and
    classes are subsequently to be pickled by value, meaning that they can
    be loaded in Python processes where the module is not importable.

    This is especially useful when developing a module in a distributed
    execution environment: restarting the client Python process with the new
    source code is enough: there is no need to re-install the new version
    of the module on all the worker nodes nor to restart the workers.

    Note: this feature is considered experimental. See the cloudpickle
    README.md file for more details and limitations.
    """
def unregister_pickle_by_value(module) -> None:
    """Unregister that the input module should be pickled by value."""
def list_registry_pickle_by_value(): ...

STORE_GLOBAL: Incomplete
DELETE_GLOBAL: Incomplete
LOAD_GLOBAL: Incomplete
GLOBAL_OPS: Incomplete
HAVE_ARGUMENT: Incomplete
EXTENDED_ARG: Incomplete

def is_tornado_coroutine(func):
    """Return whether `func` is a Tornado coroutine function.

    Running coroutines are not supported.
    """
def subimport(name): ...
def dynamic_subimport(name, vars): ...
def instance(cls):
    """Create a new instance of a class.

    Parameters
    ----------
    cls : type
        The class to create an instance of.

    Returns
    -------
    instance : cls
        A new instance of ``cls``.
    """

class _empty_cell_value:
    """Sentinel for empty closures."""
    @classmethod
    def __reduce__(cls): ...

class Pickler(pickle.Pickler):
    dispatch_table: Incomplete
    def dump(self, obj): ...
    globals_ref: Incomplete
    proto: Incomplete
    def __init__(self, file, protocol=None, buffer_callback=None) -> None: ...
    dispatch = dispatch_table
    def reducer_override(self, obj):
        """Type-agnostic reducing callback for function and classes.

            For performance reasons, subclasses of the C `pickle.Pickler` class
            cannot register custom reducers for functions and classes in the
            dispatch_table attribute. Reducers for such types must instead
            implemented via the special `reducer_override` method.

            Note that this method will be called for any object except a few
            builtin-types (int, lists, dicts etc.), which differs from reducers
            in the Pickler's dispatch_table, each of them being invoked for
            objects of a specific type only.

            This property comes in handy for classes: although most classes are
            instances of the ``type`` metaclass, some of them can be instances
            of other custom metaclasses (such as enum.EnumMeta for example). In
            particular, the metaclass will likely not be known in advance, and
            thus cannot be special-cased using an entry in the dispatch_table.
            reducer_override, among other things, allows us to register a
            reducer that will be called for any class, independently of its
            type.

            Notes:

            * reducer_override has the priority over dispatch_table-registered
            reducers.
            * reducer_override can be used to fix other limitations of
              cloudpickle for other types that suffered from type-specific
              reducers, such as Exceptions. See
              https://github.com/cloudpipe/cloudpickle/issues/248
            """
    def save_global(self, obj, name=None, pack=...):
        """Main dispatch method.

            The name of this method is somewhat misleading: all types get
            dispatched here.
            """
    def save_function(self, obj, name=None):
        """Registered with the dispatch to handle all function types.

            Determines what kind of function obj is (e.g. lambda, defined at
            interactive prompt, etc) and handles the pickling appropriately.
            """
    def save_pypy_builtin_func(self, obj) -> None:
        """Save pypy equivalent of builtin functions.

            PyPy does not have the concept of builtin-functions. Instead,
            builtin-functions are simple function instances, but with a
            builtin-code attribute.
            Most of the time, builtin functions should be pickled by attribute.
            But PyPy has flaky support for __qualname__, so some builtin
            functions such as float.__new__ will be classified as dynamic. For
            this reason only, we created this special routine. Because
            builtin-functions are not expected to have closure or globals,
            there is no additional hack (compared the one already implemented
            in pickle) to protect ourselves from reference cycles. A simple
            (reconstructor, newargs, obj.__dict__) tuple is save_reduced.  Note
            also that PyPy improved their support for __qualname__ in v3.6, so
            this routing should be removed when cloudpickle supports only PyPy
            3.6 and later.
            """

def dump(obj, file, protocol=None, buffer_callback=None) -> None:
    """Serialize obj as bytes streamed into file

    protocol defaults to cloudpickle.DEFAULT_PROTOCOL which is an alias to
    pickle.HIGHEST_PROTOCOL. This setting favors maximum communication
    speed between processes running the same Python version.

    Set protocol=pickle.DEFAULT_PROTOCOL instead if you need to ensure
    compatibility with older versions of Python (although this is not always
    guaranteed to work because cloudpickle relies on some internal
    implementation details that can change from one Python version to the
    next).
    """
def dumps(obj, protocol=None, buffer_callback=None):
    """Serialize obj as a string of bytes allocated in memory

    protocol defaults to cloudpickle.DEFAULT_PROTOCOL which is an alias to
    pickle.HIGHEST_PROTOCOL. This setting favors maximum communication
    speed between processes running the same Python version.

    Set protocol=pickle.DEFAULT_PROTOCOL instead if you need to ensure
    compatibility with older versions of Python (although this is not always
    guaranteed to work because cloudpickle relies on some internal
    implementation details that can change from one Python version to the
    next).
    """

load: Incomplete
loads: Incomplete
CloudPickler = Pickler
