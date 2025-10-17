from _typeshed import Incomplete
from typing import Any
import io
import pathlib

_email_pattern: Incomplete

def is_subclass(candidate: Any, base_class: Any) -> Any:
    """
    Check if `candidate` is a subclass of `base_class`.

    You may pass in a tuple of base classes instead of just one, and it will
    check whether `candidate` is a subclass of any of these base classes.

    This has the advantage that it doesn't throw an exception if `candidate` is
    not a type. (Python issue 10569.)
    """
def get_mro_depth_of_method(type_: Any, method_name: Any) -> Any:
    """
    Get the mro-depth of a method.

    This means, the index number in `type_`'s MRO of the base class that
    defines this method.
    """
def getted_vars(thing: Any, _getattr: Any=...) -> Any:
    """
    The `vars` of an object, but after we used `getattr` to get them.

    This is useful because some magic (like descriptors or `__getattr__`
    methods) need us to use `getattr` for them to work. For example, taking
    just the `vars` of a class will show functions instead of methods, while
    the "getted vars" will have the actual method objects.

    You may provide a replacement for the built-in `getattr` as the `_getattr`
    argument.
    """

_ascii_variable_pattern: Incomplete

def is_legal_ascii_variable_name(name: Any) -> Any:
    """Return whether `name` is a legal name for a Python variable."""
def is_magic_variable_name(name: Any) -> Any:
    """Return whether `name` is a name of a magic variable (e.g. '__add__'.)."""
def get_actual_type(thing: Any) -> Any:
    """
    Get the actual type (or class) of an object.

    This used to be needed instead of `type(thing)` in Python 2.x where we had
    old-style classes. In Python 3.x we don't have them anymore, but keeping
    this function for backward compatibility.
    """
def is_number(x: Any) -> Any:
    """Return whether `x` is a number."""
def identity_function(thing: Any) -> Any:
    """
    Return `thing`.

    This function is useful when you want to use an identity function but can't
    define a lambda one because it wouldn't be pickleable. Also using this
    function might be faster as it's prepared in advance.
    """
def do_nothing(*args: Any, **kwargs: Any) -> None: ...

class OwnNameDiscoveringDescriptor:
    """A descriptor that can discover the name it's bound to on its object."""

    our_name: Incomplete
    def __init__(self, name: Any=None) -> None:
        """
        Construct the `OwnNameDiscoveringDescriptor`.

        You may optionally pass in the name that this property has in the
        class; this will save a bit of processing later.
        """
    def get_our_name(self, thing: Any, our_type: Any=None) -> Any: ...

def find_clear_place_on_circle(circle_points: Any, circle_size: int = 1) -> Any:
    """
    Find the point on a circle that's the farthest away from other points.

    Given an interval `(0, circle_size)` and a bunch of points in it, find a
    place for a new point that is as far away from the other points as
    possible. (Since this is a circle, there's wraparound, e.g. the end of the
    interval connects to the start.)
    """
def add_extension_if_plain(path: Any, extension: Any) -> Any:
    """Add `extension` to a file path if it doesn't have an extension."""
def general_sum(things: Any, start: Any=None) -> Any:
    """
    Sum a bunch of objects, adding them to each other.

    This is like the built-in `sum`, except it works for many types, not just
    numbers.
    """
def general_product(things: Any, start: Any=None) -> Any:
    """Multiply a bunch of objects by each other, not necessarily numbers."""
def is_legal_email_address(email_address_candidate: Any) -> Any:
    """Is `email_address_candidate` a legal email address?"""
def is_type(thing: Any) -> Any:
    """Is `thing` a class? Allowing both new-style and old-style classes."""

class NonInstantiable:
    """
    Class that can't be instatiated.

    Inherit from this for classes that should never be instantiated, like
    constants and settings.
    """

    def __new__(self, *args: Any, **kwargs: Any) -> None: ...

def repeat_getattr(thing: Any, query: Any) -> Any:
    """
    Perform a repeated `getattr` operation.

    i.e., when given `repeat_getattr(x, '.y.z')`, will return `x.y.z`.
    """
def set_attributes(**kwargs: Any) -> Any:
    """
    Decorator to set attributes on a function.

    Example:

        @set_attributes(meow='frrr')
        def f():
            return 'whatever'

        assert f.meow == 'frrr'

    """

_decimal_number_pattern: Incomplete

def decimal_number_from_string(string: Any) -> Any:
    """
    Turn a string like '7' or '-32.55' into the corresponding number.

    Ensures that it was given a number. (This might be more secure than using
    something like `int` directly.)

    Uses `int` for ints and `float` for floats.
    """

class AlternativeLengthMixin:
    """
    Mixin for sized types that makes it easy to return non-standard lengths.

    Due to CPython limitation, Python's built-in `__len__` (and its counterpart
    `len`) can't return really big values or floating point numbers.

    Classes which need to return such lengths can use this mixin. They'll have
    to define a property `length` where they return their length, and if
    someone tries to call `len` on it, then if the length happens to be a
    number that `len` supports, it'll return that, otherwise it'll show a
    helpful error message.
    """

    def __len__(self) -> int: ...
    def __bool__(self) -> bool: ...

class RotatingLogStream:
    """
    A stream that writes to a log file with automatic rotation.

    This class implements a file-like object that writes log messages to a file,
    automatically rotating it when it gets too large. Each log entry is prefixed
    with a timestamp.

    The log file will be rotated when it exceeds max_size_in_mb (default 10MB).
    When rotation occurs, the existing log is renamed to .old and a new log file
    is started.

    Args:
        log_path: Path where the log file will be written
        original_stream: Optional stream to also write output to (e.g. sys.stdout)
        max_size_in_mb: Maximum size of log file before rotation, in megabytes

    Example:
        >>> stream = RotatingLogStream('app.log')
        >>> stream.write('Hello world')  # Writes timestamped message to log
        >>> RotatingLogStream.install('app.log')  # Replace stdout/stderr
    """

    log_path: Incomplete
    old_log_path: Incomplete
    original_stream: Incomplete
    _write_count: int
    max_size_in_bytes: Incomplete
    def __init__(self, log_path: pathlib.Path, original_stream: io.TextIOBase | None = None, max_size_in_mb: int = 10) -> None: ...
    def write(self, s: Any) -> None: ...
    def flush(self) -> None: ...
    @staticmethod
    def install(log_path: pathlib.Path) -> None: ...



