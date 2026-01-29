from _typeshed import Incomplete
from numba.core import errors as errors, types as types
from numba.core.typing.typeof import typeof as typeof
from numba.core.utils import PYVERSION as PYVERSION

class AsNumbaTypeRegistry:
    """
    A registry for Python types. It stores a lookup table for simple cases
    (e.g. ``int``) and a list of functions for more complicated cases (e.g.
    generics like ``List[int]``).

    Python types are used in Python type annotations, and in instance checks.
    Therefore, this registry supports determining the Numba type of Python type
    annotations at compile time, along with determining the type of classinfo
    arguments to ``isinstance()``.

    This registry is not used dynamically on instances at runtime; to check the
    type of an object at runtime, use ``numba.typeof``.
    """

    lookup: Incomplete
    functions: Incomplete
    def __init__(self) -> None: ...
    def _numba_type_infer(self, py_type): ...
    def _builtin_infer(self, py_type): ...
    def register(self, func_or_py_type, numba_type=None) -> None:
        """
        Add support for new Python types (e.g. user-defined JitClasses) to the
        registry. For a simple pair of a Python type and a Numba type, this can
        be called as a function ``register(py_type, numba_type)``. If more
        complex logic is required (e.g. for generic types), ``register`` can be
        used as a decorator for a function that takes a Python type as input
        and returns a Numba type or ``None``.
        """
    def try_infer(self, py_type):
        """
        Try to determine the Numba type of a given Python type. We first
        consider the lookup dictionary. If ``py_type`` is not there, we iterate
        through the registered functions until one returns a Numba type.  If
        type inference fails, return ``None``.
        """
    def infer(self, py_type): ...
    def __call__(self, py_type): ...

as_numba_type: Incomplete
