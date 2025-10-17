from .context_manager_type_type import ContextManagerTypeType as ContextManagerTypeType
from typing import Any
import abc

class ContextManagerType(abc.ABCMeta, metaclass=ContextManagerTypeType):
    """
    Metaclass for `ContextManager`.

    Use this directly as a decorator to create a `ContextManager` from a
    generator function.

    Example:

        @ContextManagerType
        def MyContextManager():
            # preparation
            try:
                yield
            finally:
                pass # cleanup

    The resulting context manager could be called either with the `with`
    keyword or by using it as a decorator to a function.

    For more details, see documentation of the containing module,
    `python_toolbox.context_manager`.
    """

    def __new__(mcls: Any, name: Any, bases: Any, namespace: Any) -> Any:
        """
        Create either `ContextManager` itself or a subclass of it.

        For subclasses of `ContextManager`, if a `manage_context` method is
        available, we will use `__enter__` and `__exit__` that will use the
        generator returned by `manage_context`.
        """
    def __is_the_base_context_manager_class(cls) -> Any:
        """
        Return whether `cls` is `ContextManager`.

        It's an ugly method, but unfortunately it's necessary because at one
        point we want to test if a class is `ContextManager` before
        `ContextManager` is defined in this module.
        """



