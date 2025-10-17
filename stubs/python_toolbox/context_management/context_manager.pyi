from .abstract_context_manager import AbstractContextManager as AbstractContextManager
from .context_manager_type import ContextManagerType as ContextManagerType
from .mixins import _DecoratingContextManagerMixin as _DecoratingContextManagerMixin
from _typeshed import Incomplete
from typing import Any
import abc
import types

class ContextManager(AbstractContextManager, _DecoratingContextManagerMixin, metaclass=ContextManagerType):
    """
    Allows running preparation code before a given suite and cleanup after.

    To make a context manager, use `ContextManager` as a base class and either
    (a) define `__enter__` and `__exit__` methods or (b) define a
    `manage_context` method that returns a generator. An alternative way to
    create a context manager is to define a generator function and decorate it
    with `ContextManagerType`.

    In any case, the resulting context manager could be called either with the
    `with` keyword or by using it as a decorator to a function.

    For more details, see documentation of the containing module,
    `python_toolbox.context_manager`.
    """

    @abc.abstractmethod
    def __enter__(self) -> Any:
        """Prepare for suite execution."""
    @abc.abstractmethod
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, exc_traceback: types.TracebackType | None) -> Any:
        """Cleanup after suite execution."""
    _ContextManager__args: Incomplete
    _ContextManager__kwargs: Incomplete
    _ContextManager__generators: Incomplete
    def __init_lone_manage_context(self, *args: Any, **kwargs: Any) -> None:
        """Initialize a `ContextManager` made from a lone generator function."""
    def __enter_using_manage_context(self) -> Any:
        """
        Prepare for suite execution.

        This is used as `__enter__` for context managers that use a
        `manage_context` function.
        """
    def __exit_using_manage_context(self, exc_type: Any, exc_value: Any, exc_traceback: Any) -> Any:
        """
        Cleanup after suite execution.

        This is used as `__exit__` for context managers that use a
        `manage_context` function.
        """



