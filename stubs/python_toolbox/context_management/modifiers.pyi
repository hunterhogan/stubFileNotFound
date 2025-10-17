from .context_manager import ContextManager as ContextManager
from _typeshed import Incomplete
from typing import Any
import abc

def as_idempotent(context_manager: Any) -> Any:
    """
    Wrap a context manager so repeated calls to enter and exit will be ignored.

    This means that if you call `__enter__` a second time on the context
    manager, nothing will happen. The `__enter__` method won't be called and an
    exception would not be raised. Same goes for the `__exit__` method, after
    calling it once, if you try to call it again it will be a no-op. But now
    that you've called `__exit__` you can call `__enter__` and it will really
    do the enter action again, and then `__exit__` will be available again,
    etc.

    This is useful when you have a context manager that you want to put in an
    `ExitStack`, but you also possibly want to exit it manually before the
    `ExitStack` closes. This way you don't risk an exception by having the
    context manager exit twice.

    Note: The first value returned by `__enter__` will be returned by all the
    subsequent no-op `__enter__` calls.

    This can be used when calling an existing context manager:

        with as_idempotent(some_context_manager):
            # Now we're idempotent!

    Or it can be used when defining a context manager to make it idempotent:

        @as_idempotent
        class MyContextManager(ContextManager):
            def __enter__(self):
                # ...
            def __exit__(self, exc_type, exc_value, exc_traceback):
                # ...

    And also like this...


        @as_idempotent
        @ContextManagerType
        def Meow():
            yield # ...

    """
def as_reentrant(context_manager: Any) -> Any:
    """
    Wrap a context manager to make it reentant.

    A context manager wrapped with `as_reentrant` could be entered multiple
    times, and only after it's been exited the same number of times that it has
    been entered will the original `__exit__` method be called.

    Note: The first value returned by `__enter__` will be returned by all the
    subsequent no-op `__enter__` calls.

    This can be used when calling an existing context manager:

        with as_reentrant(some_context_manager):
            # Now we're reentrant!

    Or it can be used when defining a context manager to make it reentrant:

        @as_reentrant
        class MyContextManager(ContextManager):
            def __enter__(self):
                # ...
            def __exit__(self, exc_type, exc_value, exc_traceback):
                # ...

    And also like this...


        @as_reentrant
        @ContextManagerType
        def Meow():
            yield # ...

    """

class _ContextManagerWrapper(ContextManager[Any], metaclass=abc.ABCMeta):
    _enter_value: Incomplete
    __wrapped__: Incomplete
    _wrapped_enter: Incomplete
    _wrapped_exit: Incomplete
    def __init__(self, wrapped_context_manager: Any) -> None: ...
    @classmethod
    def _wrap_context_manager_or_class(cls, thing: Any) -> Any: ...

class _IdempotentContextManager(_ContextManagerWrapper):
    _entered: bool
    _enter_value: Incomplete
    def __enter__(self) -> Any: ...
    def __exit__(self, exc_type: Any=None, exc_value: Any=None, exc_traceback: Any=None) -> Any: ...

class _ReentrantContextManager(_ContextManagerWrapper):
    depth: Incomplete
    _enter_value: Incomplete
    def __enter__(self) -> Any: ...
    def __exit__(self, exc_type: Any=None, exc_value: Any=None, exc_traceback: Any=None) -> Any: ...



