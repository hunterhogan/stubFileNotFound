from .context_manager import ContextManager as ContextManager
from _typeshed import Incomplete
from typing import Any

class DelegatingContextManager(ContextManager[Any]):
    """
    Object which delegates its context manager interface to another object.

    You set the delegatee context manager as `self.delegatee_context_manager`,
    and whenever someone tries to use the current object as a context manager,
    the `__enter__` and `__exit__` methods of the delegatee object will be
    called. No other methods of the delegatee will be used.

    This is useful when you are tempted to inherit from some context manager
    class, but you don't to inherit all the other methods that it defines.
    """

    delegatee_context_manager: Incomplete
    __enter__: Incomplete
    __exit__: Incomplete



