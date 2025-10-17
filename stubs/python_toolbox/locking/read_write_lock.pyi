from . import original_read_write_lock
from _typeshed import Incomplete
from python_toolbox import context_management
from typing import Any
import types

__all__ = ['ReadWriteLock']

class ContextManager(context_management.ContextManager[Any]):
    lock: Incomplete
    acquire_func: Incomplete
    def __init__(self, lock: Any, acquire_func: Any) -> None: ...
    def __enter__(self) -> Any: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, exc_traceback: types.TracebackType | None) -> None: ...

class ReadWriteLock(original_read_write_lock.ReadWriteLock):
    """
    A ReadWriteLock subclassed from a different ReadWriteLock class defined
    in the module original_read_write_lock.py, (See the documentation of the
    original class for more details.).

    This subclass adds two context managers, one for reading and one for
    writing.

    Usage:

        read_write_lock = ReadWriteLock()
        with read_write_lock.read:
            pass # perform read operations here
        with read_write_lock.write:
            pass # perform write operations here

    """

    read: Incomplete
    write: Incomplete
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...



