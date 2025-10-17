from typing import Any
import abc
import types

class AbstractContextManager(metaclass=abc.ABCMeta):
    """
    A no-frills context manager.

    This class is used mostly to check whether an object is a context manager:

        >>> isinstance(threading.Lock(), AbstractContextManager)
        True

    """

    @abc.abstractmethod
    def __enter__(self) -> Any:
        """Prepare for suite execution."""
    @abc.abstractmethod
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, exc_traceback: types.TracebackType | None) -> Any:
        """Cleanup after suite execution."""
    @classmethod
    def __subclasshook__(cls, candidate_class: Any) -> Any: ...



