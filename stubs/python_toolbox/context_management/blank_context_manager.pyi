from .context_manager import ContextManager as ContextManager
from _typeshed import Incomplete
from collections.abc import Generator
from typing import Any
import abc

class BlankContextManager(ContextManager[Any], metaclass=abc.ABCMeta):
    """A context manager that does nothing."""

    def manage_context(self) -> Generator[Incomplete]: ...



