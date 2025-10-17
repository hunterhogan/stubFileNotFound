from _typeshed import Incomplete
from python_toolbox import context_management as context_management, misc_tools as misc_tools
from typing import Any
import types

class DelegateeContextManager(context_management.ContextManager[Any]):
    """Inner context manager used internally by `Freezer`."""

    freezer: Incomplete
    def __init__(self, freezer: Any) -> None:
        """
        Construct the `DelegateeContextManager`.

        `freezer` is the freezer to which we belong.
        """
    def __enter__(self) -> Any:
        """Call the freezer's freeze handler."""
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, exc_traceback: types.TracebackType | None) -> Any:
        """Call the freezer's thaw handler."""
    depth: Incomplete



