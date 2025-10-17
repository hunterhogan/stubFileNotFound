from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

__all__ = ['Connection', 'get_current_connection', 'pop_connection', 'push_connection', 'use_connection']

class NoSMTPConnectionException(Exception): ...

@contextmanager
def Connection(connection: Any) -> Generator[None]: ...
def push_connection(connection: Any) -> None:
    """Pushes the given connection on the stack."""
def pop_connection() -> Any:
    """Pops the topmost connection from the stack."""
def use_connection(connection: Any) -> None:
    """Clears the stack and uses the given connection.  Protects against mixed
    use of use_connection() and stacked connection contexts.
    """
def get_current_connection() -> Any:
    """Returns the current SMTP connection (i.e. the topmost on the
    connection stack).
    """



