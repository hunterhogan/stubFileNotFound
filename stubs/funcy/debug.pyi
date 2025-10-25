from .decorators import decorator
from _typeshed import Incomplete
from collections.abc import Callable, Generator
from typing import Any, Self
import types

__all__ = ['tap', 'log_calls', 'print_calls', 'log_enters', 'print_enters', 'log_exits', 'print_exits', 'log_errors', 'print_errors', 'log_durations', 'print_durations', 'log_iter_durations', 'print_iter_durations']

def tap(x: Any, label: Any=None) -> Any:
    """Prints x and then returns it."""
@decorator
def log_calls(call: Any, print_func: Any, errors: bool = True, stack: bool = True, repr_len: Any=...) -> Any:
    """Logs or prints all function calls, including arguments, results and raised exceptions."""
def print_calls(errors: bool = True, stack: bool = True, repr_len: Any=...) -> Any: ...
@decorator
def log_enters(call: Any, print_func: Any, repr_len: Any=...) -> Any:
    """Logs each entrance to a function."""
def print_enters(repr_len: Any=...) -> Any:
    """Prints on each entrance to a function."""
@decorator
def log_exits(call: Any, print_func: Any, errors: bool = True, stack: bool = True, repr_len: Any=...) -> Any:
    """Logs exits from a function."""
def print_exits(errors: bool = True, stack: bool = True, repr_len: Any=...) -> Any:
    """Prints on exits from a function."""

class LabeledContextDecorator:
    """
    A context manager which also works as decorator, passing call signature as its label.
    """
    print_func: Incomplete
    label: Incomplete
    repr_len: Incomplete
    def __init__(self, print_func: Any, label: Any=None, repr_len: Any=...) -> None: ...
    def __call__(self, label: Any=None, **kwargs: Any) -> Callable[..., Any] | Self: ...
    def decorator(self, func: Any) -> Callable[..., Any]: ...
class log_errors(LabeledContextDecorator):
    """Logs or prints all errors within a function or block."""
    stack: Incomplete
    def __init__(self, print_func: Any, label: Any=None, stack: bool = True, repr_len: Any=...) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, tb: types.TracebackType | None) -> None: ...

print_errors: Incomplete

class log_durations(LabeledContextDecorator):
    """Times each function call or block execution."""
    format_time: Incomplete
    threshold: Incomplete
    def __init__(self, print_func: Any, label: Any=None, unit: str = 'auto', threshold: int = -1, repr_len: Any=...) -> None: ...
    start: Incomplete
    def __enter__(self) -> Self: ...
    def __exit__(self, *exc: Any) -> None: ...

print_durations: Incomplete

def log_iter_durations(seq: Any, print_func: Any, label: Any=None, unit: str = 'auto') -> Generator[Incomplete]:
    """Times processing of each item in seq."""
def print_iter_durations(seq: Any, label: Any=None, unit: str = 'auto') -> Generator[Incomplete]:
    """Times processing of each item in seq."""
    ...

def signature_repr(call: Any, repr_len: Any=...) -> str:
    ...

def smart_repr(value: Any, max_len: Any=...) -> str:
    ...

