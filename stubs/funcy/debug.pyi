from .decorators import decorator
from _typeshed import Incomplete
from collections.abc import Callable, Generator
from typing import Any, Self
import types

__all__ = ['tap', 'log_calls', 'print_calls', 'log_enters', 'print_enters', 'log_exits', 'print_exits', 'log_errors', 'print_errors', 'log_durations', 'print_durations', 'log_iter_durations', 'print_iter_durations']

def tap(x, label=None):
    """Prints x and then returns it."""
@decorator
def log_calls(call, print_func, errors: bool = True, stack: bool = True, repr_len=...):
    """Logs or prints all function calls, including arguments, results and raised exceptions."""
def print_calls(errors: bool = True, stack: bool = True, repr_len=...): ...
@decorator
def log_enters(call, print_func, repr_len=...):
    """Logs each entrance to a function."""
def print_enters(repr_len=...):
    """Prints on each entrance to a function."""
@decorator
def log_exits(call, print_func, errors: bool = True, stack: bool = True, repr_len=...):
    """Logs exits from a function."""
def print_exits(errors: bool = True, stack: bool = True, repr_len=...):
    """Prints on exits from a function."""

class LabeledContextDecorator:
    """
    A context manager which also works as decorator, passing call signature as its label.
    """
    print_func: Incomplete
    label: Incomplete
    repr_len: Incomplete
    def __init__(self, print_func, label=None, repr_len=...) -> None: ...
    def __call__(self, label=None, **kwargs) -> Callable[..., Any] | Self: ...
    def decorator(self, func) -> Callable[..., Any]: ...
class log_errors(LabeledContextDecorator):
    """Logs or prints all errors within a function or block."""
    stack: Incomplete
    def __init__(self, print_func, label=None, stack: bool = True, repr_len=...) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, tb: types.TracebackType | None) -> None: ...

print_errors: Incomplete

class log_durations(LabeledContextDecorator):
    """Times each function call or block execution."""
    format_time: Incomplete
    threshold: Incomplete
    def __init__(self, print_func, label=None, unit: str = 'auto', threshold: int = -1, repr_len=...) -> None: ...
    start: Incomplete
    def __enter__(self) -> Self: ...
    def __exit__(self, *exc) -> None: ...

print_durations: Incomplete

def log_iter_durations(seq, print_func, label=None, unit: str = 'auto') -> Generator[Incomplete]:
    """Times processing of each item in seq."""
def print_iter_durations(seq, label=None, unit: str = 'auto') -> Generator[Incomplete]:
    """Times processing of each item in seq."""
    ...

def signature_repr(call, repr_len=...) -> str:
    ...

def smart_repr(value, max_len=...) -> str:
    ...

