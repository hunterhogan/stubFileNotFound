import types
from .decorators import contextmanager, decorator
from _typeshed import Incomplete
from collections.abc import Generator
from contextlib import nullcontext as nullcontext, suppress as suppress

__all__ = ['raiser', 'ignore', 'silent', 'suppress', 'nullcontext', 'reraise', 'retry', 'fallback', 'limit_error_rate', 'ErrorRateExceeded', 'throttle', 'post_processing', 'collecting', 'joining', 'once', 'once_per', 'once_per_args', 'wrap_with']

def raiser(exception_or_class=..., *args, **kwargs):
    """Constructs function that raises the given exception
       with given arguments on any invocation."""
def ignore(errors, default=None):
    """Alters function to ignore given errors, returning default instead."""
def silent(func):
    """Alters function to ignore all exceptions."""

class nullcontext:
    """Context manager that does no additional processing.

        Used as a stand-in for a normal context manager, when a particular
        block of code is only sometimes used with a normal context manager:

        cm = optional_cm if condition else nullcontext()
        with cm:
            # Perform operation, using optional_cm if condition is True
        """
    enter_result: Incomplete
    def __init__(self, enter_result=None) -> None: ...
    def __enter__(self): ...
    def __exit__(self, *excinfo) -> None: ...

@contextmanager
def reraise(errors, into) -> Generator[None]:
    """Reraises errors as other exception."""
@decorator
def retry(call, tries, errors=..., timeout: int = 0, filter_errors=None):
    """Makes decorated function retry up to tries times.
       Retries only on specified errors.
       Sleeps timeout or timeout(attempt) seconds between tries."""
def fallback(*approaches):
    """Tries several approaches until one works.
       Each approach has a form of (callable, expected_errors)."""

class ErrorRateExceeded(Exception): ...

def limit_error_rate(fails, timeout, exception=...):
    """If function fails to complete fails times in a row,
       calls to it will be intercepted for timeout with exception raised instead."""
def throttle(period):
    """Allows only one run in a period, the rest is skipped"""
@decorator
def post_processing(call, func):
    """Post processes decorated function result with func."""

collecting: Incomplete

@decorator
def joining(call, sep):
    """Joins decorated function results with sep."""
def once_per(*argnames):
    """Call function only once for every combination of the given arguments."""

once: Incomplete

def once_per_args(func):
    """Call function once for every combination of values of its arguments."""
@decorator
def wrap_with(call, ctx):
    """Turn context manager into a decorator"""
