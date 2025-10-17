from _typeshed import Incomplete
from collections.abc import Generator
from python_toolbox import (
	context_management as context_management, logic_tools as logic_tools, misc_tools as misc_tools)
from python_toolbox.exceptions import CuteException as CuteException
from typing import Any
import unittest

class Failure(CuteException, AssertionError):
    """A test has failed."""

class RaiseAssertor(context_management.ContextManager[Any]):
    """
    Asserts that a certain exception was raised in the suite. You may use a
    snippet of text that must appear in the exception message or a regex that
    the exception message must match.

    Example:

        with RaiseAssertor(ZeroDivisionError, 'modulo by zero'):
            1/0

    """

    exception_type: Incomplete
    text: Incomplete
    exception: Incomplete
    assert_exact_type: Incomplete
    def __init__(self, exception_type: Any=..., text: str = '', assert_exact_type: bool = False) -> None:
        """
        Construct the `RaiseAssertor`.

        `exception_type` is an exception type that the exception must be of;
        `text` may be either a snippet of text that must appear in the
        exception's message, or a regex pattern that the exception message must
        match. Specify `assert_exact_type=False` if you want to assert that the
        exception is of the exact `exception_type` specified, and not a
        subclass of it.
        """
    def manage_context(self) -> Generator[Incomplete]:
        """Manage the `RaiseAssertor'`s context."""

def assert_same_signature(*callables: Any) -> None:
    """Assert that all the `callables` have the same function signature."""

class _MissingAttribute:
    """Object signifying that an attribute was not found."""

def assert_polite_wrapper(wrapper: Any, wrapped: Any=None, same_signature: bool = True) -> None:
    """
    Assert that `wrapper` is a polite function wrapper around `wrapped`.

    A function wrapper (usually created by a decorator) has a few
    responsibilties; maintain the same name, signature, documentation etc. of
    the original function, and a few others. Here we check that the wrapper did
    all of those things.
    """

class TestCase(unittest.TestCase, context_management.ContextManager[Any]):
    setUp: Incomplete
    tearDown: Incomplete
    def manage_context(self) -> Generator[Incomplete]: ...
    def setup(self) -> Any: ...
    def tear_down(self) -> Any: ...



