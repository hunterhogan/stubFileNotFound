from _typeshed import Incomplete
from collections.abc import Callable
from typing import Any

__all__ = ['LazyObject', 'cached_property', 'cached_readonly', 'monkey', 'wrap_prop']

class cached_property:
    """
    Decorator that converts a method with a single self argument into
    a property cached on the instance.
    """

    fset: Incomplete
    fdel: Incomplete
    fget: Incomplete
    __doc__: Incomplete
    def __init__(self, fget: Any) -> None: ...
    def __get__(self, instance: Any, type: Any=None) -> Any: ...

class cached_readonly(cached_property):
    """Same as @cached_property, but protected against rewrites."""

    def __set__(self, instance: Any, value: Any) -> None: ...



def wrap_prop(ctx: Any) -> Callable[..., WrapperProp]:
    """Wrap a property accessors with a context manager"""

def monkey(cls, name: Any=...) -> Callable[..., Any]:
    """
    Monkey patches class or module by adding to it decorated function.

    Anything overwritten could be accessed via .original attribute of decorated object.
    """

class LazyObject:
    """
    A simplistic lazy init object.
    Rewrites itself when any attribute is accessed.
    """

    def __init__(self, init: Any) -> None: ...
    def _setup(self) -> None: ...
    def __getattr__(self, name: Any) -> Any: ...
    def __setattr__(self, name: Any, value: Any) -> Any: ...
