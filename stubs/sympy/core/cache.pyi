from _typeshed import Incomplete
from typing import Callable

class _cache(list):
    """ List of cached functions """
    def print_cache(self) -> None:
        """print cache info"""
    def clear_cache(self) -> None:
        """clear cache content"""

CACHE: Incomplete
print_cache: Incomplete
clear_cache: Incomplete

def __cacheit(maxsize):
    """caching decorator.

        important: the result of cached function must be *immutable*


        Examples
        ========

        >>> from sympy import cacheit
        >>> @cacheit
        ... def f(a, b):
        ...    return a+b

        >>> @cacheit
        ... def f(a, b): # noqa: F811
        ...    return [a, b] # <-- WRONG, returns mutable object

        to force cacheit to check returned results mutability and consistency,
        set environment variable SYMPY_USE_CACHE to 'debug'
    """
def __cacheit_nocache(func): ...
def __cacheit_debug(maxsize):
    """cacheit + code to check cache consistency"""
def _getenv(key, default=None): ...

USE_CACHE: Incomplete
scs: Incomplete
SYMPY_CACHE_SIZE: Incomplete
cacheit = __cacheit_nocache

def cached_property(func):
    """Decorator to cache property method"""
def lazy_function(module: str, name: str) -> Callable:
    """Create a lazy proxy for a function in a module.

    The module containing the function is not imported until the function is used.

    """
