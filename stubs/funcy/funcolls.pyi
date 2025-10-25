from collections.abc import Callable
from operator import itemgetter
from typing import Any
__all__ = ['all_fn', 'any_fn', 'none_fn', 'one_fn', 'some_fn']

def all_fn(*fs: Any) -> Callable[..., object] | type[bool] | Callable[..., Any] | Callable[..., bool] | Callable[..., Any | None] | itemgetter[int | slice[Any, Any, Any]] | Callable[[Any], Any]:
    """Constructs a predicate, which holds when all fs hold."""
    ...

def any_fn(*fs: Any) -> Callable[..., object] | type[bool] | Callable[..., Any] | Callable[..., bool] | Callable[..., Any | None] | itemgetter[int | slice[Any, Any, Any]] | Callable[[Any], Any]:
    """Constructs a predicate, which holds when any fs holds."""
    ...

def none_fn(*fs: Any) -> Callable[..., object] | type[bool] | Callable[..., Any] | Callable[..., bool] | Callable[..., Any | None] | itemgetter[int | slice[Any, Any, Any]] | Callable[[Any], Any]:
    """Constructs a predicate, which holds when none of fs hold."""
    ...

def one_fn(*fs: Any) -> Callable[..., object] | type[bool] | Callable[..., Any] | Callable[..., bool] | Callable[..., Any | None] | itemgetter[int | slice[Any, Any, Any]] | Callable[[Any], Any]:
    """Constructs a predicate, which holds when exactly one of fs holds."""
    ...

def some_fn(*fs: Any) -> Callable[..., object] | type[bool] | Callable[..., Any] | Callable[..., bool] | Callable[..., Any | None] | itemgetter[int | slice[Any, Any, Any]] | Callable[[Any], Any]:
    """Constructs a function, which calls fs one by one
       and returns first truthy result."""
    ...

