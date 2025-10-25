from _typeshed import Incomplete
from typing import Any

__all__ = ['memoize', 'make_lookuper', 'silent_lookuper', 'cache']

class SkipMemory(Exception): ...

def memoize(_func: Any=None, *, key_func: Any=None) -> Any:
    """@memoize(key_func=None). Makes decorated function memoize its results.

    If key_func is specified uses key_func(*func_args, **func_kwargs) as memory key.
    Otherwise uses args + tuple(sorted(kwargs.items()))

    Exposes its memory via .memory attribute.
    """
def cache(timeout: Any, *, key_func: Any=None) -> Any:
    """Caches a function results for timeout seconds."""

class CacheMemory(dict[Any, Any]):
    timeout: Incomplete
    def __init__(self, timeout: Any) -> None: ...
    def __setitem__(self, key: Any, value: Any) -> None: ...
    def __getitem__(self, key: Any) -> Any: ...
    def expire(self) -> None: ...
    _keys: Incomplete
    _expires: Incomplete
    def clear(self) -> None: ...

make_lookuper: Incomplete
silent_lookuper: Incomplete
