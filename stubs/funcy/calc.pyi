from _typeshed import Incomplete

__all__ = ['memoize', 'make_lookuper', 'silent_lookuper', 'cache']

class SkipMemory(Exception): ...

def memoize(_func=None, *, key_func=None):
    """@memoize(key_func=None). Makes decorated function memoize its results.

    If key_func is specified uses key_func(*func_args, **func_kwargs) as memory key.
    Otherwise uses args + tuple(sorted(kwargs.items()))

    Exposes its memory via .memory attribute.
    """
def cache(timeout, *, key_func=None):
    """Caches a function results for timeout seconds."""

class CacheMemory(dict):
    timeout: Incomplete
    def __init__(self, timeout) -> None: ...
    def __setitem__(self, key, value) -> None: ...
    def __getitem__(self, key): ...
    def expire(self) -> None: ...
    _keys: Incomplete
    _expires: Incomplete
    def clear(self) -> None: ...

make_lookuper: Incomplete
silent_lookuper: Incomplete
