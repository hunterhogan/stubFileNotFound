__all__ = ['identity', 'constantly', 'caller', 'partial', 'rpartial', 'func_partial', 'curry', 'rcurry', 'autocurry', 'iffy', 'compose', 'rcompose', 'complement', 'juxt', 'ljuxt']
from collections.abc import Callable, Generator
from functools import _Wrapped
from operator import itemgetter
from typing import Any


def identity(x):
    """Returns its argument."""
    ...

def constantly(x) -> Callable[..., Any]:
    """Creates a function accepting any args, but always returning x."""
    ...

def caller(*a, **kw) -> Callable[..., Any]:
    """Creates a function calling its sole argument with given *a, **kw."""
    ...

def func_partial(func, *args, **kwargs) -> Callable[..., Any]:
    """A functools.partial alternative, which returns a real function.
       Can be used to construct methods."""
    ...

def rpartial(func, *args, **kwargs) -> Callable[..., Any]:
    """Partially applies last arguments.
       New keyworded arguments extend and override kwargs."""
    ...

def curry(func, n=...) -> Callable[..., Callable[..., Any]] | Callable[..., Any | Callable[..., Callable[..., Any]] | Callable[..., Any]]:
    """Curries func into a chain of one argument functions."""
    ...

def rcurry(func, n=...) -> Callable[..., Callable[..., Any]] | Callable[..., Any | Callable[..., Callable[..., Any]] | Callable[..., Any]]:
    """Curries func into a chain of one argument functions.
       Arguments are passed from right to left."""
    ...

def autocurry(func, n=..., _spec=..., _args=..., _kwargs=...) -> Callable[..., Any | _Wrapped[Callable[..., Any], Any, Callable[..., Any], Any]]:
    """Creates a version of func returning its partial applications
       until sufficient arguments are passed."""
    ...

def iffy(pred, action=..., default=...) -> Callable[..., Any | bool | object | None]:
    """Creates a function, which conditionally applies action or default."""
    ...

def compose(*fs) -> Callable[..., object] | type[bool] | Callable[..., Any] | Callable[..., bool] | Callable[..., Any | None] | itemgetter[int | slice[Any, Any, Any]] | Callable[[Any], Any]:
    """Composes passed functions."""
    ...

def rcompose(*fs) -> Callable[..., object] | type[bool] | Callable[..., Any] | Callable[..., bool] | Callable[..., Any | None] | itemgetter[int | slice[Any, Any, Any]] | Callable[[Any], Any]:
    """Composes functions, calling them from left to right."""
    ...

def complement(pred) -> Callable[..., object] | type[bool] | Callable[..., Any] | Callable[..., bool] | Callable[..., Any | None] | itemgetter[int | slice[Any, Any, Any]] | Callable[[Any], Any]:
    """Constructs a complementary predicate."""
    ...

def ljuxt(*fs) -> Callable[..., list[Any | bool | object | None]]:
    """Constructs a juxtaposition of the given functions.
       Result returns a list of results of fs."""
    ...

def juxt(*fs) -> Callable[..., Generator[Any | bool | object | None, None, None]]:
    """Constructs a lazy juxtaposition of the given functions.
       Result returns an iterator of results of fs."""
    ...

