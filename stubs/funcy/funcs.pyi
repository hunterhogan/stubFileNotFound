__all__ = ['autocurry', 'caller', 'complement', 'compose', 'constantly', 'curry', 'func_partial', 'identity', 'iffy', 'juxt', 'ljuxt', 'partial', 'rcompose', 'rcurry', 'rpartial']
from collections.abc import Callable, Generator
from functools import _Wrapped
from operator import itemgetter
from typing import Any

def identity(x: Any) -> Any:
    """Returns its argument."""

def constantly(x: Any) -> Callable[..., Any]:
    """Creates a function accepting any args, but always returning x."""

def caller(*a: Any, **kw: Any) -> Callable[..., Any]:
    """Creates a function calling its sole argument with given *a, **kw."""

def func_partial(func: Any, *args: Any, **kwargs: Any) -> Callable[..., Any]:
    """A functools.partial alternative, which returns a real function.
    Can be used to construct methods.
    """

def rpartial(func: Any, *args: Any, **kwargs: Any) -> Callable[..., Any]:
    """Partially applies last arguments.
    New keyworded arguments extend and override kwargs.
    """

def curry(func: Any, n: Any=...) -> Callable[..., Callable[..., Any]] | Callable[..., Any | Callable[..., Callable[..., Any]] | Callable[..., Any]]:
    """Curries func into a chain of one argument functions."""

def rcurry(func: Any, n: Any=...) -> Callable[..., Callable[..., Any]] | Callable[..., Any | Callable[..., Callable[..., Any]] | Callable[..., Any]]:
    """Curries func into a chain of one argument functions.
    Arguments are passed from right to left.
    """

def autocurry(func: Any, n: Any=..., _spec: Any=..., _args: Any=..., _kwargs: Any=...) -> Callable[..., Any | _Wrapped[Callable[..., Any], Any, Callable[..., Any], Any]]:
    """Creates a version of func returning its partial applications
    until sufficient arguments are passed.
    """

def iffy(pred: Any, action: Any=..., default: Any=...) -> Callable[..., Any | bool | object | None]:
    """Creates a function, which conditionally applies action or default."""

def compose(*fs: Any) -> Callable[..., object] | type[bool] | Callable[..., Any] | Callable[..., bool] | Callable[..., Any | None] | itemgetter[int | slice[Any, Any, Any]] | Callable[[Any], Any]:
    """Composes passed functions."""

def rcompose(*fs: Any) -> Callable[..., object] | type[bool] | Callable[..., Any] | Callable[..., bool] | Callable[..., Any | None] | itemgetter[int | slice[Any, Any, Any]] | Callable[[Any], Any]:
    """Composes functions, calling them from left to right."""

def complement(pred: Any) -> Callable[..., object] | type[bool] | Callable[..., Any] | Callable[..., bool] | Callable[..., Any | None] | itemgetter[int | slice[Any, Any, Any]] | Callable[[Any], Any]:
    """Constructs a complementary predicate."""

def ljuxt(*fs: Any) -> Callable[..., list[Any | bool | object | None]]:
    """Constructs a juxtaposition of the given functions.
    Result returns a list of results of fs.
    """

def juxt(*fs: Any) -> Callable[..., Generator[Any | bool | object | None, None, None]]:
    """Constructs a lazy juxtaposition of the given functions.
    Result returns an iterator of results of fs.
    """

