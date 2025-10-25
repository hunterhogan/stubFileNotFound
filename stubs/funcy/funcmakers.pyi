from collections.abc import Callable
from operator import itemgetter
from typing import Any

__all__ = ['make_func', 'make_pred']

def make_func(f: Any, test: bool = False) -> Callable[..., object] | type[bool] | Callable[..., Any] | Callable[..., bool] | Callable[..., Any | None] | itemgetter[int | slice[Any, Any, Any]] | Callable[[Any], Any]: ...
def make_pred(pred: Any) -> Callable[..., object] | type[bool] | Callable[..., Any] | Callable[..., bool] | Callable[..., Any | None] | itemgetter[int | slice[Any, Any, Any]] | Callable[[Any], Any]: ...
