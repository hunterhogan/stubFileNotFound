from collections import defaultdict
from collections.abc import Callable, Generator
from contextlib import ContextDecorator as ContextDecorator, contextmanager as contextmanager
from functools import partial
from inspect import unwrap as unwrap
from typing import Any

__all__ = ['ContextDecorator', 'contextmanager', 'decorator', 'unwrap', 'wraps']
def decorator(deco: Any) -> Callable[..., Any]:
    """
    Transforms a flat wrapper into decorator::

        @decorator
        def func(call, methods, content_type=DEFAULT):  # These are decorator params
            # Access call arg by name
            if call.request.method not in methods:
                # ...
            # Decorated functions and all the arguments are accessible as:
            print(call._func, call_args, call._kwargs)
            # Finally make a call:
            return call()
    """

def make_decorator(deco: Any, dargs: Any=..., dkwargs: Any=...) -> Callable[..., Any]:
    ...

class Call:
    """
    A call object to pass as first argument to decorator.

    Call object is just a proxy for decorated function
    with call arguments saved in its attributes.
    """

    def __init__(self, func: Any, args: Any, kwargs: Any) -> None: ...
    def __call__(self, *a: Any, **kw: Any) -> Any: ...
    def __getattr__(self, name: Any) -> Any | defaultdict[Any, Any] | dict[Any, Any] | list[Any] | Generator[tuple[Any, Any], None, None]:
        ...
def has_single_arg(func: Any) -> bool:
    ...

def has_1pos_and_kwonly(func: Any) -> bool:
    ...

def get_argnames(func: Any) -> Any:
    ...

def arggetter(func: Any, _cache: Any=...) -> Callable[..., Any | defaultdict[Any, Any] | dict[Any, Any] | list[Any] | Generator[tuple[Any, Any], None, None]]:
    ...

def update_wrapper(wrapper: Any, wrapped: Any, assigned: Any=..., updated: Any=...) -> Any:
    ...

def wraps(wrapped: Any, assigned: Any=..., updated: Any=...) -> partial[Any]:
    ...

