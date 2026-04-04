from collections.abc import Callable
from functools import _Wrapped
from torch.nn import Module
from typing import Any

def exists(v) -> bool:
    ...

def module_device(m: Module) -> device | None:
    ...

def move_inputs_to_device(device) -> Callable[..., _Wrapped[Callable[..., Any], Any, Callable[..., Any], Any]]:
    ...

def move_inputs_to_module_device(fn) -> _Wrapped[Callable[..., Any], Any, Callable[..., Any], Any]:
    ...

