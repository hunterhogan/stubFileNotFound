from collections.abc import Callable
from torch.nn import Module
from typing import Any

def exists(v) -> bool:
    ...

def map_values(fn, v) -> list[Any] | tuple[Any, ...]:
    ...

def dehydrate_config(config, config_instance_var_name) -> list[Any] | tuple[Any, ...]:
    ...

def rehydrate_config(config) -> list[Any] | tuple[Any, ...]:
    ...

def save_load(maybe_fn=..., *, save_method_name=..., load_method_name=..., config_instance_var_name=..., init_and_load_classmethod_name=..., version: str | None = ...) -> type[Module] | Callable[..., type[Module]]:
    ...

