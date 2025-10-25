from .decorators import unwrap as unwrap
from _typeshed import Incomplete
from typing import NamedTuple
from typing import Any

ARGS: Incomplete
two_arg_funcs: str
STD_MODULES: Incomplete

class Spec(NamedTuple):
    max_n: Incomplete
    names: Incomplete
    req_n: Incomplete
    req_names: Incomplete
    varkw: Incomplete

def get_spec(func: Any, _cache: Any={}) -> Any: ...
def _code_to_spec(func: Any) -> Any: ...
def _sig_to_spec(sig: Any) -> Any: ...
