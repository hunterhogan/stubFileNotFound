from _typeshed import Incomplete
from numba.core.registry import cpu_target as cpu_target

def git_hash(): ...
def get_func_name(fn): ...
def gather_function_info(backend): ...
def bind_file_to_print(fobj): ...
def format_signature(sig): ...

github_url: str
description: str

def format_function_infos(fninfos): ...
def gen_lower_listing(path: Incomplete | None = None) -> None:
    """
    Generate lowering listing to ``path`` or (if None) to stdout.
    """
