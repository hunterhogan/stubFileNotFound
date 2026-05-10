import importlib.abc
from _typeshed import Incomplete

class HookInjectorLoader(importlib.abc.Loader):
    fullname: Incomplete
    path: Incomplete
    target: Incomplete
    previously_loaded: Incomplete
    meta_path_finder_cls: Incomplete
    init_code_callback: Incomplete
    def __init__(self, fullname, path, target, meta_path_finder_cls, init_code_callback) -> None: ...
    def find_spec(self): ...
    def create_module(self, spec): ...
    def exec_module(self, module) -> None: ...
