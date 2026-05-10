import importlib
from _typeshed import Incomplete
from google.colab._import_hooks._hook_injector import HookInjectorLoader as HookInjectorLoader

class DisabledFunctionError(ValueError):
    funcname: Incomplete
    def __init__(self, message, funcname=None, **kwargs) -> None: ...

def disable_function(func, message, env_var, name=None): ...

class _OpenCVImportHook(importlib.abc.MetaPathFinder):
    message: str
    env_var: str
    def find_spec(self, fullname, path=None, target=None): ...
