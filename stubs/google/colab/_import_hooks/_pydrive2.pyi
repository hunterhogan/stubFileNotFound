import importlib
from google.colab._import_hooks._hook_injector import HookInjectorLoader as HookInjectorLoader

class _PyDrive2ImportHook(importlib.abc.MetaPathFinder):
    env_var: str
    def find_spec(self, fullname, path=None, target=None): ...
