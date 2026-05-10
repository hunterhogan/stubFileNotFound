import importlib
from google.colab._import_hooks._hook_injector import HookInjectorLoader as HookInjectorLoader

class _GenerativeAIImportHook(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None): ...
