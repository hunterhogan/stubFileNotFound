import importlib
from google.colab._import_hooks._hook_injector import HookInjectorLoader as HookInjectorLoader

APPLICATION_NAME: str

class APICoreClientInfoImportHook(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None): ...
