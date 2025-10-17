from .temp_value_setter import TempValueSetter as TempValueSetter
from typing import Any

class TempImportHookSetter(TempValueSetter):
    """Context manager for temporarily setting a function as the import hook."""

    def __init__(self, import_hook: Any) -> None:
        """
        Construct the `TempImportHookSetter`.

        `import_hook` is the function to be used as the import hook.
        """



