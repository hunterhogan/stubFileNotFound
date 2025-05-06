from _typeshed import StrOrBytesPath
from typing import Any


# The list should be list[StrOrBytesPath] but that doesn't work because invariance
def send2trash(paths: list[Any] | StrOrBytesPath) -> None: ...
