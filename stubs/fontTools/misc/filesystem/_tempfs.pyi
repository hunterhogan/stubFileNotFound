from ._errors import OperationFailed as OperationFailed
from ._osfs import OSFS as OSFS
from _typeshed import Incomplete

class TempFS(OSFS):
    auto_clean: Incomplete
    ignore_clean_errors: Incomplete
    _temp_dir: Incomplete
    _cleaned: bool
    def __init__(self, auto_clean: bool = True, ignore_clean_errors: bool = True) -> None: ...
    def close(self) -> None: ...
    def clean(self) -> None: ...
