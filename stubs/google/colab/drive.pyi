from _typeshed import Incomplete
from typing import NamedTuple

__all__ = ['flush_and_unmount', 'mount']

class _Environment(NamedTuple):
    home: Incomplete
    root_dir: Incomplete
    dev: Incomplete
    path: Incomplete
    config_dir: Incomplete

def flush_and_unmount(timeout_ms=...) -> None: ...
def mount(mountpoint: str, force_remount: bool = False, timeout_ms: int = 120000, readonly: bool = False) -> None: ...
