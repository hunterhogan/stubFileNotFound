from . import popen_fork
from .util import Finalize
from typing import ClassVar
import sys

if sys.platform != "win32":
    __all__ = ["Popen"]

    class _DupFd:
        def __init__(self, ind: int) -> None: ...
        def detach(self) -> int: ...

    class Popen(popen_fork.Popen):
        DupFd: ClassVar[type[_DupFd]]
        finalizer: Finalize
