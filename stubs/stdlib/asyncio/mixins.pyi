from typing import Never
import sys
import threading

_global_lock: threading.Lock

class _LoopBoundMixin:
    if sys.version_info < (3, 11):
        def __init__(self, *, loop: Never = ...) -> None: ...
