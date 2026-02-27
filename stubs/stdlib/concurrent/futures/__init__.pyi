from ._base import (
	ALL_COMPLETED as ALL_COMPLETED, as_completed as as_completed, BrokenExecutor as BrokenExecutor,
	CancelledError as CancelledError, Executor as Executor, FIRST_COMPLETED as FIRST_COMPLETED,
	FIRST_EXCEPTION as FIRST_EXCEPTION, Future as Future, InvalidStateError as InvalidStateError,
	TimeoutError as TimeoutError, wait as wait)
from .process import ProcessPoolExecutor as ProcessPoolExecutor
from .thread import ThreadPoolExecutor as ThreadPoolExecutor
import sys

if sys.version_info >= (3, 14):
    from .interpreter import InterpreterPoolExecutor as InterpreterPoolExecutor

    __all__ = [
        "ALL_COMPLETED",
        "FIRST_COMPLETED",
        "FIRST_EXCEPTION",
        "BrokenExecutor",
        "CancelledError",
        "Executor",
        "Future",
        "InterpreterPoolExecutor",
        "InvalidStateError",
        "ProcessPoolExecutor",
        "ThreadPoolExecutor",
        "TimeoutError",
        "as_completed",
        "wait",
    ]

elif sys.version_info >= (3, 13):
    __all__ = (
        "ALL_COMPLETED",
        "FIRST_COMPLETED",
        "FIRST_EXCEPTION",
        "BrokenExecutor",
        "CancelledError",
        "Executor",
        "Future",
        "InvalidStateError",
        "ProcessPoolExecutor",
        "ThreadPoolExecutor",
        "TimeoutError",
        "as_completed",
        "wait",
    )
else:
    __all__ = (
        "ALL_COMPLETED",
        "FIRST_COMPLETED",
        "FIRST_EXCEPTION",
        "BrokenExecutor",
        "CancelledError",
        "Executor",
        "Future",
        "ProcessPoolExecutor",
        "ThreadPoolExecutor",
        "TimeoutError",
        "as_completed",
        "wait",
    )

def __dir__() -> tuple[str, ...]: ...
