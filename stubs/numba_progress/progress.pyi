from typing import Any, Optional, TextIO, TypeVar, Union, Dict
from threading import Thread, Event
from types import TracebackType
import numpy as np
from numpy.typing import NDArray
import sys
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
from numba import types
from numba.core.types.abstract import Type
from numba.core.datamodel.new_models import StructModel
from numba.extending import models

from .numba_atomic import atomic_add, atomic_xchg

def is_notebook() -> bool:
    """Determine if we're running within an IPython kernel.

    Returns whether the code is running in an IPython notebook environment.
    """

class ProgressBar:
    """
    Wraps the tqdm progress bar enabling it to be updated from within a numba nopython function.
    It works by spawning a separate thread that updates the tqdm progress bar based on an atomic counter which can be
    accessed within the numba function. The progress bar works with parallel as well as sequential numba functions.

    Note: As this Class contains python objects not useable or convertible into numba, it will be boxed as a
    proxy object, that only exposes the minimum subset of functionality to update the progress bar. Attempts
    to return or create a ProgressBar within a numba function will result in an error.

    Parameters
    ----------
    file: TextIO, optional
        Specifies where to output the progress messages (default: sys.stdout).
        Uses `file.write(str)` and `file.flush()` methods.

    update_interval: float, optional
        The interval in seconds used by the internal thread to check for updates (0.1).

    notebook: bool, optional
        If set, forces or forbids the use of the notebook progress bar.
        By default the best progress bar will be determined automatically.

    dynamic_ncols: bool, optional
        If true, the number of columns (the width of the progress bar) is constantly adjusted.
        This improves the output of the notebook progress bar a lot.

    kwargs: dict-like, optional
        Additional parameters passed to the tqdm class. See https://github.com/tqdm/tqdm
        for a documentation of the available parameters.
    """
    _last_value: int
    _tqdm: tqdm[Any] | tqdm_notebook
    hook: NDArray[np.uint64]
    _updater_thread: Thread | None
    _exit_event: Event
    update_interval: float
    _timer: Thread

    def __init__(
        self,
        total: int | None = None,
        file: TextIO | None = None,
        update_interval: float = 0.1,
        notebook: bool | None = None,
        dynamic_ncols: bool = True,
        **kwargs: Any
    ) -> None: ...

    def _start(self) -> None: ...

    def close(self) -> None: ...

    @property
    def n(self) -> int: ...

    def set(self, n: int = 0) -> None: ...

    def update(self, n: int = 1) -> None: ...

    def _update_tqdm(self) -> None: ...

    def _update_function(self) -> None:
        """Background thread for updating the progress bar.
        """

    def __enter__(self) -> ProgressBar: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None
    ) -> None: ...

class ProgressBarTypeImpl(Type):
    """Type class for the Numba-compiled version of ProgressBar."""
    def __init__(self) -> None: ...

# This is the numba type representation of the ProgressBar class to be used in function signatures
ProgressBarType: ProgressBarTypeImpl

class ProgressBarModel(StructModel):
    """StructModel implementation for the ProgressBar in Numba."""
    def __init__(self, dmm: Any, fe_type: Any) -> None: ...

# The following functions are internal to Numba and normally not called directly
def typeof_index(val: Any, c: Any) -> ProgressBarTypeImpl: ...

def get_value(progress_bar: ProgressBarTypeImpl) -> Any: ...

def unbox_progressbar(typ: Any, obj: Any, c: Any) -> Any:
    """Convert a ProgressBar to its native representation (proxy object)."""

def box_progressbar(typ: Any, val: Any, c: Any) -> None:
    """Cannot convert back to Python object since it contains internal Python state."""

def _ol_update(self: ProgressBarTypeImpl, n: int = 1) -> Any:
    """Numba implementation of the update method."""

def _ol_set(self: ProgressBarTypeImpl, n: int = 0) -> Any:
    """Numba implementation of the set method."""
