from .utils import Comparable
from _typeshed import Incomplete
from collections.abc import Generator
from contextlib import contextmanager

__all__ = ['tqdm', 'trange', 'TqdmTypeError', 'TqdmKeyError', 'TqdmWarning', 'TqdmExperimentalWarning', 'TqdmDeprecationWarning', 'TqdmMonitorWarning']

class TqdmTypeError(TypeError): ...
class TqdmKeyError(KeyError): ...

class TqdmWarning(Warning):
    """base class for all tqdm warnings.

    Used for non-external-code-breaking errors, such as garbled printing.
    """
    def __init__(self, msg, fp_write: Incomplete | None = None, *a, **k) -> None: ...

class TqdmExperimentalWarning(TqdmWarning, FutureWarning):
    """beta feature, unstable API and behaviour"""
class TqdmDeprecationWarning(TqdmWarning, DeprecationWarning): ...
class TqdmMonitorWarning(TqdmWarning, RuntimeWarning):
    """tqdm monitor errors which do not affect external functionality"""

class TqdmDefaultWriteLock:
    """
    Provide a default write lock for thread and multiprocessing safety.
    Works only on platforms supporting `fork` (so Windows is excluded).
    You must initialise a `tqdm` or `TqdmDefaultWriteLock` instance
    before forking in order for the write lock to work.
    On Windows, you need to supply the lock from the parent to the children as
    an argument to joblib or the parallelism lib you use.
    """
    th_lock: Incomplete
    locks: Incomplete
    def __init__(self) -> None: ...
    def acquire(self, *a, **k) -> None: ...
    def release(self) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, *exc) -> None: ...
    @classmethod
    def create_mp_lock(cls) -> None: ...
    @classmethod
    def create_th_lock(cls) -> None: ...

class Bar:
    '''
    `str.format`-able bar with format specifiers: `[width][type]`

    - `width`
      + unspecified (default): use `self.default_len`
      + `int >= 0`: overrides `self.default_len`
      + `int < 0`: subtract from `self.default_len`
    - `type`
      + `a`: ascii (`charset=self.ASCII` override)
      + `u`: unicode (`charset=self.UTF` override)
      + `b`: blank (`charset="  "` override)
    '''
    ASCII: str
    UTF: Incomplete
    BLANK: str
    COLOUR_RESET: str
    COLOUR_RGB: str
    COLOURS: Incomplete
    frac: Incomplete
    default_len: Incomplete
    charset: Incomplete
