import pickle
from collections.abc import Generator
from pandas._libs.arrays import NDArrayBacked as NDArrayBacked
from pandas._libs.tslibs.offsets import BaseOffset as BaseOffset
from pandas.core.arrays.datetimes import DatetimeArray as DatetimeArray
from pandas.core.arrays.period import PeriodArray as PeriodArray
from pandas.core.arrays.timedeltas import TimedeltaArray as TimedeltaArray
from pandas.core.indexes.base import Index as Index
from pandas.core.internals.managers import BlockManager as BlockManager
from typing import ClassVar

TYPE_CHECKING: bool
def load_reduce(self) -> None: ...

_class_locations_map: dict

class Unpickler(pickle._Unpickler):
    dispatch: ClassVar[dict] = ...
    def find_class(self, module, name): ...
def load_newobj(self) -> None: ...
def load_newobj_ex(self) -> None: ...
def load(fh, encoding: str | None, is_verbose: bool = ...):
    """
    Load a pickle, with a provided encoding,

    Parameters
    ----------
    fh : a filelike object
    encoding : an optional encoding
    is_verbose : show exception output
    """
def loads(bytes_object: bytes, *, fix_imports: bool = ..., encoding: str = ..., errors: str = ...):
    """
    Analogous to pickle._loads.
    """
def patch_pickle(*args, **kwds) -> Generator[None, None, None]:
    """
    Temporarily patch pickle to use our unpickler.
    """
