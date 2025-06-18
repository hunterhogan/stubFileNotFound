from ..utils import ObjectWrapper
from _typeshed import Incomplete
from collections.abc import Generator

__all__ = ['tenumerate', 'tzip', 'tmap']

class DummyTqdmFile(ObjectWrapper):
    """Dummy file-like that will write to tqdm"""
    _buf: Incomplete
    def __init__(self, wrapped) -> None: ...
    def write(self, x, nolock: bool = False) -> None: ...
    def __del__(self) -> None: ...

def tenumerate(iterable, start: int = 0, total: Incomplete | None = None, tqdm_class=..., **tqdm_kwargs):
    """
    Equivalent of `numpy.ndenumerate` or builtin `enumerate`.

    Parameters
    ----------
    tqdm_class  : [default: tqdm.auto.tqdm].
    """
def tzip(iter1, *iter2plus, **tqdm_kwargs) -> Generator[Incomplete]:
    """
    Equivalent of builtin `zip`.

    Parameters
    ----------
    tqdm_class  : [default: tqdm.auto.tqdm].
    """
def tmap(function, *sequences, **tqdm_kwargs) -> Generator[Incomplete]:
    """
    Equivalent of builtin `map`.

    Parameters
    ----------
    tqdm_class  : [default: tqdm.auto.tqdm].
    """
