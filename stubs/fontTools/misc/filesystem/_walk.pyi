from ._base import FS as FS
from ._info import Info as Info
from ._path import combine as combine
from _typeshed import Incomplete
from collections.abc import Collection, Iterator
from typing import Callable

class BoundWalker:
    _fs: Incomplete
    def __init__(self, fs: FS) -> None: ...
    def _iter_walk(self, path: str, namespaces: Collection[str] | None = None) -> Iterator[tuple[str, Info | None]]:
        """Walk files using a *breadth first* search."""
    def _filter(self, include: Callable[[str, Info], bool] = ..., path: str = '/', namespaces: Collection[str] | None = None) -> Iterator[str]: ...
    def files(self, path: str = '/') -> Iterator[str]: ...
    def dirs(self, path: str = '/') -> Iterator[str]: ...
