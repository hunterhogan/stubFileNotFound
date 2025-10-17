from typing import (
    TypeAlias,
    TypeVar,
)

from pandas.core.indexes.api import Index

from pandas._libs.indexing import _NDFrameIndexerBase
from pandas._typing import (
    MaskType,
    Scalar,
    ScalarT,
)
from typing import Any

_IndexSliceTuple: TypeAlias = tuple[Index[Any] | MaskType | Scalar | list[ScalarT] | slice | tuple[Scalar, ...], ...]

_IndexSliceUnion: TypeAlias = slice | _IndexSliceTuple[Any]

_IndexSliceUnionT = TypeVar(
    "_IndexSliceUnionT", bound=_IndexSliceUnion  # pyrefly: ignore
)

class _IndexSlice:
    def __getitem__(self, arg: _IndexSliceUnionT) -> _IndexSliceUnionT: ...

IndexSlice: _IndexSlice

class IndexingMixin:
    @property
    def iloc(self) -> _iLocIndexer: ...
    @property
    def loc(self) -> _LocIndexer: ...
    @property
    def at(self) -> _AtIndexer: ...
    @property
    def iat(self) -> _iAtIndexer: ...

class _NDFrameIndexer(_NDFrameIndexerBase):
    axis = ...
    def __call__(self, axis: Any=...) -> Any: ...
    def __getitem__(self, key: Any) -> Any: ...
    def __setitem__(self, key: Any, value: Any) -> None: ...

class _LocationIndexer(_NDFrameIndexer):
    def __getitem__(self, key: Any) -> Any: ...

class _LocIndexer(_LocationIndexer): ...
class _iLocIndexer(_LocationIndexer): ...

class _ScalarAccessIndexer(_NDFrameIndexerBase):
    def __getitem__(self, key: Any) -> Any: ...
    def __setitem__(self, key: Any, value: Any) -> None: ...

class _AtIndexer(_ScalarAccessIndexer): ...
class _iAtIndexer(_ScalarAccessIndexer): ...
