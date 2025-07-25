from collections.abc import (
    Hashable,
    Iterable,
)
from typing import Literal

import numpy as np
from pandas.core import accessor
from pandas.core.indexes.base import Index
from pandas.core.indexes.extension import ExtensionIndex
from typing_extensions import Self

from pandas._typing import (
    S1,
    DtypeArg,
)
from typing import Any

class CategoricalIndex(ExtensionIndex[S1], accessor.PandasDelegate):
    codes: np.ndarray[Any, Any] = ...
    categories: Index[Any] = ...
    def __new__(
        cls,
        data: Iterable[S1] = None,
        categories: Any=None,
        ordered: Any=None,
        dtype: Any=None,
        copy: bool = False,
        name: Hashable = None,
    ) -> Self: ...
    def equals(self, other: Any): ...
    @property
    def inferred_type(self) -> str: ...
    @property
    def values(self): ...
    def __contains__(self, key: Any) -> bool: ...
    def __array__(
        self, dtype: DtypeArg = None, copy: bool | None = None
    ) -> np.ndarray[Any, Any]: ...
    def astype(self, dtype: DtypeArg, copy: bool = True) -> Index[Any]: ...
    def fillna(self, value: Any=None): ...
    @property
    def is_unique(self) -> bool: ...
    @property
    def is_monotonic_increasing(self) -> bool: ...
    @property
    def is_monotonic_decreasing(self) -> bool: ...
    def unique(self, level: Any=None): ...
    def duplicated(self, keep: Literal["first", "last", False] = 'first'): ...
    def where(self, cond: Any, other: Any=None): ...
    def reindex(self, target: Any, method: Any=None, level: Any=None, limit: Any=None, tolerance: Any=None): ...
    def get_indexer(self, target: Any, method: Any=None, limit: Any=None, tolerance: Any=None): ...
    def get_indexer_non_unique(self, target: Any): ...
    def delete(self, loc: Any): ...
    def insert(self, loc: Any, item: Any): ...
