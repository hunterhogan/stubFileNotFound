from typing import (
    Any,
    overload,
)

from pandas.core.arrays import (
    ExtensionArray as ExtensionArray,
    ExtensionOpsMixin,
)
from typing_extensions import Self

from pandas._typing import (
    NpDtype,
    Scalar,
    ScalarIndexer,
    SequenceIndexer,
    np_1darray,
    npt,
)

class BaseMaskedArray(ExtensionArray, ExtensionOpsMixin):
    @overload
    def __getitem__(self, item: ScalarIndexer) -> Any: ...
    @overload
    def __getitem__(self, item: SequenceIndexer) -> Self: ...
    def __iter__(self) -> Any: ...
    def __invert__(self) -> Any: ...
    def to_numpy(
        self,
        dtype: npt.DTypeLike | None = ...,
        copy: bool = False,
        na_value: Scalar = ...,
    ) -> np_1darray: ...
    __array_priority__: int = ...
    def __array__(
        self, dtype: NpDtype | None = None, copy: bool | None = None
    ) -> np_1darray: ...
    def __arrow_array__(self, type: Any=...) -> Any: ...
    def isna(self) -> Any: ...
    @property
    def nbytes(self) -> int: ...
    def copy(self) -> Any: ...
    def value_counts(self, dropna: bool = True) -> Any: ...
