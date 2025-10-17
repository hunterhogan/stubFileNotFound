from typing import (
    Any,
    TypeAlias,
    overload,
)

import numpy as np
from pandas import (
    Index,
    Series,
)
from pandas.core.arrays.base import ExtensionArray as ExtensionArray
from typing_extensions import Self

from pandas._libs.interval import (
    Interval as Interval,
    IntervalMixin as IntervalMixin,
)
from pandas._typing import (
    Axis,
    NpDtype,
    Scalar,
    ScalarIndexer,
    SequenceIndexer,
    TakeIndexer,
    np_1darray,
)

IntervalOrNA: TypeAlias = Interval[Any] | float

class IntervalArray(IntervalMixin, ExtensionArray):
    can_hold_na: bool = ...
    def __new__(
        cls, data: Any, closed: Any=None, dtype: Any=None, copy: bool = False, verify_integrity: bool = True
    ) -> Any: ...
    @classmethod
    def from_breaks(
        cls,
        breaks: Any,
        closed: str = "right",
        copy: bool = False,
        dtype: Any=None,
    ) -> Any: ...
    @classmethod
    def from_arrays(
        cls,
        left: Any,
        right: Any,
        closed: str = "right",
        copy: bool = False,
        dtype: Any=None,
    ) -> Any: ...
    @classmethod
    def from_tuples(
        cls,
        data: Any,
        closed: str = "right",
        copy: bool = False,
        dtype: Any=None,
    ) -> Any: ...
    def __array__(
        self, dtype: NpDtype | None = None, copy: bool | None = None
    ) -> np_1darray: ...
    @overload
    def __getitem__(self, key: ScalarIndexer) -> IntervalOrNA: ...
    @overload
    def __getitem__(self, key: SequenceIndexer) -> Self: ...
    def __setitem__(self, key: Any, value: Any) -> None: ...
    def __eq__(self, other: Any) -> Any: ...
    def __ne__(self, other: Any) -> Any: ...
    def fillna(self, value: Any=None, method: Any=None, limit: Any=None) -> Any: ...
    @property
    def dtype(self) -> Any: ...
    def astype(self, dtype: Any, copy: bool = True) -> Any: ...
    def copy(self) -> Any: ...
    def isna(self) -> Any: ...
    @property
    def nbytes(self) -> int: ...
    @property
    def size(self) -> int: ...
    def shift(self, periods: int = 1, fill_value: object = None) -> IntervalArray: ...
    def take(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
        self: Self,
        indices: TakeIndexer,
        *,
        allow_fill: bool = False,
        fill_value: Any=None,
        axis: Any=None,
        **kwargs: Any,
    ) -> Self: ...
    def value_counts(self, dropna: bool = True) -> Any: ...
    @property
    def left(self) -> Index[Any]: ...
    @property
    def right(self) -> Index[Any]: ...
    @property
    def closed(self) -> bool: ...
    def set_closed(self, closed: Any) -> Any: ...
    @property
    def length(self) -> Index[Any]: ...
    @property
    def mid(self) -> Index[Any]: ...
    @property
    def is_non_overlapping_monotonic(self) -> bool: ...
    def __arrow_array__(self, type: Any=None) -> Any: ...
    def to_tuples(self, na_tuple: bool = True) -> Any: ...
    def repeat(self, repeats: Any, axis: Axis | None = None) -> Any: ...
    @overload
    def contains(self, other: Series) -> Series[bool]: ...
    @overload
    def contains(
        self, other: Scalar | ExtensionArray | Index[Any] | np.ndarray[Any, Any]
    ) -> np_1darray[np.bool]: ...
    def overlaps(self, other: Interval[Any]) -> bool: ...
