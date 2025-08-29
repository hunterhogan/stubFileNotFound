from collections.abc import Hashable, Iterator
from pandas import Index, Series
from pandas._typing import (
	AxisIndex, DropKeep, DTypeLike, GenericT, GenericT_co, NDFrameT, np_1darray, S1, Scalar, SupportsDType)
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays import ExtensionArray
from pandas.core.arrays.categorical import Categorical
from pandas.util._decorators import cache_readonly
from typing import Any, final, Generic, Literal, overload
from typing_extensions import Self
import numpy as np

class NoNewAttributesMixin:
    def __setattr__(self, key: str, value: Any) -> None: ...

class SelectionMixin(Generic[NDFrameT]):
    obj: NDFrameT
    exclusions: frozenset[Hashable]
    @final
    @cache_readonly
    def ndim(self) -> int: ...
    def __getitem__(self, key: Any) -> Any: ...
    def aggregate(self, func: Any, *args: Any, **kwargs: Any) -> Any: ...

class IndexOpsMixin(OpsMixin, Generic[S1, GenericT_co]):
    __array_priority__: int = ...
    @property
    def T(self) -> Self: ...
    @property
    def shape(self) -> tuple[Any, ...]: ...
    @property
    def ndim(self) -> int: ...
    def item(self) -> S1: ...
    @property
    def nbytes(self) -> int: ...
    @property
    def size(self) -> int: ...
    @property
    def array(self) -> ExtensionArray: ...
    @overload
    def to_numpy(
        self,
        dtype: None = None,
        copy: bool = False,
        na_value: Scalar = ...,
        **kwargs: Any,
    ) -> np_1darray[GenericT_co]: ...
    @overload
    def to_numpy(
        self,
        dtype: np.dtype[GenericT] | SupportsDType[GenericT] | type[GenericT],
        copy: bool = False,
        na_value: Scalar = ...,
        **kwargs: Any,
    ) -> np_1darray[GenericT]: ...
    @overload
    def to_numpy(
        self,
        dtype: DTypeLike,
        copy: bool = False,
        na_value: Scalar = ...,
        **kwargs: Any,
    ) -> np_1darray: ...
    @property
    def empty(self) -> bool: ...
    def max(self, axis: Any=..., skipna: bool = ..., **kwargs: Any) -> Any: ...
    def min(self, axis: Any=..., skipna: bool = ..., **kwargs: Any) -> Any: ...
    def argmax(
        self,
        axis: AxisIndex | None = None,
        skipna: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> np.int64: ...
    def argmin(
        self,
        axis: AxisIndex | None = None,
        skipna: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> np.int64: ...
    def tolist(self) -> list[S1]: ...
    def to_list(self) -> list[S1]: ...
    def __iter__(self) -> Iterator[S1]: ...
    @property
    def hasnans(self) -> bool: ...
    @overload
    def value_counts(
        self,
        normalize: Literal[False] = False,
        sort: bool = True,
        ascending: bool = False,
        bins: Any=None,
        dropna: bool = True,
    ) -> Series[int]: ...
    @overload
    def value_counts(
        self,
        normalize: Literal[True],
        sort: bool = True,
        ascending: bool = False,
        bins: Any=None,
        dropna: bool = True,
    ) -> Series[float]: ...
    def nunique(self, dropna: bool = True) -> int: ...
    @property
    def is_unique(self) -> bool: ...
    @property
    def is_monotonic_decreasing(self) -> bool: ...
    @property
    def is_monotonic_increasing(self) -> bool: ...
    def factorize(
        self, sort: bool = False, use_na_sentinel: bool = True
    ) -> tuple[np_1darray, np_1darray | Index[Any] | Categorical]: ...
    def searchsorted(
        self, value: Any, side: Literal["left", "right"] = 'left', sorter: Any=None
    ) -> int | list[int]: ...
    def drop_duplicates(self, *, keep: DropKeep = 'first') -> Self: ...
