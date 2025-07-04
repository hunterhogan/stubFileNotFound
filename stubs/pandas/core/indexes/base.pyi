from builtins import str as _str
from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Sequence,
)
from datetime import (
    datetime,
    timedelta,
)
from typing import (
    Any,
    ClassVar,
    Literal,
    TypeAlias,
    final,
    overload,
)

import numpy as np
from pandas import (
    DataFrame,
    DatetimeIndex,
    Interval,
    IntervalIndex,
    MultiIndex,
    Period,
    PeriodDtype,
    PeriodIndex,
    Series,
    TimedeltaIndex,
)
from pandas.core.arrays import ExtensionArray
from pandas.core.base import IndexOpsMixin
from pandas.core.strings import StringMethods
from typing_extensions import (
    Never,
    Self,
)

from pandas._libs.interval import _OrderableT
from pandas._typing import (
    S1,
    AnyAll,
    AxesData,
    DropKeep,
    DtypeArg,
    DtypeObj,
    FillnaOptions,
    HashableT,
    Label,
    Level,
    MaskType,
    ReindexMethod,
    SliceType,
    TimedeltaDtypeArg,
    TimestampDtypeArg,
    np_ndarray_anyint,
    np_ndarray_bool,
    np_ndarray_complex,
    np_ndarray_float,
    type_t,
)

class InvalidIndexError(Exception): ...

class Index(IndexOpsMixin[S1]):
    __hash__: ClassVar[None]  # type: ignore[assignment]
    # overloads with additional dtypes
    @overload
    def __new__(  # pyright: ignore[reportOverlappingOverload]
        cls,
        data: Sequence[int | np.integer] | IndexOpsMixin[int] | np_ndarray_anyint,
        *,
        dtype: Literal["int"] | type_t[int | np.integer] = None,
        copy: bool = False,
        name: Hashable = None,
        tupleize_cols: bool = True,
        **kwargs,
    ) -> Index[int]: ...
    @overload
    def __new__(
        cls,
        data: AxesData,
        *,
        dtype: Literal["int"] | type_t[int | np.integer],
        copy: bool = False,
        name: Hashable = None,
        tupleize_cols: bool = True,
        **kwargs,
    ) -> Index[int]: ...
    @overload
    def __new__(
        cls,
        data: Sequence[float | np.floating] | IndexOpsMixin[float] | np_ndarray_float,
        *,
        dtype: Literal["float"] | type_t[float | np.floating] = None,
        copy: bool = False,
        name: Hashable = None,
        tupleize_cols: bool = True,
        **kwargs,
    ) -> Index[float]: ...
    @overload
    def __new__(
        cls,
        data: AxesData,
        *,
        dtype: Literal["float"] | type_t[float | np.floating],
        copy: bool = False,
        name: Hashable = None,
        tupleize_cols: bool = True,
        **kwargs,
    ) -> Index[float]: ...
    @overload
    def __new__(
        cls,
        data: (
            Sequence[complex | np.complexfloating]
            | IndexOpsMixin[complex]
            | np_ndarray_complex
        ),
        *,
        dtype: Literal["complex"] | type_t[complex | np.complexfloating] = None,
        copy: bool = False,
        name: Hashable = None,
        tupleize_cols: bool = True,
        **kwargs,
    ) -> Index[complex]: ...
    @overload
    def __new__(
        cls,
        data: AxesData,
        *,
        dtype: Literal["complex"] | type_t[complex | np.complexfloating],
        copy: bool = False,
        name: Hashable = None,
        tupleize_cols: bool = True,
        **kwargs,
    ) -> Index[complex]: ...
    # special overloads with dedicated Index-subclasses
    @overload
    def __new__(
        cls,
        data: Sequence[np.datetime64 | datetime] | IndexOpsMixin[datetime],
        *,
        dtype: TimestampDtypeArg = None,
        copy: bool = False,
        name: Hashable = None,
        tupleize_cols: bool = True,
        **kwargs,
    ) -> DatetimeIndex: ...
    @overload
    def __new__(
        cls,
        data: AxesData,
        *,
        dtype: TimestampDtypeArg,
        copy: bool = False,
        name: Hashable = None,
        tupleize_cols: bool = True,
        **kwargs,
    ) -> DatetimeIndex: ...
    @overload
    def __new__(
        cls,
        data: Sequence[Period] | IndexOpsMixin[Period],
        *,
        dtype: PeriodDtype = None,
        copy: bool = False,
        name: Hashable = None,
        tupleize_cols: bool = True,
        **kwargs,
    ) -> PeriodIndex: ...
    @overload
    def __new__(
        cls,
        data: AxesData,
        *,
        dtype: PeriodDtype,
        copy: bool = False,
        name: Hashable = None,
        tupleize_cols: bool = True,
        **kwargs,
    ) -> PeriodIndex: ...
    @overload
    def __new__(
        cls,
        data: Sequence[np.timedelta64 | timedelta] | IndexOpsMixin[timedelta],
        *,
        dtype: TimedeltaDtypeArg = None,
        copy: bool = False,
        name: Hashable = None,
        tupleize_cols: bool = True,
        **kwargs,
    ) -> TimedeltaIndex: ...
    @overload
    def __new__(
        cls,
        data: AxesData,
        *,
        dtype: TimedeltaDtypeArg,
        copy: bool = False,
        name: Hashable = None,
        tupleize_cols: bool = True,
        **kwargs,
    ) -> TimedeltaIndex: ...
    @overload
    def __new__(
        cls,
        data: Sequence[Interval[_OrderableT]] | IndexOpsMixin[Interval[_OrderableT]],
        *,
        dtype: Literal["Interval"] = "Interval",
        copy: bool = False,
        name: Hashable = None,
        tupleize_cols: bool = True,
        **kwargs,
    ) -> IntervalIndex[Interval[_OrderableT]]: ...
    @overload
    def __new__(
        cls,
        data: AxesData,
        *,
        dtype: Literal["Interval"],
        copy: bool = False,
        name: Hashable = None,
        tupleize_cols: bool = True,
        **kwargs,
    ) -> IntervalIndex[Interval[Any]]: ...
    # generic overloads
    @overload
    def __new__(
        cls,
        data: Iterable[S1] | IndexOpsMixin[S1],
        *,
        dtype: type[S1] = None,
        copy: bool = False,
        name: Hashable = None,
        tupleize_cols: bool = True,
        **kwargs,
    ) -> Self: ...
    @overload
    def __new__(
        cls,
        data: AxesData = None,
        *,
        dtype: type[S1],
        copy: bool = False,
        name: Hashable = None,
        tupleize_cols: bool = True,
        **kwargs,
    ) -> Self: ...
    # fallback overload
    @overload
    def __new__(
        cls,
        data: AxesData,
        *,
        dtype=None,
        copy: bool = False,
        name: Hashable = None,
        tupleize_cols: bool = True,
        **kwargs,
    ) -> Self: ...
    @property
    def str(
        self,
    ) -> StringMethods[
        Self,
        MultiIndex,
        np_ndarray_bool,
        Index[list[_str]],
        Index[int],
        Index[bytes],
        Index[_str],
        Index[type[object]],
    ]: ...
    def is_(self, other) -> bool: ...
    def __len__(self) -> int: ...
    def __array__(self, dtype=None) -> np.ndarray: ...
    def __array_wrap__(self, result, context=None): ...
    @property
    def dtype(self) -> DtypeObj: ...
    def ravel(self, order: _str = 'C'): ...
    def view(self, cls=None): ...
    def astype(self, dtype: DtypeArg, copy: bool = True) -> Index: ...
    def take(
        self, indices, axis: int = 0, allow_fill: bool = True, fill_value=None, **kwargs
    ): ...
    def repeat(self, repeats, axis=None): ...
    def copy(self, name: Hashable = None, deep: bool = False) -> Self: ...
    def __copy__(self, **kwargs): ...
    def __deepcopy__(self, memo=None): ...
    def format(
        self, name: bool = False, formatter: Callable | None = None, na_rep: _str = 'NaN'
    ) -> list[_str]: ...
    def to_flat_index(self): ...
    def to_series(self, index=None, name: Hashable = None) -> Series: ...
    def to_frame(self, index: bool = True, name=...) -> DataFrame: ...
    @property
    def name(self) -> Hashable | None: ...
    @name.setter
    def name(self, value) -> None: ...
    @property
    def names(self) -> list[Hashable]: ...
    @names.setter
    def names(self, names: Sequence[Hashable]) -> None: ...
    def set_names(self, names, *, level=None, inplace: bool = False): ...
    @overload
    def rename(self, name, inplace: Literal[False] = False) -> Self: ...
    @overload
    def rename(self, name, inplace: Literal[True]) -> None: ...
    @property
    def nlevels(self) -> int: ...
    def sortlevel(self, level=None, ascending: bool = True, sort_remaining=None): ...
    def get_level_values(self, level: int | _str) -> Index: ...
    def droplevel(self, level: Level | list[Level] = 0): ...
    @property
    def is_monotonic_increasing(self) -> bool: ...
    @property
    def is_monotonic_decreasing(self) -> bool: ...
    @property
    def is_unique(self) -> bool: ...
    @property
    def has_duplicates(self) -> bool: ...
    @property
    def inferred_type(self) -> _str: ...
    def __reduce__(self): ...
    @property
    def hasnans(self) -> bool: ...
    def isna(self): ...
    isnull = ...
    def notna(self): ...
    notnull = ...
    def fillna(self, value=None): ...
    def dropna(self, how: AnyAll = 'any') -> Self: ...
    def unique(self, level=None) -> Self: ...
    def drop_duplicates(self, *, keep: DropKeep = 'first') -> Self: ...
    def duplicated(self, keep: DropKeep = 'first') -> np_ndarray_bool: ...
    def __and__(self, other: Never) -> Never: ...
    def __rand__(self, other: Never) -> Never: ...
    def __or__(self, other: Never) -> Never: ...
    def __ror__(self, other: Never) -> Never: ...
    def __xor__(self, other: Never) -> Never: ...
    def __rxor__(self, other: Never) -> Never: ...
    def __neg__(self) -> Self: ...
    def __nonzero__(self) -> None: ...
    __bool__ = ...
    def union(self, other: list[HashableT] | Index, sort=None) -> Index: ...
    def intersection(self, other: list[S1] | Self, sort: bool = False) -> Self: ...
    def difference(self, other: list | Index, sort: bool | None = None) -> Self: ...
    def symmetric_difference(
        self, other: list[S1] | Self, result_name: Hashable = None, sort=None
    ) -> Self: ...
    def get_loc(
        self,
        key: Label,
        method: FillnaOptions | Literal["nearest"] | None = ...,
        tolerance=...,
    ) -> int | slice | np_ndarray_bool: ...
    def get_indexer(
        self, target, method: ReindexMethod | None = None, limit=None, tolerance=None
    ): ...
    def reindex(
        self,
        target,
        method: ReindexMethod | None = None,
        level=None,
        limit=None,
        tolerance=None,
    ): ...
    def join(
        self,
        other,
        *,
        how: _str = 'left',
        level=None,
        return_indexers: bool = False,
        sort: bool = False,
    ): ...
    @property
    def values(self) -> np.ndarray: ...
    @property
    def array(self) -> ExtensionArray: ...
    def memory_usage(self, deep: bool = False): ...
    def where(self, cond, other=None): ...
    def __contains__(self, key) -> bool: ...
    def __setitem__(self, key, value) -> None: ...
    @overload
    def __getitem__(
        self,
        idx: slice | np_ndarray_anyint | Sequence[int] | Index | MaskType,
    ) -> Self: ...
    @overload
    def __getitem__(self, idx: int | tuple[np_ndarray_anyint, ...]) -> S1: ...
    def append(self, other): ...
    def putmask(self, mask, value): ...
    def equals(self, other) -> bool: ...
    def identical(self, other) -> bool: ...
    def asof(self, label): ...
    def asof_locs(self, where, mask): ...
    def sort_values(self, return_indexer: bool = False, ascending: bool = True): ...
    def sort(self, *args, **kwargs) -> None: ...
    def argsort(self, *args, **kwargs): ...
    def get_indexer_non_unique(self, target): ...
    def get_indexer_for(self, target, **kwargs): ...
    @final
    def groupby(self, values) -> dict[Hashable, np.ndarray]: ...
    def map(self, mapper, na_action=None) -> Index: ...
    def isin(self, values, level=None) -> np_ndarray_bool: ...
    def slice_indexer(self, start=None, end=None, step=None): ...
    def get_slice_bound(self, label, side): ...
    def slice_locs(self, start: SliceType = None, end: SliceType = None, step=None): ...
    def delete(self, loc) -> Self: ...
    def insert(self, loc, item) -> Self: ...
    def drop(self, labels, errors: _str = 'raise') -> Self: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    # Extra methods from old stubs
    def __eq__(self, other: object) -> np_ndarray_bool: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def __iter__(self) -> Iterator[S1]: ...
    def __ne__(self, other: object) -> np_ndarray_bool: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def __le__(self, other: Self | S1) -> np_ndarray_bool: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def __ge__(self, other: Self | S1) -> np_ndarray_bool: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def __lt__(self, other: Self | S1) -> np_ndarray_bool: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def __gt__(self, other: Self | S1) -> np_ndarray_bool: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    # overwrite inherited methods from OpsMixin
    @overload
    def __mul__(
        self: Index[int] | Index[float], other: timedelta
    ) -> TimedeltaIndex: ...
    @overload
    def __mul__(self, other: Any) -> Self: ...
    def __floordiv__(
        self,
        other: (
            float
            | IndexOpsMixin[int]
            | IndexOpsMixin[float]
            | Sequence[int]
            | Sequence[float]
        ),
    ) -> Self: ...
    def __rfloordiv__(
        self,
        other: (
            float
            | IndexOpsMixin[int]
            | IndexOpsMixin[float]
            | Sequence[int]
            | Sequence[float]
        ),
    ) -> Self: ...
    def __truediv__(
        self,
        other: (
            float
            | IndexOpsMixin[int]
            | IndexOpsMixin[float]
            | Sequence[int]
            | Sequence[float]
        ),
    ) -> Self: ...
    def __rtruediv__(
        self,
        other: (
            float
            | IndexOpsMixin[int]
            | IndexOpsMixin[float]
            | Sequence[int]
            | Sequence[float]
        ),
    ) -> Self: ...

UnknownIndex: TypeAlias = Index[Any]
