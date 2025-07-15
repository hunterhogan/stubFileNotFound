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
    HashableT,
    Label,
    Level,
    MaskType,
    NaPosition,
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
        data: AxesData[Any],
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
        data: AxesData[Any],
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
        data: AxesData[Any],
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
        data: AxesData[Any],
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
        data: AxesData[Any],
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
        data: AxesData[Any],
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
        data: AxesData[Any],
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
        data: AxesData[Any] = None,
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
        data: AxesData[Any],
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
    def is_(self, other: Any) -> bool: ...
    def __len__(self) -> int: ...
    def __array__(self, dtype: Any=None) -> np.ndarray[Any, Any]: ...
    def __array_wrap__(self, result: Any, context: Any=None): ...
    @property
    def dtype(self) -> DtypeObj: ...
    def ravel(self, order: _str = 'C'): ...
    def view(self, cls: Any=None): ...
    def astype(self, dtype: DtypeArg, copy: bool = True) -> Index[Any]: ...
    def take(
        self, indices: Any, axis: int = 0, allow_fill: bool = True, fill_value: Any=None, **kwargs
    ): ...
    def repeat(self, repeats: Any, axis: Any=None): ...
    def copy(self, name: Hashable = None, deep: bool = False) -> Self: ...
    def __copy__(self, **kwargs): ...
    def __deepcopy__(self, memo: Any=None): ...
    def format(
        self, name: bool = False, formatter: Callable[..., Any] | None = None, na_rep: _str = 'NaN'
    ) -> list[_str]: ...
    def to_flat_index(self): ...
    def to_series(self, index: Any=None, name: Hashable = None) -> Series: ...
    def to_frame(self, index: bool = True, name: Any=...) -> DataFrame: ...
    @property
    def name(self) -> Hashable | None: ...
    @name.setter
    def name(self, value: Any) -> None: ...
    @property
    def names(self) -> list[Hashable]: ...
    @names.setter
    def names(self, names: Sequence[Hashable]) -> None: ...
    def set_names(self, names: Any, *, level=None, inplace: bool = False): ...
    @overload
    def rename(self, name: Any, inplace: Literal[False] = False) -> Self: ...
    @overload
    def rename(self, name: Any, inplace: Literal[True]) -> None: ...
    @property
    def nlevels(self) -> int: ...
    def get_level_values(self, level: int | _str) -> Index[Any]: ...
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
    def fillna(self, value: Any=None): ...
    def dropna(self, how: AnyAll = 'any') -> Self: ...
    def unique(self, level: Any=None) -> Self: ...
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
    def union(
        self, other: list[HashableT] | Self, sort: bool | None = None
    ) -> Index[Any]: ...
    def intersection(self, other: list[S1] | Self, sort: bool | None = False) -> Self: ...
    def difference(self, other: list[Any] | Self, sort: bool | None = None) -> Self: ...
    def symmetric_difference(
        self,
        other: list[S1] | Self,
        result_name: Hashable = None,
        sort: bool | None = None,
    ) -> Self: ...
    def get_loc(self, key: Label) -> int | slice | np_ndarray_bool: ...
    def get_indexer(
        self, target: Any, method: ReindexMethod | None = None, limit: Any=None, tolerance: Any=None
    ): ...
    def reindex(
        self,
        target: Any,
        method: ReindexMethod | None = None,
        level: Any=None,
        limit: Any=None,
        tolerance: Any=None,
    ): ...
    def join(
        self,
        other: Any,
        *,
        how: _str = 'left',
        level=None,
        return_indexers: bool = False,
        sort: bool = False,
    ): ...
    @property
    def values(self) -> np.ndarray[Any, Any]: ...
    @property
    def array(self) -> ExtensionArray: ...
    def memory_usage(self, deep: bool = False): ...
    def where(self, cond: Any, other: Any=None): ...
    def __contains__(self, key: Any) -> bool: ...
    def __setitem__(self, key: Any, value: Any) -> None: ...
    @overload
    def __getitem__(
        self,
        idx: slice | np_ndarray_anyint | Sequence[int] | Index[Any] | MaskType,
    ) -> Self: ...
    @overload
    def __getitem__(self, idx: int | tuple[np_ndarray_anyint, ...]) -> S1: ...
    def append(self, other: Any): ...
    def putmask(self, mask: Any, value: Any): ...
    def equals(self, other: Any) -> bool: ...
    def identical(self, other: Any) -> bool: ...
    def asof(self, label: Any): ...
    def asof_locs(self, where: Any, mask: Any): ...
    def sort_values(
        self,
        *,
        return_indexer: bool = False,
        ascending: bool = True,
        na_position: NaPosition = 'last',
        key: Callable[[Index], Index] | None = None,
    ): ...
    def sort(self, *args, **kwargs) -> None: ...
    def argsort(self, *args, **kwargs): ...
    def get_indexer_non_unique(self, target: Any): ...
    def get_indexer_for(self, target: Any, **kwargs): ...
    @final
    def groupby(self, values: Any) -> dict[Hashable, np.ndarray]: ...
    def map(self, mapper: Any, na_action: Any=None) -> Index[Any]: ...
    def isin(self, values: Any, level: Any=None) -> np_ndarray_bool: ...
    def slice_indexer(self, start: Any=None, end: Any=None, step: Any=None): ...
    def get_slice_bound(self, label: Any, side: Any): ...
    def slice_locs(self, start: SliceType = None, end: SliceType = None, step: Any=None): ...
    def delete(self, loc: Any) -> Self: ...
    def insert(self, loc: Any, item: Any) -> Self: ...
    def drop(self, labels: Any, errors: _str = 'raise') -> Self: ...
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
    def infer_objects(self, copy: bool = True) -> Self: ...

UnknownIndex: TypeAlias = Index[Any]
