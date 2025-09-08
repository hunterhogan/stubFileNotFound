from builtins import str as _str
from collections.abc import Callable, Hashable, Iterable, Iterator, Sequence
from datetime import datetime, timedelta
from pandas import (
	DataFrame, DatetimeIndex, Interval, IntervalIndex, MultiIndex, Period, PeriodDtype, PeriodIndex, Series,
	TimedeltaIndex)
from pandas._libs.interval import _OrderableT
from pandas._typing import (
	AnyAll, ArrayLike, AxesData, C2, DropKeep, Dtype, DtypeArg, DTypeLike, DtypeObj, GenericT, GenericT_co, HashableT,
	IgnoreRaise, Just, Label, Level, MaskType, NaPosition, np_1darray, np_ndarray_anyint, np_ndarray_bool,
	np_ndarray_complex, np_ndarray_float, np_ndarray_str, ReindexMethod, S1, Scalar, SequenceNotStr, SliceType,
	SupportsDType, T_COMPLEX, T_INT, TimedeltaDtypeArg, TimestampDtypeArg, type_t)
from pandas.core.arrays import ExtensionArray
from pandas.core.base import _ListLike, IndexOpsMixin, NumListLike
from pandas.core.strings.accessor import StringMethods
from typing import Any, ClassVar, final, Generic, Literal, overload, type_check_only
from typing_extensions import Never, Self
import numpy as np

class InvalidIndexError(Exception): ...

class Index(IndexOpsMixin[S1]):
    __hash__: ClassVar[None]  # type: ignore[assignment]
    # overloads with additional dtypes
    @overload
    def __new__(  # pyright: ignore[reportOverlappingOverload]
        cls,
        data: Sequence[bool | np.bool_] | IndexOpsMixin[bool] | np_ndarray_bool,
        *,
        dtype: Literal["bool"] | type_t[bool | np.bool_] | None = None,
        copy: bool = False,
        name: Hashable = None,
        tupleize_cols: bool = True,
    ) -> Index[bool]: ...
    @overload
    def __new__(
        cls,
        data: Sequence[int | np.integer] | IndexOpsMixin[int] | np_ndarray_anyint,
        *,
        dtype: Literal["int"] | type_t[int | np.integer] | None = None,
        copy: bool = False,
        name: Hashable = None,
        tupleize_cols: bool = True,
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
    ) -> Index[int]: ...
    @overload
    def __new__(
        cls,
        data: Sequence[float | np.floating] | IndexOpsMixin[float] | np_ndarray_float,
        *,
        dtype: Literal["float"] | type_t[float | np.floating] | None = None,
        copy: bool = False,
        name: Hashable = None,
        tupleize_cols: bool = True,
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
        dtype: Literal["complex"] | type_t[complex | np.complexfloating] | None = None,
        copy: bool = False,
        name: Hashable = None,
        tupleize_cols: bool = True,
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
    ) -> Index[complex]: ...
    # special overloads with dedicated Index-subclasses
    @overload
    def __new__(
        cls,
        data: Sequence[np.datetime64 | datetime] | IndexOpsMixin[datetime],
        *,
        dtype: TimestampDtypeArg | None = None,
        copy: bool = False,
        name: Hashable = None,
        tupleize_cols: bool = True,
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
    ) -> DatetimeIndex: ...
    @overload
    def __new__(
        cls,
        data: Sequence[Period] | IndexOpsMixin[Period],
        *,
        dtype: PeriodDtype | None = None,
        copy: bool = False,
        name: Hashable = None,
        tupleize_cols: bool = True,
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
    ) -> PeriodIndex: ...
    @overload
    def __new__(
        cls,
        data: Sequence[np.timedelta64 | timedelta] | IndexOpsMixin[timedelta],
        *,
        dtype: TimedeltaDtypeArg | None = None,
        copy: bool = False,
        name: Hashable = None,
        tupleize_cols: bool = True,
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
    ) -> IntervalIndex[Interval[Any]]: ...
    # generic overloads
    @overload
    def __new__(
        cls,
        data: Iterable[S1] | IndexOpsMixin[S1],
        *,
        dtype: type[S1] | None = None,
        copy: bool = False,
        name: Hashable = None,
        tupleize_cols: bool = True,
    ) -> Self: ...
    @overload
    def __new__(
        cls,
        data: AxesData[Any] | None = None,
        *,
        dtype: type[S1],
        copy: bool = False,
        name: Hashable = None,
        tupleize_cols: bool = True,
    ) -> Self: ...
    # fallback overload
    @overload
    def __new__(
        cls,
        data: AxesData[Any],
        *,
        dtype: Dtype | None = None,
        copy: bool = False,
        name: Hashable = None,
        tupleize_cols: bool = True,
    ) -> Self: ...
    @property
    def str(
        self,
    ) -> StringMethods[Self,
        MultiIndex,
        np_1darray[np.bool],
        Index[list[_str]],
        Index[int],
        Index[bytes],
        Index[_str],
        Index[Any],
    ]: ...
    @final
    def is_(self, other: Any) -> bool: ...
    def __len__(self) -> int: ...
    def __array__(
        self, dtype: _str | np.dtype[Any] | None = None, copy: bool | None = None
    ) -> np_1darray: ...
    def __array_wrap__(self, result: Any, context: Any=None) -> Any: ...
    @property
    def dtype(self) -> DtypeObj: ...
    @final
    def ravel(self, order: _str = 'C') -> Any: ...
    def view(self, cls:Any=None) -> Any: ...
    def astype(self, dtype: DtypeArg, copy: bool = True) -> Index[Any]: ...
    def take(
        self,
        indices: Any,
        axis: int = 0,
        allow_fill: bool = True,
        fill_value: Scalar | None = None,
        **kwargs: Any,
    ) -> Any: ...
    def repeat(self, repeats: Any, axis: Any=None) -> Any: ...
    def copy(self, name: Hashable = None, deep: bool = False) -> Self: ...
    @final
    def __copy__(self, **kwargs: Any) -> Any: ...
    @final
    def __deepcopy__(self, memo: Any=None) -> Any: ...
    def format(
        self, name: bool = False, formatter: Callable[..., Any] | None = None, na_rep: _str = 'NaN'
    ) -> list[_str]: ...
    def to_flat_index(self) -> Any: ...
    def to_series(self, index: Any=None, name: Hashable = None) -> Series: ...
    def to_frame(self, index: bool = True, name: Any=...) -> DataFrame: ...
    @property
    def name(self) -> Hashable | None: ...
    @name.setter
    def name(self, value: Hashable) -> None: ...
    @property
    def names(self) -> list[Hashable | None]: ...
    @names.setter
    def names(self, names: SequenceNotStr[Hashable | None]) -> None: ...
    def set_names(self, names: Any, *, level: Any=None, inplace: bool = False) -> Any: ...
    @overload
    def rename(self, name: Any, *, inplace: Literal[False] = False) -> Self: ...
    @overload
    def rename(self, name: Any, *, inplace: Literal[True]) -> None: ...
    @property
    def nlevels(self) -> int: ...
    def get_level_values(self, level: int | _str) -> Index[Any]: ...
    def droplevel(self, level: Level | list[Level] = 0) -> Any: ...
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
    def __reduce__(self) -> Any: ...
    @property
    def hasnans(self) -> bool: ...
    @final
    def isna(self) -> Any: ...
    isnull = ...
    @final
    def notna(self) -> Any: ...
    notnull = ...
    def fillna(self, value: Any=None) -> Any: ...
    def dropna(self, how: AnyAll = "any") -> Self: ...
    def unique(self, level: Any=None) -> Self: ...
    def drop_duplicates(self, *, keep: DropKeep = 'first') -> Self: ...
    def duplicated(self, keep: DropKeep = "first") -> np_1darray[np.bool]: ...
    def __and__(self, other: Never) -> Never: ...
    def __rand__(self, other: Never) -> Never: ...
    def __or__(self, other: Never) -> Never: ...
    def __ror__(self, other: Never) -> Never: ...
    def __xor__(self, other: Never) -> Never: ...
    def __rxor__(self, other: Never) -> Never: ...
    def __neg__(self) -> Self: ...
    @final
    def __nonzero__(self) -> None: ...
    __bool__ = ...
    def union(
        self, other: list[HashableT] | Self, sort: bool | None = None
    ) -> Index[Any]: ...
    def intersection(
        self, other: list[S1] | Self, sort: bool | None = False
    ) -> Self: ...
    def difference(self, other: list[Any] | Self, sort: bool | None = None) -> Self: ...
    def symmetric_difference(
        self,
        other: list[S1] | Self,
        result_name: Hashable = None,
        sort: bool | None = None,
    ) -> Self: ...
    def get_loc(self, key: Label) -> int | slice | np_1darray[np.bool]: ...
    def get_indexer(
        self, target: Any, method: ReindexMethod | None = None, limit: Any=None, tolerance: Any=None
    ) -> Any: ...
    def reindex(
        self,
        target: Any,
        method: ReindexMethod | None = None,
        level: Any=None,
        limit: Any=None,
        tolerance: Any=None,
    ) -> Any: ...
    def join(
        self,
        other: Any,
        *,
        how: _str = 'left',
        level: Any=None,
        return_indexers: bool = False,
        sort: bool = False,
    ) -> Any: ...
    @property
    def values(self) -> np_1darray: ...
    @property
    def array(self) -> ExtensionArray: ...
    def memory_usage(self, deep: bool = False) -> Any: ...
    def where(self, cond: Any, other: Scalar | ArrayLike | None = None) -> Any: ...
    def __contains__(self, key: Any) -> bool: ...
    @final
    def __setitem__(self, key: Any, value: Any) -> None: ...
    @overload
    def __getitem__(
        self,
        idx: slice | np_ndarray_anyint | Sequence[int] | Index[Any] | MaskType,
    ) -> Self: ...
    @overload
    def __getitem__(self, idx: int | tuple[np_ndarray_anyint, ...]) -> S1: ...
    @overload
    def append(
        self: Index[C2], other: Index[C2] | Sequence[Index[C2]]
    ) -> Index[C2]: ...
    @overload
    def append(self, other: Index[Any] | Sequence[Index[Any]]) -> Index[Any]: ...
    def putmask(self, mask: Any, value: Any) -> Any: ...
    def equals(self, other: Any) -> bool: ...
    @final
    def identical(self, other: Any) -> bool: ...
    @final
    def asof(self, label: Any) -> Any: ...
    def asof_locs(self, where: Any, mask: Any) -> Any: ...
    def sort_values(
        self,
        *,
        return_indexer: bool = False,
        ascending: bool = True,
        na_position: NaPosition = 'last',
        key: Callable[[Index], Index[Any]] | None = None,
    ) -> Any: ...
    @final
    def sort(self, *args: Any, **kwargs: Any) -> None: ...
    def argsort(self, *args: Any, **kwargs: Any) -> Any: ...
    def get_indexer_non_unique(self, target: Any) -> Any: ...
    @final
    def get_indexer_for(self, target: Any, **kwargs: Any) -> Any: ...
    @final
    def groupby(self, values: Any) -> dict[Hashable, np.ndarray[Any, Any]]: ...
    def map(self, mapper: Any, na_action: Any=None) -> Index[Any]: ...
    def isin(self, values: Any, level: Any=None) -> np_1darray[np.bool]: ...
    def slice_indexer(
        self,
        start: Label | None = None,
        end: Label | None = None,
        step: int | None = None,
    ) -> Any: ...
    def get_slice_bound(self, label: Any, side: Any) -> Any: ...
    def slice_locs(
        self, start: SliceType = None, end: SliceType = None, step: int | None = None
    ) -> Any: ...
    def delete(self, loc: Any) -> Self: ...
    def insert(self, loc: Any, item: Any) -> Self: ...
    def drop(self, labels: Any, errors: IgnoreRaise = "raise") -> Self: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    # Extra methods from old stubs
    def __eq__(self, other: object) -> np_1darray[np.bool]: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def __iter__(self) -> Iterator[S1]: ...
    def __ne__(self, other: object) -> np_1darray[np.bool]: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def __le__(self, other: Self | S1) -> np_1darray[np.bool]: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def __ge__(self, other: Self | S1) -> np_1darray[np.bool]: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def __lt__(self, other: Self | S1) -> np_1darray[np.bool]: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def __gt__(self, other: Self | S1) -> np_1darray[np.bool]: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    # overwrite inherited methods from OpsMixin
    @overload
    def __add__(self: Index[Never], other: _str) -> Never: ...
    @overload
    def __add__(self: Index[Never], other: complex | _ListLike | Index[Any]) -> Index[Any]: ...
    @overload
    def __add__(self, other: Index[Never]) -> Index[Any]: ...
    @overload
    def __add__(
        self: Index[bool],
        other: T_COMPLEX | Sequence[T_COMPLEX] | Index[T_COMPLEX],
    ) -> Index[T_COMPLEX]: ...
    @overload
    def __add__(self: Index[bool], other: np_ndarray_bool) -> Index[bool]: ...
    @overload
    def __add__(self: Index[bool], other: np_ndarray_anyint) -> Index[int]: ...
    @overload
    def __add__(self: Index[bool], other: np_ndarray_float) -> Index[float]: ...
    @overload
    def __add__(self: Index[bool], other: np_ndarray_complex) -> Index[complex]: ...
    @overload
    def __add__(
        self: Index[int],
        other: (
            bool | Sequence[bool] | np_ndarray_bool | np_ndarray_anyint | Index[bool]
        ),
    ) -> Index[int]: ...
    @overload
    def __add__(
        self: Index[int],
        other: T_COMPLEX | Sequence[T_COMPLEX] | Index[T_COMPLEX],
    ) -> Index[T_COMPLEX]: ...
    @overload
    def __add__(self: Index[int], other: np_ndarray_float) -> Index[float]: ...
    @overload
    def __add__(self: Index[int], other: np_ndarray_complex) -> Index[complex]: ...
    @overload
    def __add__(
        self: Index[float],
        other: (
            int
            | Sequence[int]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Index[T_INT]
        ),
    ) -> Index[float]: ...
    @overload
    def __add__(
        self: Index[float],
        other: T_COMPLEX | Sequence[T_COMPLEX] | Index[T_COMPLEX],
    ) -> Index[T_COMPLEX]: ...
    @overload
    def __add__(self: Index[float], other: np_ndarray_complex) -> Index[complex]: ...
    @overload
    def __add__(
        self: Index[complex],
        other: (
            T_COMPLEX
            | Sequence[T_COMPLEX]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | np_ndarray_complex
            | Index[T_COMPLEX]
        ),
    ) -> Index[complex]: ...
    @overload
    def __add__(
        self: Index[_str],
        other: (
            np_ndarray_bool | np_ndarray_anyint | np_ndarray_float | np_ndarray_complex
        ),
    ) -> Never: ...
    @overload
    def __add__(
        self: Index[_str], other: _str | Sequence[_str] | np_ndarray_str | Index[_str]
    ) -> Index[_str]: ...
    @overload
    def __radd__(self: Index[Never], other: _str) -> Never: ...
    @overload
    def __radd__(self: Index[Never], other: complex | _ListLike | Index[Any]) -> Index[Any]: ...
    @overload
    def __radd__(
        self: Index[bool],
        other: T_COMPLEX | Sequence[T_COMPLEX] | Index[T_COMPLEX],
    ) -> Index[T_COMPLEX]: ...
    @overload
    def __radd__(self: Index[bool], other: np_ndarray_bool) -> Index[bool]: ...
    @overload
    def __radd__(self: Index[bool], other: np_ndarray_anyint) -> Index[int]: ...
    @overload
    def __radd__(self: Index[bool], other: np_ndarray_float) -> Index[float]: ...
    @overload
    def __radd__(
        self: Index[int],
        other: (
            bool | Sequence[bool] | np_ndarray_bool | np_ndarray_anyint | Index[bool]
        ),
    ) -> Index[int]: ...
    @overload
    def __radd__(
        self: Index[int], other: T_COMPLEX | Sequence[T_COMPLEX] | Index[T_COMPLEX]
    ) -> Index[T_COMPLEX]: ...
    @overload
    def __radd__(self: Index[int], other: np_ndarray_float) -> Index[float]: ...
    @overload
    def __radd__(
        self: Index[float],
        other: (
            int
            | Sequence[int]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Index[T_INT]
        ),
    ) -> Index[float]: ...
    @overload
    def __radd__(
        self: Index[float], other: T_COMPLEX | Sequence[T_COMPLEX] | Index[T_COMPLEX]
    ) -> Index[T_COMPLEX]: ...
    @overload
    def __radd__(
        self: Index[complex],
        other: (
            T_COMPLEX
            | Sequence[T_COMPLEX]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Index[T_COMPLEX]
        ),
    ) -> Index[complex]: ...
    @overload
    def __radd__(
        self: Index[T_COMPLEX], other: np_ndarray_complex
    ) -> Index[complex]: ...
    @overload
    def __radd__(
        self: Index[_str],
        other: (
            np_ndarray_bool | np_ndarray_anyint | np_ndarray_float | np_ndarray_complex
        ),
    ) -> Never: ...
    @overload
    def __radd__(
        self: Index[_str], other: _str | Sequence[_str] | np_ndarray_str | Index[_str]
    ) -> Index[_str]: ...
    @overload
    def __sub__(self: Index[Never], other: DatetimeIndex) -> Never: ...
    @overload
    def __sub__(self: Index[Never], other: complex | NumListLike | Index[Any]) -> Index[Any]: ...
    @overload
    def __sub__(self, other: Index[Never]) -> Index[Any]: ...
    @overload
    def __sub__(
        self: Index[bool],
        other: Just[int] | Sequence[Just[int]] | np_ndarray_anyint | Index[int],
    ) -> Index[int]: ...
    @overload
    def __sub__(
        self: Index[bool],
        other: Just[float] | Sequence[Just[float]] | np_ndarray_float | Index[float],
    ) -> Index[float]: ...
    @overload
    def __sub__(
        self: Index[int],
        other: (
            int
            | Sequence[int]
            | np_ndarray_bool
            | np_ndarray_anyint
            | Index[bool]
            | Index[int]
        ),
    ) -> Index[int]: ...
    @overload
    def __sub__(
        self: Index[int],
        other: Just[float] | Sequence[Just[float]] | np_ndarray_float | Index[float],
    ) -> Index[float]: ...
    @overload
    def __sub__(
        self: Index[float],
        other: (
            float
            | Sequence[float]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Index[bool]
            | Index[int]
            | Index[float]
        ),
    ) -> Index[float]: ...
    @overload
    def __sub__(
        self: Index[complex],
        other: (
            T_COMPLEX
            | Sequence[T_COMPLEX]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Index[T_COMPLEX]
        ),
    ) -> Index[complex]: ...
    @overload
    def __sub__(
        self: Index[T_COMPLEX],
        other: (
            Just[complex]
            | Sequence[Just[complex]]
            | np_ndarray_complex
            | Index[complex]
        ),
    ) -> Index[complex]: ...
    @overload
    def __rsub__(self: Index[Never], other: DatetimeIndex) -> Never: ...  # type: ignore[misc]
    @overload
    def __rsub__(self: Index[Never], other: complex | NumListLike | Index[Any]) -> Index[Any]: ...
    @overload
    def __rsub__(self, other: Index[Never]) -> Index[Any]: ...
    @overload
    def __rsub__(
        self: Index[bool],
        other: Just[int] | Sequence[Just[int]] | np_ndarray_anyint | Index[int],
    ) -> Index[int]: ...
    @overload
    def __rsub__(
        self: Index[bool],
        other: Just[float] | Sequence[Just[float]] | np_ndarray_float | Index[float],
    ) -> Index[float]: ...
    @overload
    def __rsub__(
        self: Index[int],
        other: (
            int
            | Sequence[int]
            | np_ndarray_bool
            | np_ndarray_anyint
            | Index[bool]
            | Index[int]
        ),
    ) -> Index[int]: ...
    @overload
    def __rsub__(
        self: Index[int],
        other: Just[float] | Sequence[Just[float]] | np_ndarray_float | Index[float],
    ) -> Index[float]: ...
    @overload
    def __rsub__(
        self: Index[float],
        other: (
            float
            | Sequence[float]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Index[bool]
            | Index[int]
            | Index[float]
        ),
    ) -> Index[float]: ...
    @overload
    def __rsub__(
        self: Index[complex],
        other: (
            T_COMPLEX
            | Sequence[T_COMPLEX]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Index[T_COMPLEX]
        ),
    ) -> Index[complex]: ...
    @overload
    def __rsub__(
        self: Index[T_COMPLEX],
        other: (
            Just[complex]
            | Sequence[Just[complex]]
            | np_ndarray_complex
            | Index[complex]
        ),
    ) -> Index[complex]: ...
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

@type_check_only
class _IndexSubclassBase(Index[S1], Generic[S1, GenericT_co]):
    @overload
    def to_numpy(  # pyrefly: ignore
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
