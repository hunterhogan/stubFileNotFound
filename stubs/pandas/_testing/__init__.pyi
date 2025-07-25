from collections.abc import (
    Container,
    Generator,
    Iterable,
)
from contextlib import contextmanager
from typing import (
    Any,
    Literal,
    overload,
)
import warnings

from matplotlib.artist import Artist
import numpy as np
from pandas import (
    Categorical,
    DataFrame,
    Index,
    Series,
)
from pandas.arrays import (
    DatetimeArray,
    IntervalArray,
    PeriodArray,
    SparseArray,
    TimedeltaArray,
)
from pandas.core.arrays.base import ExtensionArray

from pandas._typing import (
    AnyArrayLike,
    T,
)

def assert_almost_equal(
    left: T,
    right: T,
    check_dtype: bool | Literal["equiv"] = 'equiv',
    rtol: float = 1e-05,
    atol: float = 1e-08,
    **kwargs,
) -> None: ...
def assert_dict_equal(left: dict[Any, Any], right: dict[Any, Any], compare_keys: bool = True) -> None: ...
def assert_index_equal(
    left: Index[Any],
    right: Index[Any],
    exact: bool | Literal["equiv"] = 'equiv',
    check_names: bool = True,
    check_exact: bool = True,
    check_categorical: bool = True,
    check_order: bool = True,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    obj: str = 'Index',
) -> None: ...
def assert_class_equal(
    left: T, right: T, exact: bool | Literal["equiv"] = True, obj: str = 'Input'
) -> None: ...
def assert_attr_equal(
    attr: str, left: object, right: object, obj: str = 'Attributes'
) -> None: ...
def assert_is_valid_plot_return_object(
    objs: Series | np.ndarray[Any, Any] | Artist | tuple[Any, ...] | dict[Any, Any],
) -> None: ...
def assert_is_sorted(seq: AnyArrayLike) -> None: ...
def assert_categorical_equal(
    left: Categorical,
    right: Categorical,
    check_dtype: bool = True,
    check_category_order: bool = True,
    obj: str = 'Categorical',
) -> None: ...
def assert_interval_array_equal(
    left: IntervalArray,
    right: IntervalArray,
    exact: bool | Literal["equiv"] = 'equiv',
    obj: str = 'IntervalArray',
) -> None: ...
def assert_period_array_equal(
    left: PeriodArray, right: PeriodArray, obj: str = 'PeriodArray'
) -> None: ...
def assert_datetime_array_equal(
    left: DatetimeArray, right: DatetimeArray, check_freq: bool = True
) -> None: ...
def assert_timedelta_array_equal(
    left: TimedeltaArray, right: TimedeltaArray, check_freq: bool = True
) -> None: ...
def assert_numpy_array_equal(
    left: Any,
    right: Any,
    strict_nan: bool = False,
    check_dtype: bool | Literal["equiv"] = True,
    err_msg: str | None = None,
    check_same: Literal["copy", "same"] | None = None,
    obj: str = 'numpy array',
    index_values: Index[Any] | np.ndarray[Any, Any] | None = None,
) -> None: ...
def assert_extension_array_equal(
    left: ExtensionArray,
    right: ExtensionArray,
    check_dtype: bool | Literal["equiv"] = True,
    index_values: Index[Any] | np.ndarray[Any, Any] | None = None,
    check_exact: bool = ...,
    rtol: float = ...,
    atol: float = ...,
    obj: str = 'ExtensionArray',
) -> None: ...
@overload
def assert_series_equal(
    left: Series,
    right: Series,
    check_dtype: bool | Literal["equiv"] = True,
    check_index_type: bool | Literal["equiv"] = 'equiv',
    check_series_type: bool = True,
    check_names: bool = True,
    check_exact: bool = ...,
    check_datetimelike_compat: bool = False,
    check_categorical: bool = True,
    check_category_order: bool = True,
    check_freq: bool = True,
    check_flags: bool = True,
    rtol: float = ...,
    atol: float = ...,
    obj: str = 'Series',
    *,
    check_index: Literal[False],
    check_like: Literal[False] = False,
) -> None: ...
@overload
def assert_series_equal(
    left: Series,
    right: Series,
    check_dtype: bool | Literal["equiv"] = True,
    check_index_type: bool | Literal["equiv"] = 'equiv',
    check_series_type: bool = True,
    check_names: bool = True,
    check_exact: bool = ...,
    check_datetimelike_compat: bool = False,
    check_categorical: bool = True,
    check_category_order: bool = True,
    check_freq: bool = True,
    check_flags: bool = True,
    rtol: float = ...,
    atol: float = ...,
    obj: str = 'Series',
    *,
    check_index: Literal[True] = True,
    check_like: bool = False,
) -> None: ...
def assert_frame_equal(
    left: DataFrame,
    right: DataFrame,
    check_dtype: bool | Literal["equiv"] = True,
    check_index_type: bool | Literal["equiv"] = 'equiv',
    check_column_type: bool | Literal["equiv"] = 'equiv',
    check_frame_type: bool = True,
    check_names: bool = True,
    by_blocks: bool = False,
    check_exact: bool = ...,
    check_datetimelike_compat: bool = False,
    check_categorical: bool = True,
    check_like: bool = False,
    check_freq: bool = True,
    check_flags: bool = True,
    rtol: float = ...,
    atol: float = ...,
    obj: str = 'DataFrame',
) -> None: ...
def assert_equal(left: Any, right: Any, **kwargs) -> None: ...
def assert_sp_array_equal(left: SparseArray, right: SparseArray) -> None: ...
def assert_contains_all(iterable: Iterable[T], dic: Container[T]) -> None: ...
def assert_copy(iter1: Iterable[T], iter2: Iterable[T], **eql_kwargs) -> None: ...
@contextmanager
def assert_produces_warning(
    expected_warning: (
        type[Warning] | Literal[False] | tuple[type[Warning], ...] | None
    ) = ...,
    filter_level: Literal[
        "error", "ignore", "always", "default", "module", "once"
    ] = 'always',
    check_stacklevel: bool = True,
    raise_on_extra_warnings: bool = True,
    match: str | None = None,
) -> Generator[list[warnings.WarningMessage], None, None]: ...
@contextmanager
def ensure_clean(
    filename: str | None = None, return_filelike: bool = False, **kwargs: Any
) -> Generator[str, None, None]: ...
