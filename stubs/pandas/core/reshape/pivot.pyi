from collections.abc import (
    Callable,
    Hashable,
    Mapping,
    Sequence,
)
import datetime
from typing import (
    Literal,
    overload,
    Any,
)

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from pandas.core.groupby.grouper import Grouper
from pandas.core.indexes.base import Index
from pandas.core.series import Series
from typing_extensions import TypeAlias

from pandas._typing import (
    AnyArrayLike,
    ArrayLike,
    HashableT1,
    HashableT2,
    HashableT3,
    Label,
    Scalar,
    ScalarT,
    npt,
)

_PivotAggCallable: TypeAlias = Callable[[Series], ScalarT]

_PivotAggFunc: TypeAlias = (
    _PivotAggCallable[Any]
    | np.ufunc
    | Literal["mean", "sum", "count", "min", "max", "median", "std", "var"]
)

_NonIterableHashable: TypeAlias = (
    str
    | datetime.date
    | datetime.datetime
    | datetime.timedelta
    | bool
    | int
    | float
    | complex
    | pd.Timestamp
    | pd.Timedelta
)

_PivotTableIndexTypes: TypeAlias = (
    Label | Sequence[HashableT1] | Series | Grouper | None
)
_PivotTableColumnsTypes: TypeAlias = (
    Label | Sequence[HashableT2] | Series | Grouper | None
)
_PivotTableValuesTypes: TypeAlias = Label | Sequence[HashableT3] | None

_ExtendedAnyArrayLike: TypeAlias = AnyArrayLike | ArrayLike

@overload
def pivot_table(
    data: DataFrame,
    values: _PivotTableValuesTypes[Any] = None,
    index: _PivotTableIndexTypes[Any] = None,
    columns: _PivotTableColumnsTypes[Any] = None,
    aggfunc: (
        _PivotAggFunc | Sequence[_PivotAggFunc] | Mapping[Hashable, _PivotAggFunc]
    ) = 'mean',
    fill_value: Scalar | None = None,
    margins: bool = False,
    dropna: bool = True,
    margins_name: str = 'All',
    observed: bool = ...,
    sort: bool = True,
) -> DataFrame: ...

# Can only use Index or ndarray when index or columns is a Grouper
@overload
def pivot_table(
    data: DataFrame,
    values: _PivotTableValuesTypes[Any] = None,
    *,
    index: Grouper,
    columns: _PivotTableColumnsTypes[Any] | Index | npt.NDArray[Any] = None,
    aggfunc: (
        _PivotAggFunc | Sequence[_PivotAggFunc] | Mapping[Hashable, _PivotAggFunc]
    ) = 'mean',
    fill_value: Scalar | None = None,
    margins: bool = False,
    dropna: bool = True,
    margins_name: str = 'All',
    observed: bool = ...,
    sort: bool = True,
) -> DataFrame: ...
@overload
def pivot_table(
    data: DataFrame,
    values: _PivotTableValuesTypes[Any] = None,
    index: _PivotTableIndexTypes[Any] | Index | npt.NDArray[Any] = None,
    *,
    columns: Grouper,
    aggfunc: (
        _PivotAggFunc | Sequence[_PivotAggFunc] | Mapping[Hashable, _PivotAggFunc]
    ) = 'mean',
    fill_value: Scalar | None = None,
    margins: bool = False,
    dropna: bool = True,
    margins_name: str = 'All',
    observed: bool = ...,
    sort: bool = True,
) -> DataFrame: ...
def pivot(
    data: DataFrame,
    *,
    index: _NonIterableHashable | Sequence[HashableT1] = ...,
    columns: _NonIterableHashable | Sequence[HashableT2] = ...,
    values: _NonIterableHashable | Sequence[HashableT3] = ...,
) -> DataFrame: ...
@overload
def crosstab(
    index: list[Any] | _ExtendedAnyArrayLike | list[Sequence[Any] | _ExtendedAnyArrayLike],
    columns: list[Any] | _ExtendedAnyArrayLike | list[Sequence[Any] | _ExtendedAnyArrayLike],
    values: list[Any] | _ExtendedAnyArrayLike,
    rownames: list[HashableT1] | None = None,
    colnames: list[HashableT2] | None = None,
    *,
    aggfunc: str | np.ufunc | Callable[[Series], float],
    margins: bool = False,
    margins_name: str = 'All',
    dropna: bool = True,
    normalize: bool | Literal[0, 1, "all", "index", "columns"] = False,
) -> DataFrame: ...
@overload
def crosstab(
    index: list[Any] | _ExtendedAnyArrayLike | list[Sequence[Any] | _ExtendedAnyArrayLike],
    columns: list[Any] | _ExtendedAnyArrayLike | list[Sequence[Any] | _ExtendedAnyArrayLike],
    values: None = None,
    rownames: list[HashableT1] | None = None,
    colnames: list[HashableT2] | None = None,
    aggfunc: None = None,
    margins: bool = False,
    margins_name: str = 'All',
    dropna: bool = True,
    normalize: bool | Literal[0, 1, "all", "index", "columns"] = False,
) -> DataFrame: ...
