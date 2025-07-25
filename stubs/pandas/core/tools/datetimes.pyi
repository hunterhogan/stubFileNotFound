from collections.abc import Sequence
from datetime import (
    date,
    datetime,
)
from typing import (
    Literal,
    TypedDict,
    overload,
)

import numpy as np
from pandas import (
    Index,
    Timestamp,
)
from pandas.core.arrays import ExtensionArray
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.series import (
    Series,
    TimestampSeries,
)
from typing_extensions import TypeAlias

from pandas._libs.tslibs import NaTType
from pandas._typing import (
    AnyArrayLike,
    DictConvertible,
    IgnoreRaise,
    RaiseCoerce,
    TimestampConvertibleTypes,
    npt,
)
from typing import Any

ArrayConvertible: TypeAlias = list | tuple | AnyArrayLike
Scalar: TypeAlias = float | str
DatetimeScalar: TypeAlias = Scalar | datetime | np.datetime64 | date

DatetimeScalarOrArrayConvertible: TypeAlias = DatetimeScalar | ArrayConvertible

DatetimeDictArg: TypeAlias = list[Scalar] | tuple[Scalar, ...] | AnyArrayLike

class YearMonthDayDict(TypedDict, total=True):
    year: DatetimeDictArg
    month: DatetimeDictArg
    day: DatetimeDictArg

class FulldatetimeDict(YearMonthDayDict, total=False):
    hour: DatetimeDictArg
    hours: DatetimeDictArg
    minute: DatetimeDictArg
    minutes: DatetimeDictArg
    second: DatetimeDictArg
    seconds: DatetimeDictArg
    ms: DatetimeDictArg
    us: DatetimeDictArg
    ns: DatetimeDictArg

@overload
def to_datetime(
    arg: DatetimeScalar,
    errors: IgnoreRaise = 'raise',
    dayfirst: bool = False,
    yearfirst: bool = False,
    utc: bool = False,
    format: str | None = None,
    exact: bool = ...,
    unit: str | None = None,
    origin: Literal["julian", "unix"] | TimestampConvertibleTypes = 'unix',
    cache: bool = True,
) -> Timestamp: ...
@overload
def to_datetime(
    arg: DatetimeScalar,
    errors: Literal["coerce"],
    dayfirst: bool = False,
    yearfirst: bool = False,
    utc: bool = False,
    format: str | None = None,
    exact: bool = ...,
    unit: str | None = None,
    origin: Literal["julian", "unix"] | TimestampConvertibleTypes = 'unix',
    cache: bool = True,
) -> Timestamp | NaTType: ...
@overload
def to_datetime(
    arg: Series | DictConvertible,
    errors: RaiseCoerce = 'raise',
    dayfirst: bool = False,
    yearfirst: bool = False,
    utc: bool = False,
    format: str | None = None,
    exact: bool = ...,
    unit: str | None = None,
    origin: Literal["julian", "unix"] | TimestampConvertibleTypes = 'unix',
    cache: bool = True,
) -> TimestampSeries: ...
@overload
def to_datetime(
    arg: (
        Sequence[float | date]
        | list[str]
        | tuple[float | str | date, ...]
        | npt.NDArray[np.datetime64]
        | npt.NDArray[np.str_]
        | npt.NDArray[np.int_]
        | Index[Any]
        | ExtensionArray
    ),
    errors: RaiseCoerce = 'raise',
    dayfirst: bool = False,
    yearfirst: bool = False,
    utc: bool = False,
    format: str | None = None,
    exact: bool = ...,
    unit: str | None = None,
    origin: Literal["julian", "unix"] | TimestampConvertibleTypes = 'unix',
    cache: bool = True,
) -> DatetimeIndex: ...
