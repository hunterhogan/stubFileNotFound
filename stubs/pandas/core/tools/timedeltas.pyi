from collections.abc import Sequence
from datetime import timedelta
from typing import overload

from pandas import Index
from pandas.core.indexes.timedeltas import TimedeltaIndex
from pandas.core.series import (
    Series,
    TimedeltaSeries,
)

from pandas._libs.tslibs import Timedelta
from pandas._libs.tslibs.timedeltas import TimeDeltaUnitChoices
from pandas._typing import (
    ArrayLike,
    RaiseCoerce,
    SequenceNotStr,
)
from typing import Any

@overload
def to_timedelta(
    arg: str | float | timedelta,
    unit: TimeDeltaUnitChoices | None = None,
    errors: RaiseCoerce = 'raise',
) -> Timedelta: ...
@overload
def to_timedelta(
    arg: Series,
    unit: TimeDeltaUnitChoices | None = None,
    errors: RaiseCoerce = 'raise',
) -> TimedeltaSeries: ...
@overload
def to_timedelta(
    arg: (
        SequenceNotStr
        | Sequence[float | timedelta]
        | tuple[str | float | timedelta, ...]
        | range
        | ArrayLike
        | Index[Any]
    ),
    unit: TimeDeltaUnitChoices | None = None,
    errors: RaiseCoerce = 'raise',
) -> TimedeltaIndex: ...
