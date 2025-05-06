from collections.abc import Hashable
from datetime import timedelta
from pandas import Index as Index, Series as Series, TimedeltaIndex as TimedeltaIndex
from pandas._libs.tslibs import NaT as NaT, NaTType as NaTType
from pandas._libs.tslibs.timedeltas import Timedelta as Timedelta, UnitChoices as UnitChoices, disallow_ambiguous_unit as disallow_ambiguous_unit, parse_timedelta_unit as parse_timedelta_unit
from pandas._typing import ArrayLike as ArrayLike, DateTimeErrorChoices as DateTimeErrorChoices
from pandas.core.dtypes.generic import ABCIndex as ABCIndex, ABCSeries as ABCSeries
from typing import overload

@overload
def to_timedelta(arg: str | float | timedelta, unit: UnitChoices | None = ..., errors: DateTimeErrorChoices = ...) -> Timedelta: ...
@overload
def to_timedelta(arg: Series, unit: UnitChoices | None = ..., errors: DateTimeErrorChoices = ...) -> Series: ...
@overload
def to_timedelta(arg: list | tuple | range | ArrayLike | Index, unit: UnitChoices | None = ..., errors: DateTimeErrorChoices = ...) -> TimedeltaIndex: ...
def _coerce_scalar_to_timedelta_type(r, unit: UnitChoices | None = 'ns', errors: DateTimeErrorChoices = 'raise'):
    """Convert string 'r' to a timedelta object."""
def _convert_listlike(arg, unit: UnitChoices | None = None, errors: DateTimeErrorChoices = 'raise', name: Hashable | None = None):
    """Convert a list of objects to a timedelta index object."""
