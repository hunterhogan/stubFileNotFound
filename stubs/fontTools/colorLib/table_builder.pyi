from _typeshed import Incomplete
from fontTools.misc.roundTools import otRound as otRound
from fontTools.ttLib.tables.otBase import (
	BaseTable as BaseTable, FormatSwitchingBaseTable as FormatSwitchingBaseTable,
	UInt8FormatSwitchingBaseTable as UInt8FormatSwitchingBaseTable)
from fontTools.ttLib.tables.otConverters import (
	ComputedInt as ComputedInt, FloatValue as FloatValue, IntValue as IntValue, OptionalValue as OptionalValue,
	Short as Short, SimpleValue as SimpleValue, Struct as Struct, UInt8 as UInt8, UShort as UShort)
import enum

class BuildCallback(enum.Enum):
    """Keyed on (BEFORE_BUILD, class[, Format if available]).
    Receives (dest, source).
    Should return (dest, source), which can be new objects.
    """

    BEFORE_BUILD = ...
    AFTER_BUILD = ...
    CREATE_DEFAULT = ...

def _assignable(convertersByName): ...
def _isNonStrSequence(value): ...
def _split_format(cls, source): ...

class TableBuilder:
    """
    Helps to populate things derived from BaseTable from maps, tuples, etc.

    A table of lifecycle callbacks may be provided to add logic beyond what is possible
    based on otData info for the target class. See BuildCallbacks.
    """

    _callbackTable: Incomplete
    def __init__(self, callbackTable=None) -> None: ...
    def _convert(self, dest, field, converter, value) -> None: ...
    def build(self, cls, source): ...

class TableUnbuilder:
    _callbackTable: Incomplete
    def __init__(self, callbackTable=None) -> None: ...
    def unbuild(self, table): ...
