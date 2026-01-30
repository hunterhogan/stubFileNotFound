from . import DefaultTable as DefaultTable
from _typeshed import Incomplete
from fontTools.misc import sstruct as sstruct
from fontTools.misc.textTools import bytesjoin as bytesjoin
from fontTools.ttLib.tables.TupleVariation import TupleVariation as TupleVariation, compileTupleVariationStore as compileTupleVariationStore, decompileTupleVariationStore as decompileTupleVariationStore

CVAR_HEADER_FORMAT: str
CVAR_HEADER_SIZE: Incomplete

class table__c_v_a_r(DefaultTable.DefaultTable):
    """Control Value Table (CVT) variations table

    The ``cvar`` table contains variations for the values in a ``cvt``
    table.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/cvar
    """
    dependencies: Incomplete
    variations: Incomplete
    def __init__(self, tag=None) -> None: ...
    def compile(self, ttFont, useSharedPoints: bool = False): ...
    majorVersion: Incomplete
    minorVersion: Incomplete
    def decompile(self, data, ttFont) -> None: ...
    def fromXML(self, name, attrs, content, ttFont) -> None: ...
    def toXML(self, writer, ttFont) -> None: ...
