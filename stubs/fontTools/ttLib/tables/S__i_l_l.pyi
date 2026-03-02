from . import DefaultTable as DefaultTable, grUtils as grUtils
from _typeshed import Incomplete
from fontTools.misc import sstruct as sstruct
from fontTools.misc.fixedTools import floatToFixedToStr as floatToFixedToStr
from fontTools.misc.textTools import safeEval as safeEval
from fontTools.ttLib import TTFont

Sill_hdr: str

class table_S__i_l_l(DefaultTable.DefaultTable):
    """Graphite Languages table

    See also https://graphite.sil.org/graphite_techAbout#graphite-font-tables
    """

    langs: Incomplete
    def __init__(self, tag=None) -> None: ...
    version: Incomplete
    def decompile(self, data, ttFont: TTFont) -> None: ...
    def compile(self, ttFont: TTFont): ...
    def toXML(self, writer, ttFont: TTFont) -> None: ...
    def fromXML(self, name, attrs, content, ttFont: TTFont) -> None: ...
