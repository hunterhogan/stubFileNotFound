from . import asciiTable as asciiTable
from _typeshed import Incomplete
from fontTools.misc.textTools import strjoin as strjoin, tobytes as tobytes, tostr as tostr
from fontTools.ttLib import TTFont

class table_T_S_I_V_(asciiTable.asciiTable):
    def toXML(self, writer, ttFont: TTFont) -> None: ...
    data: Incomplete
    def fromXML(self, name, attrs, content, ttFont: TTFont) -> None: ...
