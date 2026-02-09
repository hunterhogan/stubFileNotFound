from . import DefaultTable as DefaultTable
from _typeshed import Incomplete
from fontTools.misc.loggingTools import LogMixin as LogMixin
from fontTools.misc.textTools import strjoin as strjoin, tobytes as tobytes, tostr as tostr
from fontTools.ttLib import TTFont

class table_T_S_I__1(LogMixin, DefaultTable.DefaultTable):
    extras: Incomplete
    indextable: str
    extraPrograms: Incomplete
    glyphPrograms: Incomplete
    def decompile(self, data, ttFont: TTFont) -> None: ...
    def compile(self, ttFont: TTFont): ...
    def toXML(self, writer, ttFont: TTFont) -> None: ...
    def fromXML(self, name, attrs, content, ttFont: TTFont) -> None: ...
