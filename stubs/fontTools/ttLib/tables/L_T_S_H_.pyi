from . import DefaultTable as DefaultTable
from _typeshed import Incomplete
from fontTools.misc.textTools import safeEval as safeEval
from fontTools.ttLib import TTFont

class table_L_T_S_H_(DefaultTable.DefaultTable):
    """Linear Threshold table

    The ``LTSH`` table contains per-glyph settings indicating the ppem sizes
    at which the advance width metric should be scaled linearly, despite the
    effects of any TrueType instructions that might otherwise alter the
    advance width.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/ltsh
    """

    yPels: Incomplete
    def decompile(self, data, ttFont: TTFont) -> None: ...
    def compile(self, ttFont: TTFont): ...
    def toXML(self, writer, ttFont: TTFont) -> None: ...
    def fromXML(self, name, attrs, content, ttFont: TTFont) -> None: ...
