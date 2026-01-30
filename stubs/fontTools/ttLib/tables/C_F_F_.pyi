from . import DefaultTable as DefaultTable
from _typeshed import Incomplete
from fontTools import cffLib as cffLib

class table_C_F_F_(DefaultTable.DefaultTable):
    """Compact Font Format table (version 1)

    The ``CFF`` table embeds a CFF-formatted font. The CFF font format
    predates OpenType and could be used as a standalone font file, but the
    ``CFF`` table is also used to package CFF fonts into an OpenType
    container.

    .. note::
       ``CFF`` has been succeeded by ``CFF2``, which eliminates much of
       the redundancy incurred by embedding CFF version 1 in an OpenType
       font.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/cff
    """
    cff: Incomplete
    _gaveGlyphOrder: bool
    def __init__(self, tag=None) -> None: ...
    def decompile(self, data, otFont) -> None: ...
    def compile(self, otFont): ...
    def haveGlyphNames(self): ...
    def getGlyphOrder(self): ...
    def setGlyphOrder(self, glyphOrder) -> None: ...
    def toXML(self, writer, otFont) -> None: ...
    def fromXML(self, name, attrs, content, otFont) -> None: ...
