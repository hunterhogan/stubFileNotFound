from .sbixGlyph import Glyph as Glyph
from _typeshed import Incomplete
from fontTools.misc import sstruct as sstruct
from fontTools.misc.textTools import safeEval as safeEval
from fontTools.ttLib import TTFont

sbixStrikeHeaderFormat: str
sbixGlyphDataOffsetFormat: str
sbixStrikeHeaderFormatSize: Incomplete
sbixGlyphDataOffsetFormatSize: Incomplete

class Strike:
    data: Incomplete
    ppem: Incomplete
    resolution: Incomplete
    glyphs: Incomplete
    def __init__(self, rawdata=None, ppem: int = 0, resolution: int = 72) -> None: ...
    numGlyphs: Incomplete
    glyphDataOffsets: Incomplete
    def decompile(self, ttFont: TTFont) -> None: ...
    bitmapData: bytes
    def compile(self, ttFont: TTFont) -> None: ...
    def toXML(self, xmlWriter, ttFont: TTFont) -> None: ...
    def fromXML(self, name, attrs, content, ttFont: TTFont) -> None: ...
