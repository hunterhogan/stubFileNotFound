from _typeshed import Incomplete
from fontTools.misc import sstruct as sstruct
from fontTools.misc.textTools import safeEval as safeEval
from fontTools.ttLib import TTFont

log: Incomplete
bigGlyphMetricsFormat: str
smallGlyphMetricsFormat: str

class BitmapGlyphMetrics:
    def toXML(self, writer, ttFont: TTFont) -> None: ...
    def fromXML(self, name, attrs, content, ttFont: TTFont) -> None: ...

class BigGlyphMetrics(BitmapGlyphMetrics):
    binaryFormat = bigGlyphMetricsFormat

class SmallGlyphMetrics(BitmapGlyphMetrics):
    binaryFormat = smallGlyphMetricsFormat
