from . import DefaultTable as DefaultTable
from .sbixStrike import Strike as Strike
from _typeshed import Incomplete
from fontTools.misc import sstruct as sstruct
from fontTools.misc.textTools import binary2num as binary2num, num2binary as num2binary, safeEval as safeEval
from fontTools.ttLib import TTFont

sbixHeaderFormat: str
sbixHeaderFormatSize: Incomplete
sbixStrikeOffsetFormat: str
sbixStrikeOffsetFormatSize: Incomplete

class table__s_b_i_x(DefaultTable.DefaultTable):
    """Standard Bitmap Graphics table

    The ``sbix`` table stores bitmap image data in standard graphics formats
    like JPEG, PNG, or TIFF. The glyphs for which the ``sbix`` table provides
    data are indexed by Glyph ID. For each such glyph, the ``sbix`` table can
    hold different data for different sizes, called "strikes."

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/sbix
    """

    version: int
    flags: int
    numStrikes: int
    strikes: Incomplete
    strikeOffsets: Incomplete
    def __init__(self, tag=None) -> None: ...
    def decompile(self, data, ttFont: TTFont) -> None: ...
    def compile(self, ttFont: TTFont): ...
    def toXML(self, xmlWriter, ttFont: TTFont) -> None: ...
    def fromXML(self, name, attrs, content, ttFont: TTFont) -> None: ...

class sbixStrikeOffset: ...
