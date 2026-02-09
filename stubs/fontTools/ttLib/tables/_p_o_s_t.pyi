from . import DefaultTable as DefaultTable
from _typeshed import Incomplete
from fontTools import ttLib as ttLib
from fontTools.misc import sstruct as sstruct
from fontTools.misc.textTools import (
	bytechr as bytechr, byteord as byteord, readHex as readHex, safeEval as safeEval, tobytes as tobytes, tostr as tostr)
from fontTools.ttLib import TTFont
from fontTools.ttLib.standardGlyphOrder import standardGlyphOrder as standardGlyphOrder

log: Incomplete
postFormat: str
postFormatSize: Incomplete

class table__p_o_s_t(DefaultTable.DefaultTable):
    """PostScript table

    The ``post`` table contains information needed to use the font on
    PostScript printers, including the PostScript names of glyphs and
    data that was stored in the ``FontInfo`` dictionary for Type 1 fonts.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/post
    """

    formatType: float
    def decompile(self, data, ttFont: TTFont) -> None: ...
    def compile(self, ttFont: TTFont) -> bytes: ...
    def getGlyphOrder(self):
        """This function will get called by a ttLib.TTFont instance.
        Do not call this function yourself, use TTFont().getGlyphOrder()
        or its relatives instead!
        """
    glyphOrder: list[str]
    def decode_format_1_0(self, data, ttFont: TTFont) -> None: ...
    extraNames: list[str]
    def decode_format_2_0(self, data, ttFont: TTFont) -> None: ...
    mapping: dict[str, int]
    def build_psNameMapping(self, ttFont: TTFont) -> None: ...
    def decode_format_3_0(self, data, ttFont: TTFont) -> None: ...
    def decode_format_4_0(self, data, ttFont: TTFont) -> None: ...
    def encode_format_2_0(self, ttFont: TTFont): ...
    def encode_format_4_0(self, ttFont: TTFont): ...
    def toXML(self, writer, ttFont: TTFont) -> None: ...
    data: Incomplete
    def fromXML(self, name, attrs, content, ttFont: TTFont) -> None: ...

def unpackPStrings(data, n): ...
def packPStrings(strings): ...
