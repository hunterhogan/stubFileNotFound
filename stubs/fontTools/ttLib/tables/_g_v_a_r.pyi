from . import DefaultTable as DefaultTable
from _typeshed import Incomplete
from fontTools.misc import sstruct as sstruct
from fontTools.misc.lazyTools import LazyDict as LazyDict
from fontTools.misc.textTools import safeEval as safeEval
from fontTools.ttLib import OPTIMIZE_FONT_SPEED as OPTIMIZE_FONT_SPEED, TTFont
from fontTools.ttLib.tables.TupleVariation import TupleVariation as TupleVariation
from functools import partial as partial

log: Incomplete
GVAR_HEADER_FORMAT_HEAD: str
GVAR_HEADER_FORMAT_TAIL: str
GVAR_HEADER_SIZE_HEAD: Incomplete
GVAR_HEADER_SIZE_TAIL: Incomplete

class table__g_v_a_r(DefaultTable.DefaultTable):
    """Glyph Variations table

    The ``gvar`` table provides the per-glyph variation data that
    describe how glyph outlines in the ``glyf`` table change across
    the variation space that is defined for the font in the ``fvar``
    table.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/gvar
    """

    dependencies: Incomplete
    gid_size: int
    variations: Incomplete
    def __init__(self, tag=None) -> None: ...
    def compile(self, ttFont: TTFont): ...
    def compileGlyphs_(self, ttFont: TTFont, axisTags, sharedCoordIndices): ...
    glyphCount: Incomplete
    def decompile(self, data, ttFont: TTFont): ...
    def ensureDecompiled(self, recurse: bool = False) -> None: ...
    @staticmethod
    def decompileOffsets_(data, tableFormat, glyphCount): ...
    @staticmethod
    def compileOffsets_(offsets):
        """Packs a list of offsets into a 'gvar' offset table.

        Returns a pair (bytestring, tableFormat). Bytestring is the
        packed offset table. Format indicates whether the table
        uses short (tableFormat=0) or long (tableFormat=1) integers.
        The returned tableFormat should get packed into the flags field
        of the 'gvar' header.
        """
    def toXML(self, writer, ttFont: TTFont) -> None: ...
    version: Incomplete
    reserved: Incomplete
    def fromXML(self, name, attrs, content, ttFont: TTFont) -> None: ...
    @staticmethod
    def getNumPoints_(glyph): ...

def compileGlyph_(dataOffsetSize, variations, pointCount, axisTags, sharedCoordIndices, *, optimizeSize: bool = True): ...
def decompileGlyph_(dataOffsetSize, pointCount, sharedTuples, axisTags, data): ...
