from _typeshed import Incomplete
from fontTools.merge.unicode import is_Default_Ignorable as is_Default_Ignorable
from fontTools.pens.recordingPen import DecomposingRecordingPen as DecomposingRecordingPen

def computeMegaGlyphOrder(merger, glyphOrders) -> None:
    """Modifies passed-in glyphOrders to reflect new glyph names.
    
    Stores merger.glyphOrder.
    """
def computeMegaUvs(merger, uvsTables):
    """Returns merged UVS subtable (cmap format=14)."""

class _CmapUnicodePlatEncodings:
    BMP: Incomplete
    FullRepertoire: Incomplete
    UVS: Incomplete

def computeMegaCmap(merger, cmapTables) -> None:
    """Sets merger.cmap and merger.uvsDict."""
def renameCFFCharStrings(merger, glyphOrder, cffTable) -> None:
    """Rename topDictIndex charStrings based on glyphOrder."""
