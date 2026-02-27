from afdko.otfautohint.glyphData import glyphData

log = ...
def removeOverlap(glyph) -> glyphData | None:
    """
    If the glyphData 'glyph' object has overlap, create a new glyphData
    object with the overlap removed and return it.  If it has no overlap
    return None
    """
