from .glyphData import glyphData as glyphData
from _typeshed import Incomplete

log: Incomplete

def removeOverlap(glyph):
    """
    If the glyphData 'glyph' object has overlap, create a new glyphData
    object with the overlap removed and return it.  If it has no overlap
    return None
    """
