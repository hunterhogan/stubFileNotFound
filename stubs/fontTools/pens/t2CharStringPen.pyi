from _typeshed import Incomplete
from fontTools.misc.psCharStrings import T2CharString as T2CharString
from fontTools.pens.basePen import BasePen as BasePen
from fontTools.ttLib.ttGlyphSet import _TTGlyphSet

class T2CharStringPen(BasePen):
    """Pen to draw Type 2 CharStrings.

    The 'roundTolerance' argument controls the rounding of point coordinates.
    It is defined as the maximum absolute difference between the original
    float and the rounded integer value.
    The default tolerance of 0.5 means that all floats are rounded to integer;
    a value of 0 disables rounding; values in between will only round floats
    which are close to their integral part within the tolerated range.
    """

    round: float
    def __init__(self, width: float | None, glyphSet: _TTGlyphSet | None, roundTolerance: float = 0.5, CFF2: bool = False) -> None: ...
    def getCharString(self, private: dict[Incomplete, Incomplete] | None = None, globalSubrs: list[Incomplete] | None = None, optimize: bool = True) -> T2CharString: ...
