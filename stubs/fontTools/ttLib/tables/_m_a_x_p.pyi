from . import DefaultTable as DefaultTable
from _typeshed import Incomplete
from fontTools.misc import sstruct as sstruct
from fontTools.misc.textTools import safeEval as safeEval
from typing import Any

maxpFormat_0_5: str
maxpFormat_1_0_add: str

class table__m_a_x_p(DefaultTable.DefaultTable):
    """Maximum Profile table

    The ``maxp`` table contains the memory requirements for the data in
    the font.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/maxp
    """
    dependencies: Incomplete
    numGlyphs: int
    def decompile(self, data, ttFont: Any) -> None: ...
    tableVersion: int
    def compile(self, ttFont: Any) -> bytes: ...
    maxZones: int
    maxTwilightPoints: int
    maxStorage: int
    maxFunctionDefs: int
    maxInstructionDefs: int
    maxStackElements: int
    maxSizeOfInstructions: int
    maxPoints: Incomplete
    maxContours: Incomplete
    maxCompositePoints: Incomplete
    maxCompositeContours: Incomplete
    maxComponentElements: int
    maxComponentDepth: Incomplete
    def recalc(self, ttFont: Any) -> None:
        """Recalculate the font bounding box, and most other maxp values except
        for the TT instructions values. Also recalculate the value of bit 1
        of the flags field and the font bounding box of the 'head' table.
        """
    def testrepr(self) -> None: ...
    def toXML(self, writer, ttFont) -> None: ...
    def fromXML(self, name, attrs, content, ttFont) -> None: ...
