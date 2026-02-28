from fontTools.ttLib import TTFont
from . import DefaultTable as DefaultTable
from _typeshed import Incomplete
from fontTools.misc import sstruct as sstruct
from fontTools.misc.textTools import safeEval as safeEval

VDMX_HeaderFmt: str
VDMX_RatRangeFmt: str
VDMX_GroupFmt: str
VDMX_vTableFmt: str

class table_V_D_M_X_(DefaultTable.DefaultTable):
    """Vertical Device Metrics table

    The ``VDMX`` table records changes to the vertical glyph minima
    and maxima that result from Truetype instructions.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/vdmx
    """
    ratRanges: Incomplete
    groups: Incomplete
    def decompile(self, data, ttFont: TTFont) -> None: ...
    def compile(self, ttFont: TTFont): ...
    def toXML(self, writer, ttFont: TTFont) -> None: ...
    version: Incomplete
    numRatios: int
    numRecs: int
    def fromXML(self, name, attrs, content, ttFont: TTFont) -> None: ...
