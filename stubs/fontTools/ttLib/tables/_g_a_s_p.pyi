from . import DefaultTable as DefaultTable
from _typeshed import Incomplete
from fontTools.misc.textTools import safeEval as safeEval

GASP_SYMMETRIC_GRIDFIT: int
GASP_SYMMETRIC_SMOOTHING: int
GASP_DOGRAY: int
GASP_GRIDFIT: int

class table__g_a_s_p(DefaultTable.DefaultTable):
    """Grid-fitting and Scan-conversion Procedure table

    The ``gasp`` table defines the preferred rasterization settings for
    the font when rendered on monochrome and greyscale output devices.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/gasp
    """
    gaspRange: Incomplete
    def decompile(self, data, ttFont) -> None: ...
    def compile(self, ttFont): ...
    def toXML(self, writer, ttFont) -> None: ...
    def fromXML(self, name, attrs, content, ttFont) -> None: ...
