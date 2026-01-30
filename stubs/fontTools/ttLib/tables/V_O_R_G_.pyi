from . import DefaultTable as DefaultTable
from _typeshed import Incomplete
from fontTools.misc.textTools import bytesjoin as bytesjoin, safeEval as safeEval

class table_V_O_R_G_(DefaultTable.DefaultTable):
    """Vertical Origin table

    The ``VORG`` table contains the vertical origin of each glyph
    in a `CFF` or `CFF2` font.

    This table is structured so that you can treat it like a dictionary keyed by glyph name.

    ``ttFont['VORG'][<glyphName>]`` will return the vertical origin for any glyph.

    ``ttFont['VORG'][<glyphName>] = <value>`` will set the vertical origin for any glyph.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/vorg
    """
    getGlyphName: Incomplete
    VOriginRecords: Incomplete
    def decompile(self, data, ttFont) -> None: ...
    numVertOriginYMetrics: Incomplete
    def compile(self, ttFont): ...
    def toXML(self, writer, ttFont) -> None: ...
    def fromXML(self, name, attrs, content, ttFont) -> None: ...
    def __getitem__(self, glyphSelector): ...
    def __setitem__(self, glyphSelector, value) -> None: ...
    def __delitem__(self, glyphSelector) -> None: ...

class VOriginRecord:
    glyphName: Incomplete
    vOrigin: Incomplete
    def __init__(self, name=None, vOrigin=None) -> None: ...
    def toXML(self, writer, ttFont) -> None: ...
    def fromXML(self, name, attrs, content, ttFont) -> None: ...
