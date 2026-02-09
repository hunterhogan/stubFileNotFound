from . import DefaultTable as DefaultTable
from _typeshed import Incomplete
from fontTools.misc.textTools import safeEval as safeEval
from fontTools.ttLib import TTFont

class table_C_O_L_R_(DefaultTable.DefaultTable):
    """Color table

    The ``COLR`` table defines color presentation of outline glyphs. It must
    be used in concert with the ``CPAL`` table, which contains the color
    descriptors used.

    This table is structured so that you can treat it like a dictionary keyed by glyph name.

    ``ttFont['COLR'][<glyphName>]`` will return the color layers for any glyph.

    ``ttFont['COLR'][<glyphName>] = <value>`` will set the color layers for any glyph.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/colr
    """

    @staticmethod
    def _decompileColorLayersV0(table): ...
    def _toOTTable(self, ttFont: TTFont): ...
    version: Incomplete
    ColorLayers: Incomplete
    table: Incomplete
    def decompile(self, data, ttFont: TTFont) -> None: ...
    def compile(self, ttFont: TTFont): ...
    def toXML(self, writer, ttFont: TTFont) -> None: ...
    def fromXML(self, name, attrs, content, ttFont: TTFont) -> None: ...
    def __getitem__(self, glyphName): ...
    def __setitem__(self, glyphName, value) -> None: ...
    def __delitem__(self, glyphName) -> None: ...

class LayerRecord:
    name: Incomplete
    colorID: Incomplete
    def __init__(self, name=None, colorID=None) -> None: ...
    def toXML(self, writer, ttFont: TTFont) -> None: ...
    def fromXML(self, eltname, attrs, content, ttFont: TTFont) -> None: ...
