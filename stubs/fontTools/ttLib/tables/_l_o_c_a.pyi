from . import DefaultTable as DefaultTable
from _typeshed import Incomplete
from fontTools.ttLib import TTFont

log: Incomplete

class table__l_o_c_a(DefaultTable.DefaultTable):
    """Index to Location table

    The ``loca`` table stores the offsets in the ``glyf`` table that correspond
    to the descriptions of each glyph. The glyphs are references by Glyph ID.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/loca
    """

    dependencies: Incomplete
    locations: Incomplete
    def decompile(self, data, ttFont: TTFont) -> None: ...
    def compile(self, ttFont: TTFont): ...
    def set(self, locations) -> None: ...
    def toXML(self, writer, ttFont: TTFont) -> None: ...
    def __getitem__(self, index): ...
    def __len__(self) -> int: ...
