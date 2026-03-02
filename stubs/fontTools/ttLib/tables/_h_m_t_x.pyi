from . import DefaultTable as DefaultTable
from _typeshed import Incomplete
from collections.abc import Sequence
from fontTools import ttLib as ttLib
from fontTools.ttLib import TTFont

class table__h_m_t_x(DefaultTable.DefaultTable):
    """Horizontal Metrics table

    The ``hmtx`` table contains per-glyph metrics for the glyphs in a
    ``glyf``, ``CFF ``, or ``CFF2`` table, as needed for horizontal text
    layout.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/hmtx
    """

    headerTag: str
    advanceName: str
    sideBearingName: str
    numberOfMetricsName: str
    longMetricFormat: str
    metrics: Incomplete
    def decompile(self, data, ttFont: TTFont) -> None: ...
    def compile(self, ttFont: TTFont): ...
    def toXML(self, writer, ttFont: TTFont) -> None: ...
    def fromXML(self, name, attrs, content, ttFont: TTFont) -> None: ...
    def __delitem__(self, glyphName: str) -> None: ...
    def __getitem__(self, glyphName: str): ...
    def __setitem__(self, glyphName: str, advance_sb_pair: Sequence[int]) -> None: ...
