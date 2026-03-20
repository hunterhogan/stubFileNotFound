from . import DefaultTable as DefaultTable
from _typeshed import Incomplete
from collections.abc import Mapping, Sequence
from fontTools.ttLib import TTFont

class table__h_m_t_x(DefaultTable.DefaultTable):
    """Horizontal Metrics table.

    The ``hmtx`` table contains per-glyph metrics for the glyphs in a ``glyf``, ``CFF ``, or ``CFF2`` table, as needed for
    horizontal text layout.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/hmtx
    """

    headerTag: str
    advanceName: str
    sideBearingName: str
    numberOfMetricsName: str
    longMetricFormat: str
    metrics: dict[str, tuple[int, int]]
    def decompile(self, data: bytes, ttFont: TTFont) -> None: ...
    def compile(self, ttFont: TTFont) -> bytes: ...
    def toXML(self, writer: Incomplete, ttFont: TTFont) -> None: ...  # pyright: ignore[reportIncompatibleMethodOverride] # ty:ignore[invalid-method-override]
    def fromXML(self, name: str, attrs: Mapping[str, str], content: Incomplete, ttFont: TTFont) -> None: ...
    def __delitem__(self, glyphName: str) -> None: ...
    def __getitem__(self, glyphName: str) -> tuple[int, int]: ...
    def __setitem__(self, glyphName: str, advance_sb_pair: Sequence[int]) -> None: ...
