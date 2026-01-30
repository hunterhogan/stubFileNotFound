from _typeshed import Incomplete
from fontTools import ttLib as ttLib

superclass: Incomplete

class table__v_m_t_x(superclass):
    """Vertical Metrics table

    The ``vmtx`` table contains per-glyph metrics for the glyphs in a
    ``glyf``, ``CFF ``, or ``CFF2`` table, as needed for vertical text
    layout.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/vmtx
    """
    headerTag: str
    advanceName: str
    sideBearingName: str
    numberOfMetricsName: str
