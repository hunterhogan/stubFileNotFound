from _typeshed import Incomplete
from fontTools.pens.basePen import BasePen
from fontTools.pens.momentsPen import MomentsPen

__all__ = ['StatisticsPen', 'StatisticsControlPen']

class StatisticsBase:
    def __init__(self) -> None: ...
    area: int
    meanX: int
    meanY: int
    varianceX: int
    varianceY: int
    stddevX: int
    stddevY: int
    covariance: int
    correlation: int
    slant: int
    def _zero(self) -> None: ...
    def _update(self) -> None: ...

class StatisticsPen(StatisticsBase, MomentsPen):
    """Pen calculating area, center of mass, variance and
    standard-deviation, covariance and correlation, and slant,
    of glyph shapes.

    Note that if the glyph shape is self-intersecting, the values
    are not correct (but well-defined). Moreover, area will be
    negative if contour directions are clockwise."""
    def __init__(self, glyphset=None) -> None: ...
    def _closePath(self) -> None: ...
    meanX: Incomplete
    meanY: Incomplete
    varianceX: Incomplete
    varianceY: Incomplete
    covariance: Incomplete
    def _update(self) -> None: ...

class StatisticsControlPen(StatisticsBase, BasePen):
    """Pen calculating area, center of mass, variance and
    standard-deviation, covariance and correlation, and slant,
    of glyph shapes, using the control polygon only.

    Note that if the glyph shape is self-intersecting, the values
    are not correct (but well-defined). Moreover, area will be
    negative if contour directions are clockwise."""
    _nodes: Incomplete
    def __init__(self, glyphset=None) -> None: ...
    _startPoint: Incomplete
    def _moveTo(self, pt) -> None: ...
    def _lineTo(self, pt) -> None: ...
    def _qCurveToOne(self, pt1, pt2) -> None: ...
    def _curveToOne(self, pt1, pt2, pt3) -> None: ...
    def _closePath(self) -> None: ...
    def _endPath(self) -> None: ...
    area: Incomplete
    meanX: Incomplete
    meanY: Incomplete
    varianceX: Incomplete
    varianceY: Incomplete
    covariance: Incomplete
    def _update(self) -> None: ...
