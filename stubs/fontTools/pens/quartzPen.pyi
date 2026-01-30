from _typeshed import Incomplete
from fontTools.pens.basePen import BasePen

__all__ = ['QuartzPen']

class QuartzPen(BasePen):
    """A pen that creates a CGPath

    Parameters
    - path: an optional CGPath to add to
    - xform: an optional CGAffineTransform to apply to the path
    """
    path: Incomplete
    xform: Incomplete
    def __init__(self, glyphSet, path=None, xform=None) -> None: ...
    def _moveTo(self, pt) -> None: ...
    def _lineTo(self, pt) -> None: ...
    def _curveToOne(self, p1, p2, p3) -> None: ...
    def _qCurveToOne(self, p1, p2) -> None: ...
    def _closePath(self) -> None: ...
