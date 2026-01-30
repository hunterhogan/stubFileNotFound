import fontTools.pens.basePen
from fontTools.pens.basePen import BasePen as BasePen, OpenContourError as OpenContourError
from typing import Any

COMPILED: bool
__test__: dict

class MomentsPen(fontTools.pens.basePen.BasePen):
    def __init__(self, glyphset=...) -> Any:
        """MomentsPen.__init__(self, glyphset=None)"""
    def _closePath(self) -> Any:
        """MomentsPen._closePath(self)"""
    def _curveToOne(self, p1, p2, p3) -> Any:
        """MomentsPen._curveToOne(self, p1, p2, p3)"""
    def _endPath(self) -> Any:
        """MomentsPen._endPath(self)"""
    def _lineTo(self, p1) -> Any:
        """MomentsPen._lineTo(self, p1)"""
    def _moveTo(self, p0) -> Any:
        """MomentsPen._moveTo(self, p0)"""
    def _qCurveToOne(self, p1, p2) -> Any:
        """MomentsPen._qCurveToOne(self, p1, p2)"""
