from _typeshed import Incomplete
from fontTools.pens.basePen import BasePen
from ufo2ft.filters import BaseFilter

logger: Incomplete

def _pointIsLeftOfLine(line, aPoint): ...

class EraseOpenCornersPen(BasePen):
    segments: Incomplete
    is_closed: bool
    affected: bool
    outpen: Incomplete
    def __init__(self, outpen) -> None: ...
    def _moveTo(self, p1) -> None: ...
    def _operate(self, *points) -> None: ...
    _qCurveTo = _operate
    _curveTo = _operate
    _lineTo = _operate
    _qCurveToOne = _operate
    _curveToOne = _operate
    def closePath(self) -> None: ...
    def endPath(self) -> None: ...

class EraseOpenCornersFilter(BaseFilter):
    def filter(self, glyph): ...
