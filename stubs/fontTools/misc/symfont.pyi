from _typeshed import Incomplete
from fontTools.pens.basePen import BasePen as BasePen

n: int
t: Incomplete
x: Incomplete
y: Incomplete
c: Incomplete
X: Incomplete
Y: Incomplete
P: Incomplete
C: Incomplete
BinomialCoefficient: Incomplete
last: Incomplete
this: Incomplete
BernsteinPolynomial: Incomplete
BezierCurve: Incomplete
BezierCurveC: Incomplete

def green(f, curveXY): ...

class _BezierFuncsLazy(dict):
    _symfunc: Incomplete
    _bezfuncs: Incomplete
    def __init__(self, symfunc) -> None: ...
    def __missing__(self, i): ...

class GreenPen(BasePen):
    _BezierFuncs: Incomplete
    @classmethod
    def _getGreenBezierFuncs(celf, func): ...
    _funcs: Incomplete
    value: int
    def __init__(self, func, glyphset=None) -> None: ...
    _startPoint: Incomplete
    def _moveTo(self, p0) -> None: ...
    def _closePath(self) -> None: ...
    def _endPath(self) -> None: ...
    def _lineTo(self, p1) -> None: ...
    def _qCurveToOne(self, p1, p2) -> None: ...
    def _curveToOne(self, p1, p2, p3) -> None: ...

AreaPen: Incomplete
MomentXPen: Incomplete
MomentYPen: Incomplete
MomentXXPen: Incomplete
MomentYYPen: Incomplete
MomentXYPen: Incomplete

def printGreenPen(penName, funcs, file=..., docstring=None) -> None: ...
