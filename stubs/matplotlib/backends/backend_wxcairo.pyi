from .backend_cairo import FigureCanvasCairo as FigureCanvasCairo, cairo as cairo
from .backend_wx import _BackendWx as _BackendWx, _FigureCanvasWxBase as _FigureCanvasWxBase
from _typeshed import Incomplete

class FigureCanvasWxCairo(FigureCanvasCairo, _FigureCanvasWxBase):
    bitmap: Incomplete
    _isDrawn: bool
    def draw(self, drawDC: Incomplete | None = None) -> None: ...

class _BackendWxCairo(_BackendWx):
    FigureCanvas = FigureCanvasWxCairo
