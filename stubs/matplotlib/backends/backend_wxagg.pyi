from .backend_agg import FigureCanvasAgg as FigureCanvasAgg
from .backend_wx import _BackendWx as _BackendWx, _FigureCanvasWxBase as _FigureCanvasWxBase
from _typeshed import Incomplete

class FigureCanvasWxAgg(FigureCanvasAgg, _FigureCanvasWxBase):
    bitmap: Incomplete
    _isDrawn: bool
    def draw(self, drawDC: Incomplete | None = None) -> None:
        """
        Render the figure using agg.
        """
    def blit(self, bbox: Incomplete | None = None) -> None: ...
    def _create_bitmap(self):
        """Create a wx.Bitmap from the renderer RGBA buffer"""

class _BackendWxAgg(_BackendWx):
    FigureCanvas = FigureCanvasWxAgg
