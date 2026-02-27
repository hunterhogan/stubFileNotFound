from . import _backend_tk as _backend_tk
from ._backend_tk import (
	_BackendTk as _BackendTk, FigureCanvasTk as FigureCanvasTk, FigureManagerTk as FigureManagerTk,
	NavigationToolbar2Tk as NavigationToolbar2Tk)
from .backend_agg import FigureCanvasAgg as FigureCanvasAgg
from _typeshed import Incomplete

class FigureCanvasTkAgg(FigureCanvasAgg, FigureCanvasTk):
    def draw(self) -> None: ...
    def blit(self, bbox: Incomplete | None = None) -> None: ...

class _BackendTkAgg(_BackendTk):
    FigureCanvas = FigureCanvasTkAgg
