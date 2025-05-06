from . import _backend_tk as _backend_tk
from ._backend_tk import FigureCanvasTk as FigureCanvasTk, _BackendTk as _BackendTk
from .backend_cairo import FigureCanvasCairo as FigureCanvasCairo, cairo as cairo

class FigureCanvasTkCairo(FigureCanvasCairo, FigureCanvasTk):
    def draw(self) -> None: ...

class _BackendTkCairo(_BackendTk):
    FigureCanvas = FigureCanvasTkCairo
