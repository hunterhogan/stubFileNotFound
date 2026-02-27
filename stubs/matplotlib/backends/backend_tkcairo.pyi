from . import _backend_tk as _backend_tk
from ._backend_tk import _BackendTk as _BackendTk, FigureCanvasTk as FigureCanvasTk
from .backend_cairo import cairo as cairo, FigureCanvasCairo as FigureCanvasCairo

class FigureCanvasTkCairo(FigureCanvasCairo, FigureCanvasTk):
    def draw(self) -> None: ...

class _BackendTkCairo(_BackendTk):
    FigureCanvas = FigureCanvasTkCairo
