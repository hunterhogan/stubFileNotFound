from .backend_cairo import FigureCanvasCairo as FigureCanvasCairo
from .backend_gtk3 import FigureCanvasGTK3 as FigureCanvasGTK3, GLib as GLib, Gtk as Gtk, _BackendGTK3 as _BackendGTK3

class FigureCanvasGTK3Cairo(FigureCanvasCairo, FigureCanvasGTK3):
    _idle_draw_id: int
    def on_draw_event(self, widget, ctx) -> None: ...

class _BackendGTK3Cairo(_BackendGTK3):
    FigureCanvas = FigureCanvasGTK3Cairo
