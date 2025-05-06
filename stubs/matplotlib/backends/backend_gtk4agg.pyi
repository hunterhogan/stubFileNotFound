from . import backend_agg as backend_agg, backend_gtk4 as backend_gtk4
from .backend_gtk4 import GLib as GLib, Gtk as Gtk, _BackendGTK4 as _BackendGTK4

class FigureCanvasGTK4Agg(backend_agg.FigureCanvasAgg, backend_gtk4.FigureCanvasGTK4):
    _idle_draw_id: int
    def on_draw_event(self, widget, ctx): ...

class _BackendGTK4Agg(_BackendGTK4):
    FigureCanvas = FigureCanvasGTK4Agg
