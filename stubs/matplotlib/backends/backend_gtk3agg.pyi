from . import backend_agg as backend_agg, backend_gtk3 as backend_gtk3
from .. import cbook as cbook, transforms as transforms
from .backend_gtk3 import GLib as GLib, Gtk as Gtk, _BackendGTK3 as _BackendGTK3
from _typeshed import Incomplete

class FigureCanvasGTK3Agg(backend_agg.FigureCanvasAgg, backend_gtk3.FigureCanvasGTK3):
    _bbox_queue: Incomplete
    def __init__(self, figure) -> None: ...
    _idle_draw_id: int
    def on_draw_event(self, widget, ctx): ...
    def blit(self, bbox: Incomplete | None = None) -> None: ...

class _BackendGTK3Agg(_BackendGTK3):
    FigureCanvas = FigureCanvasGTK3Agg
