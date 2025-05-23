from _typeshed import Incomplete
from matplotlib import _api as _api, backend_tools as backend_tools, cbook as cbook
from matplotlib._pylab_helpers import Gcf as Gcf
from matplotlib.backend_bases import FigureCanvasBase as FigureCanvasBase, FigureManagerBase as FigureManagerBase, NavigationToolbar2 as NavigationToolbar2, TimerBase as TimerBase, _Backend as _Backend
from matplotlib.backend_tools import Cursors as Cursors

_log: Incomplete
_application: Incomplete

def _shutdown_application(app) -> None: ...
def _create_application(): ...
def mpl_to_gtk_cursor_name(mpl_cursor): ...

class TimerGTK(TimerBase):
    """Subclass of `.TimerBase` using GTK timer events."""
    _timer: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def _timer_start(self) -> None: ...
    def _timer_stop(self) -> None: ...
    def _timer_set_interval(self) -> None: ...
    def _on_timer(self): ...

class _FigureCanvasGTK(FigureCanvasBase):
    _timer_cls = TimerGTK

class _FigureManagerGTK(FigureManagerBase):
    """
    Attributes
    ----------
    canvas : `FigureCanvas`
        The FigureCanvas instance
    num : int or str
        The Figure number
    toolbar : Gtk.Toolbar or Gtk.Box
        The toolbar
    vbox : Gtk.VBox
        The Gtk.VBox containing the canvas and toolbar
    window : Gtk.Window
        The Gtk.Window
    """
    _gtk_ver: Incomplete
    window: Incomplete
    vbox: Incomplete
    _destroying: bool
    def __init__(self, canvas, num) -> None: ...
    def destroy(self, *args) -> None: ...
    @classmethod
    def start_main_loop(cls) -> None: ...
    def show(self) -> None: ...
    def full_screen_toggle(self): ...
    def get_window_title(self): ...
    def set_window_title(self, title) -> None: ...
    def resize(self, width, height) -> None: ...

class _NavigationToolbar2GTK(NavigationToolbar2):
    def set_message(self, s) -> None: ...
    def draw_rubberband(self, event, x0, y0, x1, y1) -> None: ...
    def remove_rubberband(self) -> None: ...
    def _update_buttons_checked(self) -> None: ...
    def pan(self, *args) -> None: ...
    def zoom(self, *args) -> None: ...
    def set_history_buttons(self) -> None: ...

class RubberbandGTK(backend_tools.RubberbandBase):
    def draw_rubberband(self, x0, y0, x1, y1) -> None: ...
    def remove_rubberband(self) -> None: ...

class ConfigureSubplotsGTK(backend_tools.ConfigureSubplotsBase):
    def trigger(self, *args) -> None: ...

class _BackendGTK(_Backend):
    backend_version: Incomplete
    mainloop: Incomplete
