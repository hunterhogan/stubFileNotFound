from . import _macosx as _macosx
from .backend_agg import FigureCanvasAgg as FigureCanvasAgg
from _typeshed import Incomplete
from matplotlib import _api as _api, cbook as cbook
from matplotlib._pylab_helpers import Gcf as Gcf
from matplotlib.backend_bases import FigureCanvasBase as FigureCanvasBase, FigureManagerBase as FigureManagerBase, NavigationToolbar2 as NavigationToolbar2, ResizeEvent as ResizeEvent, TimerBase as TimerBase, _Backend as _Backend, _allow_interrupt as _allow_interrupt

class TimerMac(_macosx.Timer, TimerBase):
    """Subclass of `.TimerBase` using CFRunLoop timer events."""

def _allow_interrupt_macos():
    """A context manager that allows terminating a plot by sending a SIGINT."""

class FigureCanvasMac(FigureCanvasAgg, _macosx.FigureCanvas, FigureCanvasBase):
    required_interactive_framework: str
    _timer_cls = TimerMac
    manager_class: Incomplete
    _draw_pending: bool
    _is_drawing: bool
    _timers: Incomplete
    def __init__(self, figure) -> None: ...
    def draw(self) -> None:
        """Render the figure and update the macosx canvas."""
    def draw_idle(self) -> None: ...
    def _single_shot_timer(self, callback) -> None:
        """Add a single shot timer with the given callback"""
    def _draw_idle(self) -> None:
        """
        Draw method for singleshot timer

        This draw method can be added to a singleshot timer, which can
        accumulate draws while the eventloop is spinning. This method will
        then only draw the first time and short-circuit the others.
        """
    def blit(self, bbox: Incomplete | None = None) -> None: ...
    def resize(self, width, height) -> None: ...
    def start_event_loop(self, timeout: int = 0) -> None: ...

class NavigationToolbar2Mac(_macosx.NavigationToolbar2, NavigationToolbar2):
    def __init__(self, canvas) -> None: ...
    def draw_rubberband(self, event, x0, y0, x1, y1) -> None: ...
    def remove_rubberband(self) -> None: ...
    def save_figure(self, *args): ...

class FigureManagerMac(_macosx.FigureManager, FigureManagerBase):
    _toolbar2_class = NavigationToolbar2Mac
    _shown: bool
    def __init__(self, canvas, num) -> None: ...
    def _close_button_pressed(self) -> None: ...
    def destroy(self) -> None: ...
    @classmethod
    def start_main_loop(cls) -> None: ...
    def show(self) -> None: ...

class _BackendMac(_Backend):
    FigureCanvas = FigureCanvasMac
    FigureManager = FigureManagerMac
    mainloop: Incomplete
