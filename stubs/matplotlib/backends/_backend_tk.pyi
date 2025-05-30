import tkinter as tk
from . import _tkagg as _tkagg
from ._tkagg import TK_PHOTO_COMPOSITE_OVERLAY as TK_PHOTO_COMPOSITE_OVERLAY, TK_PHOTO_COMPOSITE_SET as TK_PHOTO_COMPOSITE_SET
from _typeshed import Incomplete
from collections.abc import Generator
from matplotlib import _api as _api, _c_internal_utils as _c_internal_utils, backend_tools as backend_tools, cbook as cbook
from matplotlib._pylab_helpers import Gcf as Gcf
from matplotlib.backend_bases import CloseEvent as CloseEvent, FigureCanvasBase as FigureCanvasBase, FigureManagerBase as FigureManagerBase, KeyEvent as KeyEvent, LocationEvent as LocationEvent, MouseButton as MouseButton, MouseEvent as MouseEvent, NavigationToolbar2 as NavigationToolbar2, ResizeEvent as ResizeEvent, TimerBase as TimerBase, ToolContainerBase as ToolContainerBase, _Backend as _Backend, _Mode as _Mode, cursors as cursors

_log: Incomplete
cursord: Incomplete

def _restore_foreground_window_at_end() -> Generator[None]: ...

_blit_args: Incomplete
_blit_tcl_name: Incomplete

def _blit(argsid) -> None:
    """
    Thin wrapper to blit called via tkapp.call.

    *argsid* is a unique string identifier to fetch the correct arguments from
    the ``_blit_args`` dict, since arguments cannot be passed directly.
    """
def blit(photoimage, aggimage, offsets, bbox: Incomplete | None = None) -> None:
    """
    Blit *aggimage* to *photoimage*.

    *offsets* is a tuple describing how to fill the ``offset`` field of the
    ``Tk_PhotoImageBlock`` struct: it should be (0, 1, 2, 3) for RGBA8888 data,
    (2, 1, 0, 3) for little-endian ARBG32 (i.e. GBRA8888) data and (1, 2, 3, 0)
    for big-endian ARGB32 (i.e. ARGB8888) data.

    If *bbox* is passed, it defines the region that gets blitted. That region
    will be composed with the previous data according to the alpha channel.
    Blitting will be clipped to pixels inside the canvas, including silently
    doing nothing if the *bbox* region is entirely outside the canvas.

    Tcl events must be dispatched to trigger a blit from a non-Tcl thread.
    """

class TimerTk(TimerBase):
    """Subclass of `backend_bases.TimerBase` using Tk timer events."""
    _timer: Incomplete
    parent: Incomplete
    def __init__(self, parent, *args, **kwargs) -> None: ...
    def _timer_start(self) -> None: ...
    def _timer_stop(self) -> None: ...
    def _on_timer(self): ...

class FigureCanvasTk(FigureCanvasBase):
    required_interactive_framework: str
    manager_class: Incomplete
    _idle_draw_id: Incomplete
    _event_loop_id: Incomplete
    _tkcanvas: Incomplete
    _tkphoto: Incomplete
    _tkcanvas_image_region: Incomplete
    _rubberband_rect_black: Incomplete
    _rubberband_rect_white: Incomplete
    def __init__(self, figure: Incomplete | None = None, master: Incomplete | None = None) -> None: ...
    def _update_device_pixel_ratio(self, event: Incomplete | None = None) -> None: ...
    def resize(self, event) -> None: ...
    def draw_idle(self) -> None: ...
    def get_tk_widget(self):
        """
        Return the Tk widget used to implement FigureCanvasTkAgg.

        Although the initial implementation uses a Tk canvas,  this routine
        is intended to hide that fact.
        """
    def _event_mpl_coords(self, event): ...
    def motion_notify_event(self, event) -> None: ...
    def enter_notify_event(self, event) -> None: ...
    def leave_notify_event(self, event) -> None: ...
    def button_press_event(self, event, dblclick: bool = False) -> None: ...
    def button_dblclick_event(self, event) -> None: ...
    def button_release_event(self, event) -> None: ...
    def scroll_event(self, event) -> None: ...
    def scroll_event_windows(self, event) -> None:
        """MouseWheel event processor"""
    @staticmethod
    def _mpl_buttons(event): ...
    @staticmethod
    def _mpl_modifiers(event, *, exclude: Incomplete | None = None): ...
    def _get_key(self, event): ...
    def key_press(self, event) -> None: ...
    def key_release(self, event) -> None: ...
    def new_timer(self, *args, **kwargs): ...
    def flush_events(self) -> None: ...
    def start_event_loop(self, timeout: int = 0) -> None: ...
    def stop_event_loop(self) -> None: ...
    def set_cursor(self, cursor) -> None: ...

class FigureManagerTk(FigureManagerBase):
    """
    Attributes
    ----------
    canvas : `FigureCanvas`
        The FigureCanvas instance
    num : int or str
        The Figure number
    toolbar : tk.Toolbar
        The tk.Toolbar
    window : tk.Window
        The tk.Window
    """
    _owns_mainloop: bool
    window: Incomplete
    _window_dpi: Incomplete
    _window_dpi_cbname: str
    _shown: bool
    def __init__(self, canvas, num, window) -> None: ...
    @classmethod
    def create_with_canvas(cls, canvas_class, figure, num): ...
    @classmethod
    def start_main_loop(cls) -> None: ...
    def _update_window_dpi(self, *args) -> None: ...
    def resize(self, width, height) -> None: ...
    def show(self) -> None: ...
    def destroy(self, *args) -> None: ...
    def get_window_title(self): ...
    def set_window_title(self, title) -> None: ...
    def full_screen_toggle(self) -> None: ...

class NavigationToolbar2Tk(NavigationToolbar2, tk.Frame):
    _buttons: Incomplete
    _label_font: Incomplete
    message: Incomplete
    _message_label: Incomplete
    def __init__(self, canvas, window: Incomplete | None = None, *, pack_toolbar: bool = True) -> None:
        '''
        Parameters
        ----------
        canvas : `FigureCanvas`
            The figure canvas on which to operate.
        window : tk.Window
            The tk.Window which owns this toolbar.
        pack_toolbar : bool, default: True
            If True, add the toolbar to the parent\'s pack manager\'s packing
            list during initialization with ``side="bottom"`` and ``fill="x"``.
            If you want to use the toolbar with a different layout manager, use
            ``pack_toolbar=False``.
        '''
    def _rescale(self) -> None:
        """
        Scale all children of the toolbar to current DPI setting.

        Before this is called, the Tk scaling setting will have been updated to
        match the new DPI. Tk widgets do not update for changes to scaling, but
        all measurements made after the change will match the new scaling. Thus
        this function re-applies all the same sizes in points, which Tk will
        scale correctly to pixels.
        """
    def _update_buttons_checked(self) -> None: ...
    def pan(self, *args) -> None: ...
    def zoom(self, *args) -> None: ...
    def set_message(self, s) -> None: ...
    def draw_rubberband(self, event, x0, y0, x1, y1) -> None: ...
    def remove_rubberband(self) -> None: ...
    def _set_image_for_button(self, button):
        """
        Set the image for a button based on its pixel size.

        The pixel size is determined by the DPI scaling of the window.
        """
    def _Button(self, text, image_file, toggle, command): ...
    def _Spacer(self): ...
    def save_figure(self, *args): ...
    def set_history_buttons(self) -> None: ...

def add_tooltip(widget, text) -> None: ...

class RubberbandTk(backend_tools.RubberbandBase):
    def draw_rubberband(self, x0, y0, x1, y1) -> None: ...
    def remove_rubberband(self) -> None: ...

class ToolbarTk(ToolContainerBase, tk.Frame):
    _label_font: Incomplete
    _message: Incomplete
    _message_label: Incomplete
    _toolitems: Incomplete
    _groups: Incomplete
    def __init__(self, toolmanager, window: Incomplete | None = None) -> None: ...
    def _rescale(self): ...
    def add_toolitem(self, name, group, position, image_file, description, toggle): ...
    def _get_groupframe(self, group): ...
    def _add_separator(self): ...
    def _button_click(self, name) -> None: ...
    def toggle_toolitem(self, name, toggled) -> None: ...
    def remove_toolitem(self, name) -> None: ...
    def set_message(self, s) -> None: ...

class SaveFigureTk(backend_tools.SaveFigureBase):
    def trigger(self, *args) -> None: ...

class ConfigureSubplotsTk(backend_tools.ConfigureSubplotsBase):
    def trigger(self, *args) -> None: ...

class HelpTk(backend_tools.ToolHelpBase):
    def trigger(self, *args): ...
Toolbar = ToolbarTk

class _BackendTk(_Backend):
    backend_version: Incomplete
    FigureCanvas = FigureCanvasTk
    FigureManager = FigureManagerTk
    mainloop: Incomplete
