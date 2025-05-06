import wx
from _typeshed import Incomplete
from matplotlib import _api as _api, _c_internal_utils as _c_internal_utils, backend_tools as backend_tools, cbook as cbook
from matplotlib._pylab_helpers import Gcf as Gcf
from matplotlib.backend_bases import CloseEvent as CloseEvent, FigureCanvasBase as FigureCanvasBase, FigureManagerBase as FigureManagerBase, GraphicsContextBase as GraphicsContextBase, KeyEvent as KeyEvent, LocationEvent as LocationEvent, MouseButton as MouseButton, MouseEvent as MouseEvent, NavigationToolbar2 as NavigationToolbar2, RendererBase as RendererBase, ResizeEvent as ResizeEvent, TimerBase as TimerBase, ToolContainerBase as ToolContainerBase, _Backend as _Backend, cursors as cursors
from matplotlib.path import Path as Path
from matplotlib.transforms import Affine2D as Affine2D

_log: Incomplete
PIXELS_PER_INCH: int

def _create_wxapp(): ...

class TimerWx(TimerBase):
    """Subclass of `.TimerBase` using wx.Timer events."""
    _timer: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def _timer_start(self) -> None: ...
    def _timer_stop(self) -> None: ...
    def _timer_set_interval(self) -> None: ...

class RendererWx(RendererBase):
    """
    The renderer handles all the drawing primitives using a graphics
    context instance that controls the colors/styles. It acts as the
    'renderer' instance used by many classes in the hierarchy.
    """
    fontweights: Incomplete
    fontangles: Incomplete
    fontnames: Incomplete
    width: Incomplete
    height: Incomplete
    bitmap: Incomplete
    fontd: Incomplete
    dpi: Incomplete
    gc: Incomplete
    def __init__(self, bitmap, dpi) -> None:
        """Initialise a wxWindows renderer instance."""
    def flipy(self): ...
    def get_text_width_height_descent(self, s, prop, ismath): ...
    def get_canvas_width_height(self): ...
    def handle_clip_rectangle(self, gc) -> None: ...
    @staticmethod
    def convert_path(gfx_ctx, path, transform): ...
    def draw_path(self, gc, path, transform, rgbFace: Incomplete | None = None) -> None: ...
    def draw_image(self, gc, x, y, im) -> None: ...
    def draw_text(self, gc, x, y, s, prop, angle, ismath: bool = False, mtext: Incomplete | None = None) -> None: ...
    def new_gc(self): ...
    def get_wx_font(self, s, prop):
        """Return a wx font.  Cache font instances for efficiency."""
    def points_to_pixels(self, points): ...

class GraphicsContextWx(GraphicsContextBase):
    """
    The graphics context provides the color, line styles, etc.

    This class stores a reference to a wxMemoryDC, and a
    wxGraphicsContext that draws to it.  Creating a wxGraphicsContext
    seems to be fairly heavy, so these objects are cached based on the
    bitmap object that is passed in.

    The base GraphicsContext stores colors as an RGB tuple on the unit
    interval, e.g., (0.5, 0.0, 1.0).  wxPython uses an int interval, but
    since wxPython colour management is rather simple, I have not chosen
    to implement a separate colour manager class.
    """
    _capd: Incomplete
    _joind: Incomplete
    _cache: Incomplete
    bitmap: Incomplete
    dc: Incomplete
    gfx_ctx: Incomplete
    _pen: Incomplete
    renderer: Incomplete
    def __init__(self, bitmap, renderer) -> None: ...
    IsSelected: bool
    def select(self) -> None:
        """Select the current bitmap into this wxDC instance."""
    def unselect(self) -> None:
        """Select a Null bitmap into this wxDC instance."""
    def set_foreground(self, fg, isRGBA: Incomplete | None = None) -> None: ...
    def set_linewidth(self, w) -> None: ...
    def set_capstyle(self, cs) -> None: ...
    def set_joinstyle(self, js) -> None: ...
    def get_wxcolour(self, color):
        """Convert an RGB(A) color to a wx.Colour."""

class _FigureCanvasWxBase(FigureCanvasBase, wx.Panel):
    """
    The FigureCanvas contains the figure and does event handling.

    In the wxPython backend, it is derived from wxPanel, and (usually) lives
    inside a frame instantiated by a FigureManagerWx. The parent window
    probably implements a wx.Sizer to control the displayed control size - but
    we give a hint as to our preferred minimum size.
    """
    required_interactive_framework: str
    _timer_cls = TimerWx
    manager_class: Incomplete
    keyvald: Incomplete
    bitmap: Incomplete
    _isDrawn: bool
    _rubberband_rect: Incomplete
    _rubberband_pen_black: Incomplete
    _rubberband_pen_white: Incomplete
    def __init__(self, parent, id, figure: Incomplete | None = None) -> None:
        """
        Initialize a FigureWx instance.

        - Initialize the FigureCanvasBase and wxPanel parents.
        - Set event handlers for resize, paint, and keyboard and mouse
          interaction.
        """
    def Copy_to_Clipboard(self, event: Incomplete | None = None) -> None:
        """Copy bitmap of canvas to system clipboard."""
    def _update_device_pixel_ratio(self, *args, **kwargs) -> None: ...
    def draw_idle(self) -> None: ...
    def flush_events(self) -> None: ...
    _event_loop: Incomplete
    def start_event_loop(self, timeout: int = 0) -> None: ...
    def stop_event_loop(self, event: Incomplete | None = None) -> None: ...
    def _get_imagesave_wildcards(self):
        """Return the wildcard string for the filesave dialog."""
    def gui_repaint(self, drawDC: Incomplete | None = None) -> None:
        """
        Update the displayed image on the GUI canvas, using the supplied
        wx.PaintDC device context.
        """
    filetypes: Incomplete
    def _on_paint(self, event) -> None:
        """Called when wxPaintEvt is generated."""
    def _on_size(self, event) -> None:
        """
        Called when wxEventSize is generated.

        In this application we attempt to resize to fit the window, so it
        is better to take the performance hit and redraw the whole window.
        """
    @staticmethod
    def _mpl_buttons(): ...
    @staticmethod
    def _mpl_modifiers(event: Incomplete | None = None, *, exclude: Incomplete | None = None): ...
    def _get_key(self, event): ...
    def _mpl_coords(self, pos: Incomplete | None = None):
        """
        Convert a wx position, defaulting to the current cursor position, to
        Matplotlib coordinates.
        """
    def _on_key_down(self, event) -> None:
        """Capture key press."""
    def _on_key_up(self, event) -> None:
        """Release key."""
    def set_cursor(self, cursor) -> None: ...
    def _set_capture(self, capture: bool = True) -> None:
        """Control wx mouse capture."""
    def _on_capture_lost(self, event) -> None:
        """Capture changed or lost"""
    def _on_mouse_button(self, event) -> None:
        """Start measuring on an axis."""
    _skipwheelevent: bool
    def _on_mouse_wheel(self, event) -> None:
        """Translate mouse wheel events into matplotlib events"""
    def _on_motion(self, event) -> None:
        """Start measuring on an axis."""
    def _on_enter(self, event) -> None:
        """Mouse has entered the window."""
    def _on_leave(self, event) -> None:
        """Mouse has left the window."""

class FigureCanvasWx(_FigureCanvasWxBase):
    renderer: Incomplete
    _isDrawn: bool
    def draw(self, drawDC: Incomplete | None = None) -> None:
        """
        Render the figure using RendererWx instance renderer, or using a
        previously defined renderer if none is specified.
        """
    def _print_image(self, filetype, filename) -> None: ...
    print_bmp: Incomplete
    print_jpeg: Incomplete
    print_jpg: Incomplete
    print_pcx: Incomplete
    print_png: Incomplete
    print_tiff: Incomplete
    print_tif: Incomplete
    print_xpm: Incomplete

class FigureFrameWx(wx.Frame):
    canvas: Incomplete
    def __init__(self, num, fig, *, canvas_class) -> None: ...
    def _on_close(self, event) -> None: ...

class FigureManagerWx(FigureManagerBase):
    """
    Container/controller for the FigureCanvas and GUI frame.

    It is instantiated by Gcf whenever a new figure is created.  Gcf is
    responsible for managing multiple instances of FigureManagerWx.

    Attributes
    ----------
    canvas : `FigureCanvas`
        a FigureCanvasWx(wx.Panel) instance
    window : wxFrame
        a wxFrame instance - wxpython.org/Phoenix/docs/html/Frame.html
    """
    frame: Incomplete
    def __init__(self, canvas, num, frame) -> None: ...
    @classmethod
    def create_with_canvas(cls, canvas_class, figure, num): ...
    @classmethod
    def start_main_loop(cls) -> None: ...
    def show(self) -> None: ...
    def destroy(self, *args) -> None: ...
    def full_screen_toggle(self) -> None: ...
    def get_window_title(self): ...
    def set_window_title(self, title) -> None: ...
    def resize(self, width, height) -> None: ...

def _load_bitmap(filename):
    '''
    Load a wx.Bitmap from a file in the "images" directory of the Matplotlib
    data.
    '''
def _set_frame_icon(frame) -> None: ...

class NavigationToolbar2Wx(NavigationToolbar2, wx.ToolBar):
    wx_ids: Incomplete
    _coordinates: Incomplete
    _label_text: Incomplete
    def __init__(self, canvas, coordinates: bool = True, *, style=...) -> None: ...
    @staticmethod
    def _icon(name):
        '''
        Construct a `wx.Bitmap` suitable for use as icon from an image file
        *name*, including the extension and relative to Matplotlib\'s "images"
        data directory.
        '''
    def _update_buttons_checked(self) -> None: ...
    def zoom(self, *args) -> None: ...
    def pan(self, *args) -> None: ...
    def save_figure(self, *args): ...
    def draw_rubberband(self, event, x0, y0, x1, y1) -> None: ...
    def remove_rubberband(self) -> None: ...
    def set_message(self, s) -> None: ...
    def set_history_buttons(self) -> None: ...

class ToolbarWx(ToolContainerBase, wx.ToolBar):
    _icon_extension: str
    _space: Incomplete
    _label_text: Incomplete
    _toolitems: Incomplete
    _groups: Incomplete
    def __init__(self, toolmanager, parent: Incomplete | None = None, style=...) -> None: ...
    def _get_tool_pos(self, tool):
        """
        Find the position (index) of a wx.ToolBarToolBase in a ToolBar.

        ``ToolBar.GetToolPos`` is not useful because wx assigns the same Id to
        all Separators and StretchableSpaces.
        """
    def add_toolitem(self, name, group, position, image_file, description, toggle) -> None: ...
    def toggle_toolitem(self, name, toggled) -> None: ...
    def remove_toolitem(self, name) -> None: ...
    def set_message(self, s) -> None: ...

class ConfigureSubplotsWx(backend_tools.ConfigureSubplotsBase):
    def trigger(self, *args) -> None: ...

class SaveFigureWx(backend_tools.SaveFigureBase):
    def trigger(self, *args) -> None: ...

class RubberbandWx(backend_tools.RubberbandBase):
    def draw_rubberband(self, x0, y0, x1, y1) -> None: ...
    def remove_rubberband(self) -> None: ...

class _HelpDialog(wx.Dialog):
    _instance: Incomplete
    headers: Incomplete
    widths: Incomplete
    def __init__(self, parent, help_entries) -> None: ...
    def _on_close(self, event) -> None: ...
    @classmethod
    def show(cls, parent, help_entries) -> None: ...

class HelpWx(backend_tools.ToolHelpBase):
    def trigger(self, *args) -> None: ...

class ToolCopyToClipboardWx(backend_tools.ToolCopyToClipboardBase):
    def trigger(self, *args, **kwargs) -> None: ...

class _BackendWx(_Backend):
    FigureCanvas = FigureCanvasWx
    FigureManager = FigureManagerWx
    mainloop: Incomplete
