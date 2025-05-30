from _typeshed import Incomplete
from matplotlib import _api as _api, backend_bases as backend_bases, backend_tools as backend_tools
from matplotlib.backend_bases import KeyEvent as KeyEvent, LocationEvent as LocationEvent, MouseButton as MouseButton, MouseEvent as MouseEvent, ResizeEvent as ResizeEvent, _Backend as _Backend
from matplotlib.backends import backend_agg as backend_agg

_log: Incomplete
_SPECIAL_KEYS_LUT: Incomplete

def _handle_key(key):
    """Handle key values"""

class TimerTornado(backend_bases.TimerBase):
    _timer: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def _timer_start(self) -> None: ...
    def _timer_stop(self) -> None: ...
    def _timer_set_interval(self) -> None: ...

class TimerAsyncio(backend_bases.TimerBase):
    _task: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    async def _timer_task(self, interval) -> None: ...
    def _timer_start(self) -> None: ...
    def _timer_stop(self) -> None: ...
    def _timer_set_interval(self) -> None: ...

class FigureCanvasWebAggCore(backend_agg.FigureCanvasAgg):
    manager_class: Incomplete
    _timer_cls = TimerAsyncio
    supports_blit: bool
    _png_is_old: bool
    _force_full: bool
    _last_buff: Incomplete
    _current_image_mode: str
    _last_mouse_xy: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def show(self) -> None: ...
    def draw(self) -> None: ...
    def blit(self, bbox: Incomplete | None = None) -> None: ...
    def draw_idle(self) -> None: ...
    def set_cursor(self, cursor) -> None: ...
    def set_image_mode(self, mode) -> None:
        """
        Set the image mode for any subsequent images which will be sent
        to the clients. The modes may currently be either 'full' or 'diff'.

        Note: diff images may not contain transparency, therefore upon
        draw this mode may be changed if the resulting image has any
        transparent component.
        """
    def get_diff_image(self): ...
    def handle_event(self, event): ...
    def handle_unknown_event(self, event) -> None: ...
    def handle_ack(self, event) -> None: ...
    def handle_draw(self, event) -> None: ...
    def _handle_mouse(self, event) -> None: ...
    handle_button_press = _handle_mouse
    handle_button_release = _handle_mouse
    handle_dblclick = _handle_mouse
    handle_figure_enter = _handle_mouse
    handle_figure_leave = _handle_mouse
    handle_motion_notify = _handle_mouse
    handle_scroll = _handle_mouse
    def _handle_key(self, event) -> None: ...
    handle_key_press = _handle_key
    handle_key_release = _handle_key
    def handle_toolbar_button(self, event) -> None: ...
    def handle_refresh(self, event) -> None: ...
    def handle_resize(self, event) -> None: ...
    def handle_send_image_mode(self, event) -> None: ...
    def handle_set_device_pixel_ratio(self, event) -> None: ...
    def handle_set_dpi_ratio(self, event) -> None: ...
    def _handle_set_device_pixel_ratio(self, device_pixel_ratio) -> None: ...
    def send_event(self, event_type, **kwargs) -> None: ...

_ALLOWED_TOOL_ITEMS: Incomplete

class NavigationToolbar2WebAgg(backend_bases.NavigationToolbar2):
    toolitems: Incomplete
    message: str
    def __init__(self, canvas) -> None: ...
    def set_message(self, message) -> None: ...
    def draw_rubberband(self, event, x0, y0, x1, y1) -> None: ...
    def remove_rubberband(self) -> None: ...
    def save_figure(self, *args):
        """Save the current figure."""
    def pan(self) -> None: ...
    def zoom(self) -> None: ...
    def set_history_buttons(self) -> None: ...

class FigureManagerWebAgg(backend_bases.FigureManagerBase):
    _toolbar2_class: Incomplete
    ToolbarCls = NavigationToolbar2WebAgg
    _window_title: str
    web_sockets: Incomplete
    def __init__(self, canvas, num) -> None: ...
    def show(self) -> None: ...
    def resize(self, w, h, forward: bool = True) -> None: ...
    def set_window_title(self, title) -> None: ...
    def get_window_title(self): ...
    def add_web_socket(self, web_socket) -> None: ...
    def remove_web_socket(self, web_socket) -> None: ...
    def handle_json(self, content) -> None: ...
    def refresh_all(self) -> None: ...
    @classmethod
    def get_javascript(cls, stream: Incomplete | None = None): ...
    @classmethod
    def get_static_file_path(cls): ...
    def _send_event(self, event_type, **kwargs) -> None: ...

class _BackendWebAggCoreAgg(_Backend):
    FigureCanvas = FigureCanvasWebAggCore
    FigureManager = FigureManagerWebAgg
