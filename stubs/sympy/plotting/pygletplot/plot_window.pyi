from _typeshed import Incomplete
from sympy.plotting.pygletplot.managed_window import ManagedWindow as ManagedWindow

class PlotWindow(ManagedWindow):
    plot: Incomplete
    camera: Incomplete
    _calculating: bool
    antialiasing: Incomplete
    ortho: Incomplete
    invert_mouse_zoom: Incomplete
    linewidth: Incomplete
    title: Incomplete
    last_caption_update: int
    caption_update_interval: float
    drawing_first_object: bool
    def __init__(self, plot, antialiasing: bool = True, ortho: bool = False, invert_mouse_zoom: bool = False, linewidth: float = 1.5, caption: str = 'SymPy Plot', **kwargs) -> None:
        """
        Named Arguments
        ===============

        antialiasing = True
            True OR False
        ortho = False
            True OR False
        invert_mouse_zoom = False
            True OR False
        """
    controller: Incomplete
    def setup(self) -> None: ...
    def on_resize(self, w, h) -> None: ...
    def update(self, dt) -> None: ...
    def draw(self) -> None: ...
    def update_caption(self, calc_verts_pos, calc_verts_len, calc_cverts_pos, calc_cverts_len) -> None: ...
