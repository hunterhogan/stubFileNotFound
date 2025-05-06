from _typeshed import Incomplete
from sympy.plotting.pygletplot.plot_mode_base import PlotModeBase as PlotModeBase

class PlotSurface(PlotModeBase):
    default_rot_preset: str
    u_interval: Incomplete
    u_set: Incomplete
    v_interval: Incomplete
    v_set: Incomplete
    bounds: Incomplete
    _calculating_verts_pos: float
    _calculating_verts_len: Incomplete
    verts: Incomplete
    def _on_calculate_verts(self) -> None: ...
    _calculating_cverts_len: Incomplete
    _calculating_cverts_pos: int
    cverts: Incomplete
    def _on_calculate_cverts(self) -> None: ...
    def calculate_one_cvert(self, u, v): ...
    def draw_verts(self, use_cverts, use_solid_color): ...
