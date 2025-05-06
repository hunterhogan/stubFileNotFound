from _typeshed import Incomplete
from sympy.plotting.pygletplot.plot_mode_base import PlotModeBase as PlotModeBase

class PlotCurve(PlotModeBase):
    style_override: str
    t_interval: Incomplete
    t_set: Incomplete
    bounds: Incomplete
    _calculating_verts_pos: float
    _calculating_verts_len: Incomplete
    verts: Incomplete
    def _on_calculate_verts(self) -> None: ...
    _calculating_cverts_len: Incomplete
    _calculating_cverts_pos: int
    cverts: Incomplete
    def _on_calculate_cverts(self) -> None: ...
    def calculate_one_cvert(self, t): ...
    def draw_verts(self, use_cverts): ...
