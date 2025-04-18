from .util import create_bounds as create_bounds, interpolate as interpolate, rinterpolate as rinterpolate, update_bounds as update_bounds
from _typeshed import Incomplete
from sympy.core.basic import Basic as Basic
from sympy.core.symbol import Symbol as Symbol, symbols as symbols
from sympy.utilities.iterables import sift as sift
from sympy.utilities.lambdify import lambdify as lambdify

class ColorGradient:
    colors: Incomplete
    intervals: Incomplete
    def __init__(self, *args) -> None: ...
    def copy(self): ...
    def _find_interval(self, v): ...
    def _interpolate_axis(self, axis, v): ...
    def __call__(self, r, g, b): ...

default_color_schemes: Incomplete

class ColorScheme:
    args: Incomplete
    f: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def _interpret_args(self, args): ...
    def _pop_symbol_list(self, lists): ...
    def _fill_in_vars(self, args): ...
    def _sort_args(self, args): ...
    def _test_color_function(self) -> None: ...
    def __call__(self, x, y, z, u, v): ...
    def apply_to_curve(self, verts, u_set, set_len: Incomplete | None = None, inc_pos: Incomplete | None = None):
        """
        Apply this color scheme to a
        set of vertices over a single
        independent variable u.
        """
    def apply_to_surface(self, verts, u_set, v_set, set_len: Incomplete | None = None, inc_pos: Incomplete | None = None):
        """
        Apply this color scheme to a
        set of vertices over two
        independent variables u and v.
        """
    def str_base(self): ...
    def __repr__(self) -> str: ...

x: Incomplete
y: Incomplete
z: Incomplete
t: Incomplete
u: Incomplete
v: Incomplete
