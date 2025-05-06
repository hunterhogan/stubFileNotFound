from . import _api as _api, cbook as cbook
from ._enums import CapStyle as CapStyle, JoinStyle as JoinStyle
from .path import Path as Path
from .transforms import Affine2D as Affine2D, IdentityTransform as IdentityTransform
from _typeshed import Incomplete

TICKLEFT: Incomplete
TICKRIGHT: Incomplete
TICKUP: Incomplete
TICKDOWN: Incomplete
CARETLEFT: Incomplete
CARETRIGHT: Incomplete
CARETUP: Incomplete
CARETDOWN: Incomplete
CARETLEFTBASE: Incomplete
CARETRIGHTBASE: Incomplete
CARETUPBASE: Incomplete
CARETDOWNBASE: Incomplete
_empty_path: Incomplete

class MarkerStyle:
    """
    A class representing marker types.

    Instances are immutable. If you need to change anything, create a new
    instance.

    Attributes
    ----------
    markers : dict
        All known markers.
    filled_markers : tuple
        All known filled markers. This is a subset of *markers*.
    fillstyles : tuple
        The supported fillstyles.
    """
    markers: Incomplete
    filled_markers: Incomplete
    fillstyles: Incomplete
    _half_fillstyles: Incomplete
    _marker_function: Incomplete
    _user_transform: Incomplete
    _user_capstyle: Incomplete
    _user_joinstyle: Incomplete
    def __init__(self, marker, fillstyle: Incomplete | None = None, transform: Incomplete | None = None, capstyle: Incomplete | None = None, joinstyle: Incomplete | None = None) -> None:
        """
        Parameters
        ----------
        marker : str, array-like, Path, MarkerStyle
            - Another instance of `MarkerStyle` copies the details of that *marker*.
            - For other possible marker values, see the module docstring
              `matplotlib.markers`.

        fillstyle : str, default: :rc:`markers.fillstyle`
            One of 'full', 'left', 'right', 'bottom', 'top', 'none'.

        transform : `~matplotlib.transforms.Transform`, optional
            Transform that will be combined with the native transform of the
            marker.

        capstyle : `.CapStyle` or %(CapStyle)s, optional
            Cap style that will override the default cap style of the marker.

        joinstyle : `.JoinStyle` or %(JoinStyle)s, optional
            Join style that will override the default join style of the marker.
        """
    _path: Incomplete
    _transform: Incomplete
    _alt_path: Incomplete
    _alt_transform: Incomplete
    _snap_threshold: Incomplete
    _joinstyle: Incomplete
    _capstyle: Incomplete
    _filled: Incomplete
    def _recache(self) -> None: ...
    def __bool__(self) -> bool: ...
    def is_filled(self): ...
    def get_fillstyle(self): ...
    _fillstyle: Incomplete
    def _set_fillstyle(self, fillstyle) -> None:
        """
        Set the fillstyle.

        Parameters
        ----------
        fillstyle : {'full', 'left', 'right', 'bottom', 'top', 'none'}
            The part of the marker surface that is colored with
            markerfacecolor.
        """
    def get_joinstyle(self): ...
    def get_capstyle(self): ...
    def get_marker(self): ...
    __dict__: Incomplete
    _marker: Incomplete
    def _set_marker(self, marker) -> None:
        """
        Set the marker.

        Parameters
        ----------
        marker : str, array-like, Path, MarkerStyle
            - Another instance of `MarkerStyle` copies the details of that *marker*.
            - For other possible marker values see the module docstring
              `matplotlib.markers`.
        """
    def get_path(self):
        """
        Return a `.Path` for the primary part of the marker.

        For unfilled markers this is the whole marker, for filled markers,
        this is the area to be drawn with *markerfacecolor*.
        """
    def get_transform(self):
        """
        Return the transform to be applied to the `.Path` from
        `MarkerStyle.get_path()`.
        """
    def get_alt_path(self):
        """
        Return a `.Path` for the alternate part of the marker.

        For unfilled markers, this is *None*; for filled markers, this is the
        area to be drawn with *markerfacecoloralt*.
        """
    def get_alt_transform(self):
        """
        Return the transform to be applied to the `.Path` from
        `MarkerStyle.get_alt_path()`.
        """
    def get_snap_threshold(self): ...
    def get_user_transform(self):
        """Return user supplied part of marker transform."""
    def transformed(self, transform):
        """
        Return a new version of this marker with the transform applied.

        Parameters
        ----------
        transform : `~matplotlib.transforms.Affine2D`
            Transform will be combined with current user supplied transform.
        """
    def rotated(self, *, deg: Incomplete | None = None, rad: Incomplete | None = None):
        """
        Return a new version of this marker rotated by specified angle.

        Parameters
        ----------
        deg : float, optional
            Rotation angle in degrees.

        rad : float, optional
            Rotation angle in radians.

        .. note:: You must specify exactly one of deg or rad.
        """
    def scaled(self, sx, sy: Incomplete | None = None):
        """
        Return new marker scaled by specified scale factors.

        If *sy* is not given, the same scale is applied in both the *x*- and
        *y*-directions.

        Parameters
        ----------
        sx : float
            *X*-direction scaling factor.
        sy : float, optional
            *Y*-direction scaling factor.
        """
    def _set_nothing(self) -> None: ...
    def _set_custom_marker(self, path) -> None: ...
    def _set_path_marker(self) -> None: ...
    def _set_vertices(self) -> None: ...
    def _set_tuple_marker(self) -> None: ...
    _snap: bool
    def _set_mathtext_path(self) -> None:
        """
        Draw mathtext markers '$...$' using `.TextPath` object.

        Submitted by tcb
        """
    def _half_fill(self): ...
    def _set_circle(self, size: float = 1.0) -> None: ...
    def _set_point(self) -> None: ...
    def _set_pixel(self) -> None: ...
    _triangle_path: Incomplete
    _triangle_path_u: Incomplete
    _triangle_path_d: Incomplete
    _triangle_path_l: Incomplete
    _triangle_path_r: Incomplete
    def _set_triangle(self, rot, skip) -> None: ...
    def _set_triangle_up(self): ...
    def _set_triangle_down(self): ...
    def _set_triangle_left(self): ...
    def _set_triangle_right(self): ...
    def _set_square(self) -> None: ...
    def _set_diamond(self) -> None: ...
    def _set_thin_diamond(self) -> None: ...
    def _set_pentagon(self) -> None: ...
    def _set_star(self) -> None: ...
    def _set_hexagon1(self) -> None: ...
    def _set_hexagon2(self) -> None: ...
    def _set_octagon(self) -> None: ...
    _line_marker_path: Incomplete
    def _set_vline(self) -> None: ...
    def _set_hline(self) -> None: ...
    _tickhoriz_path: Incomplete
    def _set_tickleft(self) -> None: ...
    def _set_tickright(self) -> None: ...
    _tickvert_path: Incomplete
    def _set_tickup(self) -> None: ...
    def _set_tickdown(self) -> None: ...
    _tri_path: Incomplete
    def _set_tri_down(self) -> None: ...
    def _set_tri_up(self) -> None: ...
    def _set_tri_left(self) -> None: ...
    def _set_tri_right(self) -> None: ...
    _caret_path: Incomplete
    def _set_caretdown(self) -> None: ...
    def _set_caretup(self) -> None: ...
    def _set_caretleft(self) -> None: ...
    def _set_caretright(self) -> None: ...
    _caret_path_base: Incomplete
    def _set_caretdownbase(self) -> None: ...
    def _set_caretupbase(self) -> None: ...
    def _set_caretleftbase(self) -> None: ...
    def _set_caretrightbase(self) -> None: ...
    _plus_path: Incomplete
    def _set_plus(self) -> None: ...
    _x_path: Incomplete
    def _set_x(self) -> None: ...
    _plus_filled_path: Incomplete
    _plus_filled_path_t: Incomplete
    def _set_plus_filled(self) -> None: ...
    _x_filled_path: Incomplete
    _x_filled_path_t: Incomplete
    def _set_x_filled(self) -> None: ...
