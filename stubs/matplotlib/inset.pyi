from . import _api as _api, artist as artist, transforms as transforms
from _typeshed import Incomplete
from matplotlib.patches import ConnectionPatch as ConnectionPatch, PathPatch as PathPatch, Rectangle as Rectangle
from matplotlib.path import Path as Path

_shared_properties: Incomplete

class InsetIndicator(artist.Artist):
    """
    An artist to highlight an area of interest.

    An inset indicator is a rectangle on the plot at the position indicated by
    *bounds* that optionally has lines that connect the rectangle to an inset
    Axes (`.Axes.inset_axes`).

    .. versionadded:: 3.10
    """
    zorder: float
    _inset_ax: Incomplete
    _auto_update_bounds: bool
    _rectangle: Incomplete
    _connectors: Incomplete
    def __init__(self, bounds: Incomplete | None = None, inset_ax: Incomplete | None = None, zorder: Incomplete | None = None, **kwargs) -> None:
        """
        Parameters
        ----------
        bounds : [x0, y0, width, height], optional
            Lower-left corner of rectangle to be marked, and its width
            and height.  If not set, the bounds will be calculated from the
            data limits of inset_ax, which must be supplied.

        inset_ax : `~.axes.Axes`, optional
            An optional inset Axes to draw connecting lines to.  Two lines are
            drawn connecting the indicator box to the inset Axes on corners
            chosen so as to not overlap with the indicator box.

        zorder : float, default: 4.99
            Drawing order of the rectangle and connector lines.  The default,
            4.99, is just below the default level of inset Axes.

        **kwargs
            Other keyword arguments are passed on to the `.Rectangle` patch.
        """
    def _shared_setter(self, prop, val) -> None:
        """
        Helper function to set the same style property on the artist and its children.
        """
    def set_alpha(self, alpha) -> None: ...
    def set_edgecolor(self, color) -> None:
        """
        Set the edge color of the rectangle and the connectors.

        Parameters
        ----------
        color : :mpltype:`color` or None
        """
    def set_color(self, c) -> None:
        """
        Set the edgecolor of the rectangle and the connectors, and the
        facecolor for the rectangle.

        Parameters
        ----------
        c : :mpltype:`color`
        """
    def set_linewidth(self, w) -> None:
        """
        Set the linewidth in points of the rectangle and the connectors.

        Parameters
        ----------
        w : float or None
        """
    def set_linestyle(self, ls) -> None:
        """
        Set the linestyle of the rectangle and the connectors.

        ==========================================  =================
        linestyle                                   description
        ==========================================  =================
        ``'-'`` or ``'solid'``                      solid line
        ``'--'`` or ``'dashed'``                    dashed line
        ``'-.'`` or ``'dashdot'``                   dash-dotted line
        ``':'`` or ``'dotted'``                     dotted line
        ``'none'``, ``'None'``, ``' '``, or ``''``  draw nothing
        ==========================================  =================

        Alternatively a dash tuple of the following form can be provided::

            (offset, onoffseq)

        where ``onoffseq`` is an even length tuple of on and off ink in points.

        Parameters
        ----------
        ls : {'-', '--', '-.', ':', '', (offset, on-off-seq), ...}
            The line style.
        """
    def _bounds_from_inset_ax(self): ...
    def _update_connectors(self) -> None: ...
    @property
    def rectangle(self):
        """`.Rectangle`: the indicator frame."""
    @property
    def connectors(self):
        """
        4-tuple of `.patches.ConnectionPatch` or None
            The four connector lines connecting to (lower_left, upper_left,
            lower_right upper_right) corners of *inset_ax*. Two lines are
            set with visibility to *False*,  but the user can set the
            visibility to True if the automatic choice is not deemed correct.
        """
    def draw(self, renderer) -> None: ...
    def __getitem__(self, key): ...
