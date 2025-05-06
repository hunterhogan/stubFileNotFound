from . import _api as _api, _docstring as _docstring, _path as _path, cbook as cbook
from ._enums import CapStyle as CapStyle, JoinStyle as JoinStyle
from .artist import Artist as Artist, allow_rasterization as allow_rasterization
from .cbook import STEP_LOOKUP_MAP as STEP_LOOKUP_MAP, _to_unmasked_float_array as _to_unmasked_float_array, ls_mapper as ls_mapper, ls_mapper_r as ls_mapper_r
from .markers import CARETDOWN as CARETDOWN, CARETDOWNBASE as CARETDOWNBASE, CARETLEFT as CARETLEFT, CARETLEFTBASE as CARETLEFTBASE, CARETRIGHT as CARETRIGHT, CARETRIGHTBASE as CARETRIGHTBASE, CARETUP as CARETUP, CARETUPBASE as CARETUPBASE, MarkerStyle as MarkerStyle, TICKDOWN as TICKDOWN, TICKLEFT as TICKLEFT, TICKRIGHT as TICKRIGHT, TICKUP as TICKUP
from .transforms import Bbox as Bbox, BboxTransformTo as BboxTransformTo, TransformedPath as TransformedPath
from _typeshed import Incomplete

_log: Incomplete

def _get_dash_pattern(style):
    """Convert linestyle to dash pattern."""
def _get_inverse_dash_pattern(offset, dashes):
    """Return the inverse of the given dash pattern, for filling the gaps."""
def _scale_dashes(offset, dashes, lw): ...
def segment_hits(cx, cy, x, y, radius):
    """
    Return the indices of the segments in the polyline with coordinates (*cx*,
    *cy*) that are within a distance *radius* of the point (*x*, *y*).
    """
def _mark_every_path(markevery, tpath, affine, ax):
    """
    Helper function that sorts out how to deal the input
    `markevery` and returns the points where markers should be drawn.

    Takes in the `markevery` value and the line path and returns the
    sub-sampled path.
    """

class Line2D(Artist):
    '''
    A line - the line can have both a solid linestyle connecting all
    the vertices, and a marker at each vertex.  Additionally, the
    drawing of the solid line is influenced by the drawstyle, e.g., one
    can create "stepped" lines in various styles.
    '''
    lineStyles: Incomplete
    _lineStyles: Incomplete
    _drawStyles_l: Incomplete
    _drawStyles_s: Incomplete
    drawStyles: Incomplete
    drawStyleKeys: Incomplete
    markers: Incomplete
    filled_markers: Incomplete
    fillStyles: Incomplete
    zorder: int
    _subslice_optim_min_size: int
    def __str__(self) -> str: ...
    _dashcapstyle: Incomplete
    _dashjoinstyle: Incomplete
    _solidjoinstyle: Incomplete
    _solidcapstyle: Incomplete
    _linestyles: Incomplete
    _drawstyle: Incomplete
    _linewidth: Incomplete
    _unscaled_dash_pattern: Incomplete
    _dash_pattern: Incomplete
    _color: Incomplete
    _marker: Incomplete
    _gapcolor: Incomplete
    _markevery: Incomplete
    _markersize: Incomplete
    _antialiased: Incomplete
    _markeredgecolor: Incomplete
    _markeredgewidth: Incomplete
    _markerfacecolor: Incomplete
    _markerfacecoloralt: Incomplete
    pickradius: Incomplete
    ind_offset: int
    _pickradius: Incomplete
    _xorig: Incomplete
    _yorig: Incomplete
    _invalidx: bool
    _invalidy: bool
    _x: Incomplete
    _y: Incomplete
    _xy: Incomplete
    _path: Incomplete
    _transformed_path: Incomplete
    _subslice: bool
    _x_filled: Incomplete
    def __init__(self, xdata, ydata, *, linewidth: Incomplete | None = None, linestyle: Incomplete | None = None, color: Incomplete | None = None, gapcolor: Incomplete | None = None, marker: Incomplete | None = None, markersize: Incomplete | None = None, markeredgewidth: Incomplete | None = None, markeredgecolor: Incomplete | None = None, markerfacecolor: Incomplete | None = None, markerfacecoloralt: str = 'none', fillstyle: Incomplete | None = None, antialiased: Incomplete | None = None, dash_capstyle: Incomplete | None = None, solid_capstyle: Incomplete | None = None, dash_joinstyle: Incomplete | None = None, solid_joinstyle: Incomplete | None = None, pickradius: int = 5, drawstyle: Incomplete | None = None, markevery: Incomplete | None = None, **kwargs) -> None:
        """
        Create a `.Line2D` instance with *x* and *y* data in sequences of
        *xdata*, *ydata*.

        Additional keyword arguments are `.Line2D` properties:

        %(Line2D:kwdoc)s

        See :meth:`set_linestyle` for a description of the line styles,
        :meth:`set_marker` for a description of the markers, and
        :meth:`set_drawstyle` for a description of the draw styles.

        """
    def contains(self, mouseevent):
        '''
        Test whether *mouseevent* occurred on the line.

        An event is deemed to have occurred "on" the line if it is less
        than ``self.pickradius`` (default: 5 points) away from it.  Use
        `~.Line2D.get_pickradius` or `~.Line2D.set_pickradius` to get or set
        the pick radius.

        Parameters
        ----------
        mouseevent : `~matplotlib.backend_bases.MouseEvent`

        Returns
        -------
        contains : bool
            Whether any values are within the radius.
        details : dict
            A dictionary ``{\'ind\': pointlist}``, where *pointlist* is a
            list of points of the line that are within the pickradius around
            the event position.

            TODO: sort returned indices by distance
        '''
    def get_pickradius(self):
        """
        Return the pick radius used for containment tests.

        See `.contains` for more details.
        """
    def set_pickradius(self, pickradius) -> None:
        """
        Set the pick radius used for containment tests.

        See `.contains` for more details.

        Parameters
        ----------
        pickradius : float
            Pick radius, in points.
        """
    def get_fillstyle(self):
        """
        Return the marker fill style.

        See also `~.Line2D.set_fillstyle`.
        """
    stale: bool
    def set_fillstyle(self, fs) -> None:
        """
        Set the marker fill style.

        Parameters
        ----------
        fs : {'full', 'left', 'right', 'bottom', 'top', 'none'}
            Possible values:

            - 'full': Fill the whole marker with the *markerfacecolor*.
            - 'left', 'right', 'bottom', 'top': Fill the marker half at
              the given side with the *markerfacecolor*. The other
              half of the marker is filled with *markerfacecoloralt*.
            - 'none': No filling.

            For examples see :ref:`marker_fill_styles`.
        """
    def set_markevery(self, every) -> None:
        """
        Set the markevery property to subsample the plot when using markers.

        e.g., if ``every=5``, every 5-th marker will be plotted.

        Parameters
        ----------
        every : None or int or (int, int) or slice or list[int] or float or (float, float) or list[bool]
            Which markers to plot.

            - ``every=None``: every point will be plotted.
            - ``every=N``: every N-th marker will be plotted starting with
              marker 0.
            - ``every=(start, N)``: every N-th marker, starting at index
              *start*, will be plotted.
            - ``every=slice(start, end, N)``: every N-th marker, starting at
              index *start*, up to but not including index *end*, will be
              plotted.
            - ``every=[i, j, m, ...]``: only markers at the given indices
              will be plotted.
            - ``every=[True, False, True, ...]``: only positions that are True
              will be plotted. The list must have the same length as the data
              points.
            - ``every=0.1``, (i.e. a float): markers will be spaced at
              approximately equal visual distances along the line; the distance
              along the line between markers is determined by multiplying the
              display-coordinate distance of the Axes bounding-box diagonal
              by the value of *every*.
            - ``every=(0.5, 0.1)`` (i.e. a length-2 tuple of float): similar
              to ``every=0.1`` but the first marker will be offset along the
              line by 0.5 multiplied by the
              display-coordinate-diagonal-distance along the line.

            For examples see
            :doc:`/gallery/lines_bars_and_markers/markevery_demo`.

        Notes
        -----
        Setting *markevery* will still only draw markers at actual data points.
        While the float argument form aims for uniform visual spacing, it has
        to coerce from the ideal spacing to the nearest available data point.
        Depending on the number and distribution of data points, the result
        may still not look evenly spaced.

        When using a start offset to specify the first marker, the offset will
        be from the first data point which may be different from the first
        the visible data point if the plot is zoomed in.

        If zooming in on a plot when using float arguments then the actual
        data points that have markers will change because the distance between
        markers is always determined from the display-coordinates
        axes-bounding-box-diagonal regardless of the actual axes data limits.

        """
    def get_markevery(self):
        """
        Return the markevery setting for marker subsampling.

        See also `~.Line2D.set_markevery`.
        """
    _picker: Incomplete
    def set_picker(self, p) -> None:
        """
        Set the event picker details for the line.

        Parameters
        ----------
        p : float or callable[[Artist, Event], tuple[bool, dict]]
            If a float, it is used as the pick radius in points.
        """
    def get_bbox(self):
        """Get the bounding box of this line."""
    def get_window_extent(self, renderer: Incomplete | None = None): ...
    def set_data(self, *args) -> None:
        """
        Set the x and y data.

        Parameters
        ----------
        *args : (2, N) array or two 1D arrays

        See Also
        --------
        set_xdata
        set_ydata
        """
    def recache_always(self) -> None: ...
    def recache(self, always: bool = False) -> None: ...
    def _transform_path(self, subslice: Incomplete | None = None) -> None:
        """
        Put a TransformedPath instance at self._transformed_path;
        all invalidation of the transform is then handled by the
        TransformedPath instance.
        """
    def _get_transformed_path(self):
        """Return this line's `~matplotlib.transforms.TransformedPath`."""
    def set_transform(self, t) -> None: ...
    def draw(self, renderer) -> None: ...
    def get_antialiased(self):
        """Return whether antialiased rendering is used."""
    def get_color(self):
        """
        Return the line color.

        See also `~.Line2D.set_color`.
        """
    def get_drawstyle(self):
        """
        Return the drawstyle.

        See also `~.Line2D.set_drawstyle`.
        """
    def get_gapcolor(self):
        """
        Return the line gapcolor.

        See also `~.Line2D.set_gapcolor`.
        """
    def get_linestyle(self):
        """
        Return the linestyle.

        See also `~.Line2D.set_linestyle`.
        """
    def get_linewidth(self):
        """
        Return the linewidth in points.

        See also `~.Line2D.set_linewidth`.
        """
    def get_marker(self):
        """
        Return the line marker.

        See also `~.Line2D.set_marker`.
        """
    def get_markeredgecolor(self):
        """
        Return the marker edge color.

        See also `~.Line2D.set_markeredgecolor`.
        """
    def get_markeredgewidth(self):
        """
        Return the marker edge width in points.

        See also `~.Line2D.set_markeredgewidth`.
        """
    def _get_markerfacecolor(self, alt: bool = False): ...
    def get_markerfacecolor(self):
        """
        Return the marker face color.

        See also `~.Line2D.set_markerfacecolor`.
        """
    def get_markerfacecoloralt(self):
        """
        Return the alternate marker face color.

        See also `~.Line2D.set_markerfacecoloralt`.
        """
    def get_markersize(self):
        """
        Return the marker size in points.

        See also `~.Line2D.set_markersize`.
        """
    def get_data(self, orig: bool = True):
        """
        Return the line data as an ``(xdata, ydata)`` pair.

        If *orig* is *True*, return the original data.
        """
    def get_xdata(self, orig: bool = True):
        """
        Return the xdata.

        If *orig* is *True*, return the original data, else the
        processed data.
        """
    def get_ydata(self, orig: bool = True):
        """
        Return the ydata.

        If *orig* is *True*, return the original data, else the
        processed data.
        """
    def get_path(self):
        """Return the `~matplotlib.path.Path` associated with this line."""
    def get_xydata(self):
        """Return the *xy* data as a (N, 2) array."""
    def set_antialiased(self, b) -> None:
        """
        Set whether to use antialiased rendering.

        Parameters
        ----------
        b : bool
        """
    def set_color(self, color) -> None:
        """
        Set the color of the line.

        Parameters
        ----------
        color : :mpltype:`color`
        """
    def set_drawstyle(self, drawstyle) -> None:
        """
        Set the drawstyle of the plot.

        The drawstyle determines how the points are connected.

        Parameters
        ----------
        drawstyle : {'default', 'steps', 'steps-pre', 'steps-mid', 'steps-post'}, default: 'default'
            For 'default', the points are connected with straight lines.

            The steps variants connect the points with step-like lines,
            i.e. horizontal lines with vertical steps. They differ in the
            location of the step:

            - 'steps-pre': The step is at the beginning of the line segment,
              i.e. the line will be at the y-value of point to the right.
            - 'steps-mid': The step is halfway between the points.
            - 'steps-post: The step is at the end of the line segment,
              i.e. the line will be at the y-value of the point to the left.
            - 'steps' is equal to 'steps-pre' and is maintained for
              backward-compatibility.

            For examples see :doc:`/gallery/lines_bars_and_markers/step_demo`.
        """
    def set_gapcolor(self, gapcolor) -> None:
        """
        Set a color to fill the gaps in the dashed line style.

        .. note::

            Striped lines are created by drawing two interleaved dashed lines.
            There can be overlaps between those two, which may result in
            artifacts when using transparency.

            This functionality is experimental and may change.

        Parameters
        ----------
        gapcolor : :mpltype:`color` or None
            The color with which to fill the gaps. If None, the gaps are
            unfilled.
        """
    def set_linewidth(self, w) -> None:
        """
        Set the line width in points.

        Parameters
        ----------
        w : float
            Line width, in points.
        """
    _linestyle: Incomplete
    def set_linestyle(self, ls) -> None:
        """
        Set the linestyle of the line.

        Parameters
        ----------
        ls : {'-', '--', '-.', ':', '', (offset, on-off-seq), ...}
            Possible values:

            - A string:

              ==========================================  =================
              linestyle                                   description
              ==========================================  =================
              ``'-'`` or ``'solid'``                      solid line
              ``'--'`` or  ``'dashed'``                   dashed line
              ``'-.'`` or  ``'dashdot'``                  dash-dotted line
              ``':'`` or ``'dotted'``                     dotted line
              ``'none'``, ``'None'``, ``' '``, or ``''``  draw nothing
              ==========================================  =================

            - Alternatively a dash tuple of the following form can be
              provided::

                  (offset, onoffseq)

              where ``onoffseq`` is an even length tuple of on and off ink
              in points. See also :meth:`set_dashes`.

            For examples see :doc:`/gallery/lines_bars_and_markers/linestyles`.
        """
    def set_marker(self, marker) -> None:
        """
        Set the line marker.

        Parameters
        ----------
        marker : marker style string, `~.path.Path` or `~.markers.MarkerStyle`
            See `~matplotlib.markers` for full description of possible
            arguments.
        """
    def _set_markercolor(self, name, has_rcdefault, val) -> None: ...
    def set_markeredgecolor(self, ec) -> None:
        """
        Set the marker edge color.

        Parameters
        ----------
        ec : :mpltype:`color`
        """
    def set_markerfacecolor(self, fc) -> None:
        """
        Set the marker face color.

        Parameters
        ----------
        fc : :mpltype:`color`
        """
    def set_markerfacecoloralt(self, fc) -> None:
        """
        Set the alternate marker face color.

        Parameters
        ----------
        fc : :mpltype:`color`
        """
    def set_markeredgewidth(self, ew) -> None:
        """
        Set the marker edge width in points.

        Parameters
        ----------
        ew : float
             Marker edge width, in points.
        """
    def set_markersize(self, sz) -> None:
        """
        Set the marker size in points.

        Parameters
        ----------
        sz : float
             Marker size, in points.
        """
    def set_xdata(self, x) -> None:
        """
        Set the data array for x.

        Parameters
        ----------
        x : 1D array

        See Also
        --------
        set_data
        set_ydata
        """
    def set_ydata(self, y) -> None:
        """
        Set the data array for y.

        Parameters
        ----------
        y : 1D array

        See Also
        --------
        set_data
        set_xdata
        """
    def set_dashes(self, seq) -> None:
        """
        Set the dash sequence.

        The dash sequence is a sequence of floats of even length describing
        the length of dashes and spaces in points.

        For example, (5, 2, 1, 2) describes a sequence of 5 point and 1 point
        dashes separated by 2 point spaces.

        See also `~.Line2D.set_gapcolor`, which allows those spaces to be
        filled with a color.

        Parameters
        ----------
        seq : sequence of floats (on/off ink in points) or (None, None)
            If *seq* is empty or ``(None, None)``, the linestyle will be set
            to solid.
        """
    def update_from(self, other) -> None:
        """Copy properties from *other* to self."""
    def set_dash_joinstyle(self, s) -> None:
        """
        How to join segments of the line if it `~Line2D.is_dashed`.

        The default joinstyle is :rc:`lines.dash_joinstyle`.

        Parameters
        ----------
        s : `.JoinStyle` or %(JoinStyle)s
        """
    def set_solid_joinstyle(self, s) -> None:
        """
        How to join segments if the line is solid (not `~Line2D.is_dashed`).

        The default joinstyle is :rc:`lines.solid_joinstyle`.

        Parameters
        ----------
        s : `.JoinStyle` or %(JoinStyle)s
        """
    def get_dash_joinstyle(self):
        """
        Return the `.JoinStyle` for dashed lines.

        See also `~.Line2D.set_dash_joinstyle`.
        """
    def get_solid_joinstyle(self):
        """
        Return the `.JoinStyle` for solid lines.

        See also `~.Line2D.set_solid_joinstyle`.
        """
    def set_dash_capstyle(self, s) -> None:
        """
        How to draw the end caps if the line is `~Line2D.is_dashed`.

        The default capstyle is :rc:`lines.dash_capstyle`.

        Parameters
        ----------
        s : `.CapStyle` or %(CapStyle)s
        """
    def set_solid_capstyle(self, s) -> None:
        """
        How to draw the end caps if the line is solid (not `~Line2D.is_dashed`)

        The default capstyle is :rc:`lines.solid_capstyle`.

        Parameters
        ----------
        s : `.CapStyle` or %(CapStyle)s
        """
    def get_dash_capstyle(self):
        """
        Return the `.CapStyle` for dashed lines.

        See also `~.Line2D.set_dash_capstyle`.
        """
    def get_solid_capstyle(self):
        """
        Return the `.CapStyle` for solid lines.

        See also `~.Line2D.set_solid_capstyle`.
        """
    def is_dashed(self):
        """
        Return whether line has a dashed linestyle.

        A custom linestyle is assumed to be dashed, we do not inspect the
        ``onoffseq`` directly.

        See also `~.Line2D.set_linestyle`.
        """

class AxLine(Line2D):
    """
    A helper class that implements `~.Axes.axline`, by recomputing the artist
    transform at draw time.
    """
    _slope: Incomplete
    _xy1: Incomplete
    _xy2: Incomplete
    def __init__(self, xy1, xy2, slope, **kwargs) -> None:
        """
        Parameters
        ----------
        xy1 : (float, float)
            The first set of (x, y) coordinates for the line to pass through.
        xy2 : (float, float) or None
            The second set of (x, y) coordinates for the line to pass through.
            Both *xy2* and *slope* must be passed, but one of them must be None.
        slope : float or None
            The slope of the line. Both *xy2* and *slope* must be passed, but one of
            them must be None.
        """
    def get_transform(self): ...
    _transformed_path: Incomplete
    def draw(self, renderer) -> None: ...
    def get_xy1(self):
        """Return the *xy1* value of the line."""
    def get_xy2(self):
        """Return the *xy2* value of the line."""
    def get_slope(self):
        """Return the *slope* value of the line."""
    def set_xy1(self, *args, **kwargs):
        """
        Set the *xy1* value of the line.

        Parameters
        ----------
        xy1 : tuple[float, float]
            Points for the line to pass through.
        """
    def set_xy2(self, *args, **kwargs):
        """
        Set the *xy2* value of the line.

        .. note::

            You can only set *xy2* if the line was created using the *xy2*
            parameter. If the line was created using *slope*, please use
            `~.AxLine.set_slope`.

        Parameters
        ----------
        xy2 : tuple[float, float]
            Points for the line to pass through.
        """
    def set_slope(self, slope) -> None:
        """
        Set the *slope* value of the line.

        .. note::

            You can only set *slope* if the line was created using the *slope*
            parameter. If the line was created using *xy2*, please use
            `~.AxLine.set_xy2`.

        Parameters
        ----------
        slope : float
            The slope of the line.
        """

class VertexSelector:
    """
    Manage the callbacks to maintain a list of selected vertices for `.Line2D`.
    Derived classes should override the `process_selected` method to do
    something with the picks.

    Here is an example which highlights the selected verts with red circles::

        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.lines as lines

        class HighlightSelected(lines.VertexSelector):
            def __init__(self, line, fmt='ro', **kwargs):
                super().__init__(line)
                self.markers, = self.axes.plot([], [], fmt, **kwargs)

            def process_selected(self, ind, xs, ys):
                self.markers.set_data(xs, ys)
                self.canvas.draw()

        fig, ax = plt.subplots()
        x, y = np.random.rand(2, 30)
        line, = ax.plot(x, y, 'bs-', picker=5)

        selector = HighlightSelected(line)
        plt.show()
    """
    axes: Incomplete
    line: Incomplete
    cid: Incomplete
    ind: Incomplete
    def __init__(self, line) -> None:
        """
        Parameters
        ----------
        line : `~matplotlib.lines.Line2D`
            The line must already have been added to an `~.axes.Axes` and must
            have its picker property set.
        """
    canvas: Incomplete
    def process_selected(self, ind, xs, ys) -> None:
        '''
        Default "do nothing" implementation of the `process_selected` method.

        Parameters
        ----------
        ind : list of int
            The indices of the selected vertices.
        xs, ys : array-like
            The coordinates of the selected vertices.
        '''
    def onpick(self, event) -> None:
        """When the line is picked, update the set of selected indices."""

lineStyles: Incomplete
lineMarkers: Incomplete
drawStyles: Incomplete
fillStyles: Incomplete
