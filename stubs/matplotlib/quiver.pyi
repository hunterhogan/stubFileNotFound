import matplotlib.artist as martist
import matplotlib.collections as mcollections
from _typeshed import Incomplete
from matplotlib import _api as _api, _docstring as _docstring, cbook as cbook
from matplotlib.patches import CirclePolygon as CirclePolygon

_quiver_doc: Incomplete

class QuiverKey(martist.Artist):
    """Labelled arrow for use as a quiver plot scale key."""
    halign: Incomplete
    valign: Incomplete
    pivot: Incomplete
    Q: Incomplete
    X: Incomplete
    Y: Incomplete
    U: Incomplete
    angle: Incomplete
    coord: Incomplete
    color: Incomplete
    label: Incomplete
    _labelsep_inches: Incomplete
    labelpos: Incomplete
    labelcolor: Incomplete
    fontproperties: Incomplete
    kw: Incomplete
    text: Incomplete
    _dpi_at_last_init: Incomplete
    zorder: Incomplete
    def __init__(self, Q, X, Y, U, label, *, angle: int = 0, coordinates: str = 'axes', color: Incomplete | None = None, labelsep: float = 0.1, labelpos: str = 'N', labelcolor: Incomplete | None = None, fontproperties: Incomplete | None = None, zorder: Incomplete | None = None, **kwargs) -> None:
        """
        Add a key to a quiver plot.

        The positioning of the key depends on *X*, *Y*, *coordinates*, and
        *labelpos*.  If *labelpos* is 'N' or 'S', *X*, *Y* give the position of
        the middle of the key arrow.  If *labelpos* is 'E', *X*, *Y* positions
        the head, and if *labelpos* is 'W', *X*, *Y* positions the tail; in
        either of these two cases, *X*, *Y* is somewhere in the middle of the
        arrow+label key object.

        Parameters
        ----------
        Q : `~matplotlib.quiver.Quiver`
            A `.Quiver` object as returned by a call to `~.Axes.quiver()`.
        X, Y : float
            The location of the key.
        U : float
            The length of the key.
        label : str
            The key label (e.g., length and units of the key).
        angle : float, default: 0
            The angle of the key arrow, in degrees anti-clockwise from the
            horizontal axis.
        coordinates : {'axes', 'figure', 'data', 'inches'}, default: 'axes'
            Coordinate system and units for *X*, *Y*: 'axes' and 'figure' are
            normalized coordinate systems with (0, 0) in the lower left and
            (1, 1) in the upper right; 'data' are the axes data coordinates
            (used for the locations of the vectors in the quiver plot itself);
            'inches' is position in the figure in inches, with (0, 0) at the
            lower left corner.
        color : :mpltype:`color`
            Overrides face and edge colors from *Q*.
        labelpos : {'N', 'S', 'E', 'W'}
            Position the label above, below, to the right, to the left of the
            arrow, respectively.
        labelsep : float, default: 0.1
            Distance in inches between the arrow and the label.
        labelcolor : :mpltype:`color`, default: :rc:`text.color`
            Label color.
        fontproperties : dict, optional
            A dictionary with keyword arguments accepted by the
            `~matplotlib.font_manager.FontProperties` initializer:
            *family*, *style*, *variant*, *size*, *weight*.
        zorder : float
            The zorder of the key. The default is 0.1 above *Q*.
        **kwargs
            Any additional keyword arguments are used to override vector
            properties taken from *Q*.
        """
    @property
    def labelsep(self): ...
    verts: Incomplete
    vector: Incomplete
    def _init(self) -> None: ...
    def _text_shift(self): ...
    stale: bool
    def draw(self, renderer) -> None: ...
    def _set_transform(self) -> None: ...
    def set_figure(self, fig) -> None: ...
    def contains(self, mouseevent): ...

def _parse_args(*args, caller_name: str = 'function'):
    """
    Helper function to parse positional parameters for colored vector plots.

    This is currently used for Quiver and Barbs.

    Parameters
    ----------
    *args : list
        list of 2-5 arguments. Depending on their number they are parsed to::

            U, V
            U, V, C
            X, Y, U, V
            X, Y, U, V, C

    caller_name : str
        Name of the calling method (used in error messages).
    """
def _check_consistent_shapes(*arrays) -> None: ...

class Quiver(mcollections.PolyCollection):
    """
    Specialized PolyCollection for arrows.

    The only API method is set_UVC(), which can be used
    to change the size, orientation, and color of the
    arrows; their locations are fixed when the class is
    instantiated.  Possibly this method will be useful
    in animations.

    Much of the work in this class is done in the draw()
    method so that as much information as possible is available
    about the plot.  In subsequent draw() calls, recalculation
    is limited to things that might have changed, so there
    should be no performance penalty from putting the calculations
    in the draw() method.
    """
    _PIVOT_VALS: Incomplete
    _axes: Incomplete
    X: Incomplete
    Y: Incomplete
    XY: Incomplete
    N: Incomplete
    scale: Incomplete
    headwidth: Incomplete
    headlength: Incomplete
    headaxislength: Incomplete
    minshaft: Incomplete
    minlength: Incomplete
    units: Incomplete
    scale_units: Incomplete
    angles: Incomplete
    width: Incomplete
    pivot: Incomplete
    transform: Incomplete
    polykw: Incomplete
    _dpi_at_last_init: Incomplete
    def __init__(self, ax, *args, scale: Incomplete | None = None, headwidth: int = 3, headlength: int = 5, headaxislength: float = 4.5, minshaft: int = 1, minlength: int = 1, units: str = 'width', scale_units: Incomplete | None = None, angles: str = 'uv', width: Incomplete | None = None, color: str = 'k', pivot: str = 'tail', **kwargs) -> None:
        """
        The constructor takes one required argument, an Axes
        instance, followed by the args and kwargs described
        by the following pyplot interface documentation:
        %s
        """
    span: Incomplete
    def _init(self) -> None:
        """
        Initialization delayed until first draw;
        allow time for axes setup.
        """
    def get_datalim(self, transData): ...
    stale: bool
    def draw(self, renderer) -> None: ...
    U: Incomplete
    V: Incomplete
    Umask: Incomplete
    def set_UVC(self, U, V, C: Incomplete | None = None) -> None: ...
    def _dots_per_unit(self, units):
        """Return a scale factor for converting from units to pixels."""
    _trans_scale: Incomplete
    def _set_transform(self):
        """
        Set the PolyCollection transform to go
        from arrow width units to pixels.
        """
    def _angles_lengths(self, XY, U, V, eps: int = 1): ...
    def _make_verts(self, XY, U, V, angles): ...
    def _h_arrows(self, length):
        """Length is in arrow width units."""

_barbs_doc: Incomplete

class Barbs(mcollections.PolyCollection):
    """
    Specialized PolyCollection for barbs.

    The only API method is :meth:`set_UVC`, which can be used to
    change the size, orientation, and color of the arrows.  Locations
    are changed using the :meth:`set_offsets` collection method.
    Possibly this method will be useful in animations.

    There is one internal function :meth:`_find_tails` which finds
    exactly what should be put on the barb given the vector magnitude.
    From there :meth:`_make_barbs` is used to find the vertices of the
    polygon to represent the barb based on this information.
    """
    sizes: Incomplete
    fill_empty: Incomplete
    barb_increments: Incomplete
    rounding: Incomplete
    flip: Incomplete
    _pivot: Incomplete
    _length: Incomplete
    x: Incomplete
    y: Incomplete
    def __init__(self, ax, *args, pivot: str = 'tip', length: int = 7, barbcolor: Incomplete | None = None, flagcolor: Incomplete | None = None, sizes: Incomplete | None = None, fill_empty: bool = False, barb_increments: Incomplete | None = None, rounding: bool = True, flip_barb: bool = False, **kwargs) -> None:
        """
        The constructor takes one required argument, an Axes
        instance, followed by the args and kwargs described
        by the following pyplot interface documentation:
        %(barbs_doc)s
        """
    def _find_tails(self, mag, rounding: bool = True, half: int = 5, full: int = 10, flag: int = 50):
        """
        Find how many of each of the tail pieces is necessary.

        Parameters
        ----------
        mag : `~numpy.ndarray`
            Vector magnitudes; must be non-negative (and an actual ndarray).
        rounding : bool, default: True
            Whether to round or to truncate to the nearest half-barb.
        half, full, flag : float, defaults: 5, 10, 50
            Increments for a half-barb, a barb, and a flag.

        Returns
        -------
        n_flags, n_barbs : int array
            For each entry in *mag*, the number of flags and barbs.
        half_flag : bool array
            For each entry in *mag*, whether a half-barb is needed.
        empty_flag : bool array
            For each entry in *mag*, whether nothing is drawn.
        """
    def _make_barbs(self, u, v, nflags, nbarbs, half_barb, empty_flag, length, pivot, sizes, fill_empty, flip):
        '''
        Create the wind barbs.

        Parameters
        ----------
        u, v
            Components of the vector in the x and y directions, respectively.

        nflags, nbarbs, half_barb, empty_flag
            Respectively, the number of flags, number of barbs, flag for
            half a barb, and flag for empty barb, ostensibly obtained from
            :meth:`_find_tails`.

        length
            The length of the barb staff in points.

        pivot : {"tip", "middle"} or number
            The point on the barb around which the entire barb should be
            rotated.  If a number, the start of the barb is shifted by that
            many points from the origin.

        sizes : dict
            Coefficients specifying the ratio of a given feature to the length
            of the barb. These features include:

            - *spacing*: space between features (flags, full/half barbs).
            - *height*: distance from shaft of top of a flag or full barb.
            - *width*: width of a flag, twice the width of a full barb.
            - *emptybarb*: radius of the circle used for low magnitudes.

        fill_empty : bool
            Whether the circle representing an empty barb should be filled or
            not (this changes the drawing of the polygon).

        flip : list of bool
            Whether the features should be flipped to the other side of the
            barb (useful for winds in the southern hemisphere).

        Returns
        -------
        list of arrays of vertices
            Polygon vertices for each of the wind barbs.  These polygons have
            been rotated to properly align with the vector direction.
        '''
    u: Incomplete
    v: Incomplete
    _offsets: Incomplete
    stale: bool
    def set_UVC(self, U, V, C: Incomplete | None = None) -> None: ...
    def set_offsets(self, xy) -> None:
        """
        Set the offsets for the barb polygons.  This saves the offsets passed
        in and masks them as appropriate for the existing U/V data.

        Parameters
        ----------
        xy : sequence of pairs of floats
        """
