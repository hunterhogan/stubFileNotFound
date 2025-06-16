from . import _api as _api, _path as _path
from .bezier import BezierSegment as BezierSegment
from .cbook import _to_unmasked_float_array as _to_unmasked_float_array, simple_linear_interpolation as simple_linear_interpolation
from _typeshed import Incomplete
from collections.abc import Generator

class Path:
    """
    A series of possibly disconnected, possibly closed, line and curve
    segments.

    The underlying storage is made up of two parallel numpy arrays:

    - *vertices*: an (N, 2) float array of vertices
    - *codes*: an N-length `numpy.uint8` array of path codes, or None

    These two arrays always have the same length in the first
    dimension.  For example, to represent a cubic curve, you must
    provide three vertices and three `CURVE4` codes.

    The code types are:

    - `STOP`   :  1 vertex (ignored)
        A marker for the end of the entire path (currently not required and
        ignored)

    - `MOVETO` :  1 vertex
        Pick up the pen and move to the given vertex.

    - `LINETO` :  1 vertex
        Draw a line from the current position to the given vertex.

    - `CURVE3` :  1 control point, 1 endpoint
        Draw a quadratic B�zier curve from the current position, with the given
        control point, to the given end point.

    - `CURVE4` :  2 control points, 1 endpoint
        Draw a cubic B�zier curve from the current position, with the given
        control points, to the given end point.

    - `CLOSEPOLY` : 1 vertex (ignored)
        Draw a line segment to the start point of the current polyline.

    If *codes* is None, it is interpreted as a `MOVETO` followed by a series
    of `LINETO`.

    Users of Path objects should not access the vertices and codes arrays
    directly.  Instead, they should use `iter_segments` or `cleaned` to get the
    vertex/code pairs.  This helps, in particular, to consistently handle the
    case of *codes* being None.

    Some behavior of Path objects can be controlled by rcParams. See the
    rcParams whose keys start with 'path.'.

    .. note::

        The vertices and codes arrays should be treated as
        immutable -- there are a number of optimizations and assumptions
        made up front in the constructor that will not change when the
        data changes.
    """
    code_type: Incomplete
    STOP: Incomplete
    MOVETO: Incomplete
    LINETO: Incomplete
    CURVE3: Incomplete
    CURVE4: Incomplete
    CLOSEPOLY: Incomplete
    NUM_VERTICES_FOR_CODE: Incomplete
    _vertices: Incomplete
    _codes: Incomplete
    _interpolation_steps: Incomplete
    _readonly: bool
    def __init__(self, vertices, codes: Incomplete | None = None, _interpolation_steps: int = 1, closed: bool = False, readonly: bool = False) -> None:
        """
        Create a new path with the given vertices and codes.

        Parameters
        ----------
        vertices : (N, 2) array-like
            The path vertices, as an array, masked array or sequence of pairs.
            Masked values, if any, will be converted to NaNs, which are then
            handled correctly by the Agg PathIterator and other consumers of
            path data, such as :meth:`iter_segments`.
        codes : array-like or None, optional
            N-length array of integers representing the codes of the path.
            If not None, codes must be the same length as vertices.
            If None, *vertices* will be treated as a series of line segments.
        _interpolation_steps : int, optional
            Used as a hint to certain projections, such as Polar, that this
            path should be linearly interpolated immediately before drawing.
            This attribute is primarily an implementation detail and is not
            intended for public use.
        closed : bool, optional
            If *codes* is None and closed is True, vertices will be treated as
            line segments of a closed polygon.  Note that the last vertex will
            then be ignored (as the corresponding code will be set to
            `CLOSEPOLY`).
        readonly : bool, optional
            Makes the path behave in an immutable way and sets the vertices
            and codes as read-only arrays.
        """
    @classmethod
    def _fast_from_codes_and_verts(cls, verts, codes, internals_from: Incomplete | None = None):
        """
        Create a Path instance without the expense of calling the constructor.

        Parameters
        ----------
        verts : array-like
        codes : array
        internals_from : Path or None
            If not None, another `Path` from which the attributes
            ``should_simplify``, ``simplify_threshold``, and
            ``interpolation_steps`` will be copied.  Note that ``readonly`` is
            never copied, and always set to ``False`` by this constructor.
        """
    @classmethod
    def _create_closed(cls, vertices):
        """
        Create a closed polygonal path going through *vertices*.

        Unlike ``Path(..., closed=True)``, *vertices* should **not** end with
        an entry for the CLOSEPATH; this entry is added by `._create_closed`.
        """
    _simplify_threshold: Incomplete
    _should_simplify: Incomplete
    def _update_values(self) -> None: ...
    @property
    def vertices(self):
        """The vertices of the `Path` as an (N, 2) array."""
    @vertices.setter
    def vertices(self, vertices) -> None: ...
    @property
    def codes(self):
        """
        The list of codes in the `Path` as a 1D array.

        Each code is one of `STOP`, `MOVETO`, `LINETO`, `CURVE3`, `CURVE4` or
        `CLOSEPOLY`.  For codes that correspond to more than one vertex
        (`CURVE3` and `CURVE4`), that code will be repeated so that the length
        of `vertices` and `codes` is always the same.
        """
    @codes.setter
    def codes(self, codes) -> None: ...
    @property
    def simplify_threshold(self):
        """
        The fraction of a pixel difference below which vertices will
        be simplified out.
        """
    @simplify_threshold.setter
    def simplify_threshold(self, threshold) -> None: ...
    @property
    def should_simplify(self):
        """
        `True` if the vertices array should be simplified.
        """
    @should_simplify.setter
    def should_simplify(self, should_simplify) -> None: ...
    @property
    def readonly(self):
        """
        `True` if the `Path` is read-only.
        """
    def copy(self):
        """
        Return a shallow copy of the `Path`, which will share the
        vertices and codes with the source `Path`.
        """
    def __deepcopy__(self, memo: Incomplete | None = None):
        """
        Return a deepcopy of the `Path`.  The `Path` will not be
        readonly, even if the source `Path` is.
        """
    deepcopy = __deepcopy__
    @classmethod
    def make_compound_path_from_polys(cls, XY):
        """
        Make a compound `Path` object to draw a number of polygons with equal
        numbers of sides.

        .. plot:: gallery/misc/histogram_path.py

        Parameters
        ----------
        XY : (numpolys, numsides, 2) array
        """
    @classmethod
    def make_compound_path(cls, *args):
        """
        Concatenate a list of `Path`\\s into a single `Path`, removing all `STOP`\\s.
        """
    def __repr__(self) -> str: ...
    def __len__(self) -> int: ...
    def iter_segments(self, transform: Incomplete | None = None, remove_nans: bool = True, clip: Incomplete | None = None, snap: bool = False, stroke_width: float = 1.0, simplify: Incomplete | None = None, curves: bool = True, sketch: Incomplete | None = None) -> Generator[Incomplete]:
        """
        Iterate over all curve segments in the path.

        Each iteration returns a pair ``(vertices, code)``, where ``vertices``
        is a sequence of 1-3 coordinate pairs, and ``code`` is a `Path` code.

        Additionally, this method can provide a number of standard cleanups and
        conversions to the path.

        Parameters
        ----------
        transform : None or :class:`~matplotlib.transforms.Transform`
            If not None, the given affine transformation will be applied to the
            path.
        remove_nans : bool, optional
            Whether to remove all NaNs from the path and skip over them using
            MOVETO commands.
        clip : None or (float, float, float, float), optional
            If not None, must be a four-tuple (x1, y1, x2, y2)
            defining a rectangle in which to clip the path.
        snap : None or bool, optional
            If True, snap all nodes to pixels; if False, don't snap them.
            If None, snap if the path contains only segments
            parallel to the x or y axes, and no more than 1024 of them.
        stroke_width : float, optional
            The width of the stroke being drawn (used for path snapping).
        simplify : None or bool, optional
            Whether to simplify the path by removing vertices
            that do not affect its appearance.  If None, use the
            :attr:`should_simplify` attribute.  See also :rc:`path.simplify`
            and :rc:`path.simplify_threshold`.
        curves : bool, optional
            If True, curve segments will be returned as curve segments.
            If False, all curves will be converted to line segments.
        sketch : None or sequence, optional
            If not None, must be a 3-tuple of the form
            (scale, length, randomness), representing the sketch parameters.
        """
    def iter_bezier(self, **kwargs) -> Generator[Incomplete]:
        """
        Iterate over each B�zier curve (lines included) in a `Path`.

        Parameters
        ----------
        **kwargs
            Forwarded to `.iter_segments`.

        Yields
        ------
        B : `~matplotlib.bezier.BezierSegment`
            The B�zier curves that make up the current path. Note in particular
            that freestanding points are B�zier curves of order 0, and lines
            are B�zier curves of order 1 (with two control points).
        code : `~matplotlib.path.Path.code_type`
            The code describing what kind of curve is being returned.
            `MOVETO`, `LINETO`, `CURVE3`, and `CURVE4` correspond to
            B�zier curves with 1, 2, 3, and 4 control points (respectively).
            `CLOSEPOLY` is a `LINETO` with the control points correctly
            chosen based on the start/end points of the current stroke.
        """
    def _iter_connected_components(self) -> Generator[Incomplete]:
        """Return subpaths split at MOVETOs."""
    def cleaned(self, transform: Incomplete | None = None, remove_nans: bool = False, clip: Incomplete | None = None, *, simplify: bool = False, curves: bool = False, stroke_width: float = 1.0, snap: bool = False, sketch: Incomplete | None = None):
        """
        Return a new `Path` with vertices and codes cleaned according to the
        parameters.

        See Also
        --------
        Path.iter_segments : for details of the keyword arguments.
        """
    def transformed(self, transform):
        """
        Return a transformed copy of the path.

        See Also
        --------
        matplotlib.transforms.TransformedPath
            A specialized path class that will cache the transformed result and
            automatically update when the transform changes.
        """
    def contains_point(self, point, transform: Incomplete | None = None, radius: float = 0.0):
        """
        Return whether the area enclosed by the path contains the given point.

        The path is always treated as closed; i.e. if the last code is not
        `CLOSEPOLY` an implicit segment connecting the last vertex to the first
        vertex is assumed.

        Parameters
        ----------
        point : (float, float)
            The point (x, y) to check.
        transform : `~matplotlib.transforms.Transform`, optional
            If not ``None``, *point* will be compared to ``self`` transformed
            by *transform*; i.e. for a correct check, *transform* should
            transform the path into the coordinate system of *point*.
        radius : float, default: 0
            Additional margin on the path in coordinates of *point*.
            The path is extended tangentially by *radius/2*; i.e. if you would
            draw the path with a linewidth of *radius*, all points on the line
            would still be considered to be contained in the area. Conversely,
            negative values shrink the area: Points on the imaginary line
            will be considered outside the area.

        Returns
        -------
        bool

        Notes
        -----
        The current algorithm has some limitations:

        - The result is undefined for points exactly at the boundary
          (i.e. at the path shifted by *radius/2*).
        - The result is undefined if there is no enclosed area, i.e. all
          vertices are on a straight line.
        - If bounding lines start to cross each other due to *radius* shift,
          the result is not guaranteed to be correct.
        """
    def contains_points(self, points, transform: Incomplete | None = None, radius: float = 0.0):
        """
        Return whether the area enclosed by the path contains the given points.

        The path is always treated as closed; i.e. if the last code is not
        `CLOSEPOLY` an implicit segment connecting the last vertex to the first
        vertex is assumed.

        Parameters
        ----------
        points : (N, 2) array
            The points to check. Columns contain x and y values.
        transform : `~matplotlib.transforms.Transform`, optional
            If not ``None``, *points* will be compared to ``self`` transformed
            by *transform*; i.e. for a correct check, *transform* should
            transform the path into the coordinate system of *points*.
        radius : float, default: 0
            Additional margin on the path in coordinates of *points*.
            The path is extended tangentially by *radius/2*; i.e. if you would
            draw the path with a linewidth of *radius*, all points on the line
            would still be considered to be contained in the area. Conversely,
            negative values shrink the area: Points on the imaginary line
            will be considered outside the area.

        Returns
        -------
        length-N bool array

        Notes
        -----
        The current algorithm has some limitations:

        - The result is undefined for points exactly at the boundary
          (i.e. at the path shifted by *radius/2*).
        - The result is undefined if there is no enclosed area, i.e. all
          vertices are on a straight line.
        - If bounding lines start to cross each other due to *radius* shift,
          the result is not guaranteed to be correct.
        """
    def contains_path(self, path, transform: Incomplete | None = None):
        """
        Return whether this (closed) path completely contains the given path.

        If *transform* is not ``None``, the path will be transformed before
        checking for containment.
        """
    def get_extents(self, transform: Incomplete | None = None, **kwargs):
        """
        Get Bbox of the path.

        Parameters
        ----------
        transform : `~matplotlib.transforms.Transform`, optional
            Transform to apply to path before computing extents, if any.
        **kwargs
            Forwarded to `.iter_bezier`.

        Returns
        -------
        matplotlib.transforms.Bbox
            The extents of the path Bbox([[xmin, ymin], [xmax, ymax]])
        """
    def intersects_path(self, other, filled: bool = True):
        """
        Return whether if this path intersects another given path.

        If *filled* is True, then this also returns True if one path completely
        encloses the other (i.e., the paths are treated as filled).
        """
    def intersects_bbox(self, bbox, filled: bool = True):
        """
        Return whether this path intersects a given `~.transforms.Bbox`.

        If *filled* is True, then this also returns True if the path completely
        encloses the `.Bbox` (i.e., the path is treated as filled).

        The bounding box is always considered filled.
        """
    def interpolated(self, steps):
        """
        Return a new path resampled to length N x *steps*.

        Codes other than `LINETO` are not handled correctly.
        """
    def to_polygons(self, transform: Incomplete | None = None, width: int = 0, height: int = 0, closed_only: bool = True):
        """
        Convert this path to a list of polygons or polylines.  Each
        polygon/polyline is an (N, 2) array of vertices.  In other words,
        each polygon has no `MOVETO` instructions or curves.  This
        is useful for displaying in backends that do not support
        compound paths or B�zier curves.

        If *width* and *height* are both non-zero then the lines will
        be simplified so that vertices outside of (0, 0), (width,
        height) will be clipped.

        The resulting polygons will be simplified if the
        :attr:`Path.should_simplify` attribute of the path is `True`.

        If *closed_only* is `True` (default), only closed polygons,
        with the last point being the same as the first point, will be
        returned.  Any unclosed polylines in the path will be
        explicitly closed.  If *closed_only* is `False`, any unclosed
        polygons in the path will be returned as unclosed polygons,
        and the closed polygons will be returned explicitly closed by
        setting the last point to the same as the first point.
        """
    _unit_rectangle: Incomplete
    @classmethod
    def unit_rectangle(cls):
        """
        Return a `Path` instance of the unit rectangle from (0, 0) to (1, 1).
        """
    _unit_regular_polygons: Incomplete
    @classmethod
    def unit_regular_polygon(cls, numVertices):
        """
        Return a :class:`Path` instance for a unit regular polygon with the
        given *numVertices* such that the circumscribing circle has radius 1.0,
        centered at (0, 0).
        """
    _unit_regular_stars: Incomplete
    @classmethod
    def unit_regular_star(cls, numVertices, innerCircle: float = 0.5):
        """
        Return a :class:`Path` for a unit regular star with the given
        numVertices and radius of 1.0, centered at (0, 0).
        """
    @classmethod
    def unit_regular_asterisk(cls, numVertices):
        """
        Return a :class:`Path` for a unit regular asterisk with the given
        numVertices and radius of 1.0, centered at (0, 0).
        """
    _unit_circle: Incomplete
    @classmethod
    def unit_circle(cls):
        """
        Return the readonly :class:`Path` of the unit circle.

        For most cases, :func:`Path.circle` will be what you want.
        """
    @classmethod
    def circle(cls, center=(0.0, 0.0), radius: float = 1.0, readonly: bool = False):
        '''
        Return a `Path` representing a circle of a given radius and center.

        Parameters
        ----------
        center : (float, float), default: (0, 0)
            The center of the circle.
        radius : float, default: 1
            The radius of the circle.
        readonly : bool
            Whether the created path should have the "readonly" argument
            set when creating the Path instance.

        Notes
        -----
        The circle is approximated using 8 cubic B�zier curves, as described in

          Lancaster, Don.  `Approximating a Circle or an Ellipse Using Four
          Bezier Cubic Splines <https://www.tinaja.com/glib/ellipse4.pdf>`_.
        '''
    _unit_circle_righthalf: Incomplete
    @classmethod
    def unit_circle_righthalf(cls):
        """
        Return a `Path` of the right half of a unit circle.

        See `Path.circle` for the reference on the approximation used.
        """
    @classmethod
    def arc(cls, theta1, theta2, n: Incomplete | None = None, is_wedge: bool = False):
        """
        Return a `Path` for the unit circle arc from angles *theta1* to
        *theta2* (in degrees).

        *theta2* is unwrapped to produce the shortest arc within 360 degrees.
        That is, if *theta2* > *theta1* + 360, the arc will be from *theta1* to
        *theta2* - 360 and not a full circle plus some extra overlap.

        If *n* is provided, it is the number of spline segments to make.
        If *n* is not provided, the number of spline segments is
        determined based on the delta between *theta1* and *theta2*.

           Masionobe, L.  2003.  `Drawing an elliptical arc using
           polylines, quadratic or cubic Bezier curves
           <https://web.archive.org/web/20190318044212/http://www.spaceroots.org/documents/ellipse/index.html>`_.
        """
    @classmethod
    def wedge(cls, theta1, theta2, n: Incomplete | None = None):
        """
        Return a `Path` for the unit circle wedge from angles *theta1* to
        *theta2* (in degrees).

        *theta2* is unwrapped to produce the shortest wedge within 360 degrees.
        That is, if *theta2* > *theta1* + 360, the wedge will be from *theta1*
        to *theta2* - 360 and not a full circle plus some extra overlap.

        If *n* is provided, it is the number of spline segments to make.
        If *n* is not provided, the number of spline segments is
        determined based on the delta between *theta1* and *theta2*.

        See `Path.arc` for the reference on the approximation used.
        """
    @staticmethod
    def hatch(hatchpattern, density: int = 6):
        """
        Given a hatch specifier, *hatchpattern*, generates a `Path` that
        can be used in a repeated hatching pattern.  *density* is the
        number of lines per unit square.
        """
    def clip_to_bbox(self, bbox, inside: bool = True):
        """
        Clip the path to the given bounding box.

        The path must be made up of one or more closed polygons.  This
        algorithm will not behave correctly for unclosed paths.

        If *inside* is `True`, clip to the inside of the box, otherwise
        to the outside of the box.
        """

def get_path_collection_extents(master_transform, paths, transforms, offsets, offset_transform):
    """
    Get bounding box of a `.PathCollection`\\s internal objects.

    That is, given a sequence of `Path`\\s, `.Transform`\\s objects, and offsets, as found
    in a `.PathCollection`, return the bounding box that encapsulates all of them.

    Parameters
    ----------
    master_transform : `~matplotlib.transforms.Transform`
        Global transformation applied to all paths.
    paths : list of `Path`
    transforms : list of `~matplotlib.transforms.Affine2DBase`
        If non-empty, this overrides *master_transform*.
    offsets : (N, 2) array-like
    offset_transform : `~matplotlib.transforms.Affine2DBase`
        Transform applied to the offsets before offsetting the path.

    Notes
    -----
    The way that *paths*, *transforms* and *offsets* are combined follows the same
    method as for collections: each is iterated over independently, so if you have 3
    """
