import matplotlib.artist as martist
import matplotlib.text as mtext
from _typeshed import Incomplete
from matplotlib import _api as _api, _docstring as _docstring
from matplotlib.patches import FancyArrowPatch as FancyArrowPatch, FancyBboxPatch as FancyBboxPatch
from matplotlib.transforms import Bbox as Bbox, BboxBase as BboxBase, TransformedBbox as TransformedBbox

DEBUG: bool

def _compat_get_offset(meth):
    """
    Decorator for the get_offset method of OffsetBox and subclasses, that
    allows supporting both the new signature (self, bbox, renderer) and the old
    signature (self, width, height, xdescent, ydescent, renderer).
    """
def _bbox_artist(*args, **kwargs) -> None: ...
def _get_packed_offsets(widths, total, sep, mode: str = 'fixed'):
    """
    Pack boxes specified by their *widths*.

    For simplicity of the description, the terminology used here assumes a
    horizontal layout, but the function works equally for a vertical layout.

    There are three packing *mode*\\s:

    - 'fixed': The elements are packed tight to the left with a spacing of
      *sep* in between. If *total* is *None* the returned total will be the
      right edge of the last box. A non-*None* total will be passed unchecked
      to the output. In particular this means that right edge of the last
      box may be further to the right than the returned total.

    - 'expand': Distribute the boxes with equal spacing so that the left edge
      of the first box is at 0, and the right edge of the last box is at
      *total*. The parameter *sep* is ignored in this mode. A total of *None*
      is accepted and considered equal to 1. The total is returned unchanged
      (except for the conversion *None* to 1). If the total is smaller than
      the sum of the widths, the laid out boxes will overlap.

    - 'equal': If *total* is given, the total space is divided in N equal
      ranges and each box is left-aligned within its subspace.
      Otherwise (*total* is *None*), *sep* must be provided and each box is
      left-aligned in its subspace of width ``(max(widths) + sep)``. The
      total width is then calculated to be ``N * (max(widths) + sep)``.

    Parameters
    ----------
    widths : list of float
        Widths of boxes to be packed.
    total : float or None
        Intended total length. *None* if not used.
    sep : float or None
        Spacing between boxes.
    mode : {'fixed', 'expand', 'equal'}
        The packing mode.

    Returns
    -------
    total : float
        The total width needed to accommodate the laid out boxes.
    offsets : array of float
        The left offsets of the boxes.
    """
def _get_aligned_offsets(yspans, height, align: str = 'baseline'):
    '''
    Align boxes each specified by their ``(y0, y1)`` spans.

    For simplicity of the description, the terminology used here assumes a
    horizontal layout (i.e., vertical alignment), but the function works
    equally for a vertical layout.

    Parameters
    ----------
    yspans
        List of (y0, y1) spans of boxes to be aligned.
    height : float or None
        Intended total height. If None, the maximum of the heights
        (``y1 - y0``) in *yspans* is used.
    align : {\'baseline\', \'left\', \'top\', \'right\', \'bottom\', \'center\'}
        The alignment anchor of the boxes.

    Returns
    -------
    (y0, y1)
        y range spanned by the packing.  If a *height* was originally passed
        in, then for all alignments other than "baseline", a span of ``(0,
        height)`` is used without checking that it is actually large enough).
    descent
        The descent of the packing.
    offsets
        The bottom offsets of the boxes.
    '''

class OffsetBox(martist.Artist):
    """
    The OffsetBox is a simple container artist.

    The child artists are meant to be drawn at a relative position to its
    parent.

    Being an artist itself, all parameters are passed on to `.Artist`.
    """
    _children: Incomplete
    _offset: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def set_figure(self, fig) -> None:
        """
        Set the `.Figure` for the `.OffsetBox` and all its children.

        Parameters
        ----------
        fig : `~matplotlib.figure.Figure`
        """
    @martist.Artist.axes.setter
    def axes(self, ax) -> None: ...
    def contains(self, mouseevent):
        """
        Delegate the mouse event contains-check to the children.

        As a container, the `.OffsetBox` does not respond itself to
        mouseevents.

        Parameters
        ----------
        mouseevent : `~matplotlib.backend_bases.MouseEvent`

        Returns
        -------
        contains : bool
            Whether any values are within the radius.
        details : dict
            An artist-specific dictionary of details of the event context,
            such as which points are contained in the pick radius. See the
            individual Artist subclasses for details.

        See Also
        --------
        .Artist.contains
        """
    stale: bool
    def set_offset(self, xy) -> None:
        """
        Set the offset.

        Parameters
        ----------
        xy : (float, float) or callable
            The (x, y) coordinates of the offset in display units. These can
            either be given explicitly as a tuple (x, y), or by providing a
            function that converts the extent into the offset. This function
            must have the signature::

                def offset(width, height, xdescent, ydescent, renderer) -> (float, float)
        """
    def get_offset(self, bbox, renderer):
        """
        Return the offset as a tuple (x, y).

        The extent parameters have to be provided to handle the case where the
        offset is dynamically determined by a callable (see
        `~.OffsetBox.set_offset`).

        Parameters
        ----------
        bbox : `.Bbox`
        renderer : `.RendererBase` subclass
        """
    width: Incomplete
    def set_width(self, width) -> None:
        """
        Set the width of the box.

        Parameters
        ----------
        width : float
        """
    height: Incomplete
    def set_height(self, height) -> None:
        """
        Set the height of the box.

        Parameters
        ----------
        height : float
        """
    def get_visible_children(self):
        """Return a list of the visible child `.Artist`\\s."""
    def get_children(self):
        """Return a list of the child `.Artist`\\s."""
    def _get_bbox_and_child_offsets(self, renderer) -> None:
        """
        Return the bbox of the offsetbox and the child offsets.

        The bbox should satisfy ``x0 <= x1 and y0 <= y1``.

        Parameters
        ----------
        renderer : `.RendererBase` subclass

        Returns
        -------
        bbox
        list of (xoffset, yoffset) pairs
        """
    def get_bbox(self, renderer):
        """Return the bbox of the offsetbox, ignoring parent offsets."""
    def get_window_extent(self, renderer: Incomplete | None = None): ...
    def draw(self, renderer) -> None:
        """
        Update the location of children if necessary and draw them
        to the given *renderer*.
        """

class PackerBase(OffsetBox):
    height: Incomplete
    width: Incomplete
    sep: Incomplete
    pad: Incomplete
    mode: Incomplete
    align: Incomplete
    _children: Incomplete
    def __init__(self, pad: float = 0.0, sep: float = 0.0, width: Incomplete | None = None, height: Incomplete | None = None, align: str = 'baseline', mode: str = 'fixed', children: Incomplete | None = None) -> None:
        """
        Parameters
        ----------
        pad : float, default: 0.0
            The boundary padding in points.

        sep : float, default: 0.0
            The spacing between items in points.

        width, height : float, optional
            Width and height of the container box in pixels, calculated if
            *None*.

        align : {'top', 'bottom', 'left', 'right', 'center', 'baseline'}, default: 'baseline'
            Alignment of boxes.

        mode : {'fixed', 'expand', 'equal'}, default: 'fixed'
            The packing mode.

            - 'fixed' packs the given `.Artist`\\s tight with *sep* spacing.
            - 'expand' uses the maximal available space to distribute the
              artists with equal spacing in between.
            - 'equal': Each artist an equal fraction of the available space
              and is left-aligned (or top-aligned) therein.

        children : list of `.Artist`
            The artists to pack.

        Notes
        -----
        *pad* and *sep* are in points and will be scaled with the renderer
        dpi, while *width* and *height* are in pixels.
        """

class VPacker(PackerBase):
    """
    VPacker packs its children vertically, automatically adjusting their
    relative positions at draw time.

    .. code-block:: none

       +---------+
       | Child 1 |
       | Child 2 |
       | Child 3 |
       +---------+
    """
    def _get_bbox_and_child_offsets(self, renderer): ...

class HPacker(PackerBase):
    """
    HPacker packs its children horizontally, automatically adjusting their
    relative positions at draw time.

    .. code-block:: none

       +-------------------------------+
       | Child 1    Child 2    Child 3 |
       +-------------------------------+
    """
    def _get_bbox_and_child_offsets(self, renderer): ...

class PaddedBox(OffsetBox):
    """
    A container to add a padding around an `.Artist`.

    The `.PaddedBox` contains a `.FancyBboxPatch` that is used to visualize
    it when rendering.

    .. code-block:: none

       +----------------------------+
       |                            |
       |                            |
       |                            |
       | <--pad--> Artist           |
       |             ^              |
       |            pad             |
       |             v              |
       +----------------------------+

    Attributes
    ----------
    pad : float
        The padding in points.
    patch : `.FancyBboxPatch`
        When *draw_frame* is True, this `.FancyBboxPatch` is made visible and
        creates a border around the box.
    """
    pad: Incomplete
    _children: Incomplete
    patch: Incomplete
    def __init__(self, child, pad: float = 0.0, *, draw_frame: bool = False, patch_attrs: Incomplete | None = None) -> None:
        """
        Parameters
        ----------
        child : `~matplotlib.artist.Artist`
            The contained `.Artist`.
        pad : float, default: 0.0
            The padding in points. This will be scaled with the renderer dpi.
            In contrast, *width* and *height* are in *pixels* and thus not
            scaled.
        draw_frame : bool
            Whether to draw the contained `.FancyBboxPatch`.
        patch_attrs : dict or None
            Additional parameters passed to the contained `.FancyBboxPatch`.
        """
    def _get_bbox_and_child_offsets(self, renderer): ...
    stale: bool
    def draw(self, renderer) -> None: ...
    def update_frame(self, bbox, fontsize: Incomplete | None = None) -> None: ...
    def draw_frame(self, renderer) -> None: ...

class DrawingArea(OffsetBox):
    """
    The DrawingArea can contain any Artist as a child. The DrawingArea
    has a fixed width and height. The position of children relative to
    the parent is fixed. The children can be clipped at the
    boundaries of the parent.
    """
    width: Incomplete
    height: Incomplete
    xdescent: Incomplete
    ydescent: Incomplete
    _clip_children: Incomplete
    offset_transform: Incomplete
    dpi_transform: Incomplete
    def __init__(self, width, height, xdescent: float = 0.0, ydescent: float = 0.0, clip: bool = False) -> None:
        """
        Parameters
        ----------
        width, height : float
            Width and height of the container box.
        xdescent, ydescent : float
            Descent of the box in x- and y-direction.
        clip : bool
            Whether to clip the children to the box.
        """
    @property
    def clip_children(self):
        """
        If the children of this DrawingArea should be clipped
        by DrawingArea bounding box.
        """
    stale: bool
    @clip_children.setter
    def clip_children(self, val) -> None: ...
    def get_transform(self):
        """
        Return the `~matplotlib.transforms.Transform` applied to the children.
        """
    def set_transform(self, t) -> None:
        """
        set_transform is ignored.
        """
    _offset: Incomplete
    def set_offset(self, xy) -> None:
        """
        Set the offset of the container.

        Parameters
        ----------
        xy : (float, float)
            The (x, y) coordinates of the offset in display units.
        """
    def get_offset(self):
        """Return offset of the container."""
    def get_bbox(self, renderer): ...
    def add_artist(self, a) -> None:
        """Add an `.Artist` to the container box."""
    def draw(self, renderer) -> None: ...

class TextArea(OffsetBox):
    """
    The TextArea is a container artist for a single Text instance.

    The text is placed at (0, 0) with baseline+left alignment, by default. The
    width and height of the TextArea instance is the width and height of its
    child text.
    """
    _text: Incomplete
    _children: Incomplete
    offset_transform: Incomplete
    _baseline_transform: Incomplete
    _multilinebaseline: Incomplete
    def __init__(self, s, *, textprops: Incomplete | None = None, multilinebaseline: bool = False) -> None:
        """
        Parameters
        ----------
        s : str
            The text to be displayed.
        textprops : dict, default: {}
            Dictionary of keyword parameters to be passed to the `.Text`
            instance in the TextArea.
        multilinebaseline : bool, default: False
            Whether the baseline for multiline text is adjusted so that it
            is (approximately) center-aligned with single-line text.
        """
    stale: bool
    def set_text(self, s) -> None:
        """Set the text of this area as a string."""
    def get_text(self):
        """Return the string representation of this area's text."""
    def set_multilinebaseline(self, t) -> None:
        '''
        Set multilinebaseline.

        If True, the baseline for multiline text is adjusted so that it is
        (approximately) center-aligned with single-line text.  This is used
        e.g. by the legend implementation so that single-line labels are
        baseline-aligned, but multiline labels are "center"-aligned with them.
        '''
    def get_multilinebaseline(self):
        """
        Get multilinebaseline.
        """
    def set_transform(self, t) -> None:
        """
        set_transform is ignored.
        """
    _offset: Incomplete
    def set_offset(self, xy) -> None:
        """
        Set the offset of the container.

        Parameters
        ----------
        xy : (float, float)
            The (x, y) coordinates of the offset in display units.
        """
    def get_offset(self):
        """Return offset of the container."""
    def get_bbox(self, renderer): ...
    def draw(self, renderer) -> None: ...

class AuxTransformBox(OffsetBox):
    """
    Offset Box with the aux_transform. Its children will be
    transformed with the aux_transform first then will be
    offsetted. The absolute coordinate of the aux_transform is meaning
    as it will be automatically adjust so that the left-lower corner
    of the bounding box of children will be set to (0, 0) before the
    offset transform.

    It is similar to drawing area, except that the extent of the box
    is not predetermined but calculated from the window extent of its
    children. Furthermore, the extent of the children will be
    calculated in the transformed coordinate.
    """
    aux_transform: Incomplete
    offset_transform: Incomplete
    ref_offset_transform: Incomplete
    def __init__(self, aux_transform) -> None: ...
    stale: bool
    def add_artist(self, a) -> None:
        """Add an `.Artist` to the container box."""
    def get_transform(self):
        """
        Return the :class:`~matplotlib.transforms.Transform` applied
        to the children
        """
    def set_transform(self, t) -> None:
        """
        set_transform is ignored.
        """
    _offset: Incomplete
    def set_offset(self, xy) -> None:
        """
        Set the offset of the container.

        Parameters
        ----------
        xy : (float, float)
            The (x, y) coordinates of the offset in display units.
        """
    def get_offset(self):
        """Return offset of the container."""
    def get_bbox(self, renderer): ...
    def draw(self, renderer) -> None: ...

class AnchoredOffsetbox(OffsetBox):
    """
    An offset box placed according to location *loc*.

    AnchoredOffsetbox has a single child.  When multiple children are needed,
    use an extra OffsetBox to enclose them.  By default, the offset box is
    anchored against its parent Axes. You may explicitly specify the
    *bbox_to_anchor*.
    """
    zorder: int
    codes: Incomplete
    loc: Incomplete
    borderpad: Incomplete
    pad: Incomplete
    prop: Incomplete
    patch: Incomplete
    def __init__(self, loc, *, pad: float = 0.4, borderpad: float = 0.5, child: Incomplete | None = None, prop: Incomplete | None = None, frameon: bool = True, bbox_to_anchor: Incomplete | None = None, bbox_transform: Incomplete | None = None, **kwargs) -> None:
        """
        Parameters
        ----------
        loc : str
            The box location.  Valid locations are
            'upper left', 'upper center', 'upper right',
            'center left', 'center', 'center right',
            'lower left', 'lower center', 'lower right'.
            For backward compatibility, numeric values are accepted as well.
            See the parameter *loc* of `.Legend` for details.
        pad : float, default: 0.4
            Padding around the child as fraction of the fontsize.
        borderpad : float, default: 0.5
            Padding between the offsetbox frame and the *bbox_to_anchor*.
        child : `.OffsetBox`
            The box that will be anchored.
        prop : `.FontProperties`
            This is only used as a reference for paddings. If not given,
            :rc:`legend.fontsize` is used.
        frameon : bool
            Whether to draw a frame around the box.
        bbox_to_anchor : `.BboxBase`, 2-tuple, or 4-tuple of floats
            Box that is used to position the legend in conjunction with *loc*.
        bbox_transform : None or :class:`matplotlib.transforms.Transform`
            The transform for the bounding box (*bbox_to_anchor*).
        **kwargs
            All other parameters are passed on to `.OffsetBox`.

        Notes
        -----
        See `.Legend` for a detailed description of the anchoring mechanism.
        """
    _child: Incomplete
    stale: bool
    def set_child(self, child) -> None:
        """Set the child to be anchored."""
    def get_child(self):
        """Return the child."""
    def get_children(self):
        """Return the list of children."""
    def get_bbox(self, renderer): ...
    def get_bbox_to_anchor(self):
        """Return the bbox that the box is anchored to."""
    _bbox_to_anchor: Incomplete
    _bbox_to_anchor_transform: Incomplete
    def set_bbox_to_anchor(self, bbox, transform: Incomplete | None = None) -> None:
        """
        Set the bbox that the box is anchored to.

        *bbox* can be a Bbox instance, a list of [left, bottom, width,
        height], or a list of [left, bottom] where the width and
        height will be assumed to be zero. The bbox will be
        transformed to display coordinate by the given transform.
        """
    def get_offset(self, bbox, renderer): ...
    def update_frame(self, bbox, fontsize: Incomplete | None = None) -> None: ...
    def draw(self, renderer) -> None: ...

def _get_anchored_bbox(loc, bbox, parentbbox, borderpad):
    """
    Return the (x, y) position of the *bbox* anchored at the *parentbbox* with
    the *loc* code with the *borderpad*.
    """

class AnchoredText(AnchoredOffsetbox):
    """
    AnchoredOffsetbox with Text.
    """
    txt: Incomplete
    def __init__(self, s, loc, *, pad: float = 0.4, borderpad: float = 0.5, prop: Incomplete | None = None, **kwargs) -> None:
        """
        Parameters
        ----------
        s : str
            Text.

        loc : str
            Location code. See `AnchoredOffsetbox`.

        pad : float, default: 0.4
            Padding around the text as fraction of the fontsize.

        borderpad : float, default: 0.5
            Spacing between the offsetbox frame and the *bbox_to_anchor*.

        prop : dict, optional
            Dictionary of keyword parameters to be passed to the
            `~matplotlib.text.Text` instance contained inside AnchoredText.

        **kwargs
            All other parameters are passed to `AnchoredOffsetbox`.
        """

class OffsetImage(OffsetBox):
    _dpi_cor: Incomplete
    image: Incomplete
    _children: Incomplete
    def __init__(self, arr, *, zoom: int = 1, cmap: Incomplete | None = None, norm: Incomplete | None = None, interpolation: Incomplete | None = None, origin: Incomplete | None = None, filternorm: bool = True, filterrad: float = 4.0, resample: bool = False, dpi_cor: bool = True, **kwargs) -> None: ...
    _data: Incomplete
    stale: bool
    def set_data(self, arr) -> None: ...
    def get_data(self): ...
    _zoom: Incomplete
    def set_zoom(self, zoom) -> None: ...
    def get_zoom(self): ...
    def get_offset(self):
        """Return offset of the container."""
    def get_children(self): ...
    def get_bbox(self, renderer): ...
    def draw(self, renderer) -> None: ...

class AnnotationBbox(martist.Artist, mtext._AnnotationBase):
    """
    Container for an `OffsetBox` referring to a specific position *xy*.

    Optionally an arrow pointing from the offsetbox to *xy* can be drawn.

    This is like `.Annotation`, but with `OffsetBox` instead of `.Text`.
    """
    zorder: int
    def __str__(self) -> str: ...
    offsetbox: Incomplete
    arrowprops: Incomplete
    xybox: Incomplete
    boxcoords: Incomplete
    _box_alignment: Incomplete
    _arrow_relpos: Incomplete
    arrow_patch: Incomplete
    patch: Incomplete
    def __init__(self, offsetbox, xy, xybox: Incomplete | None = None, xycoords: str = 'data', boxcoords: Incomplete | None = None, *, frameon: bool = True, pad: float = 0.4, annotation_clip: Incomplete | None = None, box_alignment=(0.5, 0.5), bboxprops: Incomplete | None = None, arrowprops: Incomplete | None = None, fontsize: Incomplete | None = None, **kwargs) -> None:
        """
        Parameters
        ----------
        offsetbox : `OffsetBox`

        xy : (float, float)
            The point *(x, y)* to annotate. The coordinate system is determined
            by *xycoords*.

        xybox : (float, float), default: *xy*
            The position *(x, y)* to place the text at. The coordinate system
            is determined by *boxcoords*.

        xycoords : single or two-tuple of str or `.Artist` or `.Transform` or callable, default: 'data'
            The coordinate system that *xy* is given in. See the parameter
            *xycoords* in `.Annotation` for a detailed description.

        boxcoords : single or two-tuple of str or `.Artist` or `.Transform` or callable, default: value of *xycoords*
            The coordinate system that *xybox* is given in. See the parameter
            *textcoords* in `.Annotation` for a detailed description.

        frameon : bool, default: True
            By default, the text is surrounded by a white `.FancyBboxPatch`
            (accessible as the ``patch`` attribute of the `.AnnotationBbox`).
            If *frameon* is set to False, this patch is made invisible.

        annotation_clip: bool or None, default: None
            Whether to clip (i.e. not draw) the annotation when the annotation
            point *xy* is outside the Axes area.

            - If *True*, the annotation will be clipped when *xy* is outside
              the Axes.
            - If *False*, the annotation will always be drawn.
            - If *None*, the annotation will be clipped when *xy* is outside
              the Axes and *xycoords* is 'data'.

        pad : float, default: 0.4
            Padding around the offsetbox.

        box_alignment : (float, float)
            A tuple of two floats for a vertical and horizontal alignment of
            the offset box w.r.t. the *boxcoords*.
            The lower-left corner is (0, 0) and upper-right corner is (1, 1).

        bboxprops : dict, optional
            A dictionary of properties to set for the annotation bounding box,
            for example *boxstyle* and *alpha*.  See `.FancyBboxPatch` for
            details.

        arrowprops: dict, optional
            Arrow properties, see `.Annotation` for description.

        fontsize: float or str, optional
            Translated to points and passed as *mutation_scale* into
            `.FancyBboxPatch` to scale attributes of the box style (e.g. pad
            or rounding_size).  The name is chosen in analogy to `.Text` where
            *fontsize* defines the mutation scale as well.  If not given,
            :rc:`legend.fontsize` is used.  See `.Text.set_fontsize` for valid
            values.

        **kwargs
            Other `AnnotationBbox` properties.  See `.AnnotationBbox.set` for
            a list.
        """
    @property
    def xyann(self): ...
    stale: bool
    @xyann.setter
    def xyann(self, xyann) -> None: ...
    @property
    def anncoords(self): ...
    @anncoords.setter
    def anncoords(self, coords) -> None: ...
    def contains(self, mouseevent): ...
    def get_children(self): ...
    def set_figure(self, fig) -> None: ...
    prop: Incomplete
    def set_fontsize(self, s: Incomplete | None = None) -> None:
        """
        Set the fontsize in points.

        If *s* is not given, reset to :rc:`legend.fontsize`.
        """
    def get_fontsize(self):
        """Return the fontsize in points."""
    def get_window_extent(self, renderer: Incomplete | None = None): ...
    def get_tightbbox(self, renderer: Incomplete | None = None): ...
    def update_positions(self, renderer) -> None:
        """Update pixel positions for the annotated point, the text, and the arrow."""
    def draw(self, renderer) -> None: ...

class DraggableBase:
    """
    Helper base class for a draggable artist (legend, offsetbox).

    Derived classes must override the following methods::

        def save_offset(self):
            '''
            Called when the object is picked for dragging; should save the
            reference position of the artist.
            '''

        def update_offset(self, dx, dy):
            '''
            Called during the dragging; (*dx*, *dy*) is the pixel offset from
            the point where the mouse drag started.
            '''

    Optionally, you may override the following method::

        def finalize_offset(self):
            '''Called when the mouse is released.'''

    In the current implementation of `.DraggableLegend` and
    `DraggableAnnotation`, `update_offset` places the artists in display
    coordinates, and `finalize_offset` recalculates their position in axes
    coordinate and set a relevant attribute.
    """
    ref_artist: Incomplete
    got_artist: bool
    _use_blit: Incomplete
    _disconnectors: Incomplete
    def __init__(self, ref_artist, use_blit: bool = False) -> None: ...
    canvas: Incomplete
    cids: Incomplete
    def on_motion(self, evt) -> None: ...
    mouse_x: Incomplete
    mouse_y: Incomplete
    background: Incomplete
    def on_pick(self, evt) -> None: ...
    def on_release(self, event) -> None: ...
    def _check_still_parented(self): ...
    def disconnect(self) -> None:
        """Disconnect the callbacks."""
    def save_offset(self) -> None: ...
    def update_offset(self, dx, dy) -> None: ...
    def finalize_offset(self) -> None: ...

class DraggableOffsetBox(DraggableBase):
    offsetbox: Incomplete
    def __init__(self, ref_artist, offsetbox, use_blit: bool = False) -> None: ...
    def save_offset(self) -> None: ...
    def update_offset(self, dx, dy) -> None: ...
    def get_loc_in_canvas(self): ...

class DraggableAnnotation(DraggableBase):
    annotation: Incomplete
    def __init__(self, annotation, use_blit: bool = False) -> None: ...
    def save_offset(self) -> None: ...
    def update_offset(self, dx, dy) -> None: ...
