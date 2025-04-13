from _typeshed import Incomplete
from collections.abc import Generator
from enum import Enum, IntEnum
from matplotlib import _api as _api, _docstring as _docstring, _tight_bbox as _tight_bbox, cbook as cbook, colors as colors, is_interactive as is_interactive, rcParams as rcParams, text as text, transforms as transforms, widgets as widgets
from matplotlib._enums import CapStyle as CapStyle, JoinStyle as JoinStyle
from matplotlib._pylab_helpers import Gcf as Gcf
from matplotlib.backend_managers import ToolManager as ToolManager
from matplotlib.cbook import _setattr_cm as _setattr_cm
from matplotlib.layout_engine import ConstrainedLayoutEngine as ConstrainedLayoutEngine
from matplotlib.path import Path as Path
from matplotlib.texmanager import TexManager as TexManager
from matplotlib.transforms import Affine2D as Affine2D
from typing import NamedTuple

_log: Incomplete
_default_filetypes: Incomplete
_default_backends: Incomplete

def register_backend(format, backend, description: Incomplete | None = None) -> None:
    '''
    Register a backend for saving to a given file format.

    Parameters
    ----------
    format : str
        File extension
    backend : module string or canvas class
        Backend for handling file output
    description : str, default: ""
        Description of the file type.
    '''
def get_registered_canvas_class(format):
    """
    Return the registered default canvas for given file format.
    Handles deferred import of required backend.
    """

class RendererBase:
    """
    An abstract base class to handle drawing/rendering operations.

    The following methods must be implemented in the backend for full
    functionality (though just implementing `draw_path` alone would give a
    highly capable backend):

    * `draw_path`
    * `draw_image`
    * `draw_gouraud_triangles`

    The following methods *should* be implemented in the backend for
    optimization reasons:

    * `draw_text`
    * `draw_markers`
    * `draw_path_collection`
    * `draw_quad_mesh`
    """
    _texmanager: Incomplete
    _text2path: Incomplete
    _raster_depth: int
    _rasterizing: bool
    def __init__(self) -> None: ...
    def open_group(self, s, gid: Incomplete | None = None) -> None:
        """
        Open a grouping element with label *s* and *gid* (if set) as id.

        Only used by the SVG renderer.
        """
    def close_group(self, s) -> None:
        """
        Close a grouping element with label *s*.

        Only used by the SVG renderer.
        """
    def draw_path(self, gc, path, transform, rgbFace: Incomplete | None = None) -> None:
        """Draw a `~.path.Path` instance using the given affine transform."""
    def draw_markers(self, gc, marker_path, marker_trans, path, trans, rgbFace: Incomplete | None = None) -> None:
        """
        Draw a marker at each of *path*'s vertices (excluding control points).

        The base (fallback) implementation makes multiple calls to `draw_path`.
        Backends may want to override this method in order to draw the marker
        only once and reuse it multiple times.

        Parameters
        ----------
        gc : `.GraphicsContextBase`
            The graphics context.
        marker_path : `~matplotlib.path.Path`
            The path for the marker.
        marker_trans : `~matplotlib.transforms.Transform`
            An affine transform applied to the marker.
        path : `~matplotlib.path.Path`
            The locations to draw the markers.
        trans : `~matplotlib.transforms.Transform`
            An affine transform applied to the path.
        rgbFace : :mpltype:`color`, optional
        """
    def draw_path_collection(self, gc, master_transform, paths, all_transforms, offsets, offset_trans, facecolors, edgecolors, linewidths, linestyles, antialiaseds, urls, offset_position) -> None:
        """
        Draw a collection of *paths*.

        Each path is first transformed by the corresponding entry
        in *all_transforms* (a list of (3, 3) matrices) and then by
        *master_transform*.  They are then translated by the corresponding
        entry in *offsets*, which has been first transformed by *offset_trans*.

        *facecolors*, *edgecolors*, *linewidths*, *linestyles*, and
        *antialiased* are lists that set the corresponding properties.

        *offset_position* is unused now, but the argument is kept for
        backwards compatibility.

        The base (fallback) implementation makes multiple calls to `draw_path`.
        Backends may want to override this in order to render each set of
        path data only once, and then reference that path multiple times with
        the different offsets, colors, styles etc.  The generator methods
        `_iter_collection_raw_paths` and `_iter_collection` are provided to
        help with (and standardize) the implementation across backends.  It
        is highly recommended to use those generators, so that changes to the
        behavior of `draw_path_collection` can be made globally.
        """
    def draw_quad_mesh(self, gc, master_transform, meshWidth, meshHeight, coordinates, offsets, offsetTrans, facecolors, antialiased, edgecolors):
        """
        Draw a quadmesh.

        The base (fallback) implementation converts the quadmesh to paths and
        then calls `draw_path_collection`.
        """
    def draw_gouraud_triangles(self, gc, triangles_array, colors_array, transform) -> None:
        """
        Draw a series of Gouraud triangles.

        Parameters
        ----------
        gc : `.GraphicsContextBase`
            The graphics context.
        triangles_array : (N, 3, 2) array-like
            Array of *N* (x, y) points for the triangles.
        colors_array : (N, 3, 4) array-like
            Array of *N* RGBA colors for each point of the triangles.
        transform : `~matplotlib.transforms.Transform`
            An affine transform to apply to the points.
        """
    def _iter_collection_raw_paths(self, master_transform, paths, all_transforms) -> Generator[Incomplete]:
        """
        Helper method (along with `_iter_collection`) to implement
        `draw_path_collection` in a memory-efficient manner.

        This method yields all of the base path/transform combinations, given a
        master transform, a list of paths and list of transforms.

        The arguments should be exactly what is passed in to
        `draw_path_collection`.

        The backend should take each yielded path and transform and create an
        object that can be referenced (reused) later.
        """
    def _iter_collection_uses_per_path(self, paths, all_transforms, offsets, facecolors, edgecolors):
        """
        Compute how many times each raw path object returned by
        `_iter_collection_raw_paths` would be used when calling
        `_iter_collection`. This is intended for the backend to decide
        on the tradeoff between using the paths in-line and storing
        them once and reusing. Rounds up in case the number of uses
        is not the same for every path.
        """
    def _iter_collection(self, gc, path_ids, offsets, offset_trans, facecolors, edgecolors, linewidths, linestyles, antialiaseds, urls, offset_position) -> Generator[Incomplete, None, Incomplete]:
        """
        Helper method (along with `_iter_collection_raw_paths`) to implement
        `draw_path_collection` in a memory-efficient manner.

        This method yields all of the path, offset and graphics context
        combinations to draw the path collection.  The caller should already
        have looped over the results of `_iter_collection_raw_paths` to draw
        this collection.

        The arguments should be the same as that passed into
        `draw_path_collection`, with the exception of *path_ids*, which is a
        list of arbitrary objects that the backend will use to reference one of
        the paths created in the `_iter_collection_raw_paths` stage.

        Each yielded result is of the form::

           xo, yo, path_id, gc, rgbFace

        where *xo*, *yo* is an offset; *path_id* is one of the elements of
        *path_ids*; *gc* is a graphics context and *rgbFace* is a color to
        use for filling the path.
        """
    def get_image_magnification(self):
        """
        Get the factor by which to magnify images passed to `draw_image`.
        Allows a backend to have images at a different resolution to other
        artists.
        """
    def draw_image(self, gc, x, y, im, transform: Incomplete | None = None) -> None:
        """
        Draw an RGBA image.

        Parameters
        ----------
        gc : `.GraphicsContextBase`
            A graphics context with clipping information.

        x : float
            The distance in physical units (i.e., dots or pixels) from the left
            hand side of the canvas.

        y : float
            The distance in physical units (i.e., dots or pixels) from the
            bottom side of the canvas.

        im : (N, M, 4) array of `numpy.uint8`
            An array of RGBA pixels.

        transform : `~matplotlib.transforms.Affine2DBase`
            If and only if the concrete backend is written such that
            `option_scale_image` returns ``True``, an affine transformation
            (i.e., an `.Affine2DBase`) *may* be passed to `draw_image`.  The
            translation vector of the transformation is given in physical units
            (i.e., dots or pixels). Note that the transformation does not
            override *x* and *y*, and has to be applied *before* translating
            the result by *x* and *y* (this can be accomplished by adding *x*
            and *y* to the translation vector defined by *transform*).
        """
    def option_image_nocomposite(self):
        '''
        Return whether image composition by Matplotlib should be skipped.

        Raster backends should usually return False (letting the C-level
        rasterizer take care of image composition); vector backends should
        usually return ``not rcParams["image.composite_image"]``.
        '''
    def option_scale_image(self):
        """
        Return whether arbitrary affine transformations in `draw_image` are
        supported (True for most vector backends).
        """
    def draw_tex(self, gc, x, y, s, prop, angle, *, mtext: Incomplete | None = None) -> None:
        """
        Draw a TeX instance.

        Parameters
        ----------
        gc : `.GraphicsContextBase`
            The graphics context.
        x : float
            The x location of the text in display coords.
        y : float
            The y location of the text baseline in display coords.
        s : str
            The TeX text string.
        prop : `~matplotlib.font_manager.FontProperties`
            The font properties.
        angle : float
            The rotation angle in degrees anti-clockwise.
        mtext : `~matplotlib.text.Text`
            The original text object to be rendered.
        """
    def draw_text(self, gc, x, y, s, prop, angle, ismath: bool = False, mtext: Incomplete | None = None) -> None:
        '''
        Draw a text instance.

        Parameters
        ----------
        gc : `.GraphicsContextBase`
            The graphics context.
        x : float
            The x location of the text in display coords.
        y : float
            The y location of the text baseline in display coords.
        s : str
            The text string.
        prop : `~matplotlib.font_manager.FontProperties`
            The font properties.
        angle : float
            The rotation angle in degrees anti-clockwise.
        ismath : bool or "TeX"
            If True, use mathtext parser.
        mtext : `~matplotlib.text.Text`
            The original text object to be rendered.

        Notes
        -----
        **Notes for backend implementers:**

        `.RendererBase.draw_text` also supports passing "TeX" to the *ismath*
        parameter to use TeX rendering, but this is not required for actual
        rendering backends, and indeed many builtin backends do not support
        this.  Rather, TeX rendering is provided by `~.RendererBase.draw_tex`.
        '''
    def _draw_text_as_path(self, gc, x, y, s, prop, angle, ismath) -> None:
        '''
        Draw the text by converting them to paths using `.TextToPath`.

        This private helper supports the same parameters as
        `~.RendererBase.draw_text`; setting *ismath* to "TeX" triggers TeX
        rendering.
        '''
    def get_text_width_height_descent(self, s, prop, ismath):
        """
        Get the width, height, and descent (offset from the bottom to the baseline), in
        display coords, of the string *s* with `.FontProperties` *prop*.

        Whitespace at the start and the end of *s* is included in the reported width.
        """
    def flipy(self):
        """
        Return whether y values increase from top to bottom.

        Note that this only affects drawing of texts.
        """
    def get_canvas_width_height(self):
        """Return the canvas width and height in display coords."""
    def get_texmanager(self):
        """Return the `.TexManager` instance."""
    def new_gc(self):
        """Return an instance of a `.GraphicsContextBase`."""
    def points_to_pixels(self, points):
        """
        Convert points to display units.

        You need to override this function (unless your backend
        doesn't have a dpi, e.g., postscript or svg).  Some imaging
        systems assume some value for pixels per inch::

            points to pixels = points * pixels_per_inch/72 * dpi/72

        Parameters
        ----------
        points : float or array-like

        Returns
        -------
        Points converted to pixels
        """
    def start_rasterizing(self) -> None:
        """
        Switch to the raster renderer.

        Used by `.MixedModeRenderer`.
        """
    def stop_rasterizing(self) -> None:
        """
        Switch back to the vector renderer and draw the contents of the raster
        renderer as an image on the vector renderer.

        Used by `.MixedModeRenderer`.
        """
    def start_filter(self) -> None:
        """
        Switch to a temporary renderer for image filtering effects.

        Currently only supported by the agg renderer.
        """
    def stop_filter(self, filter_func) -> None:
        """
        Switch back to the original renderer.  The contents of the temporary
        renderer is processed with the *filter_func* and is drawn on the
        original renderer as an image.

        Currently only supported by the agg renderer.
        """
    def _draw_disabled(self):
        """
        Context manager to temporary disable drawing.

        This is used for getting the drawn size of Artists.  This lets us
        run the draw process to update any Python state but does not pay the
        cost of the draw_XYZ calls on the canvas.
        """

class GraphicsContextBase:
    """An abstract base class that provides color, line styles, etc."""
    _alpha: float
    _forced_alpha: bool
    _antialiased: int
    _capstyle: Incomplete
    _cliprect: Incomplete
    _clippath: Incomplete
    _dashes: Incomplete
    _joinstyle: Incomplete
    _linestyle: str
    _linewidth: int
    _rgb: Incomplete
    _hatch: Incomplete
    _hatch_color: Incomplete
    _hatch_linewidth: Incomplete
    _url: Incomplete
    _gid: Incomplete
    _snap: Incomplete
    _sketch: Incomplete
    def __init__(self) -> None: ...
    def copy_properties(self, gc) -> None:
        """Copy properties from *gc* to self."""
    def restore(self) -> None:
        """
        Restore the graphics context from the stack - needed only
        for backends that save graphics contexts on a stack.
        """
    def get_alpha(self):
        """
        Return the alpha value used for blending - not supported on all
        backends.
        """
    def get_antialiased(self):
        """Return whether the object should try to do antialiased rendering."""
    def get_capstyle(self):
        """Return the `.CapStyle`."""
    def get_clip_rectangle(self):
        """
        Return the clip rectangle as a `~matplotlib.transforms.Bbox` instance.
        """
    def get_clip_path(self):
        """
        Return the clip path in the form (path, transform), where path
        is a `~.path.Path` instance, and transform is
        an affine transform to apply to the path before clipping.
        """
    def get_dashes(self):
        """
        Return the dash style as an (offset, dash-list) pair.

        See `.set_dashes` for details.

        Default value is (None, None).
        """
    def get_forced_alpha(self):
        """
        Return whether the value given by get_alpha() should be used to
        override any other alpha-channel values.
        """
    def get_joinstyle(self):
        """Return the `.JoinStyle`."""
    def get_linewidth(self):
        """Return the line width in points."""
    def get_rgb(self):
        """Return a tuple of three or four floats from 0-1."""
    def get_url(self):
        """Return a url if one is set, None otherwise."""
    def get_gid(self):
        """Return the object identifier if one is set, None otherwise."""
    def get_snap(self):
        """
        Return the snap setting, which can be:

        * True: snap vertices to the nearest pixel center
        * False: leave vertices as-is
        * None: (auto) If the path contains only rectilinear line segments,
          round to the nearest pixel center
        """
    def set_alpha(self, alpha) -> None:
        """
        Set the alpha value used for blending - not supported on all backends.

        If ``alpha=None`` (the default), the alpha components of the
        foreground and fill colors will be used to set their respective
        transparencies (where applicable); otherwise, ``alpha`` will override
        them.
        """
    def set_antialiased(self, b) -> None:
        """Set whether object should be drawn with antialiased rendering."""
    def set_capstyle(self, cs) -> None:
        """
        Set how to draw endpoints of lines.

        Parameters
        ----------
        cs : `.CapStyle` or %(CapStyle)s
        """
    def set_clip_rectangle(self, rectangle) -> None:
        """Set the clip rectangle to a `.Bbox` or None."""
    def set_clip_path(self, path) -> None:
        """Set the clip path to a `.TransformedPath` or None."""
    def set_dashes(self, dash_offset, dash_list) -> None:
        """
        Set the dash style for the gc.

        Parameters
        ----------
        dash_offset : float
            Distance, in points, into the dash pattern at which to
            start the pattern. It is usually set to 0.
        dash_list : array-like or None
            The on-off sequence as points.  None specifies a solid line. All
            values must otherwise be non-negative (:math:`\\ge 0`).

        Notes
        -----
        See p. 666 of the PostScript
        `Language Reference
        <https://www.adobe.com/jp/print/postscript/pdfs/PLRM.pdf>`_
        for more info.
        """
    def set_foreground(self, fg, isRGBA: bool = False) -> None:
        """
        Set the foreground color.

        Parameters
        ----------
        fg : :mpltype:`color`
        isRGBA : bool
            If *fg* is known to be an ``(r, g, b, a)`` tuple, *isRGBA* can be
            set to True to improve performance.
        """
    def set_joinstyle(self, js) -> None:
        """
        Set how to draw connections between line segments.

        Parameters
        ----------
        js : `.JoinStyle` or %(JoinStyle)s
        """
    def set_linewidth(self, w) -> None:
        """Set the linewidth in points."""
    def set_url(self, url) -> None:
        """Set the url for links in compatible backends."""
    def set_gid(self, id) -> None:
        """Set the id."""
    def set_snap(self, snap) -> None:
        """
        Set the snap setting which may be:

        * True: snap vertices to the nearest pixel center
        * False: leave vertices as-is
        * None: (auto) If the path contains only rectilinear line segments,
          round to the nearest pixel center
        """
    def set_hatch(self, hatch) -> None:
        """Set the hatch style (for fills)."""
    def get_hatch(self):
        """Get the current hatch style."""
    def get_hatch_path(self, density: float = 6.0):
        """Return a `.Path` for the current hatch."""
    def get_hatch_color(self):
        """Get the hatch color."""
    def set_hatch_color(self, hatch_color) -> None:
        """Set the hatch color."""
    def get_hatch_linewidth(self):
        """Get the hatch linewidth."""
    def set_hatch_linewidth(self, hatch_linewidth) -> None:
        """Set the hatch linewidth."""
    def get_sketch_params(self):
        """
        Return the sketch parameters for the artist.

        Returns
        -------
        tuple or `None`

            A 3-tuple with the following elements:

            * ``scale``: The amplitude of the wiggle perpendicular to the
              source line.
            * ``length``: The length of the wiggle along the line.
            * ``randomness``: The scale factor by which the length is
              shrunken or expanded.

            May return `None` if no sketch parameters were set.
        """
    def set_sketch_params(self, scale: Incomplete | None = None, length: Incomplete | None = None, randomness: Incomplete | None = None) -> None:
        """
        Set the sketch parameters.

        Parameters
        ----------
        scale : float, optional
            The amplitude of the wiggle perpendicular to the source line, in
            pixels.  If scale is `None`, or not provided, no sketch filter will
            be provided.
        length : float, default: 128
            The length of the wiggle along the line, in pixels.
        randomness : float, default: 16
            The scale factor by which the length is shrunken or expanded.
        """

class TimerBase:
    """
    A base class for providing timer events, useful for things animations.
    Backends need to implement a few specific methods in order to use their
    own timing mechanisms so that the timer events are integrated into their
    event loops.

    Subclasses must override the following methods:

    - ``_timer_start``: Backend-specific code for starting the timer.
    - ``_timer_stop``: Backend-specific code for stopping the timer.

    Subclasses may additionally override the following methods:

    - ``_timer_set_single_shot``: Code for setting the timer to single shot
      operating mode, if supported by the timer object.  If not, the `Timer`
      class itself will store the flag and the ``_on_timer`` method should be
      overridden to support such behavior.

    - ``_timer_set_interval``: Code for setting the interval on the timer, if
      there is a method for doing so on the timer object.

    - ``_on_timer``: The internal function that any timer object should call,
      which will handle the task of running all callbacks that have been set.
    """
    callbacks: Incomplete
    def __init__(self, interval: Incomplete | None = None, callbacks: Incomplete | None = None) -> None:
        """
        Parameters
        ----------
        interval : int, default: 1000ms
            The time between timer events in milliseconds.  Will be stored as
            ``timer.interval``.
        callbacks : list[tuple[callable, tuple, dict]]
            List of (func, args, kwargs) tuples that will be called upon timer
            events.  This list is accessible as ``timer.callbacks`` and can be
            manipulated directly, or the functions `~.TimerBase.add_callback`
            and `~.TimerBase.remove_callback` can be used.
        """
    def __del__(self) -> None:
        """Need to stop timer and possibly disconnect timer."""
    def start(self, interval: Incomplete | None = None) -> None:
        """
        Start the timer object.

        Parameters
        ----------
        interval : int, optional
            Timer interval in milliseconds; overrides a previously set interval
            if provided.
        """
    def stop(self) -> None:
        """Stop the timer."""
    def _timer_start(self) -> None: ...
    def _timer_stop(self) -> None: ...
    @property
    def interval(self):
        """The time between timer events, in milliseconds."""
    _interval: Incomplete
    @interval.setter
    def interval(self, interval) -> None: ...
    @property
    def single_shot(self):
        """Whether this timer should stop after a single run."""
    _single: Incomplete
    @single_shot.setter
    def single_shot(self, ss) -> None: ...
    def add_callback(self, func, *args, **kwargs):
        """
        Register *func* to be called by timer when the event fires. Any
        additional arguments provided will be passed to *func*.

        This function returns *func*, which makes it possible to use it as a
        decorator.
        """
    def remove_callback(self, func, *args, **kwargs) -> None:
        """
        Remove *func* from list of callbacks.

        *args* and *kwargs* are optional and used to distinguish between copies
        of the same function registered to be called with different arguments.
        This behavior is deprecated.  In the future, ``*args, **kwargs`` won't
        be considered anymore; to keep a specific callback removable by itself,
        pass it to `add_callback` as a `functools.partial` object.
        """
    def _timer_set_interval(self) -> None:
        """Used to set interval on underlying timer object."""
    def _timer_set_single_shot(self) -> None:
        """Used to set single shot on underlying timer object."""
    def _on_timer(self) -> None:
        """
        Runs all function that have been registered as callbacks. Functions
        can return False (or 0) if they should not be called any more. If there
        are no callbacks, the timer is automatically stopped.
        """

class Event:
    """
    A Matplotlib event.

    The following attributes are defined and shown with their default values.
    Subclasses may define additional attributes.

    Attributes
    ----------
    name : str
        The event name.
    canvas : `FigureCanvasBase`
        The backend-specific canvas instance generating the event.
    guiEvent
        The GUI event that triggered the Matplotlib event.
    """
    name: Incomplete
    canvas: Incomplete
    guiEvent: Incomplete
    def __init__(self, name, canvas, guiEvent: Incomplete | None = None) -> None: ...
    def _process(self) -> None:
        """Process this event on ``self.canvas``, then unset ``guiEvent``."""

class DrawEvent(Event):
    """
    An event triggered by a draw operation on the canvas.

    In most backends, callbacks subscribed to this event will be fired after
    the rendering is complete but before the screen is updated. Any extra
    artists drawn to the canvas's renderer will be reflected without an
    explicit call to ``blit``.

    .. warning::

       Calling ``canvas.draw`` and ``canvas.blit`` in these callbacks may
       not be safe with all backends and may cause infinite recursion.

    A DrawEvent has a number of special attributes in addition to those defined
    by the parent `Event` class.

    Attributes
    ----------
    renderer : `RendererBase`
        The renderer for the draw event.
    """
    renderer: Incomplete
    def __init__(self, name, canvas, renderer) -> None: ...

class ResizeEvent(Event):
    """
    An event triggered by a canvas resize.

    A ResizeEvent has a number of special attributes in addition to those
    defined by the parent `Event` class.

    Attributes
    ----------
    width : int
        Width of the canvas in pixels.
    height : int
        Height of the canvas in pixels.
    """
    def __init__(self, name, canvas) -> None: ...

class CloseEvent(Event):
    """An event triggered by a figure being closed."""

class LocationEvent(Event):
    """
    An event that has a screen location.

    A LocationEvent has a number of special attributes in addition to those
    defined by the parent `Event` class.

    Attributes
    ----------
    x, y : int or None
        Event location in pixels from bottom left of canvas.
    inaxes : `~matplotlib.axes.Axes` or None
        The `~.axes.Axes` instance over which the mouse is, if any.
    xdata, ydata : float or None
        Data coordinates of the mouse within *inaxes*, or *None* if the mouse
        is not over an Axes.
    modifiers : frozenset
        The keyboard modifiers currently being pressed (except for KeyEvent).
    """
    _last_axes_ref: Incomplete
    x: Incomplete
    y: Incomplete
    inaxes: Incomplete
    xdata: Incomplete
    ydata: Incomplete
    modifiers: Incomplete
    def __init__(self, name, canvas, x, y, guiEvent: Incomplete | None = None, *, modifiers: Incomplete | None = None) -> None: ...
    def _set_inaxes(self, inaxes, xy: Incomplete | None = None) -> None: ...

class MouseButton(IntEnum):
    LEFT = 1
    MIDDLE = 2
    RIGHT = 3
    BACK = 8
    FORWARD = 9

class MouseEvent(LocationEvent):
    '''
    A mouse event (\'button_press_event\', \'button_release_event\', \'scroll_event\', \'motion_notify_event\').

    A MouseEvent has a number of special attributes in addition to those
    defined by the parent `Event` and `LocationEvent` classes.

    Attributes
    ----------
    button : None or `MouseButton` or {\'up\', \'down\'}
        The button pressed. \'up\' and \'down\' are used for scroll events.

        Note that LEFT and RIGHT actually refer to the "primary" and
        "secondary" buttons, i.e. if the user inverts their left and right
        buttons ("left-handed setting") then the LEFT button will be the one
        physically on the right.

        If this is unset, *name* is "scroll_event", and *step* is nonzero, then
        this will be set to "up" or "down" depending on the sign of *step*.

    buttons : None or frozenset
        For \'motion_notify_event\', the mouse buttons currently being pressed
        (a set of zero or more MouseButtons);
        for other events, None.

        .. note::
           For \'motion_notify_event\', this attribute is more accurate than
           the ``button`` (singular) attribute, which is obtained from the last
           \'button_press_event\' or \'button_release_event\' that occurred within
           the canvas (and thus 1. be wrong if the last change in mouse state
           occurred when the canvas did not have focus, and 2. cannot report
           when multiple buttons are pressed).

           This attribute is not set for \'button_press_event\' and
           \'button_release_event\' because GUI toolkits are inconsistent as to
           whether they report the button state *before* or *after* the
           press/release occurred.

        .. warning::
           On macOS, the Tk backends only report a single button even if
           multiple buttons are pressed.

    key : None or str
        The key pressed when the mouse event triggered, e.g. \'shift\'.
        See `KeyEvent`.

        .. warning::
           This key is currently obtained from the last \'key_press_event\' or
           \'key_release_event\' that occurred within the canvas.  Thus, if the
           last change of keyboard state occurred while the canvas did not have
           focus, this attribute will be wrong.  On the other hand, the
           ``modifiers`` attribute should always be correct, but it can only
           report on modifier keys.

    step : float
        The number of scroll steps (positive for \'up\', negative for \'down\').
        This applies only to \'scroll_event\' and defaults to 0 otherwise.

    dblclick : bool
        Whether the event is a double-click. This applies only to
        \'button_press_event\' and is False otherwise. In particular, it\'s
        not used in \'button_release_event\'.

    Examples
    --------
    ::

        def on_press(event):
            print(\'you pressed\', event.button, event.xdata, event.ydata)

        cid = fig.canvas.mpl_connect(\'button_press_event\', on_press)
    '''
    button: Incomplete
    buttons: Incomplete
    key: Incomplete
    step: Incomplete
    dblclick: Incomplete
    def __init__(self, name, canvas, x, y, button: Incomplete | None = None, key: Incomplete | None = None, step: int = 0, dblclick: bool = False, guiEvent: Incomplete | None = None, *, buttons: Incomplete | None = None, modifiers: Incomplete | None = None) -> None: ...
    def __str__(self) -> str: ...

class PickEvent(Event):
    """
    A pick event.

    This event is fired when the user picks a location on the canvas
    sufficiently close to an artist that has been made pickable with
    `.Artist.set_picker`.

    A PickEvent has a number of special attributes in addition to those defined
    by the parent `Event` class.

    Attributes
    ----------
    mouseevent : `MouseEvent`
        The mouse event that generated the pick.
    artist : `~matplotlib.artist.Artist`
        The picked artist.  Note that artists are not pickable by default
        (see `.Artist.set_picker`).
    other
        Additional attributes may be present depending on the type of the
        picked object; e.g., a `.Line2D` pick may define different extra
        attributes than a `.PatchCollection` pick.

    Examples
    --------
    Bind a function ``on_pick()`` to pick events, that prints the coordinates
    of the picked data point::

        ax.plot(np.rand(100), 'o', picker=5)  # 5 points tolerance

        def on_pick(event):
            line = event.artist
            xdata, ydata = line.get_data()
            ind = event.ind
            print(f'on pick line: {xdata[ind]:.3f}, {ydata[ind]:.3f}')

        cid = fig.canvas.mpl_connect('pick_event', on_pick)
    """
    mouseevent: Incomplete
    artist: Incomplete
    def __init__(self, name, canvas, mouseevent, artist, guiEvent: Incomplete | None = None, **kwargs) -> None: ...

class KeyEvent(LocationEvent):
    '''
    A key event (key press, key release).

    A KeyEvent has a number of special attributes in addition to those defined
    by the parent `Event` and `LocationEvent` classes.

    Attributes
    ----------
    key : None or str
        The key(s) pressed. Could be *None*, a single case sensitive Unicode
        character ("g", "G", "#", etc.), a special key ("control", "shift",
        "f1", "up", etc.) or a combination of the above (e.g., "ctrl+alt+g",
        "ctrl+alt+G").

    Notes
    -----
    Modifier keys will be prefixed to the pressed key and will be in the order
    "ctrl", "alt", "super". The exception to this rule is when the pressed key
    is itself a modifier key, therefore "ctrl+alt" and "alt+control" can both
    be valid key values.

    Examples
    --------
    ::

        def on_key(event):
            print(\'you pressed\', event.key, event.xdata, event.ydata)

        cid = fig.canvas.mpl_connect(\'key_press_event\', on_key)
    '''
    key: Incomplete
    def __init__(self, name, canvas, key, x: int = 0, y: int = 0, guiEvent: Incomplete | None = None) -> None: ...

def _key_handler(event) -> None: ...
def _mouse_handler(event) -> None: ...
def _get_renderer(figure, print_method: Incomplete | None = None):
    """
    Get the renderer that would be used to save a `.Figure`.

    If you need a renderer without any active draw methods use
    renderer._draw_disabled to temporary patch them out at your call site.
    """
def _no_output_draw(figure) -> None: ...
def _is_non_interactive_terminal_ipython(ip):
    """
    Return whether we are in a terminal IPython, but non interactive.

    When in _terminal_ IPython, ip.parent will have and `interact` attribute,
    if this attribute is False we do not setup eventloop integration as the
    user will _not_ interact with IPython. In all other case (ZMQKernel, or is
    interactive), we do.
    """
def _allow_interrupt(prepare_notifier, handle_sigint) -> Generator[None]:
    """
    A context manager that allows terminating a plot by sending a SIGINT.  It
    is necessary because the running backend prevents the Python interpreter
    from running and processing signals (i.e., to raise a KeyboardInterrupt).
    To solve this, one needs to somehow wake up the interpreter and make it
    close the plot window.  We do this by using the signal.set_wakeup_fd()
    function which organizes a write of the signal number into a socketpair.
    A backend-specific function, *prepare_notifier*, arranges to listen to
    the pair's read socket while the event loop is running.  (If it returns a
    notifier object, that object is kept alive while the context manager runs.)

    If SIGINT was indeed caught, after exiting the on_signal() function the
    interpreter reacts to the signal according to the handler function which
    had been set up by a signal.signal() call; here, we arrange to call the
    backend-specific *handle_sigint* function.  Finally, we call the old SIGINT
    handler with the same arguments that were given to our custom handler.

    We do this only if the old handler for SIGINT was not None, which means
    that a non-python handler was installed, i.e. in Julia, and not SIG_IGN
    which means we should ignore the interrupts.

    Parameters
    ----------
    prepare_notifier : Callable[[socket.socket], object]
    handle_sigint : Callable[[], object]
    """

class FigureCanvasBase:
    """
    The canvas the figure renders into.

    Attributes
    ----------
    figure : `~matplotlib.figure.Figure`
        A high-level figure instance.
    """
    required_interactive_framework: Incomplete
    manager_class: Incomplete
    events: Incomplete
    fixed_dpi: Incomplete
    filetypes = _default_filetypes
    def supports_blit(cls):
        """If this Canvas sub-class supports blitting."""
    _is_idle_drawing: bool
    _is_saving: bool
    figure: Incomplete
    manager: Incomplete
    widgetlock: Incomplete
    _button: Incomplete
    _key: Incomplete
    mouse_grabber: Incomplete
    toolbar: Incomplete
    _device_pixel_ratio: int
    def __init__(self, figure: Incomplete | None = None) -> None: ...
    callbacks: Incomplete
    button_pick_id: Incomplete
    scroll_pick_id: Incomplete
    @classmethod
    def _fix_ipython_backend2gui(cls) -> None: ...
    @classmethod
    def new_manager(cls, figure, num):
        """
        Create a new figure manager for *figure*, using this canvas class.

        Notes
        -----
        This method should not be reimplemented in subclasses.  If
        custom manager creation logic is needed, please reimplement
        ``FigureManager.create_with_canvas``.
        """
    def _idle_draw_cntx(self) -> Generator[None]: ...
    def is_saving(self):
        """
        Return whether the renderer is in the process of saving
        to a file, rather than rendering for an on-screen buffer.
        """
    def blit(self, bbox: Incomplete | None = None) -> None:
        """Blit the canvas in bbox (default entire canvas)."""
    def inaxes(self, xy):
        """
        Return the topmost visible `~.axes.Axes` containing the point *xy*.

        Parameters
        ----------
        xy : (float, float)
            (x, y) pixel positions from left/bottom of the canvas.

        Returns
        -------
        `~matplotlib.axes.Axes` or None
            The topmost visible Axes containing the point, or None if there
            is no Axes at the point.
        """
    def grab_mouse(self, ax) -> None:
        """
        Set the child `~.axes.Axes` which is grabbing the mouse events.

        Usually called by the widgets themselves. It is an error to call this
        if the mouse is already grabbed by another Axes.
        """
    def release_mouse(self, ax) -> None:
        """
        Release the mouse grab held by the `~.axes.Axes` *ax*.

        Usually called by the widgets. It is ok to call this even if *ax*
        doesn't have the mouse grab currently.
        """
    def set_cursor(self, cursor) -> None:
        """
        Set the current cursor.

        This may have no effect if the backend does not display anything.

        If required by the backend, this method should trigger an update in
        the backend event loop after the cursor is set, as this method may be
        called e.g. before a long-running task during which the GUI is not
        updated.

        Parameters
        ----------
        cursor : `.Cursors`
            The cursor to display over the canvas. Note: some backends may
            change the cursor for the entire window.
        """
    def draw(self, *args, **kwargs) -> None:
        """
        Render the `.Figure`.

        This method must walk the artist tree, even if no output is produced,
        because it triggers deferred work that users may want to access
        before saving output to disk. For example computing limits,
        auto-limits, and tick values.
        """
    def draw_idle(self, *args, **kwargs) -> None:
        """
        Request a widget redraw once control returns to the GUI event loop.

        Even if multiple calls to `draw_idle` occur before control returns
        to the GUI event loop, the figure will only be rendered once.

        Notes
        -----
        Backends may choose to override the method and implement their own
        strategy to prevent multiple renderings.

        """
    @property
    def device_pixel_ratio(self):
        """
        The ratio of physical to logical pixels used for the canvas on screen.

        By default, this is 1, meaning physical and logical pixels are the same
        size. Subclasses that support High DPI screens may set this property to
        indicate that said ratio is different. All Matplotlib interaction,
        unless working directly with the canvas, remains in logical pixels.

        """
    def _set_device_pixel_ratio(self, ratio):
        """
        Set the ratio of physical to logical pixels used for the canvas.

        Subclasses that support High DPI screens can set this property to
        indicate that said ratio is different. The canvas itself will be
        created at the physical size, while the client side will use the
        logical size. Thus the DPI of the Figure will change to be scaled by
        this ratio. Implementations that support High DPI screens should use
        physical pixels for events so that transforms back to Axes space are
        correct.

        By default, this is 1, meaning physical and logical pixels are the same
        size.

        Parameters
        ----------
        ratio : float
            The ratio of logical to physical pixels used for the canvas.

        Returns
        -------
        bool
            Whether the ratio has changed. Backends may interpret this as a
            signal to resize the window, repaint the canvas, or change any
            other relevant properties.
        """
    def get_width_height(self, *, physical: bool = False):
        """
        Return the figure width and height in integral points or pixels.

        When the figure is used on High DPI screens (and the backend supports
        it), the truncation to integers occurs after scaling by the device
        pixel ratio.

        Parameters
        ----------
        physical : bool, default: False
            Whether to return true physical pixels or logical pixels. Physical
            pixels may be used by backends that support HiDPI, but still
            configure the canvas using its actual size.

        Returns
        -------
        width, height : int
            The size of the figure, in points or pixels, depending on the
            backend.
        """
    @classmethod
    def get_supported_filetypes(cls):
        """Return dict of savefig file formats supported by this backend."""
    @classmethod
    def get_supported_filetypes_grouped(cls):
        """
        Return a dict of savefig file formats supported by this backend,
        where the keys are a file type name, such as 'Joint Photographic
        Experts Group', and the values are a list of filename extensions used
        for that filetype, such as ['jpg', 'jpeg'].
        """
    def _switch_canvas_and_return_print_method(self, fmt, backend: Incomplete | None = None) -> Generator[Incomplete, None, Incomplete]:
        """
        Context manager temporarily setting the canvas for saving the figure::

            with (canvas._switch_canvas_and_return_print_method(fmt, backend)
                  as print_method):
                # ``print_method`` is a suitable ``print_{fmt}`` method, and
                # the figure's canvas is temporarily switched to the method's
                # canvas within the with... block.  ``print_method`` is also
                # wrapped to suppress extra kwargs passed by ``print_figure``.

        Parameters
        ----------
        fmt : str
            If *backend* is None, then determine a suitable canvas class for
            saving to format *fmt* -- either the current canvas class, if it
            supports *fmt*, or whatever `get_registered_canvas_class` returns;
            switch the figure canvas to that canvas class.
        backend : str or None, default: None
            If not None, switch the figure canvas to the ``FigureCanvas`` class
            of the given backend.
        """
    def print_figure(self, filename, dpi: Incomplete | None = None, facecolor: Incomplete | None = None, edgecolor: Incomplete | None = None, orientation: str = 'portrait', format: Incomplete | None = None, *, bbox_inches: Incomplete | None = None, pad_inches: Incomplete | None = None, bbox_extra_artists: Incomplete | None = None, backend: Incomplete | None = None, **kwargs):
        '''
        Render the figure to hardcopy. Set the figure patch face and edge
        colors.  This is useful because some of the GUIs have a gray figure
        face color background and you\'ll probably want to override this on
        hardcopy.

        Parameters
        ----------
        filename : str or path-like or file-like
            The file where the figure is saved.

        dpi : float, default: :rc:`savefig.dpi`
            The dots per inch to save the figure in.

        facecolor : :mpltype:`color` or \'auto\', default: :rc:`savefig.facecolor`
            The facecolor of the figure.  If \'auto\', use the current figure
            facecolor.

        edgecolor : :mpltype:`color` or \'auto\', default: :rc:`savefig.edgecolor`
            The edgecolor of the figure.  If \'auto\', use the current figure
            edgecolor.

        orientation : {\'landscape\', \'portrait\'}, default: \'portrait\'
            Only currently applies to PostScript printing.

        format : str, optional
            Force a specific file format. If not given, the format is inferred
            from the *filename* extension, and if that fails from
            :rc:`savefig.format`.

        bbox_inches : \'tight\' or `.Bbox`, default: :rc:`savefig.bbox`
            Bounding box in inches: only the given portion of the figure is
            saved.  If \'tight\', try to figure out the tight bbox of the figure.

        pad_inches : float or \'layout\', default: :rc:`savefig.pad_inches`
            Amount of padding in inches around the figure when bbox_inches is
            \'tight\'. If \'layout\' use the padding from the constrained or
            compressed layout engine; ignored if one of those engines is not in
            use.

        bbox_extra_artists : list of `~matplotlib.artist.Artist`, optional
            A list of extra artists that will be considered when the
            tight bbox is calculated.

        backend : str, optional
            Use a non-default backend to render the file, e.g. to render a
            png file with the "cairo" backend rather than the default "agg",
            or a pdf file with the "pgf" backend rather than the default
            "pdf".  Note that the default backend is normally sufficient.  See
            :ref:`the-builtin-backends` for a list of valid backends for each
            file format.  Custom backends can be referenced as "module://...".
        '''
    @classmethod
    def get_default_filetype(cls):
        """
        Return the default savefig file format as specified in
        :rc:`savefig.format`.

        The returned string does not include a period. This method is
        overridden in backends that only support a single file type.
        """
    def get_default_filename(self):
        """
        Return a suitable default filename, including the extension.
        """
    def mpl_connect(self, s, func):
        """
        Bind function *func* to event *s*.

        Parameters
        ----------
        s : str
            One of the following events ids:

            - 'button_press_event'
            - 'button_release_event'
            - 'draw_event'
            - 'key_press_event'
            - 'key_release_event'
            - 'motion_notify_event'
            - 'pick_event'
            - 'resize_event'
            - 'scroll_event'
            - 'figure_enter_event',
            - 'figure_leave_event',
            - 'axes_enter_event',
            - 'axes_leave_event'
            - 'close_event'.

        func : callable
            The callback function to be executed, which must have the
            signature::

                def func(event: Event) -> Any

            For the location events (button and key press/release), if the
            mouse is over the Axes, the ``inaxes`` attribute of the event will
            be set to the `~matplotlib.axes.Axes` the event occurs is over, and
            additionally, the variables ``xdata`` and ``ydata`` attributes will
            be set to the mouse location in data coordinates.  See `.KeyEvent`
            and `.MouseEvent` for more info.

            .. note::

                If func is a method, this only stores a weak reference to the
                method. Thus, the figure does not influence the lifetime of
                the associated object. Usually, you want to make sure that the
                object is kept alive throughout the lifetime of the figure by
                holding a reference to it.

        Returns
        -------
        cid
            A connection id that can be used with
            `.FigureCanvasBase.mpl_disconnect`.

        Examples
        --------
        ::

            def on_press(event):
                print('you pressed', event.button, event.xdata, event.ydata)

            cid = canvas.mpl_connect('button_press_event', on_press)
        """
    def mpl_disconnect(self, cid) -> None:
        """
        Disconnect the callback with id *cid*.

        Examples
        --------
        ::

            cid = canvas.mpl_connect('button_press_event', on_press)
            # ... later
            canvas.mpl_disconnect(cid)
        """
    _timer_cls = TimerBase
    def new_timer(self, interval: Incomplete | None = None, callbacks: Incomplete | None = None):
        """
        Create a new backend-specific subclass of `.Timer`.

        This is useful for getting periodic events through the backend's native
        event loop.  Implemented only for backends with GUIs.

        Parameters
        ----------
        interval : int
            Timer interval in milliseconds.

        callbacks : list[tuple[callable, tuple, dict]]
            Sequence of (func, args, kwargs) where ``func(*args, **kwargs)``
            will be executed by the timer every *interval*.

            Callbacks which return ``False`` or ``0`` will be removed from the
            timer.

        Examples
        --------
        >>> timer = fig.canvas.new_timer(callbacks=[(f1, (1,), {'a': 3})])
        """
    def flush_events(self) -> None:
        """
        Flush the GUI events for the figure.

        Interactive backends need to reimplement this method.
        """
    _looping: bool
    def start_event_loop(self, timeout: int = 0) -> None:
        """
        Start a blocking event loop.

        Such an event loop is used by interactive functions, such as
        `~.Figure.ginput` and `~.Figure.waitforbuttonpress`, to wait for
        events.

        The event loop blocks until a callback function triggers
        `stop_event_loop`, or *timeout* is reached.

        If *timeout* is 0 or negative, never timeout.

        Only interactive backends need to reimplement this method and it relies
        on `flush_events` being properly implemented.

        Interactive backends should implement this in a more native way.
        """
    def stop_event_loop(self) -> None:
        """
        Stop the current blocking event loop.

        Interactive backends need to reimplement this to match
        `start_event_loop`
        """

def key_press_handler(event, canvas: Incomplete | None = None, toolbar: Incomplete | None = None):
    """
    Implement the default Matplotlib key bindings for the canvas and toolbar
    described at :ref:`key-event-handling`.

    Parameters
    ----------
    event : `KeyEvent`
        A key press/release event.
    canvas : `FigureCanvasBase`, default: ``event.canvas``
        The backend-specific canvas instance.  This parameter is kept for
        back-compatibility, but, if set, should always be equal to
        ``event.canvas``.
    toolbar : `NavigationToolbar2`, default: ``event.canvas.toolbar``
        The navigation cursor toolbar.  This parameter is kept for
        back-compatibility, but, if set, should always be equal to
        ``event.canvas.toolbar``.
    """
def button_press_handler(event, canvas: Incomplete | None = None, toolbar: Incomplete | None = None) -> None:
    """
    The default Matplotlib button actions for extra mouse buttons.

    Parameters are as for `key_press_handler`, except that *event* is a
    `MouseEvent`.
    """

class NonGuiException(Exception):
    """Raised when trying show a figure in a non-GUI backend."""

class FigureManagerBase:
    """
    A backend-independent abstraction of a figure container and controller.

    The figure manager is used by pyplot to interact with the window in a
    backend-independent way. It's an adapter for the real (GUI) framework that
    represents the visual figure on screen.

    The figure manager is connected to a specific canvas instance, which in turn
    is connected to a specific figure instance. To access a figure manager for
    a given figure in user code, you typically use ``fig.canvas.manager``.

    GUI backends derive from this class to translate common operations such
    as *show* or *resize* to the GUI-specific code. Non-GUI backends do not
    support these operations and can just use the base class.

    This following basic operations are accessible:

    **Window operations**

    - `~.FigureManagerBase.show`
    - `~.FigureManagerBase.destroy`
    - `~.FigureManagerBase.full_screen_toggle`
    - `~.FigureManagerBase.resize`
    - `~.FigureManagerBase.get_window_title`
    - `~.FigureManagerBase.set_window_title`

    **Key and mouse button press handling**

    The figure manager sets up default key and mouse button press handling by
    hooking up the `.key_press_handler` to the matplotlib event system. This
    ensures the same shortcuts and mouse actions across backends.

    **Other operations**

    Subclasses will have additional attributes and functions to access
    additional functionality. This is of course backend-specific. For example,
    most GUI backends have ``window`` and ``toolbar`` attributes that give
    access to the native GUI widgets of the respective framework.

    Attributes
    ----------
    canvas : `FigureCanvasBase`
        The backend-specific canvas instance.

    num : int or str
        The figure number.

    key_press_handler_id : int
        The default key handler cid, when using the toolmanager.
        To disable the default key press handling use::

            figure.canvas.mpl_disconnect(
                figure.canvas.manager.key_press_handler_id)

    button_press_handler_id : int
        The default mouse button handler cid, when using the toolmanager.
        To disable the default button press handling use::

            figure.canvas.mpl_disconnect(
                figure.canvas.manager.button_press_handler_id)
    """
    _toolbar2_class: Incomplete
    _toolmanager_toolbar_class: Incomplete
    canvas: Incomplete
    num: Incomplete
    key_press_handler_id: Incomplete
    button_press_handler_id: Incomplete
    toolmanager: Incomplete
    toolbar: Incomplete
    def __init__(self, canvas, num) -> None: ...
    @classmethod
    def create_with_canvas(cls, canvas_class, figure, num):
        """
        Create a manager for a given *figure* using a specific *canvas_class*.

        Backends should override this method if they have specific needs for
        setting up the canvas or the manager.
        """
    @classmethod
    def start_main_loop(cls) -> None:
        """
        Start the main event loop.

        This method is called by `.FigureManagerBase.pyplot_show`, which is the
        implementation of `.pyplot.show`.  To customize the behavior of
        `.pyplot.show`, interactive backends should usually override
        `~.FigureManagerBase.start_main_loop`; if more customized logic is
        necessary, `~.FigureManagerBase.pyplot_show` can also be overridden.
        """
    @classmethod
    def pyplot_show(cls, *, block: Incomplete | None = None) -> None:
        """
        Show all figures.  This method is the implementation of `.pyplot.show`.

        To customize the behavior of `.pyplot.show`, interactive backends
        should usually override `~.FigureManagerBase.start_main_loop`; if more
        customized logic is necessary, `~.FigureManagerBase.pyplot_show` can
        also be overridden.

        Parameters
        ----------
        block : bool, optional
            Whether to block by calling ``start_main_loop``.  The default,
            None, means to block if we are neither in IPython's ``%pylab`` mode
            nor in ``interactive`` mode.
        """
    def show(self) -> None:
        """
        For GUI backends, show the figure window and redraw.
        For non-GUI backends, raise an exception, unless running headless (i.e.
        on Linux with an unset DISPLAY); this exception is converted to a
        warning in `.Figure.show`.
        """
    def destroy(self) -> None: ...
    def full_screen_toggle(self) -> None: ...
    def resize(self, w, h) -> None:
        """For GUI backends, resize the window (in physical pixels)."""
    def get_window_title(self):
        """
        Return the title text of the window containing the figure, or None
        if there is no window (e.g., a PS backend).
        """
    def set_window_title(self, title) -> None:
        """
        Set the title text of the window containing the figure.

        This has no effect for non-GUI (e.g., PS) backends.

        Examples
        --------
        >>> fig = plt.figure()
        >>> fig.canvas.manager.set_window_title('My figure')
        """

cursors: Incomplete

class _Mode(str, Enum):
    NONE = ''
    PAN = 'pan/zoom'
    ZOOM = 'zoom rect'
    def __str__(self) -> str: ...
    @property
    def _navigate_mode(self): ...

class NavigationToolbar2:
    '''
    Base class for the navigation cursor, version 2.

    Backends must implement a canvas that handles connections for
    \'button_press_event\' and \'button_release_event\'.  See
    :meth:`FigureCanvasBase.mpl_connect` for more information.

    They must also define

    :meth:`save_figure`
        Save the current figure.

    :meth:`draw_rubberband` (optional)
        Draw the zoom to rect "rubberband" rectangle.

    :meth:`set_message` (optional)
        Display message.

    :meth:`set_history_buttons` (optional)
        You can change the history back / forward buttons to indicate disabled / enabled
        state.

    and override ``__init__`` to set up the toolbar -- without forgetting to
    call the base-class init.  Typically, ``__init__`` needs to set up toolbar
    buttons connected to the `home`, `back`, `forward`, `pan`, `zoom`, and
    `save_figure` methods and using standard icons in the "images" subdirectory
    of the data path.

    That\'s it, we\'ll do the rest!
    '''
    toolitems: Incomplete
    UNKNOWN_SAVED_STATUS: Incomplete
    canvas: Incomplete
    _nav_stack: Incomplete
    _last_cursor: Incomplete
    _id_press: Incomplete
    _id_release: Incomplete
    _id_drag: Incomplete
    _pan_info: Incomplete
    _zoom_info: Incomplete
    mode: Incomplete
    def __init__(self, canvas) -> None: ...
    def set_message(self, s) -> None:
        """Display a message on toolbar or in status bar."""
    def draw_rubberband(self, event, x0, y0, x1, y1) -> None:
        """
        Draw a rectangle rubberband to indicate zoom limits.

        Note that it is not guaranteed that ``x0 <= x1`` and ``y0 <= y1``.
        """
    def remove_rubberband(self) -> None:
        """Remove the rubberband."""
    def home(self, *args) -> None:
        """
        Restore the original view.

        For convenience of being directly connected as a GUI callback, which
        often get passed additional parameters, this method accepts arbitrary
        parameters, but does not use them.
        """
    def back(self, *args) -> None:
        """
        Move back up the view lim stack.

        For convenience of being directly connected as a GUI callback, which
        often get passed additional parameters, this method accepts arbitrary
        parameters, but does not use them.
        """
    def forward(self, *args) -> None:
        """
        Move forward in the view lim stack.

        For convenience of being directly connected as a GUI callback, which
        often get passed additional parameters, this method accepts arbitrary
        parameters, but does not use them.
        """
    def _update_cursor(self, event) -> None:
        """
        Update the cursor after a mouse move event or a tool (de)activation.
        """
    def _wait_cursor_for_draw_cm(self) -> Generator[None]:
        """
        Set the cursor to a wait cursor when drawing the canvas.

        In order to avoid constantly changing the cursor when the canvas
        changes frequently, do nothing if this context was triggered during the
        last second.  (Optimally we'd prefer only setting the wait cursor if
        the *current* draw takes too long, but the current draw blocks the GUI
        thread).
        """
    @staticmethod
    def _mouse_event_to_message(event): ...
    def mouse_move(self, event) -> None: ...
    def _zoom_pan_handler(self, event) -> None: ...
    def _start_event_axes_interaction(self, event, *, method): ...
    def pan(self, *args) -> None:
        """
        Toggle the pan/zoom tool.

        Pan with left button, zoom with right.
        """

    class _PanInfo(NamedTuple):
        button: Incomplete
        axes: Incomplete
        cid: Incomplete
    def press_pan(self, event) -> None:
        """Callback for mouse button press in pan/zoom mode."""
    def drag_pan(self, event) -> None:
        """Callback for dragging in pan/zoom mode."""
    def release_pan(self, event) -> None:
        """Callback for mouse button release in pan/zoom mode."""
    def zoom(self, *args) -> None: ...

    class _ZoomInfo(NamedTuple):
        direction: Incomplete
        start_xy: Incomplete
        axes: Incomplete
        cid: Incomplete
        cbar: Incomplete
    def press_zoom(self, event) -> None:
        """Callback for mouse button press in zoom to rect mode."""
    def drag_zoom(self, event) -> None:
        """Callback for dragging in zoom mode."""
    def release_zoom(self, event) -> None:
        """Callback for mouse button release in zoom to rect mode."""
    def push_current(self) -> None:
        """Push the current view limits and position onto the stack."""
    def _update_view(self) -> None:
        """
        Update the viewlim and position from the view and position stack for
        each Axes.
        """
    subplot_tool: Incomplete
    def configure_subplots(self, *args): ...
    def save_figure(self, *args) -> None:
        """
        Save the current figure.

        Backend implementations may choose to return
        the absolute path of the saved file, if any, as
        a string.

        If no file is created then `None` is returned.

        If the backend does not implement this functionality
        then `NavigationToolbar2.UNKNOWN_SAVED_STATUS` is returned.

        Returns
        -------
        str or `NavigationToolbar2.UNKNOWN_SAVED_STATUS` or `None`
            The filepath of the saved figure.
            Returns `None` if figure is not saved.
            Returns `NavigationToolbar2.UNKNOWN_SAVED_STATUS` when
            the backend does not provide the information.
        """
    def update(self) -> None:
        """Reset the Axes stack."""
    def set_history_buttons(self) -> None:
        """Enable or disable the back/forward button."""

class ToolContainerBase:
    """
    Base class for all tool containers, e.g. toolbars.

    Attributes
    ----------
    toolmanager : `.ToolManager`
        The tools with which this `ToolContainer` wants to communicate.
    """
    _icon_extension: str
    toolmanager: Incomplete
    def __init__(self, toolmanager) -> None: ...
    def _tool_toggled_cbk(self, event) -> None:
        """
        Capture the 'tool_trigger_[name]'

        This only gets used for toggled tools.
        """
    def add_tool(self, tool, group, position: int = -1) -> None:
        """
        Add a tool to this container.

        Parameters
        ----------
        tool : tool_like
            The tool to add, see `.ToolManager.get_tool`.
        group : str
            The name of the group to add this tool to.
        position : int, default: -1
            The position within the group to place this tool.
        """
    def _get_image_filename(self, tool):
        """Resolve a tool icon's filename."""
    def trigger_tool(self, name) -> None:
        """
        Trigger the tool.

        Parameters
        ----------
        name : str
            Name (id) of the tool triggered from within the container.
        """
    def add_toolitem(self, name, group, position, image, description, toggle) -> None:
        """
        A hook to add a toolitem to the container.

        This hook must be implemented in each backend and contains the
        backend-specific code to add an element to the toolbar.

        .. warning::
            This is part of the backend implementation and should
            not be called by end-users.  They should instead call
            `.ToolContainerBase.add_tool`.

        The callback associated with the button click event
        must be *exactly* ``self.trigger_tool(name)``.

        Parameters
        ----------
        name : str
            Name of the tool to add, this gets used as the tool's ID and as the
            default label of the buttons.
        group : str
            Name of the group that this tool belongs to.
        position : int
            Position of the tool within its group, if -1 it goes at the end.
        image : str
            Filename of the image for the button or `None`.
        description : str
            Description of the tool, used for the tooltips.
        toggle : bool
            * `True` : The button is a toggle (change the pressed/unpressed
              state between consecutive clicks).
            * `False` : The button is a normal button (returns to unpressed
              state after release).
        """
    def toggle_toolitem(self, name, toggled) -> None:
        """
        A hook to toggle a toolitem without firing an event.

        This hook must be implemented in each backend and contains the
        backend-specific code to silently toggle a toolbar element.

        .. warning::
            This is part of the backend implementation and should
            not be called by end-users.  They should instead call
            `.ToolManager.trigger_tool` or `.ToolContainerBase.trigger_tool`
            (which are equivalent).

        Parameters
        ----------
        name : str
            Id of the tool to toggle.
        toggled : bool
            Whether to set this tool as toggled or not.
        """
    def remove_toolitem(self, name) -> None:
        """
        A hook to remove a toolitem from the container.

        This hook must be implemented in each backend and contains the
        backend-specific code to remove an element from the toolbar; it is
        called when `.ToolManager` emits a ``tool_removed_event``.

        Because some tools are present only on the `.ToolManager` but not on
        the `ToolContainer`, this method must be a no-op when called on a tool
        absent from the container.

        .. warning::
            This is part of the backend implementation and should
            not be called by end-users.  They should instead call
            `.ToolManager.remove_tool`.

        Parameters
        ----------
        name : str
            Name of the tool to remove.
        """
    def set_message(self, s) -> None:
        """
        Display a message on the toolbar.

        Parameters
        ----------
        s : str
            Message text.
        """

class _Backend:
    backend_version: str
    FigureCanvas: Incomplete
    FigureManager = FigureManagerBase
    mainloop: Incomplete
    @classmethod
    def new_figure_manager(cls, num, *args, **kwargs):
        """Create a new figure manager instance."""
    @classmethod
    def new_figure_manager_given_figure(cls, num, figure):
        """Create a new figure manager instance for the given figure."""
    @classmethod
    def draw_if_interactive(cls) -> None: ...
    @classmethod
    def show(cls, *, block: Incomplete | None = None) -> None:
        """
        Show all figures.

        `show` blocks by calling `mainloop` if *block* is ``True``, or if it is
        ``None`` and we are not in `interactive` mode and if IPython's
        ``%matplotlib`` integration has not been activated.
        """
    @staticmethod
    def export(cls): ...

class ShowBase(_Backend):
    """
    Simple base class to generate a ``show()`` function in backends.

    Subclass must override ``mainloop()`` method.
    """
    def __call__(self, block: Incomplete | None = None): ...
