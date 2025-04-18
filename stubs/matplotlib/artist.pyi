from . import _api as _api, cbook as cbook
from .path import Path as Path
from .transforms import Bbox as Bbox, BboxBase as BboxBase, IdentityTransform as IdentityTransform, Transform as Transform, TransformedBbox as TransformedBbox, TransformedPatchPath as TransformedPatchPath, TransformedPath as TransformedPath
from _typeshed import Incomplete
from collections.abc import Generator
from typing import NamedTuple

_log: Incomplete

def _prevent_rasterization(draw): ...
def allow_rasterization(draw):
    """
    Decorator for Artist.draw method. Provides routines
    that run before and after the draw call. The before and after functions
    are useful for changing artist-dependent renderer attributes or making
    other setup function calls, such as starting and flushing a mixed-mode
    renderer.
    """
def _finalize_rasterization(draw):
    """
    Decorator for Artist.draw method. Needed on the outermost artist, i.e.
    Figure, to finish up if the render is still in rasterized mode.
    """
def _stale_axes_callback(self, val) -> None: ...

class _XYPair(NamedTuple):
    x: Incomplete
    y: Incomplete

class _Unset:
    def __repr__(self) -> str: ...

_UNSET: Incomplete

class Artist:
    """
    Abstract base class for objects that render into a FigureCanvas.

    Typically, all visible elements in a figure are subclasses of Artist.
    """
    zorder: int
    def __init_subclass__(cls): ...
    _PROPERTIES_EXCLUDED_FROM_SET: Incomplete
    @classmethod
    def _update_set_signature_and_docstring(cls) -> None:
        """
        Update the signature of the set function to list all properties
        as keyword arguments.

        Property aliases are not listed in the signature for brevity, but
        are still accepted as keyword arguments.
        """
    _stale: bool
    stale_callback: Incomplete
    _axes: Incomplete
    _parent_figure: Incomplete
    _transform: Incomplete
    _transformSet: bool
    _visible: bool
    _animated: bool
    _alpha: Incomplete
    clipbox: Incomplete
    _clippath: Incomplete
    _clipon: bool
    _label: str
    _picker: Incomplete
    _rasterized: bool
    _agg_filter: Incomplete
    _mouseover: Incomplete
    _callbacks: Incomplete
    _remove_method: Incomplete
    _url: Incomplete
    _gid: Incomplete
    _snap: Incomplete
    _sketch: Incomplete
    _path_effects: Incomplete
    _sticky_edges: Incomplete
    _in_layout: bool
    def __init__(self) -> None: ...
    def __getstate__(self): ...
    def remove(self) -> None:
        """
        Remove the artist from the figure if possible.

        The effect will not be visible until the figure is redrawn, e.g.,
        with `.FigureCanvasBase.draw_idle`.  Call `~.axes.Axes.relim` to
        update the Axes limits if desired.

        Note: `~.axes.Axes.relim` will not see collections even if the
        collection was added to the Axes with *autolim* = True.

        Note: there is no support for removing the artist's legend entry.
        """
    def have_units(self):
        """Return whether units are set on any axis."""
    def convert_xunits(self, x):
        """
        Convert *x* using the unit type of the xaxis.

        If the artist is not contained in an Axes or if the xaxis does not
        have units, *x* itself is returned.
        """
    def convert_yunits(self, y):
        """
        Convert *y* using the unit type of the yaxis.

        If the artist is not contained in an Axes or if the yaxis does not
        have units, *y* itself is returned.
        """
    @property
    def axes(self):
        """The `~.axes.Axes` instance the artist resides in, or *None*."""
    @axes.setter
    def axes(self, new_axes) -> None: ...
    @property
    def stale(self):
        """
        Whether the artist is 'stale' and needs to be re-drawn for the output
        to match the internal state of the artist.
        """
    @stale.setter
    def stale(self, val) -> None: ...
    def get_window_extent(self, renderer: Incomplete | None = None):
        '''
        Get the artist\'s bounding box in display space.

        The bounding box\' width and height are nonnegative.

        Subclasses should override for inclusion in the bounding box
        "tight" calculation. Default is to return an empty bounding
        box at 0, 0.

        Be careful when using this function, the results will not update
        if the artist window extent of the artist changes.  The extent
        can change due to any changes in the transform stack, such as
        changing the Axes limits, the figure size, or the canvas used
        (as is done when saving a figure).  This can lead to unexpected
        behavior where interactive figures will look fine on the screen,
        but will save incorrectly.
        '''
    def get_tightbbox(self, renderer: Incomplete | None = None):
        """
        Like `.Artist.get_window_extent`, but includes any clipping.

        Parameters
        ----------
        renderer : `~matplotlib.backend_bases.RendererBase` subclass, optional
            renderer that will be used to draw the figures (i.e.
            ``fig.canvas.get_renderer()``)

        Returns
        -------
        `.Bbox` or None
            The enclosing bounding box (in figure pixel coordinates).
            Returns None if clipping results in no intersection.
        """
    def add_callback(self, func):
        """
        Add a callback function that will be called whenever one of the
        `.Artist`'s properties changes.

        Parameters
        ----------
        func : callable
            The callback function. It must have the signature::

                def func(artist: Artist) -> Any

            where *artist* is the calling `.Artist`. Return values may exist
            but are ignored.

        Returns
        -------
        int
            The observer id associated with the callback. This id can be
            used for removing the callback with `.remove_callback` later.

        See Also
        --------
        remove_callback
        """
    def remove_callback(self, oid) -> None:
        """
        Remove a callback based on its observer id.

        See Also
        --------
        add_callback
        """
    def pchanged(self) -> None:
        """
        Call all of the registered callbacks.

        This function is triggered internally when a property is changed.

        See Also
        --------
        add_callback
        remove_callback
        """
    def is_transform_set(self):
        """
        Return whether the Artist has an explicitly set transform.

        This is *True* after `.set_transform` has been called.
        """
    def set_transform(self, t) -> None:
        """
        Set the artist transform.

        Parameters
        ----------
        t : `~matplotlib.transforms.Transform`
        """
    def get_transform(self):
        """Return the `.Transform` instance used by this artist."""
    def get_children(self):
        """Return a list of the child `.Artist`\\s of this `.Artist`."""
    def _different_canvas(self, event):
        """
        Check whether an *event* occurred on a canvas other that this artist's canvas.

        If this method returns True, the event definitely occurred on a different
        canvas; if it returns False, either it occurred on the same canvas, or we may
        not have enough information to know.

        Subclasses should start their definition of `contains` as follows::

            if self._different_canvas(mouseevent):
                return False, {}
            # subclass-specific implementation follows
        """
    def contains(self, mouseevent):
        """
        Test whether the artist contains the mouse event.

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
        """
    def pickable(self):
        """
        Return whether the artist is pickable.

        See Also
        --------
        .Artist.set_picker, .Artist.get_picker, .Artist.pick
        """
    def pick(self, mouseevent) -> None:
        """
        Process a pick event.

        Each child artist will fire a pick event if *mouseevent* is over
        the artist and the artist has picker set.

        See Also
        --------
        .Artist.set_picker, .Artist.get_picker, .Artist.pickable
        """
    def set_picker(self, picker) -> None:
        """
        Define the picking behavior of the artist.

        Parameters
        ----------
        picker : None or bool or float or callable
            This can be one of the following:

            - *None*: Picking is disabled for this artist (default).

            - A boolean: If *True* then picking will be enabled and the
              artist will fire a pick event if the mouse event is over
              the artist.

            - A float: If picker is a number it is interpreted as an
              epsilon tolerance in points and the artist will fire
              off an event if its data is within epsilon of the mouse
              event.  For some artists like lines and patch collections,
              the artist may provide additional data to the pick event
              that is generated, e.g., the indices of the data within
              epsilon of the pick event

            - A function: If picker is callable, it is a user supplied
              function which determines whether the artist is hit by the
              mouse event::

                hit, props = picker(artist, mouseevent)

              to determine the hit test.  if the mouse event is over the
              artist, return *hit=True* and props is a dictionary of
              properties you want added to the PickEvent attributes.
        """
    def get_picker(self):
        """
        Return the picking behavior of the artist.

        The possible values are described in `.Artist.set_picker`.

        See Also
        --------
        .Artist.set_picker, .Artist.pickable, .Artist.pick
        """
    def get_url(self):
        """Return the url."""
    def set_url(self, url) -> None:
        """
        Set the url for the artist.

        Parameters
        ----------
        url : str
        """
    def get_gid(self):
        """Return the group id."""
    def set_gid(self, gid) -> None:
        """
        Set the (group) id for the artist.

        Parameters
        ----------
        gid : str
        """
    def get_snap(self):
        """
        Return the snap setting.

        See `.set_snap` for details.
        """
    def set_snap(self, snap) -> None:
        """
        Set the snapping behavior.

        Snapping aligns positions with the pixel grid, which results in
        clearer images. For example, if a black line of 1px width was
        defined at a position in between two pixels, the resulting image
        would contain the interpolated value of that line in the pixel grid,
        which would be a grey value on both adjacent pixel positions. In
        contrast, snapping will move the line to the nearest integer pixel
        value, so that the resulting image will really contain a 1px wide
        black line.

        Snapping is currently only supported by the Agg and MacOSX backends.

        Parameters
        ----------
        snap : bool or None
            Possible values:

            - *True*: Snap vertices to the nearest pixel center.
            - *False*: Do not modify vertex positions.
            - *None*: (auto) If the path contains only rectilinear line
              segments, round to the nearest pixel center.
        """
    def get_sketch_params(self):
        """
        Return the sketch parameters for the artist.

        Returns
        -------
        tuple or None

            A 3-tuple with the following elements:

            - *scale*: The amplitude of the wiggle perpendicular to the
              source line.
            - *length*: The length of the wiggle along the line.
            - *randomness*: The scale factor by which the length is
              shrunken or expanded.

            Returns *None* if no sketch parameters were set.
        """
    def set_sketch_params(self, scale: Incomplete | None = None, length: Incomplete | None = None, randomness: Incomplete | None = None) -> None:
        """
        Set the sketch parameters.

        Parameters
        ----------
        scale : float, optional
            The amplitude of the wiggle perpendicular to the source
            line, in pixels.  If scale is `None`, or not provided, no
            sketch filter will be provided.
        length : float, optional
             The length of the wiggle along the line, in pixels
             (default 128.0)
        randomness : float, optional
            The scale factor by which the length is shrunken or
            expanded (default 16.0)

            The PGF backend uses this argument as an RNG seed and not as
            described above. Using the same seed yields the same random shape.

            .. ACCEPTS: (scale: float, length: float, randomness: float)
        """
    def set_path_effects(self, path_effects) -> None:
        """
        Set the path effects.

        Parameters
        ----------
        path_effects : list of `.AbstractPathEffect`
        """
    def get_path_effects(self): ...
    def get_figure(self, root: bool = False):
        """
        Return the `.Figure` or `.SubFigure` instance the artist belongs to.

        Parameters
        ----------
        root : bool, default=False
            If False, return the (Sub)Figure this artist is on.  If True,
            return the root Figure for a nested tree of SubFigures.
        """
    def set_figure(self, fig) -> None:
        """
        Set the `.Figure` or `.SubFigure` instance the artist belongs to.

        Parameters
        ----------
        fig : `~matplotlib.figure.Figure` or `~matplotlib.figure.SubFigure`
        """
    figure: Incomplete
    def set_clip_box(self, clipbox) -> None:
        """
        Set the artist's clip `.Bbox`.

        Parameters
        ----------
        clipbox : `~matplotlib.transforms.BboxBase` or None
            Will typically be created from a `.TransformedBbox`. For instance,
            ``TransformedBbox(Bbox([[0, 0], [1, 1]]), ax.transAxes)`` is the default
            clipping for an artist added to an Axes.

        """
    def set_clip_path(self, path, transform: Incomplete | None = None) -> None:
        """
        Set the artist's clip path.

        Parameters
        ----------
        path : `~matplotlib.patches.Patch` or `.Path` or `.TransformedPath` or None
            The clip path. If given a `.Path`, *transform* must be provided as
            well. If *None*, a previously set clip path is removed.
        transform : `~matplotlib.transforms.Transform`, optional
            Only used if *path* is a `.Path`, in which case the given `.Path`
            is converted to a `.TransformedPath` using *transform*.

        Notes
        -----
        For efficiency, if *path* is a `.Rectangle` this method will set the
        clipping box to the corresponding rectangle and set the clipping path
        to ``None``.

        For technical reasons (support of `~.Artist.set`), a tuple
        (*path*, *transform*) is also accepted as a single positional
        parameter.

        .. ACCEPTS: Patch or (Path, Transform) or None
        """
    def get_alpha(self):
        """
        Return the alpha value used for blending - not supported on all
        backends.
        """
    def get_visible(self):
        """Return the visibility."""
    def get_animated(self):
        """Return whether the artist is animated."""
    def get_in_layout(self):
        """
        Return boolean flag, ``True`` if artist is included in layout
        calculations.

        E.g. :ref:`constrainedlayout_guide`,
        `.Figure.tight_layout()`, and
        ``fig.savefig(fname, bbox_inches='tight')``.
        """
    def _fully_clipped_to_axes(self):
        """
        Return a boolean flag, ``True`` if the artist is clipped to the Axes
        and can thus be skipped in layout calculations. Requires `get_clip_on`
        is True, one of `clip_box` or `clip_path` is set, ``clip_box.extents``
        is equivalent to ``ax.bbox.extents`` (if set), and ``clip_path._patch``
        is equivalent to ``ax.patch`` (if set).
        """
    def get_clip_on(self):
        """Return whether the artist uses clipping."""
    def get_clip_box(self):
        """Return the clipbox."""
    def get_clip_path(self):
        """Return the clip path."""
    def get_transformed_clip_path_and_affine(self):
        """
        Return the clip path with the non-affine part of its
        transformation applied, and the remaining affine part of its
        transformation.
        """
    def set_clip_on(self, b) -> None:
        """
        Set whether the artist uses clipping.

        When False, artists will be visible outside the Axes which
        can lead to unexpected results.

        Parameters
        ----------
        b : bool
        """
    def _set_gc_clip(self, gc) -> None:
        """Set the clip properly for the gc."""
    def get_rasterized(self):
        """Return whether the artist is to be rasterized."""
    def set_rasterized(self, rasterized) -> None:
        """
        Force rasterized (bitmap) drawing for vector graphics output.

        Rasterized drawing is not supported by all artists. If you try to
        enable this on an artist that does not support it, the command has no
        effect and a warning will be issued.

        This setting is ignored for pixel-based output.

        See also :doc:`/gallery/misc/rasterization_demo`.

        Parameters
        ----------
        rasterized : bool
        """
    def get_agg_filter(self):
        """Return filter function to be used for agg filter."""
    def set_agg_filter(self, filter_func) -> None:
        """
        Set the agg filter.

        Parameters
        ----------
        filter_func : callable
            A filter function, which takes a (m, n, depth) float array
            and a dpi value, and returns a (m, n, depth) array and two
            offsets from the bottom left corner of the image

            .. ACCEPTS: a filter function, which takes a (m, n, 3) float array
                and a dpi value, and returns a (m, n, 3) array and two offsets
                from the bottom left corner of the image
        """
    def draw(self, renderer) -> None:
        """
        Draw the Artist (and its children) using the given renderer.

        This has no effect if the artist is not visible (`.Artist.get_visible`
        returns False).

        Parameters
        ----------
        renderer : `~matplotlib.backend_bases.RendererBase` subclass.

        Notes
        -----
        This method is overridden in the Artist subclasses.
        """
    def set_alpha(self, alpha) -> None:
        """
        Set the alpha value used for blending - not supported on all backends.

        Parameters
        ----------
        alpha : float or None
            *alpha* must be within the 0-1 range, inclusive.
        """
    def _set_alpha_for_array(self, alpha) -> None:
        """
        Set the alpha value used for blending - not supported on all backends.

        Parameters
        ----------
        alpha : array-like or float or None
            All values must be within the 0-1 range, inclusive.
            Masked values and nans are not supported.
        """
    def set_visible(self, b) -> None:
        """
        Set the artist's visibility.

        Parameters
        ----------
        b : bool
        """
    def set_animated(self, b) -> None:
        """
        Set whether the artist is intended to be used in an animation.

        If True, the artist is excluded from regular drawing of the figure.
        You have to call `.Figure.draw_artist` / `.Axes.draw_artist`
        explicitly on the artist. This approach is used to speed up animations
        using blitting.

        See also `matplotlib.animation` and
        :ref:`blitting`.

        Parameters
        ----------
        b : bool
        """
    def set_in_layout(self, in_layout) -> None:
        """
        Set if artist is to be included in layout calculations,
        E.g. :ref:`constrainedlayout_guide`,
        `.Figure.tight_layout()`, and
        ``fig.savefig(fname, bbox_inches='tight')``.

        Parameters
        ----------
        in_layout : bool
        """
    def get_label(self):
        """Return the label used for this artist in the legend."""
    def set_label(self, s) -> None:
        """
        Set a label that will be displayed in the legend.

        Parameters
        ----------
        s : object
            *s* will be converted to a string by calling `str`.
        """
    def get_zorder(self):
        """Return the artist's zorder."""
    def set_zorder(self, level) -> None:
        """
        Set the zorder for the artist.  Artists with lower zorder
        values are drawn first.

        Parameters
        ----------
        level : float
        """
    @property
    def sticky_edges(self):
        '''
        ``x`` and ``y`` sticky edge lists for autoscaling.

        When performing autoscaling, if a data limit coincides with a value in
        the corresponding sticky_edges list, then no margin will be added--the
        view limit "sticks" to the edge. A typical use case is histograms,
        where one usually expects no margin on the bottom edge (0) of the
        histogram.

        Moreover, margin expansion "bumps" against sticky edges and cannot
        cross them.  For example, if the upper data limit is 1.0, the upper
        view limit computed by simple margin application is 1.2, but there is a
        sticky edge at 1.1, then the actual upper view limit will be 1.1.

        This attribute cannot be assigned to; however, the ``x`` and ``y``
        lists can be modified in place as needed.

        Examples
        --------
        >>> artist.sticky_edges.x[:] = (xmin, xmax)
        >>> artist.sticky_edges.y[:] = (ymin, ymax)

        '''
    def update_from(self, other) -> None:
        """Copy properties from *other* to *self*."""
    def properties(self):
        """Return a dictionary of all the properties of the artist."""
    def _update_props(self, props, errfmt):
        '''
        Helper for `.Artist.set` and `.Artist.update`.

        *errfmt* is used to generate error messages for invalid property
        names; it gets formatted with ``type(self)`` for "{cls}" and the
        property name for "{prop_name}".
        '''
    def update(self, props):
        """
        Update this artist's properties from the dict *props*.

        Parameters
        ----------
        props : dict
        """
    def _internal_update(self, kwargs):
        """
        Update artist properties without prenormalizing them, but generating
        errors as if calling `set`.

        The lack of prenormalization is to maintain backcompatibility.
        """
    def set(self, **kwargs): ...
    def _cm_set(self, **kwargs) -> Generator[None]:
        """
        `.Artist.set` context-manager that restores original values at exit.
        """
    def findobj(self, match: Incomplete | None = None, include_self: bool = True):
        """
        Find artist objects.

        Recursively find all `.Artist` instances contained in the artist.

        Parameters
        ----------
        match
            A filter criterion for the matches. This can be

            - *None*: Return all objects contained in artist.
            - A function with signature ``def match(artist: Artist) -> bool``.
              The result will only contain artists for which the function
              returns *True*.
            - A class instance: e.g., `.Line2D`. The result will only contain
              artists of this class or its subclasses (``isinstance`` check).

        include_self : bool
            Include *self* in the list to be checked for a match.

        Returns
        -------
        list of `.Artist`

        """
    def get_cursor_data(self, event) -> None:
        """
        Return the cursor data for a given event.

        .. note::
            This method is intended to be overridden by artist subclasses.
            As an end-user of Matplotlib you will most likely not call this
            method yourself.

        Cursor data can be used by Artists to provide additional context
        information for a given event. The default implementation just returns
        *None*.

        Subclasses can override the method and return arbitrary data. However,
        when doing so, they must ensure that `.format_cursor_data` can convert
        the data to a string representation.

        The only current use case is displaying the z-value of an `.AxesImage`
        in the status bar of a plot window, while moving the mouse.

        Parameters
        ----------
        event : `~matplotlib.backend_bases.MouseEvent`

        See Also
        --------
        format_cursor_data

        """
    def format_cursor_data(self, data):
        """
        Return a string representation of *data*.

        .. note::
            This method is intended to be overridden by artist subclasses.
            As an end-user of Matplotlib you will most likely not call this
            method yourself.

        The default implementation converts ints and floats and arrays of ints
        and floats into a comma-separated string enclosed in square brackets,
        unless the artist has an associated colorbar, in which case scalar
        values are formatted using the colorbar's formatter.

        See Also
        --------
        get_cursor_data
        """
    def get_mouseover(self):
        """
        Return whether this artist is queried for custom context information
        when the mouse cursor moves over it.
        """
    def set_mouseover(self, mouseover) -> None:
        """
        Set whether this artist is queried for custom context information when
        the mouse cursor moves over it.

        Parameters
        ----------
        mouseover : bool

        See Also
        --------
        get_cursor_data
        .ToolCursorPosition
        .NavigationToolbar2
        """
    mouseover: Incomplete

def _get_tightbbox_for_layout_only(obj, *args, **kwargs):
    """
    Matplotlib's `.Axes.get_tightbbox` and `.Axis.get_tightbbox` support a
    *for_layout_only* kwarg; this helper tries to use the kwarg but skips it
    when encountering third-party subclasses that do not support it.
    """

class ArtistInspector:
    """
    A helper class to inspect an `~matplotlib.artist.Artist` and return
    information about its settable properties and their current values.
    """
    oorig: Incomplete
    o: Incomplete
    aliasd: Incomplete
    def __init__(self, o) -> None:
        """
        Initialize the artist inspector with an `Artist` or an iterable of
        `Artist`\\s.  If an iterable is used, we assume it is a homogeneous
        sequence (all `Artist`\\s are of the same type) and it is your
        responsibility to make sure this is so.
        """
    def get_aliases(self):
        """
        Get a dict mapping property fullnames to sets of aliases for each alias
        in the :class:`~matplotlib.artist.ArtistInspector`.

        e.g., for lines::

          {'markerfacecolor': {'mfc'},
           'linewidth'      : {'lw'},
          }
        """
    _get_valid_values_regex: Incomplete
    def get_valid_values(self, attr):
        '''
        Get the legal arguments for the setter associated with *attr*.

        This is done by querying the docstring of the setter for a line that
        begins with "ACCEPTS:" or ".. ACCEPTS:", and then by looking for a
        numpydoc-style documentation for the setter\'s first argument.
        '''
    def _replace_path(self, source_class):
        """
        Changes the full path to the public API path that is used
        in sphinx. This is needed for links to work.
        """
    def get_setters(self):
        """
        Get the attribute strings with setters for object.

        For example, for a line, return ``['markerfacecolor', 'linewidth',
        ....]``.
        """
    @staticmethod
    def number_of_parameters(func):
        """Return number of parameters of the callable *func*."""
    @staticmethod
    def is_alias(method):
        """
        Return whether the object *method* is an alias for another method.
        """
    def aliased_name(self, s):
        """
        Return 'PROPNAME or alias' if *s* has an alias, else return 'PROPNAME'.

        For example, for the line markerfacecolor property, which has an
        alias, return 'markerfacecolor or mfc' and for the transform
        property, which does not, return 'transform'.
        """
    _NOT_LINKABLE: Incomplete
    def aliased_name_rest(self, s, target):
        """
        Return 'PROPNAME or alias' if *s* has an alias, else return 'PROPNAME',
        formatted for reST.

        For example, for the line markerfacecolor property, which has an
        alias, return 'markerfacecolor or mfc' and for the transform
        property, which does not, return 'transform'.
        """
    def pprint_setters(self, prop: Incomplete | None = None, leadingspace: int = 2):
        """
        If *prop* is *None*, return a list of strings of all settable
        properties and their valid values.

        If *prop* is not *None*, it is a valid property name and that
        property will be returned as a string of property : valid
        values.
        """
    def pprint_setters_rest(self, prop: Incomplete | None = None, leadingspace: int = 4):
        '''
        If *prop* is *None*, return a list of reST-formatted strings of all
        settable properties and their valid values.

        If *prop* is not *None*, it is a valid property name and that
        property will be returned as a string of "property : valid"
        values.
        '''
    def properties(self):
        """Return a dictionary mapping property name -> value."""
    def pprint_getters(self):
        """Return the getters and actual values as list of strings."""

def getp(obj, property: Incomplete | None = None):
    """
    Return the value of an `.Artist`'s *property*, or print all of them.

    Parameters
    ----------
    obj : `~matplotlib.artist.Artist`
        The queried artist; e.g., a `.Line2D`, a `.Text`, or an `~.axes.Axes`.

    property : str or None, default: None
        If *property* is 'somename', this function returns
        ``obj.get_somename()``.

        If it's None (or unset), it *prints* all gettable properties from
        *obj*.  Many properties have aliases for shorter typing, e.g. 'lw' is
        an alias for 'linewidth'.  In the output, aliases and full property
        names will be listed as:

          property or alias = value

        e.g.:

          linewidth or lw = 2

    See Also
    --------
    setp
    """
get = getp

def setp(obj, *args, file: Incomplete | None = None, **kwargs):
    """
    Set one or more properties on an `.Artist`, or list allowed values.

    Parameters
    ----------
    obj : `~matplotlib.artist.Artist` or list of `.Artist`
        The artist(s) whose properties are being set or queried.  When setting
        properties, all artists are affected; when querying the allowed values,
        only the first instance in the sequence is queried.

        For example, two lines can be made thicker and red with a single call:

        >>> x = arange(0, 1, 0.01)
        >>> lines = plot(x, sin(2*pi*x), x, sin(4*pi*x))
        >>> setp(lines, linewidth=2, color='r')

    file : file-like, default: `sys.stdout`
        Where `setp` writes its output when asked to list allowed values.

        >>> with open('output.log') as file:
        ...     setp(line, file=file)

        The default, ``None``, means `sys.stdout`.

    *args, **kwargs
        The properties to set.  The following combinations are supported:

        - Set the linestyle of a line to be dashed:

          >>> line, = plot([1, 2, 3])
          >>> setp(line, linestyle='--')

        - Set multiple properties at once:

          >>> setp(line, linewidth=2, color='r')

        - List allowed values for a line's linestyle:

          >>> setp(line, 'linestyle')
          linestyle: {'-', '--', '-.', ':', '', (offset, on-off-seq), ...}

        - List all properties that can be set, and their allowed values:

          >>> setp(line)
          agg_filter: a filter function, ...
          [long output listing omitted]

        `setp` also supports MATLAB style string/value pairs.  For example, the
        following are equivalent:

        >>> setp(lines, 'linewidth', 2, 'color', 'r')  # MATLAB style
        >>> setp(lines, linewidth=2, color='r')        # Python style

    See Also
    --------
    getp
    """
def kwdoc(artist):
    """
    Inspect an `~matplotlib.artist.Artist` class (using `.ArtistInspector`) and
    return information about its settable properties and their current values.

    Parameters
    ----------
    artist : `~matplotlib.artist.Artist` or an iterable of `Artist`\\s

    Returns
    -------
    str
        The settable properties of *artist*, as plain text if
        :rc:`docstring.hardcopy` is False and as a rst table (intended for
        use in Sphinx) if it is True.
    """
