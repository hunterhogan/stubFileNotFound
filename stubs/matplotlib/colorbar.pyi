import matplotlib.spines as mspines
from _typeshed import Incomplete
from matplotlib import _api as _api, _docstring as _docstring, cbook as cbook, cm as cm, collections as collections, colors as colors, contour as contour, ticker as ticker

_log: Incomplete

def _set_ticks_on_axis_warn(*args, **kwargs) -> None: ...

class _ColorbarSpine(mspines.Spine):
    _ax: Incomplete
    def __init__(self, axes) -> None: ...
    def get_window_extent(self, renderer: Incomplete | None = None): ...
    _path: Incomplete
    _xy: Incomplete
    stale: bool
    def set_xy(self, xy) -> None: ...
    def draw(self, renderer): ...

class _ColorbarAxesLocator:
    """
    Shrink the Axes if there are triangular or rectangular extends.
    """
    _cbar: Incomplete
    _orig_locator: Incomplete
    def __init__(self, cbar) -> None: ...
    def __call__(self, ax, renderer): ...
    def get_subplotspec(self): ...

class Colorbar:
    """
    Draw a colorbar in an existing Axes.

    Typically, colorbars are created using `.Figure.colorbar` or
    `.pyplot.colorbar` and associated with `.ScalarMappable`\\s (such as an
    `.AxesImage` generated via `~.axes.Axes.imshow`).

    In order to draw a colorbar not associated with other elements in the
    figure, e.g. when showing a colormap by itself, one can create an empty
    `.ScalarMappable`, or directly pass *cmap* and *norm* instead of *mappable*
    to `Colorbar`.

    Useful public methods are :meth:`set_label` and :meth:`add_lines`.

    Attributes
    ----------
    ax : `~matplotlib.axes.Axes`
        The `~.axes.Axes` instance in which the colorbar is drawn.
    lines : list
        A list of `.LineCollection` (empty if no lines were drawn).
    dividers : `.LineCollection`
        A LineCollection (empty if *drawedges* is ``False``).
    """
    n_rasterize: int
    mappable: Incomplete
    ax: Incomplete
    alpha: Incomplete
    cmap: Incomplete
    norm: Incomplete
    values: Incomplete
    boundaries: Incomplete
    extend: Incomplete
    _inside: Incomplete
    spacing: Incomplete
    orientation: Incomplete
    drawedges: Incomplete
    _filled: Incomplete
    extendfrac: Incomplete
    extendrect: Incomplete
    _extend_patches: Incomplete
    solids: Incomplete
    solids_patches: Incomplete
    lines: Incomplete
    outline: Incomplete
    dividers: Incomplete
    _locator: Incomplete
    _minorlocator: Incomplete
    _formatter: Incomplete
    _minorformatter: Incomplete
    ticklocation: Incomplete
    _interactive_funcs: Incomplete
    _extend_cid1: Incomplete
    _extend_cid2: Incomplete
    def __init__(self, ax, mappable: Incomplete | None = None, *, alpha: Incomplete | None = None, location: Incomplete | None = None, extend: Incomplete | None = None, extendfrac: Incomplete | None = None, extendrect: bool = False, ticks: Incomplete | None = None, format: Incomplete | None = None, values: Incomplete | None = None, boundaries: Incomplete | None = None, spacing: str = 'uniform', drawedges: bool = False, label: str = '', cmap: Incomplete | None = None, norm: Incomplete | None = None, orientation: Incomplete | None = None, ticklocation: str = 'auto') -> None:
        """
        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            The `~.axes.Axes` instance in which the colorbar is drawn.

        mappable : `.ScalarMappable`
            The mappable whose colormap and norm will be used.

            To show the colors versus index instead of on a 0-1 scale, set the
            mappable's norm to ``colors.NoNorm()``.

        alpha : float
            The colorbar transparency between 0 (transparent) and 1 (opaque).

        location : None or {'left', 'right', 'top', 'bottom'}
            Set the colorbar's *orientation* and *ticklocation*. Colorbars on
            the left and right are vertical, colorbars at the top and bottom
            are horizontal. The *ticklocation* is the same as *location*, so if
            *location* is 'top', the ticks are on the top. *orientation* and/or
            *ticklocation* can be provided as well and overrides the value set by
            *location*, but there will be an error for incompatible combinations.

            .. versionadded:: 3.7

        %(_colormap_kw_doc)s

        Other Parameters
        ----------------
        cmap : `~matplotlib.colors.Colormap`, default: :rc:`image.cmap`
            The colormap to use.  This parameter is ignored, unless *mappable* is
            None.

        norm : `~matplotlib.colors.Normalize`
            The normalization to use.  This parameter is ignored, unless *mappable*
            is None.

        orientation : None or {'vertical', 'horizontal'}
            If None, use the value determined by *location*. If both
            *orientation* and *location* are None then defaults to 'vertical'.

        ticklocation : {'auto', 'left', 'right', 'top', 'bottom'}
            The location of the colorbar ticks. The *ticklocation* must match
            *orientation*. For example, a horizontal colorbar can only have ticks
            at the top or the bottom. If 'auto', the ticks will be the same as
            *location*, so a colorbar to the left will have ticks to the left. If
            *location* is None, the ticks will be at the bottom for a horizontal
            colorbar and at the right for a vertical.
        """
    @property
    def long_axis(self):
        """Axis that has decorations (ticks, etc) on it."""
    @property
    def locator(self):
        """Major tick `.Locator` for the colorbar."""
    @locator.setter
    def locator(self, loc) -> None: ...
    @property
    def minorlocator(self):
        """Minor tick `.Locator` for the colorbar."""
    @minorlocator.setter
    def minorlocator(self, loc) -> None: ...
    @property
    def formatter(self):
        """Major tick label `.Formatter` for the colorbar."""
    @formatter.setter
    def formatter(self, fmt) -> None: ...
    @property
    def minorformatter(self):
        """Minor tick `.Formatter` for the colorbar."""
    @minorformatter.setter
    def minorformatter(self, fmt) -> None: ...
    def _cbar_cla(self) -> None:
        """Function to clear the interactive colorbar state."""
    stale: bool
    def update_normal(self, mappable: Incomplete | None = None) -> None:
        """
        Update solid patches, lines, etc.

        This is meant to be called when the norm of the image or contour plot
        to which this colorbar belongs changes.

        If the norm on the mappable is different than before, this resets the
        locator and formatter for the axis, so if these have been customized,
        they will need to be customized again.  However, if the norm only
        changes values of *vmin*, *vmax* or *cmap* then the old formatter
        and locator will be preserved.
        """
    def _draw_all(self) -> None:
        """
        Calculate any free parameters based on the current cmap and norm,
        and do all the drawing.
        """
    def _add_solids(self, X, Y, C) -> None:
        """Draw the colors; optionally add separators."""
    def _update_dividers(self) -> None: ...
    def _add_solids_patches(self, X, Y, C, mappable) -> None: ...
    def _do_extends(self, ax: Incomplete | None = None) -> None:
        """
        Add the extend tri/rectangles on the outside of the Axes.

        ax is unused, but required due to the callbacks on xlim/ylim changed
        """
    def add_lines(self, *args, **kwargs):
        """
        Draw lines on the colorbar.

        The lines are appended to the list :attr:`lines`.

        Parameters
        ----------
        levels : array-like
            The positions of the lines.
        colors : :mpltype:`color` or list of :mpltype:`color`
            Either a single color applying to all lines or one color value for
            each line.
        linewidths : float or array-like
            Either a single linewidth applying to all lines or one linewidth
            for each line.
        erase : bool, default: True
            Whether to remove any previously added lines.

        Notes
        -----
        Alternatively, this method can also be called with the signature
        ``colorbar.add_lines(contour_set, erase=True)``, in which case
        *levels*, *colors*, and *linewidths* are taken from *contour_set*.
        """
    def update_ticks(self) -> None:
        """
        Set up the ticks and ticklabels. This should not be needed by users.
        """
    def _get_ticker_locator_formatter(self) -> None:
        """
        Return the ``locator`` and ``formatter`` of the colorbar.

        If they have not been defined (i.e. are *None*), the formatter and
        locator are retrieved from the axis, or from the value of the
        boundaries for a boundary norm.

        Called by update_ticks...
        """
    def set_ticks(self, ticks, *, labels: Incomplete | None = None, minor: bool = False, **kwargs) -> None:
        """
        Set tick locations.

        Parameters
        ----------
        ticks : 1D array-like
            List of tick locations.
        labels : list of str, optional
            List of tick labels. If not set, the labels show the data value.
        minor : bool, default: False
            If ``False``, set the major ticks; if ``True``, the minor ticks.
        **kwargs
            `.Text` properties for the labels. These take effect only if you
            pass *labels*. In other cases, please use `~.Axes.tick_params`.
        """
    def get_ticks(self, minor: bool = False):
        """
        Return the ticks as a list of locations.

        Parameters
        ----------
        minor : boolean, default: False
            if True return the minor ticks.
        """
    def set_ticklabels(self, ticklabels, *, minor: bool = False, **kwargs) -> None:
        """
        [*Discouraged*] Set tick labels.

        .. admonition:: Discouraged

            The use of this method is discouraged, because of the dependency
            on tick positions. In most cases, you'll want to use
            ``set_ticks(positions, labels=labels)`` instead.

            If you are using this method, you should always fix the tick
            positions before, e.g. by using `.Colorbar.set_ticks` or by
            explicitly setting a `~.ticker.FixedLocator` on the long axis
            of the colorbar. Otherwise, ticks are free to move and the
            labels may end up in unexpected positions.

        Parameters
        ----------
        ticklabels : sequence of str or of `.Text`
            Texts for labeling each tick location in the sequence set by
            `.Colorbar.set_ticks`; the number of labels must match the number
            of locations.

        update_ticks : bool, default: True
            This keyword argument is ignored and will be removed.
            Deprecated

        minor : bool
            If True, set minor ticks instead of major ticks.

        **kwargs
            `.Text` properties for the labels.
        """
    def minorticks_on(self) -> None:
        """
        Turn on colorbar minor ticks.
        """
    def minorticks_off(self) -> None:
        """Turn the minor ticks of the colorbar off."""
    def set_label(self, label, *, loc: Incomplete | None = None, **kwargs) -> None:
        """
        Add a label to the long axis of the colorbar.

        Parameters
        ----------
        label : str
            The label text.
        loc : str, optional
            The location of the label.

            - For horizontal orientation one of {'left', 'center', 'right'}
            - For vertical orientation one of {'bottom', 'center', 'top'}

            Defaults to :rc:`xaxis.labellocation` or :rc:`yaxis.labellocation`
            depending on the orientation.
        **kwargs
            Keyword arguments are passed to `~.Axes.set_xlabel` /
            `~.Axes.set_ylabel`.
            Supported keywords are *labelpad* and `.Text` properties.
        """
    def set_alpha(self, alpha) -> None:
        """
        Set the transparency between 0 (transparent) and 1 (opaque).

        If an array is provided, *alpha* will be set to None to use the
        transparency values associated with the colormap.
        """
    def _set_scale(self, scale, **kwargs) -> None:
        '''
        Set the colorbar long axis scale.

        Parameters
        ----------
        scale : {"linear", "log", "symlog", "logit", ...} or `.ScaleBase`
            The axis scale type to apply.

        **kwargs
            Different keyword arguments are accepted, depending on the scale.
            See the respective class keyword arguments:

            - `matplotlib.scale.LinearScale`
            - `matplotlib.scale.LogScale`
            - `matplotlib.scale.SymmetricalLogScale`
            - `matplotlib.scale.LogitScale`
            - `matplotlib.scale.FuncScale`
            - `matplotlib.scale.AsinhScale`

        Notes
        -----
        By default, Matplotlib supports the above-mentioned scales.
        Additionally, custom scales may be registered using
        `matplotlib.scale.register_scale`. These scales can then also
        be used here.
        '''
    def remove(self) -> None:
        """
        Remove this colorbar from the figure.

        If the colorbar was created with ``use_gridspec=True`` the previous
        gridspec is restored.
        """
    _values: Incomplete
    _boundaries: Incomplete
    def _process_values(self) -> None:
        """
        Set `_boundaries` and `_values` based on the self.boundaries and
        self.values if not None, or based on the size of the colormap and
        the vmin/vmax of the norm.
        """
    _y: Incomplete
    def _mesh(self):
        """
        Return the coordinate arrays for the colorbar pcolormesh/patches.

        These are scaled between vmin and vmax, and already handle colorbar
        orientation.
        """
    def _forward_boundaries(self, x): ...
    def _inverse_boundaries(self, x): ...
    def _reset_locator_formatter_scale(self) -> None:
        """
        Reset the locator et al to defaults.  Any user-hardcoded changes
        need to be re-entered if this gets called (either at init, or when
        the mappable normal gets changed: Colorbar.update_normal)
        """
    def _locate(self, x):
        """
        Given a set of color data values, return their
        corresponding colorbar data coordinates.
        """
    def _uniform_y(self, N):
        """
        Return colorbar data coordinates for *N* uniformly
        spaced boundaries, plus extension lengths if required.
        """
    def _proportional_y(self):
        """
        Return colorbar data coordinates for the boundaries of
        a proportional colorbar, plus extension lengths if required:
        """
    def _get_extension_lengths(self, frac, automin, automax, default: float = 0.05):
        """
        Return the lengths of colorbar extensions.

        This is a helper method for _uniform_y and _proportional_y.
        """
    def _extend_lower(self):
        """Return whether the lower limit is open ended."""
    def _extend_upper(self):
        """Return whether the upper limit is open ended."""
    def _short_axis(self):
        """Return the short axis"""
    def _get_view(self): ...
    def _set_view(self, view) -> None: ...
    def _set_view_from_bbox(self, bbox, direction: str = 'in', mode: Incomplete | None = None, twinx: bool = False, twiny: bool = False) -> None: ...
    def drag_pan(self, button, key, x, y) -> None: ...
ColorbarBase = Colorbar

def _normalize_location_orientation(location, orientation): ...
def _get_orientation_from_location(location): ...
def _get_ticklocation_from_orientation(orientation): ...
def make_axes(parents, location: Incomplete | None = None, orientation: Incomplete | None = None, fraction: float = 0.15, shrink: float = 1.0, aspect: int = 20, **kwargs):
    """
    Create an `~.axes.Axes` suitable for a colorbar.

    The Axes is placed in the figure of the *parents* Axes, by resizing and
    repositioning *parents*.

    Parameters
    ----------
    parents : `~matplotlib.axes.Axes` or iterable or `numpy.ndarray` of `~.axes.Axes`
        The Axes to use as parents for placing the colorbar.
    %(_make_axes_kw_doc)s

    Returns
    -------
    cax : `~matplotlib.axes.Axes`
        The child Axes.
    kwargs : dict
        The reduced keyword dictionary to be passed when creating the colorbar
        instance.
    """
def make_axes_gridspec(parent, *, location: Incomplete | None = None, orientation: Incomplete | None = None, fraction: float = 0.15, shrink: float = 1.0, aspect: int = 20, **kwargs):
    """
    Create an `~.axes.Axes` suitable for a colorbar.

    The Axes is placed in the figure of the *parent* Axes, by resizing and
    repositioning *parent*.

    This function is similar to `.make_axes` and mostly compatible with it.
    Primary differences are

    - `.make_axes_gridspec` requires the *parent* to have a subplotspec.
    - `.make_axes` positions the Axes in figure coordinates;
      `.make_axes_gridspec` positions it using a subplotspec.
    - `.make_axes` updates the position of the parent.  `.make_axes_gridspec`
      replaces the parent gridspec with a new one.

    Parameters
    ----------
    parent : `~matplotlib.axes.Axes`
        The Axes to use as parent for placing the colorbar.
    %(_make_axes_kw_doc)s

    Returns
    -------
    cax : `~matplotlib.axes.Axes`
        The child Axes.
    kwargs : dict
        The reduced keyword dictionary to be passed when creating the colorbar
        instance.
    """
