import matplotlib.artist as martist
import matplotlib.axis as maxis
import matplotlib.transforms as mtransforms
from _typeshed import Incomplete
from collections.abc import Generator, Sequence
from matplotlib import _api as _api, _docstring as _docstring, cbook as cbook, offsetbox as offsetbox
from matplotlib.cbook import _OrderedSet as _OrderedSet, _check_1d as _check_1d, index_of as index_of
from matplotlib.rcsetup import cycler as cycler, validate_axisbelow as validate_axisbelow

_log: Incomplete

class _axis_method_wrapper:
    '''
    Helper to generate Axes methods wrapping Axis methods.

    After ::

        get_foo = _axis_method_wrapper("xaxis", "get_bar")

    (in the body of a class) ``get_foo`` is a method that forwards it arguments
    to the ``get_bar`` method of the ``xaxis`` attribute, and gets its
    signature and docstring from ``Axis.get_bar``.

    The docstring of ``get_foo`` is built by replacing "this Axis" by "the
    {attr_name}" (i.e., "the xaxis", "the yaxis") in the wrapped method\'s
    dedented docstring; additional replacements can be given in *doc_sub*.
    '''
    attr_name: Incomplete
    method_name: Incomplete
    _missing_subs: Incomplete
    __doc__: Incomplete
    def __init__(self, attr_name, method_name, *, doc_sub: Incomplete | None = None) -> None: ...
    def __set_name__(self, owner, name): ...

class _TransformedBoundsLocator:
    """
    Axes locator for `.Axes.inset_axes` and similarly positioned Axes.

    The locator is a callable object used in `.Axes.set_aspect` to compute the
    Axes location depending on the renderer.
    """
    _bounds: Incomplete
    _transform: Incomplete
    def __init__(self, bounds, transform) -> None:
        """
        *bounds* (a ``[l, b, w, h]`` rectangle) and *transform* together
        specify the position of the inset Axes.
        """
    def __call__(self, ax, renderer): ...

def _process_plot_format(fmt, *, ambiguous_fmt_datakey: bool = False):
    """
    Convert a MATLAB style color/line style format string to a (*linestyle*,
    *marker*, *color*) tuple.

    Example format strings include:

    * 'ko': black circles
    * '.b': blue dots
    * 'r--': red dashed lines
    * 'C2--': the third color in the color cycle, dashed lines

    The format is absolute in the sense that if a linestyle or marker is not
    defined in *fmt*, there is no line or marker. This is expressed by
    returning 'None' for the respective quantity.

    See Also
    --------
    matplotlib.Line2D.lineStyles, matplotlib.colors.cnames
        All possible styles and color format strings.
    """

class _process_plot_var_args:
    """
    Process variable length arguments to `~.Axes.plot`, to support ::

      plot(t, s)
      plot(t1, s1, t2, s2)
      plot(t1, s1, 'ko', t2, s2)
      plot(t1, s1, 'ko', t2, s2, 'r--', t3, e3)

    an arbitrary number of *x*, *y*, *fmt* are allowed
    """
    output: Incomplete
    def __init__(self, output: str = 'Line2D') -> None: ...
    _idx: int
    _cycler_items: Incomplete
    def set_prop_cycle(self, cycler) -> None: ...
    def __call__(self, axes, *args, data: Incomplete | None = None, return_kwargs: bool = False, **kwargs) -> Generator[Incomplete, Incomplete]: ...
    def get_next_color(self):
        """Return the next color in the cycle."""
    def _getdefaults(self, kw, ignore=...):
        """
        If some keys in the property cycle (excluding those in the set
        *ignore*) are absent or set to None in the dict *kw*, return a copy
        of the next entry in the property cycle, excluding keys in *ignore*.
        Otherwise, don't advance the property cycle, and return an empty dict.
        """
    def _setdefaults(self, defaults, kw) -> None:
        """
        Add to the dict *kw* the entries in the dict *default* that are absent
        or set to None in *kw*.
        """
    def _make_line(self, axes, x, y, kw, kwargs): ...
    def _make_coordinates(self, axes, x, y, kw, kwargs): ...
    def _make_polygon(self, axes, x, y, kw, kwargs): ...
    def _plot_args(self, axes, tup, kwargs, *, return_kwargs: bool = False, ambiguous_fmt_datakey: bool = False):
        """
        Process the arguments of ``plot([x], y, [fmt], **kwargs)`` calls.

        This processes a single set of ([x], y, [fmt]) parameters; i.e. for
        ``plot(x, y, x2, y2)`` it will be called twice. Once for (x, y) and
        once for (x2, y2).

        x and y may be 2D and thus can still represent multiple datasets.

        For multiple datasets, if the keyword argument *label* is a list, this
        will unpack the list and assign the individual labels to the datasets.

        Parameters
        ----------
        tup : tuple
            A tuple of the positional parameters. This can be one of

            - (y,)
            - (x, y)
            - (y, fmt)
            - (x, y, fmt)

        kwargs : dict
            The keyword arguments passed to ``plot()``.

        return_kwargs : bool
            Whether to also return the effective keyword arguments after label
            unpacking as well.

        ambiguous_fmt_datakey : bool
            Whether the format string in *tup* could also have been a
            misspelled data key.

        Returns
        -------
        result
            If *return_kwargs* is false, a list of Artists representing the
            dataset(s).
            If *return_kwargs* is true, a list of (Artist, effective_kwargs)
            representing the dataset(s). See *return_kwargs*.
            The Artist is either `.Line2D` (if called from ``plot()``) or
            `.Polygon` otherwise.
        """

class _AxesBase(martist.Artist):
    name: str
    _axis_names: Incomplete
    _shared_axes: Incomplete
    _twinned_axes: Incomplete
    _subclass_uses_cla: bool
    dataLim: mtransforms.Bbox
    xaxis: maxis.XAxis
    yaxis: maxis.YAxis
    @property
    def _axis_map(self):
        """A mapping of axis names, e.g. 'x', to `Axis` instances."""
    def __str__(self) -> str: ...
    _position: Incomplete
    _originalPosition: Incomplete
    axes: Incomplete
    _aspect: str
    _adjustable: str
    _anchor: str
    _stale_viewlims: Incomplete
    _forward_navigation_events: Incomplete
    _sharex: Incomplete
    _sharey: Incomplete
    _subplotspec: Incomplete
    _axes_locator: Incomplete
    _children: Incomplete
    _colorbars: Incomplete
    spines: Incomplete
    _facecolor: Incomplete
    _frameon: Incomplete
    _rasterization_zorder: Incomplete
    fmt_xdata: Incomplete
    fmt_ydata: Incomplete
    def __init__(self, fig, *args, facecolor: Incomplete | None = None, frameon: bool = True, sharex: Incomplete | None = None, sharey: Incomplete | None = None, label: str = '', xscale: Incomplete | None = None, yscale: Incomplete | None = None, box_aspect: Incomplete | None = None, forward_navigation_events: str = 'auto', **kwargs) -> None:
        '''
        Build an Axes in a figure.

        Parameters
        ----------
        fig : `~matplotlib.figure.Figure`
            The Axes is built in the `.Figure` *fig*.

        *args
            ``*args`` can be a single ``(left, bottom, width, height)``
            rectangle or a single `.Bbox`.  This specifies the rectangle (in
            figure coordinates) where the Axes is positioned.

            ``*args`` can also consist of three numbers or a single three-digit
            number; in the latter case, the digits are considered as
            independent numbers.  The numbers are interpreted as ``(nrows,
            ncols, index)``: ``(nrows, ncols)`` specifies the size of an array
            of subplots, and ``index`` is the 1-based index of the subplot
            being created.  Finally, ``*args`` can also directly be a
            `.SubplotSpec` instance.

        sharex, sharey : `~matplotlib.axes.Axes`, optional
            The x- or y-`~.matplotlib.axis` is shared with the x- or y-axis in
            the input `~.axes.Axes`.  Note that it is not possible to unshare
            axes.

        frameon : bool, default: True
            Whether the Axes frame is visible.

        box_aspect : float, optional
            Set a fixed aspect for the Axes box, i.e. the ratio of height to
            width. See `~.axes.Axes.set_box_aspect` for details.

        forward_navigation_events : bool or "auto", default: "auto"
            Control whether pan/zoom events are passed through to Axes below
            this one. "auto" is *True* for axes with an invisible patch and
            *False* otherwise.

        **kwargs
            Other optional keyword arguments:

            %(Axes:kwdoc)s

        Returns
        -------
        `~.axes.Axes`
            The new `~.axes.Axes` object.
        '''
    def __init_subclass__(cls, **kwargs) -> None: ...
    def __getstate__(self): ...
    __dict__: Incomplete
    _stale: bool
    def __setstate__(self, state) -> None: ...
    def __repr__(self) -> str: ...
    def get_subplotspec(self):
        """Return the `.SubplotSpec` associated with the subplot, or None."""
    def set_subplotspec(self, subplotspec) -> None:
        """Set the `.SubplotSpec`. associated with the subplot."""
    def get_gridspec(self):
        """Return the `.GridSpec` associated with the subplot, or None."""
    def get_window_extent(self, renderer: Incomplete | None = None):
        """
        Return the Axes bounding box in display space.

        This bounding box does not include the spines, ticks, ticklabels,
        or other labels.  For a bounding box including these elements use
        `~matplotlib.axes.Axes.get_tightbbox`.

        See Also
        --------
        matplotlib.axes.Axes.get_tightbbox
        matplotlib.axis.Axis.get_tightbbox
        matplotlib.spines.Spine.get_window_extent
        """
    def _init_axis(self) -> None: ...
    bbox: Incomplete
    _viewLim: Incomplete
    transScale: Incomplete
    def set_figure(self, fig) -> None: ...
    def _unstale_viewLim(self) -> None: ...
    @property
    def viewLim(self):
        """The view limits as `.Bbox` in data coordinates."""
    _tight: Incomplete
    def _request_autoscale_view(self, axis: str = 'all', tight: Incomplete | None = None) -> None:
        '''
        Mark a single axis, or all of them, as stale wrt. autoscaling.

        No computation is performed until the next autoscaling; thus, separate
        calls to control individual axises incur negligible performance cost.

        Parameters
        ----------
        axis : str, default: "all"
            Either an element of ``self._axis_names``, or "all".
        tight : bool or None, default: None
        '''
    transAxes: Incomplete
    transLimits: Incomplete
    transData: Incomplete
    _xaxis_transform: Incomplete
    _yaxis_transform: Incomplete
    def _set_lim_and_transforms(self) -> None:
        """
        Set the *_xaxis_transform*, *_yaxis_transform*, *transScale*,
        *transData*, *transLimits* and *transAxes* transformations.

        .. note::

            This method is primarily used by rectilinear projections of the
            `~matplotlib.axes.Axes` class, and is meant to be overridden by
            new kinds of projection Axes that need different transformations
            and limits. (See `~matplotlib.projections.polar.PolarAxes` for an
            example.)
        """
    def get_xaxis_transform(self, which: str = 'grid'):
        """
        Get the transformation used for drawing x-axis labels, ticks
        and gridlines.  The x-direction is in data coordinates and the
        y-direction is in axis coordinates.

        .. note::

            This transformation is primarily used by the
            `~matplotlib.axis.Axis` class, and is meant to be
            overridden by new kinds of projections that may need to
            place axis elements in different locations.

        Parameters
        ----------
        which : {'grid', 'tick1', 'tick2'}
        """
    def get_xaxis_text1_transform(self, pad_points):
        """
        Returns
        -------
        transform : Transform
            The transform used for drawing x-axis labels, which will add
            *pad_points* of padding (in points) between the axis and the label.
            The x-direction is in data coordinates and the y-direction is in
            axis coordinates
        valign : {'center', 'top', 'bottom', 'baseline', 'center_baseline'}
            The text vertical alignment.
        halign : {'center', 'left', 'right'}
            The text horizontal alignment.

        Notes
        -----
        This transformation is primarily used by the `~matplotlib.axis.Axis`
        class, and is meant to be overridden by new kinds of projections that
        may need to place axis elements in different locations.
        """
    def get_xaxis_text2_transform(self, pad_points):
        """
        Returns
        -------
        transform : Transform
            The transform used for drawing secondary x-axis labels, which will
            add *pad_points* of padding (in points) between the axis and the
            label.  The x-direction is in data coordinates and the y-direction
            is in axis coordinates
        valign : {'center', 'top', 'bottom', 'baseline', 'center_baseline'}
            The text vertical alignment.
        halign : {'center', 'left', 'right'}
            The text horizontal alignment.

        Notes
        -----
        This transformation is primarily used by the `~matplotlib.axis.Axis`
        class, and is meant to be overridden by new kinds of projections that
        may need to place axis elements in different locations.
        """
    def get_yaxis_transform(self, which: str = 'grid'):
        """
        Get the transformation used for drawing y-axis labels, ticks
        and gridlines.  The x-direction is in axis coordinates and the
        y-direction is in data coordinates.

        .. note::

            This transformation is primarily used by the
            `~matplotlib.axis.Axis` class, and is meant to be
            overridden by new kinds of projections that may need to
            place axis elements in different locations.

        Parameters
        ----------
        which : {'grid', 'tick1', 'tick2'}
        """
    def get_yaxis_text1_transform(self, pad_points):
        """
        Returns
        -------
        transform : Transform
            The transform used for drawing y-axis labels, which will add
            *pad_points* of padding (in points) between the axis and the label.
            The x-direction is in axis coordinates and the y-direction is in
            data coordinates
        valign : {'center', 'top', 'bottom', 'baseline', 'center_baseline'}
            The text vertical alignment.
        halign : {'center', 'left', 'right'}
            The text horizontal alignment.

        Notes
        -----
        This transformation is primarily used by the `~matplotlib.axis.Axis`
        class, and is meant to be overridden by new kinds of projections that
        may need to place axis elements in different locations.
        """
    def get_yaxis_text2_transform(self, pad_points):
        """
        Returns
        -------
        transform : Transform
            The transform used for drawing secondart y-axis labels, which will
            add *pad_points* of padding (in points) between the axis and the
            label.  The x-direction is in axis coordinates and the y-direction
            is in data coordinates
        valign : {'center', 'top', 'bottom', 'baseline', 'center_baseline'}
            The text vertical alignment.
        halign : {'center', 'left', 'right'}
            The text horizontal alignment.

        Notes
        -----
        This transformation is primarily used by the `~matplotlib.axis.Axis`
        class, and is meant to be overridden by new kinds of projections that
        may need to place axis elements in different locations.
        """
    def _update_transScale(self) -> None: ...
    def get_position(self, original: bool = False):
        """
        Return the position of the Axes within the figure as a `.Bbox`.

        Parameters
        ----------
        original : bool
            If ``True``, return the original position. Otherwise, return the
            active position. For an explanation of the positions see
            `.set_position`.

        Returns
        -------
        `.Bbox`

        """
    def set_position(self, pos, which: str = 'both') -> None:
        """
        Set the Axes position.

        Axes have two position attributes. The 'original' position is the
        position allocated for the Axes. The 'active' position is the
        position the Axes is actually drawn at. These positions are usually
        the same unless a fixed aspect is set to the Axes. See
        `.Axes.set_aspect` for details.

        Parameters
        ----------
        pos : [left, bottom, width, height] or `~matplotlib.transforms.Bbox`
            The new position of the Axes in `.Figure` coordinates.

        which : {'both', 'active', 'original'}, default: 'both'
            Determines which position variables to change.

        See Also
        --------
        matplotlib.transforms.Bbox.from_bounds
        matplotlib.transforms.Bbox.from_extents
        """
    stale: bool
    def _set_position(self, pos, which: str = 'both') -> None:
        """
        Private version of set_position.

        Call this internally to get the same functionality of `set_position`,
        but not to take the axis out of the constrained_layout hierarchy.
        """
    def reset_position(self) -> None:
        """
        Reset the active position to the original position.

        This undoes changes to the active position (as defined in
        `.set_position`) which may have been performed to satisfy fixed-aspect
        constraints.
        """
    def set_axes_locator(self, locator) -> None:
        """
        Set the Axes locator.

        Parameters
        ----------
        locator : Callable[[Axes, Renderer], Bbox]
        """
    def get_axes_locator(self):
        """
        Return the axes_locator.
        """
    def _set_artist_props(self, a) -> None:
        """Set the boilerplate props for artists added to Axes."""
    def _gen_axes_patch(self):
        """
        Returns
        -------
        Patch
            The patch used to draw the background of the Axes.  It is also used
            as the clipping path for any data elements on the Axes.

            In the standard Axes, this is a rectangle, but in other projections
            it may not be.

        Notes
        -----
        Intended to be overridden by new projection types.
        """
    def _gen_axes_spines(self, locations: Incomplete | None = None, offset: float = 0.0, units: str = 'inches'):
        """
        Returns
        -------
        dict
            Mapping of spine names to `.Line2D` or `.Patch` instances that are
            used to draw Axes spines.

            In the standard Axes, spines are single line segments, but in other
            projections they may not be.

        Notes
        -----
        Intended to be overridden by new projection types.
        """
    def sharex(self, other) -> None:
        """
        Share the x-axis with *other*.

        This is equivalent to passing ``sharex=other`` when constructing the
        Axes, and cannot be used if the x-axis is already being shared with
        another Axes.  Note that it is not possible to unshare axes.
        """
    def sharey(self, other) -> None:
        """
        Share the y-axis with *other*.

        This is equivalent to passing ``sharey=other`` when constructing the
        Axes, and cannot be used if the y-axis is already being shared with
        another Axes.  Note that it is not possible to unshare axes.
        """
    ignore_existing_data_limits: bool
    callbacks: Incomplete
    _xmargin: Incomplete
    _ymargin: Incomplete
    _use_sticky_edges: bool
    _get_lines: Incomplete
    _get_patches_for_fill: Incomplete
    _gridOn: Incomplete
    _mouseover_set: Incomplete
    child_axes: Incomplete
    _current_image: Incomplete
    _projection_init: Incomplete
    legend_: Incomplete
    containers: Incomplete
    _autotitlepos: bool
    title: Incomplete
    _left_title: Incomplete
    _right_title: Incomplete
    patch: Incomplete
    def __clear(self) -> None:
        """Clear the Axes."""
    def clear(self) -> None:
        """Clear the Axes."""
    def cla(self) -> None:
        """Clear the Axes."""
    class ArtistList(Sequence):
        """
        A sublist of Axes children based on their type.

        The type-specific children sublists were made immutable in Matplotlib
        3.7.  In the future these artist lists may be replaced by tuples. Use
        as if this is a tuple already.
        """
        _axes: Incomplete
        _prop_name: Incomplete
        _type_check: Incomplete
        def __init__(self, axes, prop_name, valid_types: Incomplete | None = None, invalid_types: Incomplete | None = None) -> None:
            """
            Parameters
            ----------
            axes : `~matplotlib.axes.Axes`
                The Axes from which this sublist will pull the children
                Artists.
            prop_name : str
                The property name used to access this sublist from the Axes;
                used to generate deprecation warnings.
            valid_types : list of type, optional
                A list of types that determine which children will be returned
                by this sublist. If specified, then the Artists in the sublist
                must be instances of any of these types. If unspecified, then
                any type of Artist is valid (unless limited by
                *invalid_types*.)
            invalid_types : tuple, optional
                A list of types that determine which children will *not* be
                returned by this sublist. If specified, then Artists in the
                sublist will never be an instance of these types. Otherwise, no
                types will be excluded.
            """
        def __repr__(self) -> str: ...
        def __len__(self) -> int: ...
        def __iter__(self): ...
        def __getitem__(self, key): ...
        def __add__(self, other): ...
        def __radd__(self, other): ...
    @property
    def artists(self): ...
    @property
    def collections(self): ...
    @property
    def images(self): ...
    @property
    def lines(self): ...
    @property
    def patches(self): ...
    @property
    def tables(self): ...
    @property
    def texts(self): ...
    def get_facecolor(self):
        """Get the facecolor of the Axes."""
    def set_facecolor(self, color):
        """
        Set the facecolor of the Axes.

        Parameters
        ----------
        color : :mpltype:`color`
        """
    titleOffsetTrans: Incomplete
    def _set_title_offset_trans(self, title_offset_points) -> None:
        """
        Set the offset for the title either from :rc:`axes.titlepad`
        or from set_title kwarg ``pad``.
        """
    def set_prop_cycle(self, *args, **kwargs) -> None:
        """
        Set the property cycle of the Axes.

        The property cycle controls the style properties such as color,
        marker and linestyle of future plot commands. The style properties
        of data already added to the Axes are not modified.

        Call signatures::

          set_prop_cycle(cycler)
          set_prop_cycle(label=values, label2=values2, ...)
          set_prop_cycle(label, values)

        Form 1 sets given `~cycler.Cycler` object.

        Form 2 creates a `~cycler.Cycler` which cycles over one or more
        properties simultaneously and set it as the property cycle of the
        Axes. If multiple properties are given, their value lists must have
        the same length. This is just a shortcut for explicitly creating a
        cycler and passing it to the function, i.e. it's short for
        ``set_prop_cycle(cycler(label=values, label2=values2, ...))``.

        Form 3 creates a `~cycler.Cycler` for a single property and set it
        as the property cycle of the Axes. This form exists for compatibility
        with the original `cycler.cycler` interface. Its use is discouraged
        in favor of the kwarg form, i.e. ``set_prop_cycle(label=values)``.

        Parameters
        ----------
        cycler : `~cycler.Cycler`
            Set the given Cycler. *None* resets to the cycle defined by the
            current style.

            .. ACCEPTS: `~cycler.Cycler`

        label : str
            The property key. Must be a valid `.Artist` property.
            For example, 'color' or 'linestyle'. Aliases are allowed,
            such as 'c' for 'color' and 'lw' for 'linewidth'.

        values : iterable
            Finite-length iterable of the property values. These values
            are validated and will raise a ValueError if invalid.

        See Also
        --------
        matplotlib.rcsetup.cycler
            Convenience function for creating validated cyclers for properties.
        cycler.cycler
            The original function for creating unvalidated cyclers.

        Examples
        --------
        Setting the property cycle for a single property:

        >>> ax.set_prop_cycle(color=['red', 'green', 'blue'])

        Setting the property cycle for simultaneously cycling over multiple
        properties (e.g. red circle, green plus, blue cross):

        >>> ax.set_prop_cycle(color=['red', 'green', 'blue'],
        ...                   marker=['o', '+', 'x'])

        """
    def get_aspect(self):
        '''
        Return the aspect ratio of the Axes scaling.

        This is either "auto" or a float giving the ratio of y/x-scale.
        '''
    def set_aspect(self, aspect, adjustable: Incomplete | None = None, anchor: Incomplete | None = None, share: bool = False) -> None:
        """
        Set the aspect ratio of the Axes scaling, i.e. y/x-scale.

        Parameters
        ----------
        aspect : {'auto', 'equal'} or float
            Possible values:

            - 'auto': fill the position rectangle with data.
            - 'equal': same as ``aspect=1``, i.e. same scaling for x and y.
            - *float*: The displayed size of 1 unit in y-data coordinates will
              be *aspect* times the displayed size of 1 unit in x-data
              coordinates; e.g. for ``aspect=2`` a square in data coordinates
              will be rendered with a height of twice its width.

        adjustable : None or {'box', 'datalim'}, optional
            If not ``None``, this defines which parameter will be adjusted to
            meet the required aspect. See `.set_adjustable` for further
            details.

        anchor : None or str or (float, float), optional
            If not ``None``, this defines where the Axes will be drawn if there
            is extra space due to aspect constraints. The most common way
            to specify the anchor are abbreviations of cardinal directions:

            =====   =====================
            value   description
            =====   =====================
            'C'     centered
            'SW'    lower left corner
            'S'     middle of bottom edge
            'SE'    lower right corner
            etc.
            =====   =====================

            See `~.Axes.set_anchor` for further details.

        share : bool, default: False
            If ``True``, apply the settings to all shared Axes.

        See Also
        --------
        matplotlib.axes.Axes.set_adjustable
            Set how the Axes adjusts to achieve the required aspect ratio.
        matplotlib.axes.Axes.set_anchor
            Set the position in case of extra space.
        """
    def get_adjustable(self):
        """
        Return whether the Axes will adjust its physical dimension ('box') or
        its data limits ('datalim') to achieve the desired aspect ratio.

        See Also
        --------
        matplotlib.axes.Axes.set_adjustable
            Set how the Axes adjusts to achieve the required aspect ratio.
        matplotlib.axes.Axes.set_aspect
            For a description of aspect handling.
        """
    def set_adjustable(self, adjustable, share: bool = False) -> None:
        """
        Set how the Axes adjusts to achieve the required aspect ratio.

        Parameters
        ----------
        adjustable : {'box', 'datalim'}
            If 'box', change the physical dimensions of the Axes.
            If 'datalim', change the ``x`` or ``y`` data limits. This
            may ignore explicitly defined axis limits.

        share : bool, default: False
            If ``True``, apply the settings to all shared Axes.

        See Also
        --------
        matplotlib.axes.Axes.set_aspect
            For a description of aspect handling.

        Notes
        -----
        Shared Axes (of which twinned Axes are a special case)
        impose restrictions on how aspect ratios can be imposed.
        For twinned Axes, use 'datalim'.  For Axes that share both
        x and y, use 'box'.  Otherwise, either 'datalim' or 'box'
        may be used.  These limitations are partly a requirement
        to avoid over-specification, and partly a result of the
        particular implementation we are currently using, in
        which the adjustments for aspect ratios are done sequentially
        and independently on each Axes as it is drawn.
        """
    def get_box_aspect(self):
        """
        Return the Axes box aspect, i.e. the ratio of height to width.

        The box aspect is ``None`` (i.e. chosen depending on the available
        figure space) unless explicitly specified.

        See Also
        --------
        matplotlib.axes.Axes.set_box_aspect
            for a description of box aspect.
        matplotlib.axes.Axes.set_aspect
            for a description of aspect handling.
        """
    def set_box_aspect(self, aspect: Incomplete | None = None) -> None:
        """
        Set the Axes box aspect, i.e. the ratio of height to width.

        This defines the aspect of the Axes in figure space and is not to be
        confused with the data aspect (see `~.Axes.set_aspect`).

        Parameters
        ----------
        aspect : float or None
            Changes the physical dimensions of the Axes, such that the ratio
            of the Axes height to the Axes width in physical units is equal to
            *aspect*. Defining a box aspect will change the *adjustable*
            property to 'datalim' (see `~.Axes.set_adjustable`).

            *None* will disable a fixed box aspect so that height and width
            of the Axes are chosen independently.

        See Also
        --------
        matplotlib.axes.Axes.set_aspect
            for a description of aspect handling.
        """
    def get_anchor(self):
        """
        Get the anchor location.

        See Also
        --------
        matplotlib.axes.Axes.set_anchor
            for a description of the anchor.
        matplotlib.axes.Axes.set_aspect
            for a description of aspect handling.
        """
    def set_anchor(self, anchor, share: bool = False) -> None:
        """
        Define the anchor location.

        The actual drawing area (active position) of the Axes may be smaller
        than the Bbox (original position) when a fixed aspect is required. The
        anchor defines where the drawing area will be located within the
        available space.

        Parameters
        ----------
        anchor : (float, float) or {'C', 'SW', 'S', 'SE', 'E', 'NE', ...}
            Either an (*x*, *y*) pair of relative coordinates (0 is left or
            bottom, 1 is right or top), 'C' (center), or a cardinal direction
            ('SW', southwest, is bottom left, etc.).  str inputs are shorthands
            for (*x*, *y*) coordinates, as shown in the following diagram::

               ┌─────────────────┬─────────────────┬─────────────────┐
               │ 'NW' (0.0, 1.0) │ 'N' (0.5, 1.0)  │ 'NE' (1.0, 1.0) │
               ├─────────────────┼─────────────────┼─────────────────┤
               │ 'W'  (0.0, 0.5) │ 'C' (0.5, 0.5)  │ 'E'  (1.0, 0.5) │
               ├─────────────────┼─────────────────┼─────────────────┤
               │ 'SW' (0.0, 0.0) │ 'S' (0.5, 0.0)  │ 'SE' (1.0, 0.0) │
               └─────────────────┴─────────────────┴─────────────────┘

        share : bool, default: False
            If ``True``, apply the settings to all shared Axes.

        See Also
        --------
        matplotlib.axes.Axes.set_aspect
            for a description of aspect handling.
        """
    def get_data_ratio(self):
        """
        Return the aspect ratio of the scaled data.

        Notes
        -----
        This method is intended to be overridden by new projection types.
        """
    def apply_aspect(self, position: Incomplete | None = None) -> None:
        """
        Adjust the Axes for a specified data aspect ratio.

        Depending on `.get_adjustable` this will modify either the
        Axes box (position) or the view limits. In the former case,
        `~matplotlib.axes.Axes.get_anchor` will affect the position.

        Parameters
        ----------
        position : None or .Bbox

            .. note::
                This parameter exists for historic reasons and is considered
                internal. End users should not use it.

            If not ``None``, this defines the position of the
            Axes within the figure as a Bbox. See `~.Axes.get_position`
            for further details.

        Notes
        -----
        This is called automatically when each Axes is drawn.  You may need
        to call it yourself if you need to update the Axes position and/or
        view limits before the Figure is drawn.

        An alternative with a broader scope is `.Figure.draw_without_rendering`,
        which updates all stale components of a figure, not only the positioning /
        view limits of a single Axes.

        See Also
        --------
        matplotlib.axes.Axes.set_aspect
            For a description of aspect ratio handling.
        matplotlib.axes.Axes.set_adjustable
            Set how the Axes adjusts to achieve the required aspect ratio.
        matplotlib.axes.Axes.set_anchor
            Set the position in case of extra space.
        matplotlib.figure.Figure.draw_without_rendering
            Update all stale components of a figure.

        Examples
        --------
        A typical usage example would be the following. `~.Axes.imshow` sets the
        aspect to 1, but adapting the Axes position and extent to reflect this is
        deferred until rendering for performance reasons. If you want to know the
        Axes size before, you need to call `.apply_aspect` to get the correct
        values.

        >>> fig, ax = plt.subplots()
        >>> ax.imshow(np.zeros((3, 3)))
        >>> ax.bbox.width, ax.bbox.height
        (496.0, 369.59999999999997)
        >>> ax.apply_aspect()
        >>> ax.bbox.width, ax.bbox.height
        (369.59999999999997, 369.59999999999997)
        """
    def axis(self, arg: Incomplete | None = None, /, *, emit: bool = True, **kwargs):
        """
        Convenience method to get or set some axis properties.

        Call signatures::

          xmin, xmax, ymin, ymax = axis()
          xmin, xmax, ymin, ymax = axis([xmin, xmax, ymin, ymax])
          xmin, xmax, ymin, ymax = axis(option)
          xmin, xmax, ymin, ymax = axis(**kwargs)

        Parameters
        ----------
        xmin, xmax, ymin, ymax : float, optional
            The axis limits to be set.  This can also be achieved using ::

                ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))

        option : bool or str
            If a bool, turns axis lines and labels on or off. If a string,
            possible values are:

            ================ ===========================================================
            Value            Description
            ================ ===========================================================
            'off' or `False` Hide all axis decorations, i.e. axis labels, spines,
                             tick marks, tick labels, and grid lines.
                             This is the same as `~.Axes.set_axis_off()`.
            'on' or `True`   Do not hide all axis decorations, i.e. axis labels, spines,
                             tick marks, tick labels, and grid lines.
                             This is the same as `~.Axes.set_axis_on()`.
            'equal'          Set equal scaling (i.e., make circles circular) by
                             changing the axis limits. This is the same as
                             ``ax.set_aspect('equal', adjustable='datalim')``.
                             Explicit data limits may not be respected in this case.
            'scaled'         Set equal scaling (i.e., make circles circular) by
                             changing dimensions of the plot box. This is the same as
                             ``ax.set_aspect('equal', adjustable='box', anchor='C')``.
                             Additionally, further autoscaling will be disabled.
            'tight'          Set limits just large enough to show all data, then
                             disable further autoscaling.
            'auto'           Automatic scaling (fill plot box with data).
            'image'          'scaled' with axis limits equal to data limits.
            'square'         Square plot; similar to 'scaled', but initially forcing
                             ``xmax-xmin == ymax-ymin``.
            ================ ===========================================================

        emit : bool, default: True
            Whether observers are notified of the axis limit change.
            This option is passed on to `~.Axes.set_xlim` and
            `~.Axes.set_ylim`.

        Returns
        -------
        xmin, xmax, ymin, ymax : float
            The axis limits.

        See Also
        --------
        matplotlib.axes.Axes.set_xlim
        matplotlib.axes.Axes.set_ylim

        Notes
        -----
        For 3D Axes, this method additionally takes *zmin*, *zmax* as
        parameters and likewise returns them.
        """
    def get_legend(self):
        """Return the `.Legend` instance, or None if no legend is defined."""
    def get_images(self):
        """Return a list of `.AxesImage`\\s contained by the Axes."""
    def get_lines(self):
        """Return a list of lines contained by the Axes."""
    def get_xaxis(self):
        """
        [*Discouraged*] Return the XAxis instance.

        .. admonition:: Discouraged

            The use of this function is discouraged. You should instead
            directly access the attribute `~.Axes.xaxis`.
        """
    def get_yaxis(self):
        """
        [*Discouraged*] Return the YAxis instance.

        .. admonition:: Discouraged

            The use of this function is discouraged. You should instead
            directly access the attribute `~.Axes.yaxis`.
        """
    get_xgridlines: Incomplete
    get_xticklines: Incomplete
    get_ygridlines: Incomplete
    get_yticklines: Incomplete
    def _sci(self, im) -> None:
        """
        Set the current image.

        This image will be the target of colormap functions like
        ``pyplot.viridis``, and other functions such as `~.pyplot.clim`.  The
        current image is an attribute of the current Axes.
        """
    def _gci(self):
        """Helper for `~matplotlib.pyplot.gci`; do not use elsewhere."""
    def has_data(self):
        """
        Return whether any artists have been added to the Axes.

        This should not be used to determine whether the *dataLim*
        need to be updated, and may not actually be useful for
        anything.
        """
    def add_artist(self, a):
        '''
        Add an `.Artist` to the Axes; return the artist.

        Use `add_artist` only for artists for which there is no dedicated
        "add" method; and if necessary, use a method such as `update_datalim`
        to manually update the `~.Axes.dataLim` if the artist is to be included
        in autoscaling.

        If no ``transform`` has been specified when creating the artist (e.g.
        ``artist.get_transform() == None``) then the transform is set to
        ``ax.transData``.
        '''
    def add_child_axes(self, ax):
        """
        Add an `.Axes` to the Axes' children; return the child Axes.

        This is the lowlevel version.  See `.axes.Axes.inset_axes`.
        """
    def add_collection(self, collection, autolim: bool = True):
        """
        Add a `.Collection` to the Axes; return the collection.
        """
    def add_image(self, image):
        """
        Add an `.AxesImage` to the Axes; return the image.
        """
    def _update_image_limits(self, image) -> None: ...
    def add_line(self, line):
        """
        Add a `.Line2D` to the Axes; return the line.
        """
    def _add_text(self, txt):
        """
        Add a `.Text` to the Axes; return the text.
        """
    def _update_line_limits(self, line) -> None:
        """
        Figures out the data limit of the given line, updating `.Axes.dataLim`.
        """
    def add_patch(self, p):
        """
        Add a `.Patch` to the Axes; return the patch.
        """
    def _update_patch_limits(self, patch) -> None:
        """Update the data limits for the given patch."""
    def add_table(self, tab):
        """
        Add a `.Table` to the Axes; return the table.
        """
    def add_container(self, container):
        """
        Add a `.Container` to the Axes' containers; return the container.
        """
    def _unit_change_handler(self, axis_name, event: Incomplete | None = None):
        """
        Process axis units changes: requests updates to data and view limits.
        """
    def relim(self, visible_only: bool = False) -> None:
        """
        Recompute the data limits based on current artists.

        At present, `.Collection` instances are not supported.

        Parameters
        ----------
        visible_only : bool, default: False
            Whether to exclude invisible artists.
        """
    def update_datalim(self, xys, updatex: bool = True, updatey: bool = True) -> None:
        """
        Extend the `~.Axes.dataLim` Bbox to include the given points.

        If no data is set currently, the Bbox will ignore its limits and set
        the bound to be the bounds of the xydata (*xys*). Otherwise, it will
        compute the bounds of the union of its current data and the data in
        *xys*.

        Parameters
        ----------
        xys : 2D array-like
            The points to include in the data limits Bbox. This can be either
            a list of (x, y) tuples or a (N, 2) array.

        updatex, updatey : bool, default: True
            Whether to update the x/y limits.
        """
    def _process_unit_info(self, datasets: Incomplete | None = None, kwargs: Incomplete | None = None, *, convert: bool = True):
        """
        Set axis units based on *datasets* and *kwargs*, and optionally apply
        unit conversions to *datasets*.

        Parameters
        ----------
        datasets : list
            List of (axis_name, dataset) pairs (where the axis name is defined
            as in `._axis_map`).  Individual datasets can also be None
            (which gets passed through).
        kwargs : dict
            Other parameters from which unit info (i.e., the *xunits*,
            *yunits*, *zunits* (for 3D Axes), *runits* and *thetaunits* (for
            polar) entries) is popped, if present.  Note that this dict is
            mutated in-place!
        convert : bool, default: True
            Whether to return the original datasets or the converted ones.

        Returns
        -------
        list
            Either the original datasets if *convert* is False, or the
            converted ones if *convert* is True (the default).
        """
    def in_axes(self, mouseevent):
        """
        Return whether the given event (in display coords) is in the Axes.
        """
    get_autoscalex_on: Incomplete
    get_autoscaley_on: Incomplete
    set_autoscalex_on: Incomplete
    set_autoscaley_on: Incomplete
    def get_autoscale_on(self):
        """Return True if each axis is autoscaled, False otherwise."""
    def set_autoscale_on(self, b) -> None:
        """
        Set whether autoscaling is applied to each axis on the next draw or
        call to `.Axes.autoscale_view`.

        Parameters
        ----------
        b : bool
        """
    @property
    def use_sticky_edges(self):
        """
        When autoscaling, whether to obey all `.Artist.sticky_edges`.

        Default is ``True``.

        Setting this to ``False`` ensures that the specified margins
        will be applied, even if the plot includes an image, for
        example, which would otherwise force a view limit to coincide
        with its data limit.

        The changing this property does not change the plot until
        `autoscale` or `autoscale_view` is called.
        """
    @use_sticky_edges.setter
    def use_sticky_edges(self, b) -> None: ...
    def get_xmargin(self):
        """
        Retrieve autoscaling margin of the x-axis.

        .. versionadded:: 3.9

        Returns
        -------
        xmargin : float

        See Also
        --------
        matplotlib.axes.Axes.set_xmargin
        """
    def get_ymargin(self):
        """
        Retrieve autoscaling margin of the y-axis.

        .. versionadded:: 3.9

        Returns
        -------
        ymargin : float

        See Also
        --------
        matplotlib.axes.Axes.set_ymargin
        """
    def set_xmargin(self, m) -> None:
        """
        Set padding of X data limits prior to autoscaling.

        *m* times the data interval will be added to each end of that interval
        before it is used in autoscaling.  If *m* is negative, this will clip
        the data range instead of expanding it.

        For example, if your data is in the range [0, 2], a margin of 0.1 will
        result in a range [-0.2, 2.2]; a margin of -0.1 will result in a range
        of [0.2, 1.8].

        Parameters
        ----------
        m : float greater than -0.5
        """
    def set_ymargin(self, m) -> None:
        """
        Set padding of Y data limits prior to autoscaling.

        *m* times the data interval will be added to each end of that interval
        before it is used in autoscaling.  If *m* is negative, this will clip
        the data range instead of expanding it.

        For example, if your data is in the range [0, 2], a margin of 0.1 will
        result in a range [-0.2, 2.2]; a margin of -0.1 will result in a range
        of [0.2, 1.8].

        Parameters
        ----------
        m : float greater than -0.5
        """
    def margins(self, *margins, x: Incomplete | None = None, y: Incomplete | None = None, tight: bool = True):
        '''
        Set or retrieve margins around the data for autoscaling axis limits.

        This allows to configure the padding around the data without having to
        set explicit limits using `~.Axes.set_xlim` / `~.Axes.set_ylim`.

        Autoscaling determines the axis limits by adding *margin* times the
        data interval as padding around the data. See the following illustration:

        .. plot:: _embedded_plots/axes_margins.py

        All input parameters must be floats greater than -0.5. Passing both
        positional and keyword arguments is invalid and will raise a TypeError.
        If no arguments (positional or otherwise) are provided, the current
        margins will remain unchanged and simply be returned.

        The default margins are :rc:`axes.xmargin` and :rc:`axes.ymargin`.

        Parameters
        ----------
        *margins : float, optional
            If a single positional argument is provided, it specifies
            both margins of the x-axis and y-axis limits. If two
            positional arguments are provided, they will be interpreted
            as *xmargin*, *ymargin*. If setting the margin on a single
            axis is desired, use the keyword arguments described below.

        x, y : float, optional
            Specific margin values for the x-axis and y-axis,
            respectively. These cannot be used with positional
            arguments, but can be used individually to alter on e.g.,
            only the y-axis.

        tight : bool or None, default: True
            The *tight* parameter is passed to `~.axes.Axes.autoscale_view`,
            which is executed after a margin is changed; the default
            here is *True*, on the assumption that when margins are
            specified, no additional padding to match tick marks is
            usually desired.  Setting *tight* to *None* preserves
            the previous setting.

        Returns
        -------
        xmargin, ymargin : float

        Notes
        -----
        If a previously used Axes method such as :meth:`pcolor` has set
        `~.Axes.use_sticky_edges` to `True`, only the limits not set by
        the "sticky artists" will be modified. To force all
        margins to be set, set `~.Axes.use_sticky_edges` to `False`
        before calling :meth:`margins`.

        See Also
        --------
        .Axes.set_xmargin, .Axes.set_ymargin
        '''
    def set_rasterization_zorder(self, z) -> None:
        """
        Set the zorder threshold for rasterization for vector graphics output.

        All artists with a zorder below the given value will be rasterized if
        they support rasterization.

        This setting is ignored for pixel-based output.

        See also :doc:`/gallery/misc/rasterization_demo`.

        Parameters
        ----------
        z : float or None
            The zorder below which artists are rasterized.
            If ``None`` rasterization based on zorder is deactivated.
        """
    def get_rasterization_zorder(self):
        """Return the zorder value below which artists will be rasterized."""
    def autoscale(self, enable: bool = True, axis: str = 'both', tight: Incomplete | None = None) -> None:
        """
        Autoscale the axis view to the data (toggle).

        Convenience method for simple axis view autoscaling.
        It turns autoscaling on or off, and then,
        if autoscaling for either axis is on, it performs
        the autoscaling on the specified axis or Axes.

        Parameters
        ----------
        enable : bool or None, default: True
            True turns autoscaling on, False turns it off.
            None leaves the autoscaling state unchanged.
        axis : {'both', 'x', 'y'}, default: 'both'
            The axis on which to operate.  (For 3D Axes, *axis* can also be set
            to 'z', and 'both' refers to all three Axes.)
        tight : bool or None, default: None
            If True, first set the margins to zero.  Then, this argument is
            forwarded to `~.axes.Axes.autoscale_view` (regardless of
            its value); see the description of its behavior there.
        """
    def autoscale_view(self, tight: Incomplete | None = None, scalex: bool = True, scaley: bool = True) -> None:
        """
        Autoscale the view limits using the data limits.

        Parameters
        ----------
        tight : bool or None
            If *True*, only expand the axis limits using the margins.  Note
            that unlike for `autoscale`, ``tight=True`` does *not* set the
            margins to zero.

            If *False* and :rc:`axes.autolimit_mode` is 'round_numbers', then
            after expansion by the margins, further expand the axis limits
            using the axis major locator.

            If None (the default), reuse the value set in the previous call to
            `autoscale_view` (the initial value is False, but the default style
            sets :rc:`axes.autolimit_mode` to 'data', in which case this
            behaves like True).

        scalex : bool, default: True
            Whether to autoscale the x-axis.

        scaley : bool, default: True
            Whether to autoscale the y-axis.

        Notes
        -----
        The autoscaling preserves any preexisting axis direction reversal.

        The data limits are not updated automatically when artist data are
        changed after the artist has been added to an Axes instance.  In that
        case, use :meth:`matplotlib.axes.Axes.relim` prior to calling
        autoscale_view.

        If the views of the Axes are fixed, e.g. via `set_xlim`, they will
        not be changed by autoscale_view().
        See :meth:`matplotlib.axes.Axes.autoscale` for an alternative.
        """
    def _update_title_position(self, renderer) -> None:
        """
        Update the title position based on the bounding box enclosing
        all the ticklabels and x-axis spine and xlabel...
        """
    def draw(self, renderer) -> None: ...
    def draw_artist(self, a) -> None:
        """
        Efficiently redraw a single artist.
        """
    def redraw_in_frame(self) -> None:
        """
        Efficiently redraw Axes data, but not axis ticks, labels, etc.
        """
    def get_frame_on(self):
        """Get whether the Axes rectangle patch is drawn."""
    def set_frame_on(self, b) -> None:
        """
        Set whether the Axes rectangle patch is drawn.

        Parameters
        ----------
        b : bool
        """
    def get_axisbelow(self):
        """
        Get whether axis ticks and gridlines are above or below most artists.

        Returns
        -------
        bool or 'line'

        See Also
        --------
        set_axisbelow
        """
    _axisbelow: Incomplete
    def set_axisbelow(self, b) -> None:
        """
        Set whether axis ticks and gridlines are above or below most artists.

        This controls the zorder of the ticks and gridlines. For more
        information on the zorder see :doc:`/gallery/misc/zorder_demo`.

        Parameters
        ----------
        b : bool or 'line'
            Possible values:

            - *True* (zorder = 0.5): Ticks and gridlines are below patches and
              lines, though still above images.
            - 'line' (zorder = 1.5): Ticks and gridlines are above patches
              (e.g. rectangles, with default zorder = 1) but still below lines
              and markers (with their default zorder = 2).
            - *False* (zorder = 2.5): Ticks and gridlines are above patches
              and lines / markers.

        Notes
        -----
        For more control, call the `~.Artist.set_zorder` method of each axis.

        See Also
        --------
        get_axisbelow
        """
    def grid(self, visible: Incomplete | None = None, which: str = 'major', axis: str = 'both', **kwargs) -> None:
        """
        Configure the grid lines.

        Parameters
        ----------
        visible : bool or None, optional
            Whether to show the grid lines.  If any *kwargs* are supplied, it
            is assumed you want the grid on and *visible* will be set to True.

            If *visible* is *None* and there are no *kwargs*, this toggles the
            visibility of the lines.

        which : {'major', 'minor', 'both'}, optional
            The grid lines to apply the changes on.

        axis : {'both', 'x', 'y'}, optional
            The axis to apply the changes on.

        **kwargs : `~matplotlib.lines.Line2D` properties
            Define the line properties of the grid, e.g.::

                grid(color='r', linestyle='-', linewidth=2)

            Valid keyword arguments are:

            %(Line2D:kwdoc)s

        Notes
        -----
        The axis is drawn as a unit, so the effective zorder for drawing the
        grid is determined by the zorder of each axis, not by the zorder of the
        `.Line2D` objects comprising the grid.  Therefore, to set grid zorder,
        use `.set_axisbelow` or, for more control, call the
        `~.Artist.set_zorder` method of each axis.
        """
    def ticklabel_format(self, *, axis: str = 'both', style: Incomplete | None = None, scilimits: Incomplete | None = None, useOffset: Incomplete | None = None, useLocale: Incomplete | None = None, useMathText: Incomplete | None = None) -> None:
        """
        Configure the `.ScalarFormatter` used by default for linear Axes.

        If a parameter is not set, the corresponding property of the formatter
        is left unchanged.

        Parameters
        ----------
        axis : {'x', 'y', 'both'}, default: 'both'
            The axis to configure.  Only major ticks are affected.

        style : {'sci', 'scientific', 'plain'}
            Whether to use scientific notation.
            The formatter default is to use scientific notation.
            'sci' is equivalent to 'scientific'.

        scilimits : pair of ints (m, n)
            Scientific notation is used only for numbers outside the range
            10\\ :sup:`m` to 10\\ :sup:`n` (and only if the formatter is
            configured to use scientific notation at all).  Use (0, 0) to
            include all numbers.  Use (m, m) where m != 0 to fix the order of
            magnitude to 10\\ :sup:`m`.
            The formatter default is :rc:`axes.formatter.limits`.

        useOffset : bool or float
            If True, the offset is calculated as needed.
            If False, no offset is used.
            If a numeric value, it sets the offset.
            The formatter default is :rc:`axes.formatter.useoffset`.

        useLocale : bool
            Whether to format the number using the current locale or using the
            C (English) locale.  This affects e.g. the decimal separator.  The
            formatter default is :rc:`axes.formatter.use_locale`.

        useMathText : bool
            Render the offset and scientific notation in mathtext.
            The formatter default is :rc:`axes.formatter.use_mathtext`.

        Raises
        ------
        AttributeError
            If the current formatter is not a `.ScalarFormatter`.
        """
    def locator_params(self, axis: str = 'both', tight: Incomplete | None = None, **kwargs) -> None:
        """
        Control behavior of major tick locators.

        Because the locator is involved in autoscaling, `~.Axes.autoscale_view`
        is called automatically after the parameters are changed.

        Parameters
        ----------
        axis : {'both', 'x', 'y'}, default: 'both'
            The axis on which to operate.  (For 3D Axes, *axis* can also be
            set to 'z', and 'both' refers to all three axes.)
        tight : bool or None, optional
            Parameter passed to `~.Axes.autoscale_view`.
            Default is None, for no change.

        Other Parameters
        ----------------
        **kwargs
            Remaining keyword arguments are passed to directly to the
            ``set_params()`` method of the locator. Supported keywords depend
            on the type of the locator. See for example
            `~.ticker.MaxNLocator.set_params` for the `.ticker.MaxNLocator`
            used by default for linear.

        Examples
        --------
        When plotting small subplots, one might want to reduce the maximum
        number of ticks and use tight bounds, for example::

            ax.locator_params(tight=True, nbins=4)

        """
    def tick_params(self, axis: str = 'both', **kwargs) -> None:
        """
        Change the appearance of ticks, tick labels, and gridlines.

        Tick properties that are not explicitly set using the keyword
        arguments remain unchanged unless *reset* is True. For the current
        style settings, see `.Axis.get_tick_params`.

        Parameters
        ----------
        axis : {'x', 'y', 'both'}, default: 'both'
            The axis to which the parameters are applied.
        which : {'major', 'minor', 'both'}, default: 'major'
            The group of ticks to which the parameters are applied.
        reset : bool, default: False
            Whether to reset the ticks to defaults before updating them.

        Other Parameters
        ----------------
        direction : {'in', 'out', 'inout'}
            Puts ticks inside the Axes, outside the Axes, or both.
        length : float
            Tick length in points.
        width : float
            Tick width in points.
        color : :mpltype:`color`
            Tick color.
        pad : float
            Distance in points between tick and label.
        labelsize : float or str
            Tick label font size in points or as a string (e.g., 'large').
        labelcolor : :mpltype:`color`
            Tick label color.
        labelfontfamily : str
            Tick label font.
        colors : :mpltype:`color`
            Tick color and label color.
        zorder : float
            Tick and label zorder.
        bottom, top, left, right : bool
            Whether to draw the respective ticks.
        labelbottom, labeltop, labelleft, labelright : bool
            Whether to draw the respective tick labels.
        labelrotation : float
            Tick label rotation
        grid_color : :mpltype:`color`
            Gridline color.
        grid_alpha : float
            Transparency of gridlines: 0 (transparent) to 1 (opaque).
        grid_linewidth : float
            Width of gridlines in points.
        grid_linestyle : str
            Any valid `.Line2D` line style spec.

        Examples
        --------
        ::

            ax.tick_params(direction='out', length=6, width=2, colors='r',
                           grid_color='r', grid_alpha=0.5)

        This will make all major ticks be red, pointing out of the box,
        and with dimensions 6 points by 2 points.  Tick labels will
        also be red.  Gridlines will be red and translucent.

        """
    axison: bool
    def set_axis_off(self) -> None:
        """
        Hide all visual components of the x- and y-axis.

        This sets a flag to suppress drawing of all axis decorations, i.e.
        axis labels, axis spines, and the axis tick component (tick markers,
        tick labels, and grid lines). Individual visibility settings of these
        components are ignored as long as `set_axis_off()` is in effect.
        """
    def set_axis_on(self) -> None:
        """
        Do not hide all visual components of the x- and y-axis.

        This reverts the effect of a prior `.set_axis_off()` call. Whether the
        individual axis decorations are drawn is controlled by their respective
        visibility settings.

        This is on by default.
        """
    def get_xlabel(self):
        """
        Get the xlabel text string.
        """
    def set_xlabel(self, xlabel, fontdict: Incomplete | None = None, labelpad: Incomplete | None = None, *, loc: Incomplete | None = None, **kwargs):
        """
        Set the label for the x-axis.

        Parameters
        ----------
        xlabel : str
            The label text.

        labelpad : float, default: :rc:`axes.labelpad`
            Spacing in points from the Axes bounding box including ticks
            and tick labels.  If None, the previous value is left as is.

        loc : {'left', 'center', 'right'}, default: :rc:`xaxis.labellocation`
            The label position. This is a high-level alternative for passing
            parameters *x* and *horizontalalignment*.

        Other Parameters
        ----------------
        **kwargs : `~matplotlib.text.Text` properties
            `.Text` properties control the appearance of the label.

        See Also
        --------
        text : Documents the properties supported by `.Text`.
        """
    def invert_xaxis(self) -> None:
        """
        Invert the x-axis.

        See Also
        --------
        xaxis_inverted
        get_xlim, set_xlim
        get_xbound, set_xbound
        """
    xaxis_inverted: Incomplete
    def get_xbound(self):
        """
        Return the lower and upper x-axis bounds, in increasing order.

        See Also
        --------
        set_xbound
        get_xlim, set_xlim
        invert_xaxis, xaxis_inverted
        """
    def set_xbound(self, lower: Incomplete | None = None, upper: Incomplete | None = None) -> None:
        """
        Set the lower and upper numerical bounds of the x-axis.

        This method will honor axis inversion regardless of parameter order.
        It will not change the autoscaling setting (`.get_autoscalex_on()`).

        Parameters
        ----------
        lower, upper : float or None
            The lower and upper bounds. If *None*, the respective axis bound
            is not modified.

            .. ACCEPTS: (lower: float, upper: float)

        See Also
        --------
        get_xbound
        get_xlim, set_xlim
        invert_xaxis, xaxis_inverted
        """
    def get_xlim(self):
        """
        Return the x-axis view limits.

        Returns
        -------
        left, right : (float, float)
            The current x-axis limits in data coordinates.

        See Also
        --------
        .Axes.set_xlim
        .Axes.set_xbound, .Axes.get_xbound
        .Axes.invert_xaxis, .Axes.xaxis_inverted

        Notes
        -----
        The x-axis may be inverted, in which case the *left* value will
        be greater than the *right* value.
        """
    def _validate_converted_limits(self, limit, convert):
        """
        Raise ValueError if converted limits are non-finite.

        Note that this function also accepts None as a limit argument.

        Returns
        -------
        The limit value after call to convert(), or None if limit is None.
        """
    def set_xlim(self, left: Incomplete | None = None, right: Incomplete | None = None, *, emit: bool = True, auto: bool = False, xmin: Incomplete | None = None, xmax: Incomplete | None = None):
        """
        Set the x-axis view limits.

        Parameters
        ----------
        left : float, optional
            The left xlim in data coordinates. Passing *None* leaves the
            limit unchanged.

            The left and right xlims may also be passed as the tuple
            (*left*, *right*) as the first positional argument (or as
            the *left* keyword argument).

            .. ACCEPTS: (left: float, right: float)

        right : float, optional
            The right xlim in data coordinates. Passing *None* leaves the
            limit unchanged.

        emit : bool, default: True
            Whether to notify observers of limit change.

        auto : bool or None, default: False
            Whether to turn on autoscaling of the x-axis. True turns on,
            False turns off, None leaves unchanged.

        xmin, xmax : float, optional
            They are equivalent to left and right respectively, and it is an
            error to pass both *xmin* and *left* or *xmax* and *right*.

        Returns
        -------
        left, right : (float, float)
            The new x-axis limits in data coordinates.

        See Also
        --------
        get_xlim
        set_xbound, get_xbound
        invert_xaxis, xaxis_inverted

        Notes
        -----
        The *left* value may be greater than the *right* value, in which
        case the x-axis values will decrease from left to right.

        Examples
        --------
        >>> set_xlim(left, right)
        >>> set_xlim((left, right))
        >>> left, right = set_xlim(left, right)

        One limit may be left unchanged.

        >>> set_xlim(right=right_lim)

        Limits may be passed in reverse order to flip the direction of
        the x-axis. For example, suppose *x* represents the number of
        years before present. The x-axis limits might be set like the
        following so 5000 years ago is on the left of the plot and the
        present is on the right.

        >>> set_xlim(5000, 0)
        """
    get_xscale: Incomplete
    set_xscale: Incomplete
    get_xticks: Incomplete
    set_xticks: Incomplete
    get_xmajorticklabels: Incomplete
    get_xminorticklabels: Incomplete
    get_xticklabels: Incomplete
    set_xticklabels: Incomplete
    def get_ylabel(self):
        """
        Get the ylabel text string.
        """
    def set_ylabel(self, ylabel, fontdict: Incomplete | None = None, labelpad: Incomplete | None = None, *, loc: Incomplete | None = None, **kwargs):
        """
        Set the label for the y-axis.

        Parameters
        ----------
        ylabel : str
            The label text.

        labelpad : float, default: :rc:`axes.labelpad`
            Spacing in points from the Axes bounding box including ticks
            and tick labels.  If None, the previous value is left as is.

        loc : {'bottom', 'center', 'top'}, default: :rc:`yaxis.labellocation`
            The label position. This is a high-level alternative for passing
            parameters *y* and *horizontalalignment*.

        Other Parameters
        ----------------
        **kwargs : `~matplotlib.text.Text` properties
            `.Text` properties control the appearance of the label.

        See Also
        --------
        text : Documents the properties supported by `.Text`.
        """
    def invert_yaxis(self) -> None:
        """
        Invert the y-axis.

        See Also
        --------
        yaxis_inverted
        get_ylim, set_ylim
        get_ybound, set_ybound
        """
    yaxis_inverted: Incomplete
    def get_ybound(self):
        """
        Return the lower and upper y-axis bounds, in increasing order.

        See Also
        --------
        set_ybound
        get_ylim, set_ylim
        invert_yaxis, yaxis_inverted
        """
    def set_ybound(self, lower: Incomplete | None = None, upper: Incomplete | None = None) -> None:
        """
        Set the lower and upper numerical bounds of the y-axis.

        This method will honor axis inversion regardless of parameter order.
        It will not change the autoscaling setting (`.get_autoscaley_on()`).

        Parameters
        ----------
        lower, upper : float or None
            The lower and upper bounds. If *None*, the respective axis bound
            is not modified.

         .. ACCEPTS: (lower: float, upper: float)

        See Also
        --------
        get_ybound
        get_ylim, set_ylim
        invert_yaxis, yaxis_inverted
        """
    def get_ylim(self):
        """
        Return the y-axis view limits.

        Returns
        -------
        bottom, top : (float, float)
            The current y-axis limits in data coordinates.

        See Also
        --------
        .Axes.set_ylim
        .Axes.set_ybound, .Axes.get_ybound
        .Axes.invert_yaxis, .Axes.yaxis_inverted

        Notes
        -----
        The y-axis may be inverted, in which case the *bottom* value
        will be greater than the *top* value.
        """
    def set_ylim(self, bottom: Incomplete | None = None, top: Incomplete | None = None, *, emit: bool = True, auto: bool = False, ymin: Incomplete | None = None, ymax: Incomplete | None = None):
        """
        Set the y-axis view limits.

        Parameters
        ----------
        bottom : float, optional
            The bottom ylim in data coordinates. Passing *None* leaves the
            limit unchanged.

            The bottom and top ylims may also be passed as the tuple
            (*bottom*, *top*) as the first positional argument (or as
            the *bottom* keyword argument).

            .. ACCEPTS: (bottom: float, top: float)

        top : float, optional
            The top ylim in data coordinates. Passing *None* leaves the
            limit unchanged.

        emit : bool, default: True
            Whether to notify observers of limit change.

        auto : bool or None, default: False
            Whether to turn on autoscaling of the y-axis. *True* turns on,
            *False* turns off, *None* leaves unchanged.

        ymin, ymax : float, optional
            They are equivalent to bottom and top respectively, and it is an
            error to pass both *ymin* and *bottom* or *ymax* and *top*.

        Returns
        -------
        bottom, top : (float, float)
            The new y-axis limits in data coordinates.

        See Also
        --------
        get_ylim
        set_ybound, get_ybound
        invert_yaxis, yaxis_inverted

        Notes
        -----
        The *bottom* value may be greater than the *top* value, in which
        case the y-axis values will decrease from *bottom* to *top*.

        Examples
        --------
        >>> set_ylim(bottom, top)
        >>> set_ylim((bottom, top))
        >>> bottom, top = set_ylim(bottom, top)

        One limit may be left unchanged.

        >>> set_ylim(top=top_lim)

        Limits may be passed in reverse order to flip the direction of
        the y-axis. For example, suppose ``y`` represents depth of the
        ocean in m. The y-axis limits might be set like the following
        so 5000 m depth is at the bottom of the plot and the surface,
        0 m, is at the top.

        >>> set_ylim(5000, 0)
        """
    get_yscale: Incomplete
    set_yscale: Incomplete
    get_yticks: Incomplete
    set_yticks: Incomplete
    get_ymajorticklabels: Incomplete
    get_yminorticklabels: Incomplete
    get_yticklabels: Incomplete
    set_yticklabels: Incomplete
    xaxis_date: Incomplete
    yaxis_date: Incomplete
    def format_xdata(self, x):
        """
        Return *x* formatted as an x-value.

        This function will use the `.fmt_xdata` attribute if it is not None,
        else will fall back on the xaxis major formatter.
        """
    def format_ydata(self, y):
        """
        Return *y* formatted as a y-value.

        This function will use the `.fmt_ydata` attribute if it is not None,
        else will fall back on the yaxis major formatter.
        """
    def format_coord(self, x, y):
        """Return a format string formatting the *x*, *y* coordinates."""
    def minorticks_on(self) -> None:
        """
        Display minor ticks on the Axes.

        Displaying minor ticks may reduce performance; you may turn them off
        using `minorticks_off()` if drawing speed is a problem.
        """
    def minorticks_off(self) -> None:
        """Remove minor ticks from the Axes."""
    def can_zoom(self):
        """
        Return whether this Axes supports the zoom box button functionality.
        """
    def can_pan(self):
        """
        Return whether this Axes supports any pan/zoom button functionality.
        """
    def get_navigate(self):
        """
        Get whether the Axes responds to navigation commands.
        """
    _navigate: Incomplete
    def set_navigate(self, b) -> None:
        """
        Set whether the Axes responds to navigation toolbar commands.

        Parameters
        ----------
        b : bool

        See Also
        --------
        matplotlib.axes.Axes.set_forward_navigation_events

        """
    def get_navigate_mode(self):
        """
        Get the navigation toolbar button status: 'PAN', 'ZOOM', or None.
        """
    _navigate_mode: Incomplete
    def set_navigate_mode(self, b) -> None:
        """
        Set the navigation toolbar button status.

        .. warning::
            This is not a user-API function.

        """
    def _get_view(self):
        """
        Save information required to reproduce the current view.

        This method is called before a view is changed, such as during a pan or zoom
        initiated by the user.  It returns an opaque object that describes the current
        view, in a format compatible with :meth:`_set_view`.

        The default implementation saves the view limits and autoscaling state.
        Subclasses may override this as needed, as long as :meth:`_set_view` is also
        adjusted accordingly.
        """
    def _set_view(self, view) -> None:
        """
        Apply a previously saved view.

        This method is called when restoring a view (with the return value of
        :meth:`_get_view` as argument), such as with the navigation buttons.

        Subclasses that override :meth:`_get_view` also need to override this method
        accordingly.
        """
    def _prepare_view_from_bbox(self, bbox, direction: str = 'in', mode: Incomplete | None = None, twinx: bool = False, twiny: bool = False):
        """
        Helper function to prepare the new bounds from a bbox.

        This helper function returns the new x and y bounds from the zoom
        bbox. This a convenience method to abstract the bbox logic
        out of the base setter.
        """
    def _set_view_from_bbox(self, bbox, direction: str = 'in', mode: Incomplete | None = None, twinx: bool = False, twiny: bool = False) -> None:
        """
        Update view from a selection bbox.

        .. note::

            Intended to be overridden by new projection types, but if not, the
            default implementation sets the view limits to the bbox directly.

        Parameters
        ----------
        bbox : 4-tuple or 3 tuple
            * If bbox is a 4 tuple, it is the selected bounding box limits,
              in *display* coordinates.
            * If bbox is a 3 tuple, it is an (xp, yp, scl) triple, where
              (xp, yp) is the center of zooming and scl the scale factor to
              zoom by.

        direction : str
            The direction to apply the bounding box.
                * `'in'` - The bounding box describes the view directly, i.e.,
                           it zooms in.
                * `'out'` - The bounding box describes the size to make the
                            existing view, i.e., it zooms out.

        mode : str or None
            The selection mode, whether to apply the bounding box in only the
            `'x'` direction, `'y'` direction or both (`None`).

        twinx : bool
            Whether this axis is twinned in the *x*-direction.

        twiny : bool
            Whether this axis is twinned in the *y*-direction.
        """
    _pan_start: Incomplete
    def start_pan(self, x, y, button) -> None:
        """
        Called when a pan operation has started.

        Parameters
        ----------
        x, y : float
            The mouse coordinates in display coords.
        button : `.MouseButton`
            The pressed mouse button.

        Notes
        -----
        This is intended to be overridden by new projection types.
        """
    def end_pan(self) -> None:
        """
        Called when a pan operation completes (when the mouse button is up.)

        Notes
        -----
        This is intended to be overridden by new projection types.
        """
    def _get_pan_points(self, button, key, x, y):
        """
        Helper function to return the new points after a pan.

        This helper function returns the points on the axis after a pan has
        occurred. This is a convenience method to abstract the pan logic
        out of the base setter.
        """
    def drag_pan(self, button, key, x, y) -> None:
        """
        Called when the mouse moves during a pan operation.

        Parameters
        ----------
        button : `.MouseButton`
            The pressed mouse button.
        key : str or None
            The pressed key, if any.
        x, y : float
            The mouse coordinates in display coords.

        Notes
        -----
        This is intended to be overridden by new projection types.
        """
    def get_children(self): ...
    def contains(self, mouseevent): ...
    def contains_point(self, point):
        """
        Return whether *point* (pair of pixel coordinates) is inside the Axes
        patch.
        """
    def get_default_bbox_extra_artists(self):
        """
        Return a default list of artists that are used for the bounding box
        calculation.

        Artists are excluded either by not being visible or
        ``artist.set_in_layout(False)``.
        """
    def get_tightbbox(self, renderer: Incomplete | None = None, *, call_axes_locator: bool = True, bbox_extra_artists: Incomplete | None = None, for_layout_only: bool = False):
        """
        Return the tight bounding box of the Axes, including axis and their
        decorators (xlabel, title, etc).

        Artists that have ``artist.set_in_layout(False)`` are not included
        in the bbox.

        Parameters
        ----------
        renderer : `.RendererBase` subclass
            renderer that will be used to draw the figures (i.e.
            ``fig.canvas.get_renderer()``)

        bbox_extra_artists : list of `.Artist` or ``None``
            List of artists to include in the tight bounding box.  If
            ``None`` (default), then all artist children of the Axes are
            included in the tight bounding box.

        call_axes_locator : bool, default: True
            If *call_axes_locator* is ``False``, it does not call the
            ``_axes_locator`` attribute, which is necessary to get the correct
            bounding box. ``call_axes_locator=False`` can be used if the
            caller is only interested in the relative size of the tightbbox
            compared to the Axes bbox.

        for_layout_only : default: False
            The bounding box will *not* include the x-extent of the title and
            the xlabel, or the y-extent of the ylabel.

        Returns
        -------
        `.BboxBase`
            Bounding box in figure pixel coordinates.

        See Also
        --------
        matplotlib.axes.Axes.get_window_extent
        matplotlib.axis.Axis.get_tightbbox
        matplotlib.spines.Spine.get_window_extent
        """
    def _make_twin_axes(self, *args, **kwargs):
        """Make a twinx Axes of self. This is used for twinx and twiny."""
    def twinx(self):
        """
        Create a twin Axes sharing the xaxis.

        Create a new Axes with an invisible x-axis and an independent
        y-axis positioned opposite to the original one (i.e. at right). The
        x-axis autoscale setting will be inherited from the original
        Axes.  To ensure that the tick marks of both y-axes align, see
        `~matplotlib.ticker.LinearLocator`.

        Returns
        -------
        Axes
            The newly created Axes instance

        Notes
        -----
        For those who are 'picking' artists while using twinx, pick
        events are only called for the artists in the top-most Axes.
        """
    def twiny(self):
        """
        Create a twin Axes sharing the yaxis.

        Create a new Axes with an invisible y-axis and an independent
        x-axis positioned opposite to the original one (i.e. at top). The
        y-axis autoscale setting will be inherited from the original Axes.
        To ensure that the tick marks of both x-axes align, see
        `~matplotlib.ticker.LinearLocator`.

        Returns
        -------
        Axes
            The newly created Axes instance

        Notes
        -----
        For those who are 'picking' artists while using twiny, pick
        events are only called for the artists in the top-most Axes.
        """
    def get_shared_x_axes(self):
        """Return an immutable view on the shared x-axes Grouper."""
    def get_shared_y_axes(self):
        """Return an immutable view on the shared y-axes Grouper."""
    def label_outer(self, remove_inner_ticks: bool = False) -> None:
        '''
        Only show "outer" labels and tick labels.

        x-labels are only kept for subplots on the last row (or first row, if
        labels are on the top side); y-labels only for subplots on the first
        column (or last column, if labels are on the right side).

        Parameters
        ----------
        remove_inner_ticks : bool, default: False
            If True, remove the inner ticks as well (not only tick labels).

            .. versionadded:: 3.8
        '''
    def _label_outer_xaxis(self, *, skip_non_rectangular_axes, remove_inner_ticks: bool = False) -> None: ...
    def _label_outer_yaxis(self, *, skip_non_rectangular_axes, remove_inner_ticks: bool = False) -> None: ...
    def set_forward_navigation_events(self, forward) -> None:
        '''
        Set how pan/zoom events are forwarded to Axes below this one.

        Parameters
        ----------
        forward : bool or "auto"
            Possible values:

            - True: Forward events to other axes with lower or equal zorder.
            - False: Events are only executed on this axes.
            - "auto": Default behaviour (*True* for axes with an invisible
              patch and *False* otherwise)

        See Also
        --------
        matplotlib.axes.Axes.set_navigate

        '''
    def get_forward_navigation_events(self):
        """Get how pan/zoom events are forwarded to Axes below this one."""

def _draw_rasterized(figure, artists, renderer):
    '''
    A helper function for rasterizing the list of artists.

    The bookkeeping to track if we are or are not in rasterizing mode
    with the mixed-mode backends is relatively complicated and is now
    handled in the matplotlib.artist.allow_rasterization decorator.

    This helper defines the absolute minimum methods and attributes on a
    shim class to be compatible with that decorator and then uses it to
    rasterize the list of artists.

    This is maybe too-clever, but allows us to reuse the same code that is
    used on normal artists to participate in the "are we rasterizing"
    accounting.

    Please do not use this outside of the "rasterize below a given zorder"
    functionality of Axes.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        The figure all of the artists belong to (not checked).  We need this
        because we can at the figure level suppress composition and insert each
        rasterized artist as its own image.

    artists : List[matplotlib.artist.Artist]
        The list of Artists to be rasterized.  These are assumed to all
        be in the same Figure.

    renderer : matplotlib.backendbases.RendererBase
        The currently active renderer

    Returns
    -------
    None

    '''
