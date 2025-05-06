from _typeshed import Incomplete
from matplotlib import _api as _api, _pylab_helpers as _pylab_helpers, _tight_layout as _tight_layout
from matplotlib.transforms import Bbox as Bbox

_log: Incomplete

class GridSpecBase:
    """
    A base class of GridSpec that specifies the geometry of the grid
    that a subplot will be placed.
    """
    def __init__(self, nrows, ncols, height_ratios: Incomplete | None = None, width_ratios: Incomplete | None = None) -> None:
        """
        Parameters
        ----------
        nrows, ncols : int
            The number of rows and columns of the grid.
        width_ratios : array-like of length *ncols*, optional
            Defines the relative widths of the columns. Each column gets a
            relative width of ``width_ratios[i] / sum(width_ratios)``.
            If not given, all columns will have the same width.
        height_ratios : array-like of length *nrows*, optional
            Defines the relative heights of the rows. Each row gets a
            relative height of ``height_ratios[i] / sum(height_ratios)``.
            If not given, all rows will have the same height.
        """
    def __repr__(self) -> str: ...
    nrows: Incomplete
    ncols: Incomplete
    def get_geometry(self):
        """
        Return a tuple containing the number of rows and columns in the grid.
        """
    def get_subplot_params(self, figure: Incomplete | None = None) -> None: ...
    def new_subplotspec(self, loc, rowspan: int = 1, colspan: int = 1):
        """
        Create and return a `.SubplotSpec` instance.

        Parameters
        ----------
        loc : (int, int)
            The position of the subplot in the grid as
            ``(row_index, column_index)``.
        rowspan, colspan : int, default: 1
            The number of rows and columns the subplot should span in the grid.
        """
    _col_width_ratios: Incomplete
    def set_width_ratios(self, width_ratios) -> None:
        """
        Set the relative widths of the columns.

        *width_ratios* must be of length *ncols*. Each column gets a relative
        width of ``width_ratios[i] / sum(width_ratios)``.
        """
    def get_width_ratios(self):
        """
        Return the width ratios.

        This is *None* if no width ratios have been set explicitly.
        """
    _row_height_ratios: Incomplete
    def set_height_ratios(self, height_ratios) -> None:
        """
        Set the relative heights of the rows.

        *height_ratios* must be of length *nrows*. Each row gets a relative
        height of ``height_ratios[i] / sum(height_ratios)``.
        """
    def get_height_ratios(self):
        """
        Return the height ratios.

        This is *None* if no height ratios have been set explicitly.
        """
    def get_grid_positions(self, fig):
        """
        Return the positions of the grid cells in figure coordinates.

        Parameters
        ----------
        fig : `~matplotlib.figure.Figure`
            The figure the grid should be applied to. The subplot parameters
            (margins and spacing between subplots) are taken from *fig*.

        Returns
        -------
        bottoms, tops, lefts, rights : array
            The bottom, top, left, right positions of the grid cells in
            figure coordinates.
        """
    @staticmethod
    def _check_gridspec_exists(figure, nrows, ncols):
        """
        Check if the figure already has a gridspec with these dimensions,
        or create a new one
        """
    def __getitem__(self, key):
        """Create and return a `.SubplotSpec` instance."""
    def subplots(self, *, sharex: bool = False, sharey: bool = False, squeeze: bool = True, subplot_kw: Incomplete | None = None):
        """
        Add all subplots specified by this `GridSpec` to its parent figure.

        See `.Figure.subplots` for detailed documentation.
        """

class GridSpec(GridSpecBase):
    """
    A grid layout to place subplots within a figure.

    The location of the grid cells is determined in a similar way to
    `.SubplotParams` using *left*, *right*, *top*, *bottom*, *wspace*
    and *hspace*.

    Indexing a GridSpec instance returns a `.SubplotSpec`.
    """
    left: Incomplete
    bottom: Incomplete
    right: Incomplete
    top: Incomplete
    wspace: Incomplete
    hspace: Incomplete
    figure: Incomplete
    def __init__(self, nrows, ncols, figure: Incomplete | None = None, left: Incomplete | None = None, bottom: Incomplete | None = None, right: Incomplete | None = None, top: Incomplete | None = None, wspace: Incomplete | None = None, hspace: Incomplete | None = None, width_ratios: Incomplete | None = None, height_ratios: Incomplete | None = None) -> None:
        """
        Parameters
        ----------
        nrows, ncols : int
            The number of rows and columns of the grid.

        figure : `.Figure`, optional
            Only used for constrained layout to create a proper layoutgrid.

        left, right, top, bottom : float, optional
            Extent of the subplots as a fraction of figure width or height.
            Left cannot be larger than right, and bottom cannot be larger than
            top. If not given, the values will be inferred from a figure or
            rcParams at draw time. See also `GridSpec.get_subplot_params`.

        wspace : float, optional
            The amount of width reserved for space between subplots,
            expressed as a fraction of the average axis width.
            If not given, the values will be inferred from a figure or
            rcParams when necessary. See also `GridSpec.get_subplot_params`.

        hspace : float, optional
            The amount of height reserved for space between subplots,
            expressed as a fraction of the average axis height.
            If not given, the values will be inferred from a figure or
            rcParams when necessary. See also `GridSpec.get_subplot_params`.

        width_ratios : array-like of length *ncols*, optional
            Defines the relative widths of the columns. Each column gets a
            relative width of ``width_ratios[i] / sum(width_ratios)``.
            If not given, all columns will have the same width.

        height_ratios : array-like of length *nrows*, optional
            Defines the relative heights of the rows. Each row gets a
            relative height of ``height_ratios[i] / sum(height_ratios)``.
            If not given, all rows will have the same height.

        """
    _AllowedKeys: Incomplete
    def update(self, **kwargs) -> None:
        """
        Update the subplot parameters of the grid.

        Parameters that are not explicitly given are not changed. Setting a
        parameter to *None* resets it to :rc:`figure.subplot.*`.

        Parameters
        ----------
        left, right, top, bottom : float or None, optional
            Extent of the subplots as a fraction of figure width or height.
        wspace, hspace : float, optional
            Spacing between the subplots as a fraction of the average subplot
            width / height.
        """
    def get_subplot_params(self, figure: Incomplete | None = None):
        """
        Return the `.SubplotParams` for the GridSpec.

        In order of precedence the values are taken from

        - non-*None* attributes of the GridSpec
        - the provided *figure*
        - :rc:`figure.subplot.*`

        Note that the ``figure`` attribute of the GridSpec is always ignored.
        """
    def locally_modified_subplot_params(self):
        """
        Return a list of the names of the subplot parameters explicitly set
        in the GridSpec.

        This is a subset of the attributes of `.SubplotParams`.
        """
    def tight_layout(self, figure, renderer: Incomplete | None = None, pad: float = 1.08, h_pad: Incomplete | None = None, w_pad: Incomplete | None = None, rect: Incomplete | None = None) -> None:
        """
        Adjust subplot parameters to give specified padding.

        Parameters
        ----------
        figure : `.Figure`
            The figure.
        renderer :  `.RendererBase` subclass, optional
            The renderer to be used.
        pad : float
            Padding between the figure edge and the edges of subplots, as a
            fraction of the font-size.
        h_pad, w_pad : float, optional
            Padding (height/width) between edges of adjacent subplots.
            Defaults to *pad*.
        rect : tuple (left, bottom, right, top), default: None
            (left, bottom, right, top) rectangle in normalized figure
            coordinates that the whole subplots area (including labels) will
            fit into. Default (None) is the whole figure.
        """

class GridSpecFromSubplotSpec(GridSpecBase):
    """
    GridSpec whose subplot layout parameters are inherited from the
    location specified by a given SubplotSpec.
    """
    _wspace: Incomplete
    _hspace: Incomplete
    _subplot_spec: Incomplete
    figure: Incomplete
    def __init__(self, nrows, ncols, subplot_spec, wspace: Incomplete | None = None, hspace: Incomplete | None = None, height_ratios: Incomplete | None = None, width_ratios: Incomplete | None = None) -> None:
        """
        Parameters
        ----------
        nrows, ncols : int
            Number of rows and number of columns of the grid.
        subplot_spec : SubplotSpec
            Spec from which the layout parameters are inherited.
        wspace, hspace : float, optional
            See `GridSpec` for more details. If not specified default values
            (from the figure or rcParams) are used.
        height_ratios : array-like of length *nrows*, optional
            See `GridSpecBase` for details.
        width_ratios : array-like of length *ncols*, optional
            See `GridSpecBase` for details.
        """
    def get_subplot_params(self, figure: Incomplete | None = None):
        """Return a dictionary of subplot layout parameters."""
    def get_topmost_subplotspec(self):
        """
        Return the topmost `.SubplotSpec` instance associated with the subplot.
        """

class SubplotSpec:
    """
    The location of a subplot in a `GridSpec`.

    .. note::

        Likely, you will never instantiate a `SubplotSpec` yourself. Instead,
        you will typically obtain one from a `GridSpec` using item-access.

    Parameters
    ----------
    gridspec : `~matplotlib.gridspec.GridSpec`
        The GridSpec, which the subplot is referencing.
    num1, num2 : int
        The subplot will occupy the *num1*-th cell of the given
        *gridspec*.  If *num2* is provided, the subplot will span between
        *num1*-th cell and *num2*-th cell **inclusive**.

        The index starts from 0.
    """
    _gridspec: Incomplete
    num1: Incomplete
    def __init__(self, gridspec, num1, num2: Incomplete | None = None) -> None: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def _from_subplot_args(figure, args):
        """
        Construct a `.SubplotSpec` from a parent `.Figure` and either

        - a `.SubplotSpec` -- returned as is;
        - one or three numbers -- a MATLAB-style subplot specifier.
        """
    @property
    def num2(self): ...
    _num2: Incomplete
    @num2.setter
    def num2(self, value) -> None: ...
    def get_gridspec(self): ...
    def get_geometry(self):
        """
        Return the subplot geometry as tuple ``(n_rows, n_cols, start, stop)``.

        The indices *start* and *stop* define the range of the subplot within
        the `GridSpec`. *stop* is inclusive (i.e. for a single cell
        ``start == stop``).
        """
    @property
    def rowspan(self):
        """The rows spanned by this subplot, as a `range` object."""
    @property
    def colspan(self):
        """The columns spanned by this subplot, as a `range` object."""
    def is_first_row(self): ...
    def is_last_row(self): ...
    def is_first_col(self): ...
    def is_last_col(self): ...
    def get_position(self, figure):
        """
        Update the subplot position from ``figure.subplotpars``.
        """
    def get_topmost_subplotspec(self):
        """
        Return the topmost `SubplotSpec` instance associated with the subplot.
        """
    def __eq__(self, other):
        """
        Two SubplotSpecs are considered equal if they refer to the same
        position(s) in the same `GridSpec`.
        """
    def __hash__(self): ...
    def subgridspec(self, nrows, ncols, **kwargs):
        """
        Create a GridSpec within this subplot.

        The created `.GridSpecFromSubplotSpec` will have this `SubplotSpec` as
        a parent.

        Parameters
        ----------
        nrows : int
            Number of rows in grid.

        ncols : int
            Number of columns in grid.

        Returns
        -------
        `.GridSpecFromSubplotSpec`

        Other Parameters
        ----------------
        **kwargs
            All other parameters are passed to `.GridSpecFromSubplotSpec`.

        See Also
        --------
        matplotlib.pyplot.subplots

        Examples
        --------
        Adding three subplots in the space occupied by a single subplot::

            fig = plt.figure()
            gs0 = fig.add_gridspec(3, 1)
            ax1 = fig.add_subplot(gs0[0])
            ax2 = fig.add_subplot(gs0[1])
            gssub = gs0[2].subgridspec(1, 3)
            for i in range(3):
                fig.add_subplot(gssub[0, i])
        """

class SubplotParams:
    """
    Parameters defining the positioning of a subplots grid in a figure.
    """
    def __init__(self, left: Incomplete | None = None, bottom: Incomplete | None = None, right: Incomplete | None = None, top: Incomplete | None = None, wspace: Incomplete | None = None, hspace: Incomplete | None = None) -> None:
        """
        Defaults are given by :rc:`figure.subplot.[name]`.

        Parameters
        ----------
        left : float
            The position of the left edge of the subplots,
            as a fraction of the figure width.
        right : float
            The position of the right edge of the subplots,
            as a fraction of the figure width.
        bottom : float
            The position of the bottom edge of the subplots,
            as a fraction of the figure height.
        top : float
            The position of the top edge of the subplots,
            as a fraction of the figure height.
        wspace : float
            The width of the padding between subplots,
            as a fraction of the average Axes width.
        hspace : float
            The height of the padding between subplots,
            as a fraction of the average Axes height.
        """
    left: Incomplete
    right: Incomplete
    bottom: Incomplete
    top: Incomplete
    wspace: Incomplete
    hspace: Incomplete
    def update(self, left: Incomplete | None = None, bottom: Incomplete | None = None, right: Incomplete | None = None, top: Incomplete | None = None, wspace: Incomplete | None = None, hspace: Incomplete | None = None) -> None:
        """
        Update the dimensions of the passed parameters. *None* means unchanged.
        """
