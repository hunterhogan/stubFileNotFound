from _typeshed import Incomplete

_log: Incomplete

def do_constrained_layout(fig, h_pad, w_pad, hspace: Incomplete | None = None, wspace: Incomplete | None = None, rect=(0, 0, 1, 1), compress: bool = False):
    """
    Do the constrained_layout.  Called at draw time in
     ``figure.constrained_layout()``

    Parameters
    ----------
    fig : `~matplotlib.figure.Figure`
        `.Figure` instance to do the layout in.

    h_pad, w_pad : float
      Padding around the Axes elements in figure-normalized units.

    hspace, wspace : float
       Fraction of the figure to dedicate to space between the
       Axes.  These are evenly spread between the gaps between the Axes.
       A value of 0.2 for a three-column layout would have a space
       of 0.1 of the figure width between each column.
       If h/wspace < h/w_pad, then the pads are used instead.

    rect : tuple of 4 floats
        Rectangle in figure coordinates to perform constrained layout in
        [left, bottom, width, height], each from 0-1.

    compress : bool
        Whether to shift Axes so that white space in between them is
        removed. This is useful for simple grids of fixed-aspect Axes (e.g.
        a grid of images).

    Returns
    -------
    layoutgrid : private debugging structure
    """
def make_layoutgrids(fig, layoutgrids, rect=(0, 0, 1, 1)):
    """
    Make the layoutgrid tree.

    (Sub)Figures get a layoutgrid so we can have figure margins.

    Gridspecs that are attached to Axes get a layoutgrid so Axes
    can have margins.
    """
def make_layoutgrids_gs(layoutgrids, gs):
    """
    Make the layoutgrid for a gridspec (and anything nested in the gridspec)
    """
def check_no_collapsed_axes(layoutgrids, fig):
    """
    Check that no Axes have collapsed to zero size.
    """
def compress_fixed_aspect(layoutgrids, fig): ...
def get_margin_from_padding(obj, *, w_pad: int = 0, h_pad: int = 0, hspace: int = 0, wspace: int = 0): ...
def make_layout_margins(layoutgrids, fig, renderer, *, w_pad: int = 0, h_pad: int = 0, hspace: int = 0, wspace: int = 0) -> None:
    """
    For each Axes, make a margin between the *pos* layoutbox and the
    *axes* layoutbox be a minimum size that can accommodate the
    decorations on the axis.

    Then make room for colorbars.

    Parameters
    ----------
    layoutgrids : dict
    fig : `~matplotlib.figure.Figure`
        `.Figure` instance to do the layout in.
    renderer : `~matplotlib.backend_bases.RendererBase` subclass.
        The renderer to use.
    w_pad, h_pad : float, default: 0
        Width and height padding (in fraction of figure).
    hspace, wspace : float, default: 0
        Width and height padding as fraction of figure size divided by
        number of columns or rows.
    """
def make_margin_suptitles(layoutgrids, fig, renderer, *, w_pad: int = 0, h_pad: int = 0) -> None: ...
def match_submerged_margins(layoutgrids, fig) -> None:
    '''
    Make the margins that are submerged inside an Axes the same size.

    This allows Axes that span two columns (or rows) that are offset
    from one another to have the same size.

    This gives the proper layout for something like::
        fig = plt.figure(constrained_layout=True)
        axs = fig.subplot_mosaic("AAAB
CCDD")

    Without this routine, the Axes D will be wider than C, because the
    margin width between the two columns in C has no width by default,
    whereas the margins between the two columns of D are set by the
    width of the margin between A and B. However, obviously the user would
    like C and D to be the same size, so we need to add constraints to these
    "submerged" margins.

    This routine makes all the interior margins the same, and the spacing
    between the three columns in A and the two column in C are all set to the
    margins between the two columns of D.

    See test_constrained_layout::test_constrained_layout12 for an example.
    '''
def get_cb_parent_spans(cbax):
    """
    Figure out which subplotspecs this colorbar belongs to.

    Parameters
    ----------
    cbax : `~matplotlib.axes.Axes`
        Axes for the colorbar.
    """
def get_pos_and_bbox(ax, renderer):
    """
    Get the position and the bbox for the Axes.

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
    renderer : `~matplotlib.backend_bases.RendererBase` subclass.

    Returns
    -------
    pos : `~matplotlib.transforms.Bbox`
        Position in figure coordinates.
    bbox : `~matplotlib.transforms.Bbox`
        Tight bounding box in figure coordinates.
    """
def reposition_axes(layoutgrids, fig, renderer, *, w_pad: int = 0, h_pad: int = 0, hspace: int = 0, wspace: int = 0) -> None:
    """
    Reposition all the Axes based on the new inner bounding box.
    """
def reposition_colorbar(layoutgrids, cbax, renderer, *, offset: Incomplete | None = None):
    """
    Place the colorbar in its new place.

    Parameters
    ----------
    layoutgrids : dict
    cbax : `~matplotlib.axes.Axes`
        Axes for the colorbar.
    renderer : `~matplotlib.backend_bases.RendererBase` subclass.
        The renderer to use.
    offset : array-like
        Offset the colorbar needs to be pushed to in order to
        account for multiple colorbars.
    """
def reset_margins(layoutgrids, fig) -> None:
    """
    Reset the margins in the layoutboxes of *fig*.

    Margins are usually set as a minimum, so if the figure gets smaller
    the minimum needs to be zero in order for it to grow again.
    """
def colorbar_get_pad(layoutgrids, cax): ...
