from _typeshed import Incomplete
from matplotlib._constrained_layout import do_constrained_layout as do_constrained_layout
from matplotlib._tight_layout import get_subplotspec_list as get_subplotspec_list, get_tight_layout_figure as get_tight_layout_figure

class LayoutEngine:
    """
    Base class for Matplotlib layout engines.

    A layout engine can be passed to a figure at instantiation or at any time
    with `~.figure.Figure.set_layout_engine`.  Once attached to a figure, the
    layout engine ``execute`` function is called at draw time by
    `~.figure.Figure.draw`, providing a special draw-time hook.

    .. note::

       However, note that layout engines affect the creation of colorbars, so
       `~.figure.Figure.set_layout_engine` should be called before any
       colorbars are created.

    Currently, there are two properties of `LayoutEngine` classes that are
    consulted while manipulating the figure:

    - ``engine.colorbar_gridspec`` tells `.Figure.colorbar` whether to make the
       axes using the gridspec method (see `.colorbar.make_axes_gridspec`) or
       not (see `.colorbar.make_axes`);
    - ``engine.adjust_compatible`` stops `.Figure.subplots_adjust` from being
        run if it is not compatible with the layout engine.

    To implement a custom `LayoutEngine`:

    1. override ``_adjust_compatible`` and ``_colorbar_gridspec``
    2. override `LayoutEngine.set` to update *self._params*
    3. override `LayoutEngine.execute` with your implementation

    """
    _adjust_compatible: Incomplete
    _colorbar_gridspec: Incomplete
    _params: Incomplete
    def __init__(self, **kwargs) -> None: ...
    def set(self, **kwargs) -> None:
        """
        Set the parameters for the layout engine.
        """
    @property
    def colorbar_gridspec(self):
        """
        Return a boolean if the layout engine creates colorbars using a
        gridspec.
        """
    @property
    def adjust_compatible(self):
        """
        Return a boolean if the layout engine is compatible with
        `~.Figure.subplots_adjust`.
        """
    def get(self):
        """
        Return copy of the parameters for the layout engine.
        """
    def execute(self, fig) -> None:
        """
        Execute the layout on the figure given by *fig*.
        """

class PlaceHolderLayoutEngine(LayoutEngine):
    """
    This layout engine does not adjust the figure layout at all.

    The purpose of this `.LayoutEngine` is to act as a placeholder when the user removes
    a layout engine to ensure an incompatible `.LayoutEngine` cannot be set later.

    Parameters
    ----------
    adjust_compatible, colorbar_gridspec : bool
        Allow the PlaceHolderLayoutEngine to mirror the behavior of whatever
        layout engine it is replacing.

    """
    _adjust_compatible: Incomplete
    _colorbar_gridspec: Incomplete
    def __init__(self, adjust_compatible, colorbar_gridspec, **kwargs) -> None: ...
    def execute(self, fig) -> None:
        """
        Do nothing.
        """

class TightLayoutEngine(LayoutEngine):
    """
    Implements the ``tight_layout`` geometry management.  See
    :ref:`tight_layout_guide` for details.
    """
    _adjust_compatible: bool
    _colorbar_gridspec: bool
    def __init__(self, *, pad: float = 1.08, h_pad: Incomplete | None = None, w_pad: Incomplete | None = None, rect=(0, 0, 1, 1), **kwargs) -> None:
        """
        Initialize tight_layout engine.

        Parameters
        ----------
        pad : float, default: 1.08
            Padding between the figure edge and the edges of subplots, as a
            fraction of the font size.
        h_pad, w_pad : float
            Padding (height/width) between edges of adjacent subplots.
            Defaults to *pad*.
        rect : tuple (left, bottom, right, top), default: (0, 0, 1, 1).
            rectangle in normalized figure coordinates that the subplots
            (including labels) will fit into.
        """
    def execute(self, fig) -> None:
        """
        Execute tight_layout.

        This decides the subplot parameters given the padding that
        will allow the Axes labels to not be covered by other labels
        and Axes.

        Parameters
        ----------
        fig : `.Figure` to perform layout on.

        See Also
        --------
        .figure.Figure.tight_layout
        .pyplot.tight_layout
        """
    def set(self, *, pad: Incomplete | None = None, w_pad: Incomplete | None = None, h_pad: Incomplete | None = None, rect: Incomplete | None = None) -> None:
        """
        Set the pads for tight_layout.

        Parameters
        ----------
        pad : float
            Padding between the figure edge and the edges of subplots, as a
            fraction of the font size.
        w_pad, h_pad : float
            Padding (width/height) between edges of adjacent subplots.
            Defaults to *pad*.
        rect : tuple (left, bottom, right, top)
            rectangle in normalized figure coordinates that the subplots
            (including labels) will fit into.
        """

class ConstrainedLayoutEngine(LayoutEngine):
    """
    Implements the ``constrained_layout`` geometry management.  See
    :ref:`constrainedlayout_guide` for details.
    """
    _adjust_compatible: bool
    _colorbar_gridspec: bool
    _compress: Incomplete
    def __init__(self, *, h_pad: Incomplete | None = None, w_pad: Incomplete | None = None, hspace: Incomplete | None = None, wspace: Incomplete | None = None, rect=(0, 0, 1, 1), compress: bool = False, **kwargs) -> None:
        """
        Initialize ``constrained_layout`` settings.

        Parameters
        ----------
        h_pad, w_pad : float
            Padding around the Axes elements in inches.
            Default to :rc:`figure.constrained_layout.h_pad` and
            :rc:`figure.constrained_layout.w_pad`.
        hspace, wspace : float
            Fraction of the figure to dedicate to space between the
            axes.  These are evenly spread between the gaps between the Axes.
            A value of 0.2 for a three-column layout would have a space
            of 0.1 of the figure width between each column.
            If h/wspace < h/w_pad, then the pads are used instead.
            Default to :rc:`figure.constrained_layout.hspace` and
            :rc:`figure.constrained_layout.wspace`.
        rect : tuple of 4 floats
            Rectangle in figure coordinates to perform constrained layout in
            (left, bottom, width, height), each from 0-1.
        compress : bool
            Whether to shift Axes so that white space in between them is
            removed. This is useful for simple grids of fixed-aspect Axes (e.g.
            a grid of images).  See :ref:`compressed_layout`.
        """
    def execute(self, fig):
        """
        Perform constrained_layout and move and resize Axes accordingly.

        Parameters
        ----------
        fig : `.Figure` to perform layout on.
        """
    def set(self, *, h_pad: Incomplete | None = None, w_pad: Incomplete | None = None, hspace: Incomplete | None = None, wspace: Incomplete | None = None, rect: Incomplete | None = None) -> None:
        """
        Set the pads for constrained_layout.

        Parameters
        ----------
        h_pad, w_pad : float
            Padding around the Axes elements in inches.
            Default to :rc:`figure.constrained_layout.h_pad` and
            :rc:`figure.constrained_layout.w_pad`.
        hspace, wspace : float
            Fraction of the figure to dedicate to space between the
            axes.  These are evenly spread between the gaps between the Axes.
            A value of 0.2 for a three-column layout would have a space
            of 0.1 of the figure width between each column.
            If h/wspace < h/w_pad, then the pads are used instead.
            Default to :rc:`figure.constrained_layout.hspace` and
            :rc:`figure.constrained_layout.wspace`.
        rect : tuple of 4 floats
            Rectangle in figure coordinates to perform constrained layout in
            (left, bottom, width, height), each from 0-1.
        """
