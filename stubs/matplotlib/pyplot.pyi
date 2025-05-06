import PIL.Image
import datetime
import matplotlib.axes
import matplotlib.backend_bases
import numpy as np
import os
import pathlib
from .ticker import AutoLocator as AutoLocator, FixedFormatter as FixedFormatter, FixedLocator as FixedLocator, FormatStrFormatter as FormatStrFormatter, Formatter as Formatter, FuncFormatter as FuncFormatter, IndexLocator as IndexLocator, LinearLocator as LinearLocator, Locator as Locator, LogFormatter as LogFormatter, LogFormatterExponent as LogFormatterExponent, LogFormatterMathtext as LogFormatterMathtext, LogLocator as LogLocator, MaxNLocator as MaxNLocator, MultipleLocator as MultipleLocator, NullFormatter as NullFormatter, NullLocator as NullLocator, ScalarFormatter as ScalarFormatter, TickHelper as TickHelper
from _typeshed import Incomplete
from collections.abc import Callable as Callable, Hashable, Iterable, Sequence
from contextlib import AbstractContextManager, ExitStack
from cycler import cycler as cycler
from matplotlib import _api as _api, _docstring as _docstring, _pylab_helpers as _pylab_helpers, cbook as cbook, interactive as interactive, mlab as mlab, rcParamsDefault as rcParamsDefault, rcParamsOrig as rcParamsOrig, rcsetup as rcsetup
from matplotlib.artist import Artist as Artist
from matplotlib.axes import Axes as Axes, Subplot as Subplot
from matplotlib.axes._base import _AxesBase as _AxesBase
from matplotlib.axis import Tick as Tick
from matplotlib.backend_bases import Event as Event, FigureCanvasBase as FigureCanvasBase, FigureManagerBase as FigureManagerBase, MouseButton as MouseButton
from matplotlib.backends import BackendFilter as BackendFilter, backend_registry as backend_registry
from matplotlib.cm import ScalarMappable as ScalarMappable, _colormaps as _colormaps
from matplotlib.collections import Collection as Collection, EventCollection as EventCollection, FillBetweenPolyCollection as FillBetweenPolyCollection, LineCollection as LineCollection, PathCollection as PathCollection, PolyCollection as PolyCollection, QuadMesh as QuadMesh
from matplotlib.colorbar import Colorbar as Colorbar
from matplotlib.colorizer import Colorizer as Colorizer, ColorizingArtist as ColorizingArtist, _ColorizerInterface as _ColorizerInterface
from matplotlib.colors import Colormap as Colormap, Normalize as Normalize, _color_sequences as _color_sequences
from matplotlib.container import BarContainer as BarContainer, ErrorbarContainer as ErrorbarContainer, StemContainer as StemContainer
from matplotlib.contour import ContourSet as ContourSet, QuadContourSet as QuadContourSet
from matplotlib.figure import Figure as Figure, FigureBase as FigureBase, SubFigure as SubFigure, figaspect as figaspect
from matplotlib.gridspec import GridSpec as GridSpec, SubplotSpec as SubplotSpec
from matplotlib.image import AxesImage as AxesImage, FigureImage as FigureImage
from matplotlib.legend import Legend as Legend
from matplotlib.lines import AxLine as AxLine, Line2D as Line2D
from matplotlib.mlab import GaussianKDE as GaussianKDE
from matplotlib.patches import Arrow as Arrow, Circle as Circle, FancyArrow as FancyArrow, Polygon as Polygon, Rectangle as Rectangle, StepPatch as StepPatch, Wedge as Wedge
from matplotlib.projections import PolarAxes as PolarAxes
from matplotlib.quiver import Barbs as Barbs, Quiver as Quiver, QuiverKey as QuiverKey
from matplotlib.scale import ScaleBase as ScaleBase, get_scale_names as get_scale_names
from matplotlib.text import Annotation as Annotation, Text as Text
from matplotlib.typing import ColorType as ColorType, CoordsType as CoordsType, HashableList as HashableList, LineStyleType as LineStyleType, MarkerType as MarkerType
from matplotlib.widgets import Button as Button, Slider as Slider, SubplotTool as SubplotTool, Widget as Widget
from numpy.typing import ArrayLike as ArrayLike
from typing import Any, BinaryIO, Literal, TypeVar, overload
from typing_extensions import ParamSpec

_P = ParamSpec('_P')
_R = TypeVar('_R')
_T = TypeVar('_T')
_log: Incomplete
colormaps = _colormaps
color_sequences = _color_sequences

@overload
def _copy_docstring_and_deprecators(method: Any, func: Literal[None] = None) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]: ...
@overload
def _copy_docstring_and_deprecators(method: Any, func: Callable[_P, _R]) -> Callable[_P, _R]: ...

_NO_PYPLOT_NOTE: Incomplete

def _add_pyplot_note(func, wrapped_func) -> None:
    '''
    Add a note to the docstring of *func* that it is a pyplot wrapper.

    The note is added to the "Notes" section of the docstring. If that does
    not exist, a "Notes" section is created. In numpydoc, the "Notes"
    section is the third last possible section, only potentially followed by
    "References" and "Examples".
    '''

_ReplDisplayHook: Incomplete
_REPL_DISPLAYHOOK: Incomplete

def _draw_all_if_interactive() -> None: ...
def install_repl_displayhook() -> None:
    """
    Connect to the display hook of the current shell.

    The display hook gets called when the read-evaluate-print-loop (REPL) of
    the shell has finished the execution of a command. We use this callback
    to be able to automatically update a figure in interactive mode.

    This works both with IPython and with vanilla python shells.
    """
def uninstall_repl_displayhook() -> None:
    """Disconnect from the display hook of the current shell."""

draw_all: Incomplete

def set_loglevel(*args, **kwargs) -> None: ...
def findobj(o: Artist | None = None, match: Callable[[Artist], bool] | type[Artist] | None = None, include_self: bool = True) -> list[Artist]: ...

_backend_mod: type[matplotlib.backend_bases._Backend] | None

def _get_backend_mod() -> type[matplotlib.backend_bases._Backend]:
    """
    Ensure that a backend is selected and return it.

    This is currently private, but may be made public in the future.
    """
def switch_backend(newbackend: str) -> None:
    """
    Set the pyplot backend.

    Switching to an interactive backend is possible only if no event loop for
    another interactive backend has started.  Switching to and from
    non-interactive backends is always possible.

    If the new backend is different than the current backend then all open
    Figures will be closed via ``plt.close('all')``.

    Parameters
    ----------
    newbackend : str
        The case-insensitive name of the backend to use.

    """
def _warn_if_gui_out_of_main_thread() -> None: ...
def new_figure_manager(*args, **kwargs):
    """Create a new figure manager instance."""
def draw_if_interactive(*args, **kwargs):
    """
    Redraw the current figure if in interactive mode.

    .. warning::

        End users will typically not have to call this function because the
        the interactive mode takes care of this.
    """
def show(*args, **kwargs) -> None:
    """
    Display all open figures.

    Parameters
    ----------
    block : bool, optional
        Whether to wait for all figures to be closed before returning.

        If `True` block and run the GUI main loop until all figure windows
        are closed.

        If `False` ensure that all figure windows are displayed and return
        immediately.  In this case, you are responsible for ensuring
        that the event loop is running to have responsive figures.

        Defaults to True in non-interactive mode and to False in interactive
        mode (see `.pyplot.isinteractive`).

    See Also
    --------
    ion : Enable interactive mode, which shows / updates the figure after
          every plotting command, so that calling ``show()`` is not necessary.
    ioff : Disable interactive mode.
    savefig : Save the figure to an image file instead of showing it on screen.

    Notes
    -----
    **Saving figures to file and showing a window at the same time**

    If you want an image file as well as a user interface window, use
    `.pyplot.savefig` before `.pyplot.show`. At the end of (a blocking)
    ``show()`` the figure is closed and thus unregistered from pyplot. Calling
    `.pyplot.savefig` afterwards would save a new and thus empty figure. This
    limitation of command order does not apply if the show is non-blocking or
    if you keep a reference to the figure and use `.Figure.savefig`.

    **Auto-show in jupyter notebooks**

    The jupyter backends (activated via ``%matplotlib inline``,
    ``%matplotlib notebook``, or ``%matplotlib widget``), call ``show()`` at
    the end of every cell by default. Thus, you usually don't have to call it
    explicitly there.
    """
def isinteractive() -> bool:
    """
    Return whether plots are updated after every plotting command.

    The interactive mode is mainly useful if you build plots from the command
    line and want to see the effect of each command while you are building the
    figure.

    In interactive mode:

    - newly created figures will be shown immediately;
    - figures will automatically redraw on change;
    - `.pyplot.show` will not block by default.

    In non-interactive mode:

    - newly created figures and changes to figures will not be reflected until
      explicitly asked to be;
    - `.pyplot.show` will block by default.

    See Also
    --------
    ion : Enable interactive mode.
    ioff : Disable interactive mode.
    show : Show all figures (and maybe block).
    pause : Show all figures, and block for a time.
    """
def ioff() -> AbstractContextManager:
    """
    Disable interactive mode.

    See `.pyplot.isinteractive` for more details.

    See Also
    --------
    ion : Enable interactive mode.
    isinteractive : Whether interactive mode is enabled.
    show : Show all figures (and maybe block).
    pause : Show all figures, and block for a time.

    Notes
    -----
    For a temporary change, this can be used as a context manager::

        # if interactive mode is on
        # then figures will be shown on creation
        plt.ion()
        # This figure will be shown immediately
        fig = plt.figure()

        with plt.ioff():
            # interactive mode will be off
            # figures will not automatically be shown
            fig2 = plt.figure()
            # ...

    To enable optional usage as a context manager, this function returns a
    context manager object, which is not intended to be stored or
    accessed by the user.
    """
def ion() -> AbstractContextManager:
    """
    Enable interactive mode.

    See `.pyplot.isinteractive` for more details.

    See Also
    --------
    ioff : Disable interactive mode.
    isinteractive : Whether interactive mode is enabled.
    show : Show all figures (and maybe block).
    pause : Show all figures, and block for a time.

    Notes
    -----
    For a temporary change, this can be used as a context manager::

        # if interactive mode is off
        # then figures will not be shown on creation
        plt.ioff()
        # This figure will not be shown immediately
        fig = plt.figure()

        with plt.ion():
            # interactive mode will be on
            # figures will automatically be shown
            fig2 = plt.figure()
            # ...

    To enable optional usage as a context manager, this function returns a
    context manager object, which is not intended to be stored or
    accessed by the user.
    """
def pause(interval: float) -> None:
    """
    Run the GUI event loop for *interval* seconds.

    If there is an active figure, it will be updated and displayed before the
    pause, and the GUI event loop (if any) will run during the pause.

    This can be used for crude animation.  For more complex animation use
    :mod:`matplotlib.animation`.

    If there is no active figure, sleep for *interval* seconds instead.

    See Also
    --------
    matplotlib.animation : Proper animations
    show : Show all figures and optional block until all figures are closed.
    """
def rc(group: str, **kwargs) -> None: ...
def rc_context(rc: dict[str, Any] | None = None, fname: str | pathlib.Path | os.PathLike | None = None) -> AbstractContextManager[None]: ...
def rcdefaults() -> None: ...
def getp(obj, *args, **kwargs): ...
def get(obj, *args, **kwargs): ...
def setp(obj, *args, **kwargs): ...
def xkcd(scale: float = 1, length: float = 100, randomness: float = 2) -> ExitStack:
    """
    Turn on `xkcd <https://xkcd.com/>`_ sketch-style drawing mode.

    This will only have an effect on things drawn after this function is called.

    For best results, install the `xkcd script <https://github.com/ipython/xkcd-font/>`_
    font; xkcd fonts are not packaged with Matplotlib.

    Parameters
    ----------
    scale : float, optional
        The amplitude of the wiggle perpendicular to the source line.
    length : float, optional
        The length of the wiggle along the line.
    randomness : float, optional
        The scale factor by which the length is shrunken or expanded.

    Notes
    -----
    This function works by a number of rcParams, so it will probably
    override others you have set before.

    If you want the effects of this function to be temporary, it can
    be used as a context manager, for example::

        with plt.xkcd():
            # This figure will be in XKCD-style
            fig1 = plt.figure()
            # ...

        # This figure will be in regular style
        fig2 = plt.figure()
    """
def figure(num: int | str | Figure | SubFigure | None = None, figsize: tuple[float, float] | None = None, dpi: float | None = None, *, facecolor: ColorType | None = None, edgecolor: ColorType | None = None, frameon: bool = True, FigureClass: type[Figure] = ..., clear: bool = False, **kwargs) -> Figure:
    """
    Create a new figure, or activate an existing figure.

    Parameters
    ----------
    num : int or str or `.Figure` or `.SubFigure`, optional
        A unique identifier for the figure.

        If a figure with that identifier already exists, this figure is made
        active and returned. An integer refers to the ``Figure.number``
        attribute, a string refers to the figure label.

        If there is no figure with the identifier or *num* is not given, a new
        figure is created, made active and returned.  If *num* is an int, it
        will be used for the ``Figure.number`` attribute, otherwise, an
        auto-generated integer value is used (starting at 1 and incremented
        for each new figure). If *num* is a string, the figure label and the
        window title is set to this value.  If num is a ``SubFigure``, its
        parent ``Figure`` is activated.

    figsize : (float, float), default: :rc:`figure.figsize`
        Width, height in inches.

    dpi : float, default: :rc:`figure.dpi`
        The resolution of the figure in dots-per-inch.

    facecolor : :mpltype:`color`, default: :rc:`figure.facecolor`
        The background color.

    edgecolor : :mpltype:`color`, default: :rc:`figure.edgecolor`
        The border color.

    frameon : bool, default: True
        If False, suppress drawing the figure frame.

    FigureClass : subclass of `~matplotlib.figure.Figure`
        If set, an instance of this subclass will be created, rather than a
        plain `.Figure`.

    clear : bool, default: False
        If True and the figure already exists, then it is cleared.

    layout : {'constrained', 'compressed', 'tight', 'none', `.LayoutEngine`, None}, default: None
        The layout mechanism for positioning of plot elements to avoid
        overlapping Axes decorations (labels, ticks, etc). Note that layout
        managers can measurably slow down figure display.

        - 'constrained': The constrained layout solver adjusts Axes sizes
          to avoid overlapping Axes decorations.  Can handle complex plot
          layouts and colorbars, and is thus recommended.

          See :ref:`constrainedlayout_guide`
          for examples.

        - 'compressed': uses the same algorithm as 'constrained', but
          removes extra space between fixed-aspect-ratio Axes.  Best for
          simple grids of Axes.

        - 'tight': Use the tight layout mechanism. This is a relatively
          simple algorithm that adjusts the subplot parameters so that
          decorations do not overlap. See `.Figure.set_tight_layout` for
          further details.

        - 'none': Do not use a layout engine.

        - A `.LayoutEngine` instance. Builtin layout classes are
          `.ConstrainedLayoutEngine` and `.TightLayoutEngine`, more easily
          accessible by 'constrained' and 'tight'.  Passing an instance
          allows third parties to provide their own layout engine.

        If not given, fall back to using the parameters *tight_layout* and
        *constrained_layout*, including their config defaults
        :rc:`figure.autolayout` and :rc:`figure.constrained_layout.use`.

    **kwargs
        Additional keyword arguments are passed to the `.Figure` constructor.

    Returns
    -------
    `~matplotlib.figure.Figure`

    Notes
    -----
    A newly created figure is passed to the `~.FigureCanvasBase.new_manager`
    method or the `new_figure_manager` function provided by the current
    backend, which install a canvas and a manager on the figure.

    Once this is done, :rc:`figure.hooks` are called, one at a time, on the
    figure; these hooks allow arbitrary customization of the figure (e.g.,
    attaching callbacks) or of associated elements (e.g., modifying the
    toolbar).  See :doc:`/gallery/user_interfaces/mplcvd` for an example of
    toolbar customization.

    If you are creating many figures, make sure you explicitly call
    `.pyplot.close` on the figures you are not using, because this will
    enable pyplot to properly clean up the memory.

    `~matplotlib.rcParams` defines the default values, which can be modified
    in the matplotlibrc file.
    """
def _auto_draw_if_interactive(fig, val) -> None:
    """
    An internal helper function for making sure that auto-redrawing
    works as intended in the plain python repl.

    Parameters
    ----------
    fig : Figure
        A figure object which is assumed to be associated with a canvas
    """
def gcf() -> Figure:
    """
    Get the current figure.

    If there is currently no figure on the pyplot figure stack, a new one is
    created using `~.pyplot.figure()`.  (To test whether there is currently a
    figure on the pyplot figure stack, check whether `~.pyplot.get_fignums()`
    is empty.)
    """
def fignum_exists(num: int | str) -> bool:
    """
    Return whether the figure with the given id exists.

    Parameters
    ----------
    num : int or str
        A figure identifier.

    Returns
    -------
    bool
        Whether or not a figure with id *num* exists.
    """
def get_fignums() -> list[int]:
    """Return a list of existing figure numbers."""
def get_figlabels() -> list[Any]:
    """Return a list of existing figure labels."""
def get_current_fig_manager() -> FigureManagerBase | None:
    """
    Return the figure manager of the current figure.

    The figure manager is a container for the actual backend-depended window
    that displays the figure on screen.

    If no current figure exists, a new one is created, and its figure
    manager is returned.

    Returns
    -------
    `.FigureManagerBase` or backend-dependent subclass thereof
    """
def connect(s: str, func: Callable[[Event], Any]) -> int: ...
def disconnect(cid: int) -> None: ...
def close(fig: None | int | str | Figure | Literal['all'] = None) -> None:
    """
    Close a figure window.

    Parameters
    ----------
    fig : None or int or str or `.Figure`
        The figure to close. There are a number of ways to specify this:

        - *None*: the current figure
        - `.Figure`: the given `.Figure` instance
        - ``int``: a figure number
        - ``str``: a figure name
        - 'all': all figures

    """
def clf() -> None:
    """Clear the current figure."""
def draw() -> None:
    '''
    Redraw the current figure.

    This is used to update a figure that has been altered, but not
    automatically re-drawn.  If interactive mode is on (via `.ion()`), this
    should be only rarely needed, but there may be ways to modify the state of
    a figure without marking it as "stale".  Please report these cases as bugs.

    This is equivalent to calling ``fig.canvas.draw_idle()``, where ``fig`` is
    the current figure.

    See Also
    --------
    .FigureCanvasBase.draw_idle
    .FigureCanvasBase.draw
    '''
def savefig(*args, **kwargs) -> None: ...
def figlegend(*args, **kwargs) -> Legend: ...
def axes(arg: None | tuple[float, float, float, float] = None, **kwargs) -> matplotlib.axes.Axes:
    """
    Add an Axes to the current figure and make it the current Axes.

    Call signatures::

        plt.axes()
        plt.axes(rect, projection=None, polar=False, **kwargs)
        plt.axes(ax)

    Parameters
    ----------
    arg : None or 4-tuple
        The exact behavior of this function depends on the type:

        - *None*: A new full window Axes is added using
          ``subplot(**kwargs)``.
        - 4-tuple of floats *rect* = ``(left, bottom, width, height)``.
          A new Axes is added with dimensions *rect* in normalized
          (0, 1) units using `~.Figure.add_axes` on the current figure.

    projection : {None, 'aitoff', 'hammer', 'lambert', 'mollweide', 'polar', 'rectilinear', str}, optional
        The projection type of the `~.axes.Axes`. *str* is the name of
        a custom projection, see `~matplotlib.projections`. The default
        None results in a 'rectilinear' projection.

    polar : bool, default: False
        If True, equivalent to projection='polar'.

    sharex, sharey : `~matplotlib.axes.Axes`, optional
        Share the x or y `~matplotlib.axis` with sharex and/or sharey.
        The axis will have the same limits, ticks, and scale as the axis
        of the shared Axes.

    label : str
        A label for the returned Axes.

    Returns
    -------
    `~.axes.Axes`, or a subclass of `~.axes.Axes`
        The returned Axes class depends on the projection used. It is
        `~.axes.Axes` if rectilinear projection is used and
        `.projections.polar.PolarAxes` if polar projection is used.

    Other Parameters
    ----------------
    **kwargs
        This method also takes the keyword arguments for
        the returned Axes class. The keyword arguments for the
        rectilinear Axes class `~.axes.Axes` can be found in
        the following table but there might also be other keyword
        arguments if another projection is used, see the actual Axes
        class.

        %(Axes:kwdoc)s

    See Also
    --------
    .Figure.add_axes
    .pyplot.subplot
    .Figure.add_subplot
    .Figure.subplots
    .pyplot.subplots

    Examples
    --------
    ::

        # Creating a new full window Axes
        plt.axes()

        # Creating a new Axes with specified dimensions and a grey background
        plt.axes((left, bottom, width, height), facecolor='grey')
    """
def delaxes(ax: matplotlib.axes.Axes | None = None) -> None:
    """
    Remove an `~.axes.Axes` (defaulting to the current Axes) from its figure.
    """
def sca(ax: Axes) -> None:
    """
    Set the current Axes to *ax* and the current Figure to the parent of *ax*.
    """
def cla() -> None:
    """Clear the current Axes."""
def subplot(*args, **kwargs) -> Axes:
    '''
    Add an Axes to the current figure or retrieve an existing Axes.

    This is a wrapper of `.Figure.add_subplot` which provides additional
    behavior when working with the implicit API (see the notes section).

    Call signatures::

       subplot(nrows, ncols, index, **kwargs)
       subplot(pos, **kwargs)
       subplot(**kwargs)
       subplot(ax)

    Parameters
    ----------
    *args : int, (int, int, *index*), or `.SubplotSpec`, default: (1, 1, 1)
        The position of the subplot described by one of

        - Three integers (*nrows*, *ncols*, *index*). The subplot will take the
          *index* position on a grid with *nrows* rows and *ncols* columns.
          *index* starts at 1 in the upper left corner and increases to the
          right. *index* can also be a two-tuple specifying the (*first*,
          *last*) indices (1-based, and including *last*) of the subplot, e.g.,
          ``fig.add_subplot(3, 1, (1, 2))`` makes a subplot that spans the
          upper 2/3 of the figure.
        - A 3-digit integer. The digits are interpreted as if given separately
          as three single-digit integers, i.e. ``fig.add_subplot(235)`` is the
          same as ``fig.add_subplot(2, 3, 5)``. Note that this can only be used
          if there are no more than 9 subplots.
        - A `.SubplotSpec`.

    projection : {None, \'aitoff\', \'hammer\', \'lambert\', \'mollweide\', \'polar\', \'rectilinear\', str}, optional
        The projection type of the subplot (`~.axes.Axes`). *str* is the name
        of a custom projection, see `~matplotlib.projections`. The default
        None results in a \'rectilinear\' projection.

    polar : bool, default: False
        If True, equivalent to projection=\'polar\'.

    sharex, sharey : `~matplotlib.axes.Axes`, optional
        Share the x or y `~matplotlib.axis` with sharex and/or sharey. The
        axis will have the same limits, ticks, and scale as the axis of the
        shared Axes.

    label : str
        A label for the returned Axes.

    Returns
    -------
    `~.axes.Axes`

        The Axes of the subplot. The returned Axes can actually be an instance
        of a subclass, such as `.projections.polar.PolarAxes` for polar
        projections.

    Other Parameters
    ----------------
    **kwargs
        This method also takes the keyword arguments for the returned Axes
        base class; except for the *figure* argument. The keyword arguments
        for the rectilinear base class `~.axes.Axes` can be found in
        the following table but there might also be other keyword
        arguments if another projection is used.

        %(Axes:kwdoc)s

    Notes
    -----
    .. versionchanged:: 3.8
        In versions prior to 3.8, any preexisting Axes that overlap with the new Axes
        beyond sharing a boundary was deleted. Deletion does not happen in more
        recent versions anymore. Use `.Axes.remove` explicitly if needed.

    If you do not want this behavior, use the `.Figure.add_subplot` method
    or the `.pyplot.axes` function instead.

    If no *kwargs* are passed and there exists an Axes in the location
    specified by *args* then that Axes will be returned rather than a new
    Axes being created.

    If *kwargs* are passed and there exists an Axes in the location
    specified by *args*, the projection type is the same, and the
    *kwargs* match with the existing Axes, then the existing Axes is
    returned.  Otherwise a new Axes is created with the specified
    parameters.  We save a reference to the *kwargs* which we use
    for this comparison.  If any of the values in *kwargs* are
    mutable we will not detect the case where they are mutated.
    In these cases we suggest using `.Figure.add_subplot` and the
    explicit Axes API rather than the implicit pyplot API.

    See Also
    --------
    .Figure.add_subplot
    .pyplot.subplots
    .pyplot.axes
    .Figure.subplots

    Examples
    --------
    ::

        plt.subplot(221)

        # equivalent but more general
        ax1 = plt.subplot(2, 2, 1)

        # add a subplot with no frame
        ax2 = plt.subplot(222, frameon=False)

        # add a polar subplot
        plt.subplot(223, projection=\'polar\')

        # add a red subplot that shares the x-axis with ax1
        plt.subplot(224, sharex=ax1, facecolor=\'red\')

        # delete ax2 from the figure
        plt.delaxes(ax2)

        # add ax2 to the figure again
        plt.subplot(ax2)

        # make the first Axes "current" again
        plt.subplot(221)

    '''
@overload
def subplots(nrows: Literal[1] = ..., ncols: Literal[1] = ..., *, sharex: bool | Literal['none', 'all', 'row', 'col'] = ..., sharey: bool | Literal['none', 'all', 'row', 'col'] = ..., squeeze: Literal[True] = ..., width_ratios: Sequence[float] | None = ..., height_ratios: Sequence[float] | None = ..., subplot_kw: dict[str, Any] | None = ..., gridspec_kw: dict[str, Any] | None = ..., **fig_kw) -> tuple[Figure, Axes]: ...
@overload
def subplots(nrows: int = ..., ncols: int = ..., *, sharex: bool | Literal['none', 'all', 'row', 'col'] = ..., sharey: bool | Literal['none', 'all', 'row', 'col'] = ..., squeeze: Literal[False], width_ratios: Sequence[float] | None = ..., height_ratios: Sequence[float] | None = ..., subplot_kw: dict[str, Any] | None = ..., gridspec_kw: dict[str, Any] | None = ..., **fig_kw) -> tuple[Figure, np.ndarray]: ...
@overload
def subplots(nrows: int = ..., ncols: int = ..., *, sharex: bool | Literal['none', 'all', 'row', 'col'] = ..., sharey: bool | Literal['none', 'all', 'row', 'col'] = ..., squeeze: bool = ..., width_ratios: Sequence[float] | None = ..., height_ratios: Sequence[float] | None = ..., subplot_kw: dict[str, Any] | None = ..., gridspec_kw: dict[str, Any] | None = ..., **fig_kw) -> tuple[Figure, Any]: ...
@overload
def subplot_mosaic(mosaic: str, *, sharex: bool = ..., sharey: bool = ..., width_ratios: ArrayLike | None = ..., height_ratios: ArrayLike | None = ..., empty_sentinel: str = ..., subplot_kw: dict[str, Any] | None = ..., gridspec_kw: dict[str, Any] | None = ..., per_subplot_kw: dict[str | tuple[str, ...], dict[str, Any]] | None = ..., **fig_kw: Any) -> tuple[Figure, dict[str, matplotlib.axes.Axes]]: ...
@overload
def subplot_mosaic(mosaic: list[HashableList[_T]], *, sharex: bool = ..., sharey: bool = ..., width_ratios: ArrayLike | None = ..., height_ratios: ArrayLike | None = ..., empty_sentinel: _T = ..., subplot_kw: dict[str, Any] | None = ..., gridspec_kw: dict[str, Any] | None = ..., per_subplot_kw: dict[_T | tuple[_T, ...], dict[str, Any]] | None = ..., **fig_kw: Any) -> tuple[Figure, dict[_T, matplotlib.axes.Axes]]: ...
@overload
def subplot_mosaic(mosaic: list[HashableList[Hashable]], *, sharex: bool = ..., sharey: bool = ..., width_ratios: ArrayLike | None = ..., height_ratios: ArrayLike | None = ..., empty_sentinel: Any = ..., subplot_kw: dict[str, Any] | None = ..., gridspec_kw: dict[str, Any] | None = ..., per_subplot_kw: dict[Hashable | tuple[Hashable, ...], dict[str, Any]] | None = ..., **fig_kw: Any) -> tuple[Figure, dict[Hashable, matplotlib.axes.Axes]]: ...
def subplot2grid(shape: tuple[int, int], loc: tuple[int, int], rowspan: int = 1, colspan: int = 1, fig: Figure | None = None, **kwargs) -> matplotlib.axes.Axes:
    """
    Create a subplot at a specific location inside a regular grid.

    Parameters
    ----------
    shape : (int, int)
        Number of rows and of columns of the grid in which to place axis.
    loc : (int, int)
        Row number and column number of the axis location within the grid.
    rowspan : int, default: 1
        Number of rows for the axis to span downwards.
    colspan : int, default: 1
        Number of columns for the axis to span to the right.
    fig : `.Figure`, optional
        Figure to place the subplot in. Defaults to the current figure.
    **kwargs
        Additional keyword arguments are handed to `~.Figure.add_subplot`.

    Returns
    -------
    `~.axes.Axes`

        The Axes of the subplot. The returned Axes can actually be an instance
        of a subclass, such as `.projections.polar.PolarAxes` for polar
        projections.

    Notes
    -----
    The following call ::

        ax = subplot2grid((nrows, ncols), (row, col), rowspan, colspan)

    is identical to ::

        fig = gcf()
        gs = fig.add_gridspec(nrows, ncols)
        ax = fig.add_subplot(gs[row:row+rowspan, col:col+colspan])
    """
def twinx(ax: matplotlib.axes.Axes | None = None) -> _AxesBase:
    """
    Make and return a second Axes that shares the *x*-axis.  The new Axes will
    overlay *ax* (or the current Axes if *ax* is *None*), and its ticks will be
    on the right.

    Examples
    --------
    :doc:`/gallery/subplots_axes_and_figures/two_scales`
    """
def twiny(ax: matplotlib.axes.Axes | None = None) -> _AxesBase:
    """
    Make and return a second Axes that shares the *y*-axis.  The new Axes will
    overlay *ax* (or the current Axes if *ax* is *None*), and its ticks will be
    on the top.

    Examples
    --------
    :doc:`/gallery/subplots_axes_and_figures/two_scales`
    """
def subplot_tool(targetfig: Figure | None = None) -> SubplotTool | None:
    """
    Launch a subplot tool window for a figure.

    Returns
    -------
    `matplotlib.widgets.SubplotTool`
    """
def box(on: bool | None = None) -> None:
    """
    Turn the Axes box on or off on the current Axes.

    Parameters
    ----------
    on : bool or None
        The new `~matplotlib.axes.Axes` box state. If ``None``, toggle
        the state.

    See Also
    --------
    :meth:`matplotlib.axes.Axes.set_frame_on`
    :meth:`matplotlib.axes.Axes.get_frame_on`
    """
def xlim(*args, **kwargs) -> tuple[float, float]:
    """
    Get or set the x limits of the current Axes.

    Call signatures::

        left, right = xlim()  # return the current xlim
        xlim((left, right))   # set the xlim to left, right
        xlim(left, right)     # set the xlim to left, right

    If you do not specify args, you can pass *left* or *right* as kwargs,
    i.e.::

        xlim(right=3)  # adjust the right leaving left unchanged
        xlim(left=1)  # adjust the left leaving right unchanged

    Setting limits turns autoscaling off for the x-axis.

    Returns
    -------
    left, right
        A tuple of the new x-axis limits.

    Notes
    -----
    Calling this function with no arguments (e.g. ``xlim()``) is the pyplot
    equivalent of calling `~.Axes.get_xlim` on the current Axes.
    Calling this function with arguments is the pyplot equivalent of calling
    `~.Axes.set_xlim` on the current Axes. All arguments are passed though.
    """
def ylim(*args, **kwargs) -> tuple[float, float]:
    """
    Get or set the y-limits of the current Axes.

    Call signatures::

        bottom, top = ylim()  # return the current ylim
        ylim((bottom, top))   # set the ylim to bottom, top
        ylim(bottom, top)     # set the ylim to bottom, top

    If you do not specify args, you can alternatively pass *bottom* or
    *top* as kwargs, i.e.::

        ylim(top=3)  # adjust the top leaving bottom unchanged
        ylim(bottom=1)  # adjust the bottom leaving top unchanged

    Setting limits turns autoscaling off for the y-axis.

    Returns
    -------
    bottom, top
        A tuple of the new y-axis limits.

    Notes
    -----
    Calling this function with no arguments (e.g. ``ylim()``) is the pyplot
    equivalent of calling `~.Axes.get_ylim` on the current Axes.
    Calling this function with arguments is the pyplot equivalent of calling
    `~.Axes.set_ylim` on the current Axes. All arguments are passed though.
    """
def xticks(ticks: ArrayLike | None = None, labels: Sequence[str] | None = None, *, minor: bool = False, **kwargs) -> tuple[list[Tick] | np.ndarray, list[Text]]:
    """
    Get or set the current tick locations and labels of the x-axis.

    Pass no arguments to return the current values without modifying them.

    Parameters
    ----------
    ticks : array-like, optional
        The list of xtick locations.  Passing an empty list removes all xticks.
    labels : array-like, optional
        The labels to place at the given *ticks* locations.  This argument can
        only be passed if *ticks* is passed as well.
    minor : bool, default: False
        If ``False``, get/set the major ticks/labels; if ``True``, the minor
        ticks/labels.
    **kwargs
        `.Text` properties can be used to control the appearance of the labels.

        .. warning::

            This only sets the properties of the current ticks, which is
            only sufficient if you either pass *ticks*, resulting in a
            fixed list of ticks, or if the plot is static.

            Ticks are not guaranteed to be persistent. Various operations
            can create, delete and modify the Tick instances. There is an
            imminent risk that these settings can get lost if you work on
            the figure further (including also panning/zooming on a
            displayed figure).

            Use `~.pyplot.tick_params` instead if possible.


    Returns
    -------
    locs
        The list of xtick locations.
    labels
        The list of xlabel `.Text` objects.

    Notes
    -----
    Calling this function with no arguments (e.g. ``xticks()``) is the pyplot
    equivalent of calling `~.Axes.get_xticks` and `~.Axes.get_xticklabels` on
    the current Axes.
    Calling this function with arguments is the pyplot equivalent of calling
    `~.Axes.set_xticks` and `~.Axes.set_xticklabels` on the current Axes.

    Examples
    --------
    >>> locs, labels = xticks()  # Get the current locations and labels.
    >>> xticks(np.arange(0, 1, step=0.2))  # Set label locations.
    >>> xticks(np.arange(3), ['Tom', 'Dick', 'Sue'])  # Set text labels.
    >>> xticks([0, 1, 2], ['January', 'February', 'March'],
    ...        rotation=20)  # Set text labels and properties.
    >>> xticks([])  # Disable xticks.
    """
def yticks(ticks: ArrayLike | None = None, labels: Sequence[str] | None = None, *, minor: bool = False, **kwargs) -> tuple[list[Tick] | np.ndarray, list[Text]]:
    """
    Get or set the current tick locations and labels of the y-axis.

    Pass no arguments to return the current values without modifying them.

    Parameters
    ----------
    ticks : array-like, optional
        The list of ytick locations.  Passing an empty list removes all yticks.
    labels : array-like, optional
        The labels to place at the given *ticks* locations.  This argument can
        only be passed if *ticks* is passed as well.
    minor : bool, default: False
        If ``False``, get/set the major ticks/labels; if ``True``, the minor
        ticks/labels.
    **kwargs
        `.Text` properties can be used to control the appearance of the labels.

        .. warning::

            This only sets the properties of the current ticks, which is
            only sufficient if you either pass *ticks*, resulting in a
            fixed list of ticks, or if the plot is static.

            Ticks are not guaranteed to be persistent. Various operations
            can create, delete and modify the Tick instances. There is an
            imminent risk that these settings can get lost if you work on
            the figure further (including also panning/zooming on a
            displayed figure).

            Use `~.pyplot.tick_params` instead if possible.

    Returns
    -------
    locs
        The list of ytick locations.
    labels
        The list of ylabel `.Text` objects.

    Notes
    -----
    Calling this function with no arguments (e.g. ``yticks()``) is the pyplot
    equivalent of calling `~.Axes.get_yticks` and `~.Axes.get_yticklabels` on
    the current Axes.
    Calling this function with arguments is the pyplot equivalent of calling
    `~.Axes.set_yticks` and `~.Axes.set_yticklabels` on the current Axes.

    Examples
    --------
    >>> locs, labels = yticks()  # Get the current locations and labels.
    >>> yticks(np.arange(0, 1, step=0.2))  # Set label locations.
    >>> yticks(np.arange(3), ['Tom', 'Dick', 'Sue'])  # Set text labels.
    >>> yticks([0, 1, 2], ['January', 'February', 'March'],
    ...        rotation=45)  # Set text labels and properties.
    >>> yticks([])  # Disable yticks.
    """
def rgrids(radii: ArrayLike | None = None, labels: Sequence[str | Text] | None = None, angle: float | None = None, fmt: str | None = None, **kwargs) -> tuple[list[Line2D], list[Text]]:
    """
    Get or set the radial gridlines on the current polar plot.

    Call signatures::

     lines, labels = rgrids()
     lines, labels = rgrids(radii, labels=None, angle=22.5, fmt=None, **kwargs)

    When called with no arguments, `.rgrids` simply returns the tuple
    (*lines*, *labels*). When called with arguments, the labels will
    appear at the specified radial distances and angle.

    Parameters
    ----------
    radii : tuple with floats
        The radii for the radial gridlines

    labels : tuple with strings or None
        The labels to use at each radial gridline. The
        `matplotlib.ticker.ScalarFormatter` will be used if None.

    angle : float
        The angular position of the radius labels in degrees.

    fmt : str or None
        Format string used in `matplotlib.ticker.FormatStrFormatter`.
        For example '%f'.

    Returns
    -------
    lines : list of `.lines.Line2D`
        The radial gridlines.

    labels : list of `.text.Text`
        The tick labels.

    Other Parameters
    ----------------
    **kwargs
        *kwargs* are optional `.Text` properties for the labels.

    See Also
    --------
    .pyplot.thetagrids
    .projections.polar.PolarAxes.set_rgrids
    .Axis.get_gridlines
    .Axis.get_ticklabels

    Examples
    --------
    ::

      # set the locations of the radial gridlines
      lines, labels = rgrids( (0.25, 0.5, 1.0) )

      # set the locations and labels of the radial gridlines
      lines, labels = rgrids( (0.25, 0.5, 1.0), ('Tom', 'Dick', 'Harry' ))
    """
def thetagrids(angles: ArrayLike | None = None, labels: Sequence[str | Text] | None = None, fmt: str | None = None, **kwargs) -> tuple[list[Line2D], list[Text]]:
    """
    Get or set the theta gridlines on the current polar plot.

    Call signatures::

     lines, labels = thetagrids()
     lines, labels = thetagrids(angles, labels=None, fmt=None, **kwargs)

    When called with no arguments, `.thetagrids` simply returns the tuple
    (*lines*, *labels*). When called with arguments, the labels will
    appear at the specified angles.

    Parameters
    ----------
    angles : tuple with floats, degrees
        The angles of the theta gridlines.

    labels : tuple with strings or None
        The labels to use at each radial gridline. The
        `.projections.polar.ThetaFormatter` will be used if None.

    fmt : str or None
        Format string used in `matplotlib.ticker.FormatStrFormatter`.
        For example '%f'. Note that the angle in radians will be used.

    Returns
    -------
    lines : list of `.lines.Line2D`
        The theta gridlines.

    labels : list of `.text.Text`
        The tick labels.

    Other Parameters
    ----------------
    **kwargs
        *kwargs* are optional `.Text` properties for the labels.

    See Also
    --------
    .pyplot.rgrids
    .projections.polar.PolarAxes.set_thetagrids
    .Axis.get_gridlines
    .Axis.get_ticklabels

    Examples
    --------
    ::

      # set the locations of the angular gridlines
      lines, labels = thetagrids(range(45, 360, 90))

      # set the locations and labels of the angular gridlines
      lines, labels = thetagrids(range(45, 360, 90), ('NE', 'NW', 'SW', 'SE'))
    """
def get_plot_commands() -> list[str]:
    """
    Get a sorted list of all of the plotting commands.
    """
def _get_pyplot_commands() -> list[str]: ...
def colorbar(mappable: ScalarMappable | ColorizingArtist | None = None, cax: matplotlib.axes.Axes | None = None, ax: matplotlib.axes.Axes | Iterable[matplotlib.axes.Axes] | None = None, **kwargs) -> Colorbar: ...
def clim(vmin: float | None = None, vmax: float | None = None) -> None:
    """
    Set the color limits of the current image.

    If either *vmin* or *vmax* is None, the image min/max respectively
    will be used for color scaling.

    If you want to set the clim of multiple images, use
    `~.ScalarMappable.set_clim` on every image, for example::

      for im in gca().get_images():
          im.set_clim(0, 0.5)

    """
def get_cmap(name: Colormap | str | None = None, lut: int | None = None) -> Colormap:
    """
    Get a colormap instance, defaulting to rc values if *name* is None.

    Parameters
    ----------
    name : `~matplotlib.colors.Colormap` or str or None, default: None
        If a `.Colormap` instance, it will be returned. Otherwise, the name of
        a colormap known to Matplotlib, which will be resampled by *lut*. The
        default, None, means :rc:`image.cmap`.
    lut : int or None, default: None
        If *name* is not already a Colormap instance and *lut* is not None, the
        colormap will be resampled to have *lut* entries in the lookup table.

    Returns
    -------
    Colormap
    """
def set_cmap(cmap: Colormap | str) -> None:
    """
    Set the default colormap, and applies it to the current image if any.

    Parameters
    ----------
    cmap : `~matplotlib.colors.Colormap` or str
        A colormap instance or the name of a registered colormap.

    See Also
    --------
    colormaps
    get_cmap
    """
def imread(fname: str | pathlib.Path | BinaryIO, format: str | None = None) -> np.ndarray: ...
def imsave(fname: str | os.PathLike | BinaryIO, arr: ArrayLike, **kwargs) -> None: ...
def matshow(A: ArrayLike, fignum: None | int = None, **kwargs) -> AxesImage:
    """
    Display a 2D array as a matrix in a new figure window.

    The origin is set at the upper left hand corner.
    The indexing is ``(row, column)`` so that the first index runs vertically
    and the second index runs horizontally in the figure:

    .. code-block:: none

        A[0, 0]   ⋯ A[0, M-1]
           ⋮             ⋮
        A[N-1, 0] ⋯ A[N-1, M-1]

    The aspect ratio of the figure window is that of the array,
    unless this would make an excessively short or narrow figure.

    Tick labels for the xaxis are placed on top.

    Parameters
    ----------
    A : 2D array-like
        The matrix to be displayed.

    fignum : None or int
        If *None*, create a new, appropriately sized figure window.

        If 0, use the current Axes (creating one if there is none, without ever
        adjusting the figure size).

        Otherwise, create a new Axes on the figure with the given number
        (creating it at the appropriate size if it does not exist, but not
        adjusting the figure size otherwise).  Note that this will be drawn on
        top of any preexisting Axes on the figure.

    Returns
    -------
    `~matplotlib.image.AxesImage`

    Other Parameters
    ----------------
    **kwargs : `~matplotlib.axes.Axes.imshow` arguments

    """
def polar(*args, **kwargs) -> list[Line2D]:
    """
    Make a polar plot.

    call signature::

      polar(theta, r, [fmt], **kwargs)

    This is a convenience wrapper around `.pyplot.plot`. It ensures that the
    current Axes is polar (or creates one if needed) and then passes all parameters
    to ``.pyplot.plot``.

    .. note::
        When making polar plots using the :ref:`pyplot API <pyplot_interface>`,
        ``polar()`` should typically be the first command because that makes sure
        a polar Axes is created. Using other commands such as ``plt.title()``
        before this can lead to the implicit creation of a rectangular Axes, in which
        case a subsequent ``polar()`` call will fail.
    """
def figimage(X: ArrayLike, xo: int = 0, yo: int = 0, alpha: float | None = None, norm: str | Normalize | None = None, cmap: str | Colormap | None = None, vmin: float | None = None, vmax: float | None = None, origin: Literal['upper', 'lower'] | None = None, resize: bool = False, *, colorizer: Colorizer | None = None, **kwargs) -> FigureImage: ...
def figtext(x: float, y: float, s: str, fontdict: dict[str, Any] | None = None, **kwargs) -> Text: ...
def gca() -> Axes: ...
def gci() -> ColorizingArtist | None: ...
def ginput(n: int = 1, timeout: float = 30, show_clicks: bool = True, mouse_add: MouseButton = ..., mouse_pop: MouseButton = ..., mouse_stop: MouseButton = ...) -> list[tuple[int, int]]: ...
def subplots_adjust(left: float | None = None, bottom: float | None = None, right: float | None = None, top: float | None = None, wspace: float | None = None, hspace: float | None = None) -> None: ...
def suptitle(t: str, **kwargs) -> Text: ...
def tight_layout(*, pad: float = 1.08, h_pad: float | None = None, w_pad: float | None = None, rect: tuple[float, float, float, float] | None = None) -> None: ...
def waitforbuttonpress(timeout: float = -1) -> None | bool: ...
def acorr(x: ArrayLike, *, data: Incomplete | None = None, **kwargs) -> tuple[np.ndarray, np.ndarray, LineCollection | Line2D, Line2D | None]: ...
def angle_spectrum(x: ArrayLike, Fs: float | None = None, Fc: int | None = None, window: Callable[[ArrayLike], ArrayLike] | ArrayLike | None = None, pad_to: int | None = None, sides: Literal['default', 'onesided', 'twosided'] | None = None, *, data: Incomplete | None = None, **kwargs) -> tuple[np.ndarray, np.ndarray, Line2D]: ...
def annotate(text: str, xy: tuple[float, float], xytext: tuple[float, float] | None = None, xycoords: CoordsType = 'data', textcoords: CoordsType | None = None, arrowprops: dict[str, Any] | None = None, annotation_clip: bool | None = None, **kwargs) -> Annotation: ...
def arrow(x: float, y: float, dx: float, dy: float, **kwargs) -> FancyArrow: ...
def autoscale(enable: bool = True, axis: Literal['both', 'x', 'y'] = 'both', tight: bool | None = None) -> None: ...
def axhline(y: float = 0, xmin: float = 0, xmax: float = 1, **kwargs) -> Line2D: ...
def axhspan(ymin: float, ymax: float, xmin: float = 0, xmax: float = 1, **kwargs) -> Rectangle: ...
def axis(arg: tuple[float, float, float, float] | bool | str | None = None, /, *, emit: bool = True, **kwargs) -> tuple[float, float, float, float]: ...
def axline(xy1: tuple[float, float], xy2: tuple[float, float] | None = None, *, slope: float | None = None, **kwargs) -> AxLine: ...
def axvline(x: float = 0, ymin: float = 0, ymax: float = 1, **kwargs) -> Line2D: ...
def axvspan(xmin: float, xmax: float, ymin: float = 0, ymax: float = 1, **kwargs) -> Rectangle: ...
def bar(x: float | ArrayLike, height: float | ArrayLike, width: float | ArrayLike = 0.8, bottom: float | ArrayLike | None = None, *, align: Literal['center', 'edge'] = 'center', data: Incomplete | None = None, **kwargs) -> BarContainer: ...
def barbs(*args, data: Incomplete | None = None, **kwargs) -> Barbs: ...
def barh(y: float | ArrayLike, width: float | ArrayLike, height: float | ArrayLike = 0.8, left: float | ArrayLike | None = None, *, align: Literal['center', 'edge'] = 'center', data: Incomplete | None = None, **kwargs) -> BarContainer: ...
def bar_label(container: BarContainer, labels: ArrayLike | None = None, *, fmt: str | Callable[[float], str] = '%g', label_type: Literal['center', 'edge'] = 'edge', padding: float = 0, **kwargs) -> list[Annotation]: ...
def boxplot(x: ArrayLike | Sequence[ArrayLike], notch: bool | None = None, sym: str | None = None, vert: bool | None = None, orientation: Literal['vertical', 'horizontal'] = 'vertical', whis: float | tuple[float, float] | None = None, positions: ArrayLike | None = None, widths: float | ArrayLike | None = None, patch_artist: bool | None = None, bootstrap: int | None = None, usermedians: ArrayLike | None = None, conf_intervals: ArrayLike | None = None, meanline: bool | None = None, showmeans: bool | None = None, showcaps: bool | None = None, showbox: bool | None = None, showfliers: bool | None = None, boxprops: dict[str, Any] | None = None, tick_labels: Sequence[str] | None = None, flierprops: dict[str, Any] | None = None, medianprops: dict[str, Any] | None = None, meanprops: dict[str, Any] | None = None, capprops: dict[str, Any] | None = None, whiskerprops: dict[str, Any] | None = None, manage_ticks: bool = True, autorange: bool = False, zorder: float | None = None, capwidths: float | ArrayLike | None = None, label: Sequence[str] | None = None, *, data: Incomplete | None = None) -> dict[str, Any]: ...
def broken_barh(xranges: Sequence[tuple[float, float]], yrange: tuple[float, float], *, data: Incomplete | None = None, **kwargs) -> PolyCollection: ...
def clabel(CS: ContourSet, levels: ArrayLike | None = None, **kwargs) -> list[Text]: ...
def cohere(x: ArrayLike, y: ArrayLike, NFFT: int = 256, Fs: float = 2, Fc: int = 0, detrend: Literal['none', 'mean', 'linear'] | Callable[[ArrayLike], ArrayLike] = ..., window: Callable[[ArrayLike], ArrayLike] | ArrayLike = ..., noverlap: int = 0, pad_to: int | None = None, sides: Literal['default', 'onesided', 'twosided'] = 'default', scale_by_freq: bool | None = None, *, data: Incomplete | None = None, **kwargs) -> tuple[np.ndarray, np.ndarray]: ...
def contour(*args, data: Incomplete | None = None, **kwargs) -> QuadContourSet: ...
def contourf(*args, data: Incomplete | None = None, **kwargs) -> QuadContourSet: ...
def csd(x: ArrayLike, y: ArrayLike, NFFT: int | None = None, Fs: float | None = None, Fc: int | None = None, detrend: Literal['none', 'mean', 'linear'] | Callable[[ArrayLike], ArrayLike] | None = None, window: Callable[[ArrayLike], ArrayLike] | ArrayLike | None = None, noverlap: int | None = None, pad_to: int | None = None, sides: Literal['default', 'onesided', 'twosided'] | None = None, scale_by_freq: bool | None = None, return_line: bool | None = None, *, data: Incomplete | None = None, **kwargs) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, Line2D]: ...
def ecdf(x: ArrayLike, weights: ArrayLike | None = None, *, complementary: bool = False, orientation: Literal['vertical', 'horizonatal'] = 'vertical', compress: bool = False, data: Incomplete | None = None, **kwargs) -> Line2D: ...
def errorbar(x: float | ArrayLike, y: float | ArrayLike, yerr: float | ArrayLike | None = None, xerr: float | ArrayLike | None = None, fmt: str = '', ecolor: ColorType | None = None, elinewidth: float | None = None, capsize: float | None = None, barsabove: bool = False, lolims: bool | ArrayLike = False, uplims: bool | ArrayLike = False, xlolims: bool | ArrayLike = False, xuplims: bool | ArrayLike = False, errorevery: int | tuple[int, int] = 1, capthick: float | None = None, *, data: Incomplete | None = None, **kwargs) -> ErrorbarContainer: ...
def eventplot(positions: ArrayLike | Sequence[ArrayLike], orientation: Literal['horizontal', 'vertical'] = 'horizontal', lineoffsets: float | Sequence[float] = 1, linelengths: float | Sequence[float] = 1, linewidths: float | Sequence[float] | None = None, colors: ColorType | Sequence[ColorType] | None = None, alpha: float | Sequence[float] | None = None, linestyles: LineStyleType | Sequence[LineStyleType] = 'solid', *, data: Incomplete | None = None, **kwargs) -> EventCollection: ...
def fill(*args, data: Incomplete | None = None, **kwargs) -> list[Polygon]: ...
def fill_between(x: ArrayLike, y1: ArrayLike | float, y2: ArrayLike | float = 0, where: Sequence[bool] | None = None, interpolate: bool = False, step: Literal['pre', 'post', 'mid'] | None = None, *, data: Incomplete | None = None, **kwargs) -> FillBetweenPolyCollection: ...
def fill_betweenx(y: ArrayLike, x1: ArrayLike | float, x2: ArrayLike | float = 0, where: Sequence[bool] | None = None, step: Literal['pre', 'post', 'mid'] | None = None, interpolate: bool = False, *, data: Incomplete | None = None, **kwargs) -> FillBetweenPolyCollection: ...
def grid(visible: bool | None = None, which: Literal['major', 'minor', 'both'] = 'major', axis: Literal['both', 'x', 'y'] = 'both', **kwargs) -> None: ...
def hexbin(x: ArrayLike, y: ArrayLike, C: ArrayLike | None = None, gridsize: int | tuple[int, int] = 100, bins: Literal['log'] | int | Sequence[float] | None = None, xscale: Literal['linear', 'log'] = 'linear', yscale: Literal['linear', 'log'] = 'linear', extent: tuple[float, float, float, float] | None = None, cmap: str | Colormap | None = None, norm: str | Normalize | None = None, vmin: float | None = None, vmax: float | None = None, alpha: float | None = None, linewidths: float | None = None, edgecolors: Literal['face', 'none'] | ColorType = 'face', reduce_C_function: Callable[[np.ndarray | list[float]], float] = ..., mincnt: int | None = None, marginals: bool = False, colorizer: Colorizer | None = None, *, data: Incomplete | None = None, **kwargs) -> PolyCollection: ...
def hist(x: ArrayLike | Sequence[ArrayLike], bins: int | Sequence[float] | str | None = None, range: tuple[float, float] | None = None, density: bool = False, weights: ArrayLike | None = None, cumulative: bool | float = False, bottom: ArrayLike | float | None = None, histtype: Literal['bar', 'barstacked', 'step', 'stepfilled'] = 'bar', align: Literal['left', 'mid', 'right'] = 'mid', orientation: Literal['vertical', 'horizontal'] = 'vertical', rwidth: float | None = None, log: bool = False, color: ColorType | Sequence[ColorType] | None = None, label: str | Sequence[str] | None = None, stacked: bool = False, *, data: Incomplete | None = None, **kwargs) -> tuple[np.ndarray | list[np.ndarray], np.ndarray, BarContainer | Polygon | list[BarContainer | Polygon]]: ...
def stairs(values: ArrayLike, edges: ArrayLike | None = None, *, orientation: Literal['vertical', 'horizontal'] = 'vertical', baseline: float | ArrayLike | None = 0, fill: bool = False, data: Incomplete | None = None, **kwargs) -> StepPatch: ...
def hist2d(x: ArrayLike, y: ArrayLike, bins: None | int | tuple[int, int] | ArrayLike | tuple[ArrayLike, ArrayLike] = 10, range: ArrayLike | None = None, density: bool = False, weights: ArrayLike | None = None, cmin: float | None = None, cmax: float | None = None, *, data: Incomplete | None = None, **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray, QuadMesh]: ...
def hlines(y: float | ArrayLike, xmin: float | ArrayLike, xmax: float | ArrayLike, colors: ColorType | Sequence[ColorType] | None = None, linestyles: LineStyleType = 'solid', label: str = '', *, data: Incomplete | None = None, **kwargs) -> LineCollection: ...
def imshow(X: ArrayLike | PIL.Image.Image, cmap: str | Colormap | None = None, norm: str | Normalize | None = None, *, aspect: Literal['equal', 'auto'] | float | None = None, interpolation: str | None = None, alpha: float | ArrayLike | None = None, vmin: float | None = None, vmax: float | None = None, colorizer: Colorizer | None = None, origin: Literal['upper', 'lower'] | None = None, extent: tuple[float, float, float, float] | None = None, interpolation_stage: Literal['data', 'rgba', 'auto'] | None = None, filternorm: bool = True, filterrad: float = 4.0, resample: bool | None = None, url: str | None = None, data: Incomplete | None = None, **kwargs) -> AxesImage: ...
def legend(*args, **kwargs) -> Legend: ...
def locator_params(axis: Literal['both', 'x', 'y'] = 'both', tight: bool | None = None, **kwargs) -> None: ...
def loglog(*args, **kwargs) -> list[Line2D]: ...
def magnitude_spectrum(x: ArrayLike, Fs: float | None = None, Fc: int | None = None, window: Callable[[ArrayLike], ArrayLike] | ArrayLike | None = None, pad_to: int | None = None, sides: Literal['default', 'onesided', 'twosided'] | None = None, scale: Literal['default', 'linear', 'dB'] | None = None, *, data: Incomplete | None = None, **kwargs) -> tuple[np.ndarray, np.ndarray, Line2D]: ...
def margins(*margins: float, x: float | None = None, y: float | None = None, tight: bool | None = True) -> tuple[float, float] | None: ...
def minorticks_off() -> None: ...
def minorticks_on() -> None: ...
def pcolor(*args: ArrayLike, shading: Literal['flat', 'nearest', 'auto'] | None = None, alpha: float | None = None, norm: str | Normalize | None = None, cmap: str | Colormap | None = None, vmin: float | None = None, vmax: float | None = None, colorizer: Colorizer | None = None, data: Incomplete | None = None, **kwargs) -> Collection: ...
def pcolormesh(*args: ArrayLike, alpha: float | None = None, norm: str | Normalize | None = None, cmap: str | Colormap | None = None, vmin: float | None = None, vmax: float | None = None, colorizer: Colorizer | None = None, shading: Literal['flat', 'nearest', 'gouraud', 'auto'] | None = None, antialiased: bool = False, data: Incomplete | None = None, **kwargs) -> QuadMesh: ...
def phase_spectrum(x: ArrayLike, Fs: float | None = None, Fc: int | None = None, window: Callable[[ArrayLike], ArrayLike] | ArrayLike | None = None, pad_to: int | None = None, sides: Literal['default', 'onesided', 'twosided'] | None = None, *, data: Incomplete | None = None, **kwargs) -> tuple[np.ndarray, np.ndarray, Line2D]: ...
def pie(x: ArrayLike, explode: ArrayLike | None = None, labels: Sequence[str] | None = None, colors: ColorType | Sequence[ColorType] | None = None, autopct: str | Callable[[float], str] | None = None, pctdistance: float = 0.6, shadow: bool = False, labeldistance: float | None = 1.1, startangle: float = 0, radius: float = 1, counterclock: bool = True, wedgeprops: dict[str, Any] | None = None, textprops: dict[str, Any] | None = None, center: tuple[float, float] = (0, 0), frame: bool = False, rotatelabels: bool = False, *, normalize: bool = True, hatch: str | Sequence[str] | None = None, data: Incomplete | None = None) -> tuple[list[Wedge], list[Text]] | tuple[list[Wedge], list[Text], list[Text]]: ...
def plot(*args: float | ArrayLike | str, scalex: bool = True, scaley: bool = True, data: Incomplete | None = None, **kwargs) -> list[Line2D]: ...
def plot_date(x: ArrayLike, y: ArrayLike, fmt: str = 'o', tz: str | datetime.tzinfo | None = None, xdate: bool = True, ydate: bool = False, *, data: Incomplete | None = None, **kwargs) -> list[Line2D]: ...
def psd(x: ArrayLike, NFFT: int | None = None, Fs: float | None = None, Fc: int | None = None, detrend: Literal['none', 'mean', 'linear'] | Callable[[ArrayLike], ArrayLike] | None = None, window: Callable[[ArrayLike], ArrayLike] | ArrayLike | None = None, noverlap: int | None = None, pad_to: int | None = None, sides: Literal['default', 'onesided', 'twosided'] | None = None, scale_by_freq: bool | None = None, return_line: bool | None = None, *, data: Incomplete | None = None, **kwargs) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, Line2D]: ...
def quiver(*args, data: Incomplete | None = None, **kwargs) -> Quiver: ...
def quiverkey(Q: Quiver, X: float, Y: float, U: float, label: str, **kwargs) -> QuiverKey: ...
def scatter(x: float | ArrayLike, y: float | ArrayLike, s: float | ArrayLike | None = None, c: ArrayLike | Sequence[ColorType] | ColorType | None = None, marker: MarkerType | None = None, cmap: str | Colormap | None = None, norm: str | Normalize | None = None, vmin: float | None = None, vmax: float | None = None, alpha: float | None = None, linewidths: float | Sequence[float] | None = None, *, edgecolors: Literal['face', 'none'] | ColorType | Sequence[ColorType] | None = None, colorizer: Colorizer | None = None, plotnonfinite: bool = False, data: Incomplete | None = None, **kwargs) -> PathCollection: ...
def semilogx(*args, **kwargs) -> list[Line2D]: ...
def semilogy(*args, **kwargs) -> list[Line2D]: ...
def specgram(x: ArrayLike, NFFT: int | None = None, Fs: float | None = None, Fc: int | None = None, detrend: Literal['none', 'mean', 'linear'] | Callable[[ArrayLike], ArrayLike] | None = None, window: Callable[[ArrayLike], ArrayLike] | ArrayLike | None = None, noverlap: int | None = None, cmap: str | Colormap | None = None, xextent: tuple[float, float] | None = None, pad_to: int | None = None, sides: Literal['default', 'onesided', 'twosided'] | None = None, scale_by_freq: bool | None = None, mode: Literal['default', 'psd', 'magnitude', 'angle', 'phase'] | None = None, scale: Literal['default', 'linear', 'dB'] | None = None, vmin: float | None = None, vmax: float | None = None, *, data: Incomplete | None = None, **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray, AxesImage]: ...
def spy(Z: ArrayLike, precision: float | Literal['present'] = 0, marker: str | None = None, markersize: float | None = None, aspect: Literal['equal', 'auto'] | float | None = 'equal', origin: Literal['upper', 'lower'] = 'upper', **kwargs) -> AxesImage: ...
def stackplot(x, *args, labels=(), colors: Incomplete | None = None, hatch: Incomplete | None = None, baseline: str = 'zero', data: Incomplete | None = None, **kwargs): ...
def stem(*args: ArrayLike | str, linefmt: str | None = None, markerfmt: str | None = None, basefmt: str | None = None, bottom: float = 0, label: str | None = None, orientation: Literal['vertical', 'horizontal'] = 'vertical', data: Incomplete | None = None) -> StemContainer: ...
def step(x: ArrayLike, y: ArrayLike, *args, where: Literal['pre', 'post', 'mid'] = 'pre', data: Incomplete | None = None, **kwargs) -> list[Line2D]: ...
def streamplot(x, y, u, v, density: int = 1, linewidth: Incomplete | None = None, color: Incomplete | None = None, cmap: Incomplete | None = None, norm: Incomplete | None = None, arrowsize: int = 1, arrowstyle: str = '-|>', minlength: float = 0.1, transform: Incomplete | None = None, zorder: Incomplete | None = None, start_points: Incomplete | None = None, maxlength: float = 4.0, integration_direction: str = 'both', broken_streamlines: bool = True, *, data: Incomplete | None = None): ...
def table(cellText: Incomplete | None = None, cellColours: Incomplete | None = None, cellLoc: str = 'right', colWidths: Incomplete | None = None, rowLabels: Incomplete | None = None, rowColours: Incomplete | None = None, rowLoc: str = 'left', colLabels: Incomplete | None = None, colColours: Incomplete | None = None, colLoc: str = 'center', loc: str = 'bottom', bbox: Incomplete | None = None, edges: str = 'closed', **kwargs): ...
def text(x: float, y: float, s: str, fontdict: dict[str, Any] | None = None, **kwargs) -> Text: ...
def tick_params(axis: Literal['both', 'x', 'y'] = 'both', **kwargs) -> None: ...
def ticklabel_format(*, axis: Literal['both', 'x', 'y'] = 'both', style: Literal['', 'sci', 'scientific', 'plain'] | None = None, scilimits: tuple[int, int] | None = None, useOffset: bool | float | None = None, useLocale: bool | None = None, useMathText: bool | None = None) -> None: ...
def tricontour(*args, **kwargs): ...
def tricontourf(*args, **kwargs): ...
def tripcolor(*args, alpha: float = 1.0, norm: Incomplete | None = None, cmap: Incomplete | None = None, vmin: Incomplete | None = None, vmax: Incomplete | None = None, shading: str = 'flat', facecolors: Incomplete | None = None, **kwargs): ...
def triplot(*args, **kwargs): ...
def violinplot(dataset: ArrayLike | Sequence[ArrayLike], positions: ArrayLike | None = None, vert: bool | None = None, orientation: Literal['vertical', 'horizontal'] = 'vertical', widths: float | ArrayLike = 0.5, showmeans: bool = False, showextrema: bool = True, showmedians: bool = False, quantiles: Sequence[float | Sequence[float]] | None = None, points: int = 100, bw_method: Literal['scott', 'silverman'] | float | Callable[[GaussianKDE], float] | None = None, side: Literal['both', 'low', 'high'] = 'both', *, data: Incomplete | None = None) -> dict[str, Collection]: ...
def vlines(x: float | ArrayLike, ymin: float | ArrayLike, ymax: float | ArrayLike, colors: ColorType | Sequence[ColorType] | None = None, linestyles: LineStyleType = 'solid', label: str = '', *, data: Incomplete | None = None, **kwargs) -> LineCollection: ...
def xcorr(x: ArrayLike, y: ArrayLike, normed: bool = True, detrend: Callable[[ArrayLike], ArrayLike] = ..., usevlines: bool = True, maxlags: int = 10, *, data: Incomplete | None = None, **kwargs) -> tuple[np.ndarray, np.ndarray, LineCollection | Line2D, Line2D | None]: ...
def sci(im: ColorizingArtist) -> None: ...
def title(label: str, fontdict: dict[str, Any] | None = None, loc: Literal['left', 'center', 'right'] | None = None, pad: float | None = None, *, y: float | None = None, **kwargs) -> Text: ...
def xlabel(xlabel: str, fontdict: dict[str, Any] | None = None, labelpad: float | None = None, *, loc: Literal['left', 'center', 'right'] | None = None, **kwargs) -> Text: ...
def ylabel(ylabel: str, fontdict: dict[str, Any] | None = None, labelpad: float | None = None, *, loc: Literal['bottom', 'center', 'top'] | None = None, **kwargs) -> Text: ...
def xscale(value: str | ScaleBase, **kwargs) -> None: ...
def yscale(value: str | ScaleBase, **kwargs) -> None: ...
def autumn() -> None:
    """
    Set the colormap to 'autumn'.

    This changes the default colormap as well as the colormap of the current
    image if there is one. See ``help(colormaps)`` for more information.
    """
def bone() -> None:
    """
    Set the colormap to 'bone'.

    This changes the default colormap as well as the colormap of the current
    image if there is one. See ``help(colormaps)`` for more information.
    """
def cool() -> None:
    """
    Set the colormap to 'cool'.

    This changes the default colormap as well as the colormap of the current
    image if there is one. See ``help(colormaps)`` for more information.
    """
def copper() -> None:
    """
    Set the colormap to 'copper'.

    This changes the default colormap as well as the colormap of the current
    image if there is one. See ``help(colormaps)`` for more information.
    """
def flag() -> None:
    """
    Set the colormap to 'flag'.

    This changes the default colormap as well as the colormap of the current
    image if there is one. See ``help(colormaps)`` for more information.
    """
def gray() -> None:
    """
    Set the colormap to 'gray'.

    This changes the default colormap as well as the colormap of the current
    image if there is one. See ``help(colormaps)`` for more information.
    """
def hot() -> None:
    """
    Set the colormap to 'hot'.

    This changes the default colormap as well as the colormap of the current
    image if there is one. See ``help(colormaps)`` for more information.
    """
def hsv() -> None:
    """
    Set the colormap to 'hsv'.

    This changes the default colormap as well as the colormap of the current
    image if there is one. See ``help(colormaps)`` for more information.
    """
def jet() -> None:
    """
    Set the colormap to 'jet'.

    This changes the default colormap as well as the colormap of the current
    image if there is one. See ``help(colormaps)`` for more information.
    """
def pink() -> None:
    """
    Set the colormap to 'pink'.

    This changes the default colormap as well as the colormap of the current
    image if there is one. See ``help(colormaps)`` for more information.
    """
def prism() -> None:
    """
    Set the colormap to 'prism'.

    This changes the default colormap as well as the colormap of the current
    image if there is one. See ``help(colormaps)`` for more information.
    """
def spring() -> None:
    """
    Set the colormap to 'spring'.

    This changes the default colormap as well as the colormap of the current
    image if there is one. See ``help(colormaps)`` for more information.
    """
def summer() -> None:
    """
    Set the colormap to 'summer'.

    This changes the default colormap as well as the colormap of the current
    image if there is one. See ``help(colormaps)`` for more information.
    """
def winter() -> None:
    """
    Set the colormap to 'winter'.

    This changes the default colormap as well as the colormap of the current
    image if there is one. See ``help(colormaps)`` for more information.
    """
def magma() -> None:
    """
    Set the colormap to 'magma'.

    This changes the default colormap as well as the colormap of the current
    image if there is one. See ``help(colormaps)`` for more information.
    """
def inferno() -> None:
    """
    Set the colormap to 'inferno'.

    This changes the default colormap as well as the colormap of the current
    image if there is one. See ``help(colormaps)`` for more information.
    """
def plasma() -> None:
    """
    Set the colormap to 'plasma'.

    This changes the default colormap as well as the colormap of the current
    image if there is one. See ``help(colormaps)`` for more information.
    """
def viridis() -> None:
    """
    Set the colormap to 'viridis'.

    This changes the default colormap as well as the colormap of the current
    image if there is one. See ``help(colormaps)`` for more information.
    """
def nipy_spectral() -> None:
    """
    Set the colormap to 'nipy_spectral'.

    This changes the default colormap as well as the colormap of the current
    image if there is one. See ``help(colormaps)`` for more information.
    """
