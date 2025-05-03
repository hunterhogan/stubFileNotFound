import numpy as np
from _typeshed import Incomplete
from collections.abc import Iterable, Sequence
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.table import Table
from pandas import DataFrame as DataFrame, Series as Series
from pandas.core.dtypes.common import is_list_like as is_list_like
from pandas.core.dtypes.generic import ABCDataFrame as ABCDataFrame, ABCIndex as ABCIndex, ABCSeries as ABCSeries
from pandas.util._exceptions import find_stack_level as find_stack_level

def do_adjust_figure(fig: Figure) -> bool:
    """Whether fig has constrained_layout enabled."""
def maybe_adjust_figure(fig: Figure, *args, **kwargs) -> None:
    """Call fig.subplots_adjust unless fig has constrained_layout enabled."""
def format_date_labels(ax: Axes, rot) -> None: ...
def table(ax, data: DataFrame | Series, rowLabels: Incomplete | None = None, colLabels: Incomplete | None = None, **kwargs) -> Table: ...
def _get_layout(nplots: int, layout: tuple[int, int] | None = None, layout_type: str = 'box') -> tuple[int, int]: ...
def create_subplots(naxes: int, sharex: bool = False, sharey: bool = False, squeeze: bool = True, subplot_kw: Incomplete | None = None, ax: Incomplete | None = None, layout: Incomplete | None = None, layout_type: str = 'box', **fig_kw):
    """
    Create a figure with a set of subplots already made.

    This utility wrapper makes it convenient to create common layouts of
    subplots, including the enclosing figure object, in a single call.

    Parameters
    ----------
    naxes : int
      Number of required axes. Exceeded axes are set invisible. Default is
      nrows * ncols.

    sharex : bool
      If True, the X axis will be shared amongst all subplots.

    sharey : bool
      If True, the Y axis will be shared amongst all subplots.

    squeeze : bool

      If True, extra dimensions are squeezed out from the returned axis object:
        - if only one subplot is constructed (nrows=ncols=1), the resulting
        single Axis object is returned as a scalar.
        - for Nx1 or 1xN subplots, the returned object is a 1-d numpy object
        array of Axis objects are returned as numpy 1-d arrays.
        - for NxM subplots with N>1 and M>1 are returned as a 2d array.

      If False, no squeezing is done: the returned axis object is always
      a 2-d array containing Axis instances, even if it ends up being 1x1.

    subplot_kw : dict
      Dict with keywords passed to the add_subplot() call used to create each
      subplots.

    ax : Matplotlib axis object, optional

    layout : tuple
      Number of rows and columns of the subplot grid.
      If not specified, calculated from naxes and layout_type

    layout_type : {'box', 'horizontal', 'vertical'}, default 'box'
      Specify how to layout the subplot grid.

    fig_kw : Other keyword arguments to be passed to the figure() call.
        Note that all keywords not recognized above will be
        automatically included here.

    Returns
    -------
    fig, ax : tuple
      - fig is the Matplotlib Figure object
      - ax can be either a single axis object or an array of axis objects if
      more than one subplot was created.  The dimensions of the resulting array
      can be controlled with the squeeze keyword, see above.

    Examples
    --------
    x = np.linspace(0, 2*np.pi, 400)
    y = np.sin(x**2)

    # Just a figure and one subplot
    f, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title('Simple plot')

    # Two subplots, unpack the output array immediately
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.plot(x, y)
    ax1.set_title('Sharing Y axis')
    ax2.scatter(x, y)

    # Four polar axes
    plt.subplots(2, 2, subplot_kw=dict(polar=True))
    """
def _remove_labels_from_axis(axis: Axis) -> None: ...
def _has_externally_shared_axis(ax1: Axes, compare_axis: str) -> bool:
    '''
    Return whether an axis is externally shared.

    Parameters
    ----------
    ax1 : matplotlib.axes.Axes
        Axis to query.
    compare_axis : str
        `"x"` or `"y"` according to whether the X-axis or Y-axis is being
        compared.

    Returns
    -------
    bool
        `True` if the axis is externally shared. Otherwise `False`.

    Notes
    -----
    If two axes with different positions are sharing an axis, they can be
    referred to as *externally* sharing the common axis.

    If two axes sharing an axis also have the same position, they can be
    referred to as *internally* sharing the common axis (a.k.a twinning).

    _handle_shared_axes() is only interested in axes externally sharing an
    axis, regardless of whether either of the axes is also internally sharing
    with a third axis.
    '''
def handle_shared_axes(axarr: Iterable[Axes], nplots: int, naxes: int, nrows: int, ncols: int, sharex: bool, sharey: bool) -> None: ...
def flatten_axes(axes: Axes | Sequence[Axes]) -> np.ndarray: ...
def set_ticks_props(axes: Axes | Sequence[Axes], xlabelsize: int | None = None, xrot: Incomplete | None = None, ylabelsize: int | None = None, yrot: Incomplete | None = None): ...
def get_all_lines(ax: Axes) -> list[Line2D]: ...
def get_xlim(lines: Iterable[Line2D]) -> tuple[float, float]: ...
