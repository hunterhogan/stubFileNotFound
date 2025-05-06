from . import _api as _api, _docstring as _docstring
from .artist import Artist as Artist, allow_rasterization as allow_rasterization
from .patches import Rectangle as Rectangle
from _typeshed import Incomplete

class Cell(Rectangle):
    """
    A cell is a `.Rectangle` with some associated `.Text`.

    As a user, you'll most likely not creates cells yourself. Instead, you
    should use either the `~matplotlib.table.table` factory function or
    `.Table.add_cell`.
    """
    PAD: float
    _edges: str
    _edge_aliases: Incomplete
    _loc: Incomplete
    _text: Incomplete
    def __init__(self, xy, width, height, *, edgecolor: str = 'k', facecolor: str = 'w', fill: bool = True, text: str = '', loc: str = 'right', fontproperties: Incomplete | None = None, visible_edges: str = 'closed') -> None:
        """
        Parameters
        ----------
        xy : 2-tuple
            The position of the bottom left corner of the cell.
        width : float
            The cell width.
        height : float
            The cell height.
        edgecolor : :mpltype:`color`, default: 'k'
            The color of the cell border.
        facecolor : :mpltype:`color`, default: 'w'
            The cell facecolor.
        fill : bool, default: True
            Whether the cell background is filled.
        text : str, optional
            The cell text.
        loc : {'right', 'center', 'left'}
            The alignment of the text within the cell.
        fontproperties : dict, optional
            A dict defining the font properties of the text. Supported keys and
            values are the keyword arguments accepted by `.FontProperties`.
        visible_edges : {'closed', 'open', 'horizontal', 'vertical'} or substring of 'BRTL'
            The cell edges to be drawn with a line: a substring of 'BRTL'
            (bottom, right, top, left), or one of 'open' (no edges drawn),
            'closed' (all edges drawn), 'horizontal' (bottom and top),
            'vertical' (right and left).
        """
    stale: bool
    def set_transform(self, t) -> None: ...
    def set_figure(self, fig) -> None: ...
    def get_text(self):
        """Return the cell `.Text` instance."""
    def set_fontsize(self, size) -> None:
        """Set the text fontsize."""
    def get_fontsize(self):
        """Return the cell fontsize."""
    def auto_set_font_size(self, renderer):
        """Shrink font size until the text fits into the cell width."""
    def draw(self, renderer) -> None: ...
    def _set_text_position(self, renderer) -> None:
        """Set text up so it is drawn in the right place."""
    def get_text_bounds(self, renderer):
        """
        Return the text bounds as *(x, y, width, height)* in table coordinates.
        """
    def get_required_width(self, renderer):
        """Return the minimal required width for the cell."""
    def set_text_props(self, **kwargs) -> None:
        """
        Update the text properties.

        Valid keyword arguments are:

        %(Text:kwdoc)s
        """
    @property
    def visible_edges(self):
        """
        The cell edges to be drawn with a line.

        Reading this property returns a substring of 'BRTL' (bottom, right,
        top, left').

        When setting this property, you can use a substring of 'BRTL' or one
        of {'open', 'closed', 'horizontal', 'vertical'}.
        """
    _visible_edges: Incomplete
    @visible_edges.setter
    def visible_edges(self, value) -> None: ...
    def get_path(self):
        """Return a `.Path` for the `.visible_edges`."""
CustomCell = Cell

class Table(Artist):
    """
    A table of cells.

    The table consists of a grid of cells, which are indexed by (row, column).

    For a simple table, you'll have a full grid of cells with indices from
    (0, 0) to (num_rows-1, num_cols-1), in which the cell (0, 0) is positioned
    at the top left. However, you can also add cells with negative indices.
    You don't have to add a cell to every grid position, so you can create
    tables that have holes.

    *Note*: You'll usually not create an empty table from scratch. Instead use
    `~matplotlib.table.table` to create a table from data.
    """
    codes: Incomplete
    FONTSIZE: int
    AXESPAD: float
    _axes: Incomplete
    _loc: Incomplete
    _bbox: Incomplete
    _cells: Incomplete
    _edges: Incomplete
    _autoColumns: Incomplete
    _autoFontsize: bool
    def __init__(self, ax, loc: Incomplete | None = None, bbox: Incomplete | None = None, **kwargs) -> None:
        """
        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            The `~.axes.Axes` to plot the table into.
        loc : str, optional
            The position of the cell with respect to *ax*. This must be one of
            the `~.Table.codes`.
        bbox : `.Bbox` or [xmin, ymin, width, height], optional
            A bounding box to draw the table into. If this is not *None*, this
            overrides *loc*.

        Other Parameters
        ----------------
        **kwargs
            `.Artist` properties.
        """
    def add_cell(self, row, col, *args, **kwargs):
        """
        Create a cell and add it to the table.

        Parameters
        ----------
        row : int
            Row index.
        col : int
            Column index.
        *args, **kwargs
            All other parameters are passed on to `Cell`.

        Returns
        -------
        `.Cell`
            The created cell.

        """
    stale: bool
    def __setitem__(self, position, cell) -> None:
        """
        Set a custom cell in a given position.
        """
    def __getitem__(self, position):
        """Retrieve a custom cell from a given position."""
    @property
    def edges(self):
        """
        The default value of `~.Cell.visible_edges` for newly added
        cells using `.add_cell`.

        Notes
        -----
        This setting does currently only affect newly created cells using
        `.add_cell`.

        To change existing cells, you have to set their edges explicitly::

            for c in tab.get_celld().values():
                c.visible_edges = 'horizontal'

        """
    @edges.setter
    def edges(self, value) -> None: ...
    def _approx_text_height(self): ...
    def draw(self, renderer) -> None: ...
    def _get_grid_bbox(self, renderer):
        """
        Get a bbox, in axes coordinates for the cells.

        Only include those in the range (0, 0) to (maxRow, maxCol).
        """
    def contains(self, mouseevent): ...
    def get_children(self):
        """Return the Artists contained by the table."""
    def get_window_extent(self, renderer: Incomplete | None = None): ...
    def _do_cell_alignment(self) -> None:
        """
        Calculate row heights and column widths; position cells accordingly.
        """
    def auto_set_column_width(self, col) -> None:
        """
        Automatically set the widths of given columns to optimal sizes.

        Parameters
        ----------
        col : int or sequence of ints
            The indices of the columns to auto-scale.
        """
    def _auto_set_column_width(self, col, renderer) -> None:
        """Automatically set width for column."""
    def auto_set_font_size(self, value: bool = True) -> None:
        """Automatically set font size."""
    def _auto_set_font_size(self, renderer) -> None: ...
    def scale(self, xscale, yscale) -> None:
        """Scale column widths by *xscale* and row heights by *yscale*."""
    def set_fontsize(self, size) -> None:
        """
        Set the font size, in points, of the cell text.

        Parameters
        ----------
        size : float

        Notes
        -----
        As long as auto font size has not been disabled, the value will be
        clipped such that the text fits horizontally into the cell.

        You can disable this behavior using `.auto_set_font_size`.

        >>> the_table.auto_set_font_size(False)
        >>> the_table.set_fontsize(20)

        However, there is no automatic scaling of the row height so that the
        text may exceed the cell boundary.
        """
    def _offset(self, ox, oy) -> None:
        """Move all the artists by ox, oy (axes coords)."""
    def _update_positions(self, renderer) -> None: ...
    def get_celld(self):
        """
        Return a dict of cells in the table mapping *(row, column)* to
        `.Cell`\\s.

        Notes
        -----
        You can also directly index into the Table object to access individual
        cells::

            cell = table[row, col]

        """

def table(ax, cellText: Incomplete | None = None, cellColours: Incomplete | None = None, cellLoc: str = 'right', colWidths: Incomplete | None = None, rowLabels: Incomplete | None = None, rowColours: Incomplete | None = None, rowLoc: str = 'left', colLabels: Incomplete | None = None, colColours: Incomplete | None = None, colLoc: str = 'center', loc: str = 'bottom', bbox: Incomplete | None = None, edges: str = 'closed', **kwargs):
    """
    Add a table to an `~.axes.Axes`.

    At least one of *cellText* or *cellColours* must be specified. These
    parameters must be 2D lists, in which the outer lists define the rows and
    the inner list define the column values per row. Each row must have the
    same number of elements.

    The table can optionally have row and column headers, which are configured
    using *rowLabels*, *rowColours*, *rowLoc* and *colLabels*, *colColours*,
    *colLoc* respectively.

    For finer grained control over tables, use the `.Table` class and add it to
    the Axes with `.Axes.add_table`.

    Parameters
    ----------
    cellText : 2D list of str or pandas.DataFrame, optional
        The texts to place into the table cells.

        *Note*: Line breaks in the strings are currently not accounted for and
        will result in the text exceeding the cell boundaries.

    cellColours : 2D list of :mpltype:`color`, optional
        The background colors of the cells.

    cellLoc : {'right', 'center', 'left'}
        The alignment of the text within the cells.

    colWidths : list of float, optional
        The column widths in units of the axes. If not given, all columns will
        have a width of *1 / ncols*.

    rowLabels : list of str, optional
        The text of the row header cells.

    rowColours : list of :mpltype:`color`, optional
        The colors of the row header cells.

    rowLoc : {'left', 'center', 'right'}
        The text alignment of the row header cells.

    colLabels : list of str, optional
        The text of the column header cells.

    colColours : list of :mpltype:`color`, optional
        The colors of the column header cells.

    colLoc : {'center', 'left', 'right'}
        The text alignment of the column header cells.

    loc : str, default: 'bottom'
        The position of the cell with respect to *ax*. This must be one of
        the `~.Table.codes`.

    bbox : `.Bbox` or [xmin, ymin, width, height], optional
        A bounding box to draw the table into. If this is not *None*, this
        overrides *loc*.

    edges : {'closed', 'open', 'horizontal', 'vertical'} or substring of 'BRTL'
        The cell edges to be drawn with a line. See also
        `~.Cell.visible_edges`.

    Returns
    -------
    `~matplotlib.table.Table`
        The created table.

    Other Parameters
    ----------------
    **kwargs
        `.Table` properties.

    %(Table:kwdoc)s
    """
