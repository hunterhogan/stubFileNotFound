from _typeshed import Incomplete
from collections.abc import Sequence
from pandas import DataFrame as DataFrame, Index as Index, IndexSlice as IndexSlice, MultiIndex as MultiIndex, Series as Series, isna as isna
from pandas._config import get_option as get_option
from pandas._libs import lib as lib
from pandas._typing import Axis as Axis, Level as Level
from pandas.api.types import is_list_like as is_list_like
from pandas.compat._optional import import_optional_dependency as import_optional_dependency
from pandas.core.dtypes.common import is_complex as is_complex, is_float as is_float, is_integer as is_integer
from pandas.core.dtypes.generic import ABCSeries as ABCSeries
from typing import Any, Callable, DefaultDict, TypedDict

jinja2: Incomplete
BaseFormatter = str | Callable
ExtFormatter = BaseFormatter | dict[Any, BaseFormatter | None]
CSSPair = tuple[str, str | float]
CSSList = list[CSSPair]
CSSProperties = str | CSSList

class CSSDict(TypedDict):
    selector: str
    props: CSSProperties
CSSStyles = list[CSSDict]
Subset = slice | Sequence | Index

class StylerRenderer:
    """
    Base class to process rendering a Styler with a specified jinja2 template.
    """
    loader: Incomplete
    env: Incomplete
    template_html: Incomplete
    template_html_table: Incomplete
    template_html_style: Incomplete
    template_latex: Incomplete
    template_string: Incomplete
    data: DataFrame
    index: Index
    columns: Index
    uuid: Incomplete
    uuid_len: Incomplete
    table_styles: Incomplete
    table_attributes: Incomplete
    caption: Incomplete
    cell_ids: Incomplete
    css: Incomplete
    concatenated: list[StylerRenderer]
    hide_index_names: bool
    hide_column_names: bool
    hide_index_: list
    hide_columns_: list
    hidden_rows: Sequence[int]
    hidden_columns: Sequence[int]
    ctx: DefaultDict[tuple[int, int], CSSList]
    ctx_index: DefaultDict[tuple[int, int], CSSList]
    ctx_columns: DefaultDict[tuple[int, int], CSSList]
    cell_context: DefaultDict[tuple[int, int], str]
    _todo: list[tuple[Callable, tuple, dict]]
    tooltips: Tooltips | None
    _display_funcs: DefaultDict[tuple[int, int], Callable[[Any], str]]
    _display_funcs_index: DefaultDict[tuple[int, int], Callable[[Any], str]]
    _display_funcs_columns: DefaultDict[tuple[int, int], Callable[[Any], str]]
    def __init__(self, data: DataFrame | Series, uuid: str | None = None, uuid_len: int = 5, table_styles: CSSStyles | None = None, table_attributes: str | None = None, caption: str | tuple | list | None = None, cell_ids: bool = True, precision: int | None = None) -> None: ...
    def _render(self, sparse_index: bool, sparse_columns: bool, max_rows: int | None = None, max_cols: int | None = None, blank: str = ''):
        """
        Computes and applies styles and then generates the general render dicts.

        Also extends the `ctx` and `ctx_index` attributes with those of concatenated
        stylers for use within `_translate_latex`
        """
    def _render_html(self, sparse_index: bool, sparse_columns: bool, max_rows: int | None = None, max_cols: int | None = None, **kwargs) -> str:
        """
        Renders the ``Styler`` including all applied styles to HTML.
        Generates a dict with necessary kwargs passed to jinja2 template.
        """
    def _render_latex(self, sparse_index: bool, sparse_columns: bool, clines: str | None, **kwargs) -> str:
        """
        Render a Styler in latex format
        """
    def _render_string(self, sparse_index: bool, sparse_columns: bool, max_rows: int | None = None, max_cols: int | None = None, **kwargs) -> str:
        """
        Render a Styler in string format
        """
    def _compute(self):
        """
        Execute the style functions built up in `self._todo`.

        Relies on the conventions that all style functions go through
        .apply or .map. The append styles to apply as tuples of

        (application method, *args, **kwargs)
        """
    cellstyle_map_columns: DefaultDict[tuple[CSSPair, ...], list[str]]
    cellstyle_map: DefaultDict[tuple[CSSPair, ...], list[str]]
    cellstyle_map_index: DefaultDict[tuple[CSSPair, ...], list[str]]
    def _translate(self, sparse_index: bool, sparse_cols: bool, max_rows: int | None = None, max_cols: int | None = None, blank: str = '&nbsp;', dxs: list[dict] | None = None):
        """
        Process Styler data and settings into a dict for template rendering.

        Convert data and settings from ``Styler`` attributes such as ``self.data``,
        ``self.tooltips`` including applying any methods in ``self._todo``.

        Parameters
        ----------
        sparse_index : bool
            Whether to sparsify the index or print all hierarchical index elements.
            Upstream defaults are typically to `pandas.options.styler.sparse.index`.
        sparse_cols : bool
            Whether to sparsify the columns or print all hierarchical column elements.
            Upstream defaults are typically to `pandas.options.styler.sparse.columns`.
        max_rows, max_cols : int, optional
            Specific max rows and cols. max_elements always take precedence in render.
        blank : str
            Entry to top-left blank cells.
        dxs : list[dict]
            The render dicts of the concatenated Stylers.

        Returns
        -------
        d : dict
            The following structure: {uuid, table_styles, caption, head, body,
            cellstyle, table_attributes}
        """
    def _translate_header(self, sparsify_cols: bool, max_cols: int):
        """
        Build each <tr> within table <head> as a list

        Using the structure:
             +----------------------------+---------------+---------------------------+
             |  index_blanks ...          | column_name_0 |  column_headers (level_0) |
          1) |       ..                   |       ..      |             ..            |
             |  index_blanks ...          | column_name_n |  column_headers (level_n) |
             +----------------------------+---------------+---------------------------+
          2) |  index_names (level_0 to level_n) ...      | column_blanks ...         |
             +----------------------------+---------------+---------------------------+

        Parameters
        ----------
        sparsify_cols : bool
            Whether column_headers section will add colspan attributes (>1) to elements.
        max_cols : int
            Maximum number of columns to render. If exceeded will contain `...` filler.

        Returns
        -------
        head : list
            The associated HTML elements needed for template rendering.
        """
    def _generate_col_header_row(self, iter: Sequence, max_cols: int, col_lengths: dict):
        """
        Generate the row containing column headers:

         +----------------------------+---------------+---------------------------+
         |  index_blanks ...          | column_name_i |  column_headers (level_i) |
         +----------------------------+---------------+---------------------------+

        Parameters
        ----------
        iter : tuple
            Looping variables from outer scope
        max_cols : int
            Permissible number of columns
        col_lengths :
            c

        Returns
        -------
        list of elements
        """
    def _generate_index_names_row(self, iter: Sequence, max_cols: int, col_lengths: dict):
        """
        Generate the row containing index names

         +----------------------------+---------------+---------------------------+
         |  index_names (level_0 to level_n) ...      | column_blanks ...         |
         +----------------------------+---------------+---------------------------+

        Parameters
        ----------
        iter : tuple
            Looping variables from outer scope
        max_cols : int
            Permissible number of columns

        Returns
        -------
        list of elements
        """
    def _translate_body(self, idx_lengths: dict, max_rows: int, max_cols: int):
        """
        Build each <tr> within table <body> as a list

        Use the following structure:
          +--------------------------------------------+---------------------------+
          |  index_header_0    ...    index_header_n   |  data_by_column   ...     |
          +--------------------------------------------+---------------------------+

        Also add elements to the cellstyle_map for more efficient grouped elements in
        <style></style> block

        Parameters
        ----------
        sparsify_index : bool
            Whether index_headers section will add rowspan attributes (>1) to elements.

        Returns
        -------
        body : list
            The associated HTML elements needed for template rendering.
        """
    def _check_trim(self, count: int, max: int, obj: list, element: str, css: str | None = None, value: str = '...') -> bool:
        """
        Indicates whether to break render loops and append a trimming indicator

        Parameters
        ----------
        count : int
            The loop count of previous visible items.
        max : int
            The allowable rendered items in the loop.
        obj : list
            The current render collection of the rendered items.
        element : str
            The type of element to append in the case a trimming indicator is needed.
        css : str, optional
            The css to add to the trimming indicator element.
        value : str, optional
            The value of the elements display if necessary.

        Returns
        -------
        result : bool
            Whether a trimming element was required and appended.
        """
    def _generate_trimmed_row(self, max_cols: int) -> list:
        '''
        When a render has too many rows we generate a trimming row containing "..."

        Parameters
        ----------
        max_cols : int
            Number of permissible columns

        Returns
        -------
        list of elements
        '''
    def _generate_body_row(self, iter: tuple, max_cols: int, idx_lengths: dict):
        """
        Generate a regular row for the body section of appropriate format.

          +--------------------------------------------+---------------------------+
          |  index_header_0    ...    index_header_n   |  data_by_column   ...     |
          +--------------------------------------------+---------------------------+

        Parameters
        ----------
        iter : tuple
            Iterable from outer scope: row number, row data tuple, row index labels.
        max_cols : int
            Number of permissible columns.
        idx_lengths : dict
            A map of the sparsification structure of the index

        Returns
        -------
            list of elements
        """
    def _translate_latex(self, d: dict, clines: str | None) -> None:
        """
        Post-process the default render dict for the LaTeX template format.

        Processing items included are:
          - Remove hidden columns from the non-headers part of the body.
          - Place cellstyles directly in td cells rather than use cellstyle_map.
          - Remove hidden indexes or reinsert missing th elements if part of multiindex
            or multirow sparsification (so that \\multirow and \\multicol work correctly).
        """
    def format(self, formatter: ExtFormatter | None = None, subset: Subset | None = None, na_rep: str | None = None, precision: int | None = None, decimal: str = '.', thousands: str | None = None, escape: str | None = None, hyperlinks: str | None = None) -> StylerRenderer:
        '''
        Format the text display value of cells.

        Parameters
        ----------
        formatter : str, callable, dict or None
            Object to define how values are displayed. See notes.
        subset : label, array-like, IndexSlice, optional
            A valid 2d input to `DataFrame.loc[<subset>]`, or, in the case of a 1d input
            or single key, to `DataFrame.loc[:, <subset>]` where the columns are
            prioritised, to limit ``data`` to *before* applying the function.
        na_rep : str, optional
            Representation for missing values.
            If ``na_rep`` is None, no special formatting is applied.
        precision : int, optional
            Floating point precision to use for display purposes, if not determined by
            the specified ``formatter``.

            .. versionadded:: 1.3.0

        decimal : str, default "."
            Character used as decimal separator for floats, complex and integers.

            .. versionadded:: 1.3.0

        thousands : str, optional, default None
            Character used as thousands separator for floats, complex and integers.

            .. versionadded:: 1.3.0

        escape : str, optional
            Use \'html\' to replace the characters ``&``, ``<``, ``>``, ``\'``, and ``"``
            in cell display string with HTML-safe sequences.
            Use \'latex\' to replace the characters ``&``, ``%``, ``$``, ``#``, ``_``,
            ``{``, ``}``, ``~``, ``^``, and ``\\`` in the cell display string with
            LaTeX-safe sequences.
            Use \'latex-math\' to replace the characters the same way as in \'latex\' mode,
            except for math substrings, which either are surrounded
            by two characters ``$`` or start with the character ``\\(`` and
            end with ``\\)``. Escaping is done before ``formatter``.

            .. versionadded:: 1.3.0

        hyperlinks : {"html", "latex"}, optional
            Convert string patterns containing https://, http://, ftp:// or www. to
            HTML <a> tags as clickable URL hyperlinks if "html", or LaTeX \\href
            commands if "latex".

            .. versionadded:: 1.4.0

        Returns
        -------
        Styler

        See Also
        --------
        Styler.format_index: Format the text display value of index labels.

        Notes
        -----
        This method assigns a formatting function, ``formatter``, to each cell in the
        DataFrame. If ``formatter`` is ``None``, then the default formatter is used.
        If a callable then that function should take a data value as input and return
        a displayable representation, such as a string. If ``formatter`` is
        given as a string this is assumed to be a valid Python format specification
        and is wrapped to a callable as ``string.format(x)``. If a ``dict`` is given,
        keys should correspond to column names, and values should be string or
        callable, as above.

        The default formatter currently expresses floats and complex numbers with the
        pandas display precision unless using the ``precision`` argument here. The
        default formatter does not adjust the representation of missing values unless
        the ``na_rep`` argument is used.

        The ``subset`` argument defines which region to apply the formatting function
        to. If the ``formatter`` argument is given in dict form but does not include
        all columns within the subset then these columns will have the default formatter
        applied. Any columns in the formatter dict excluded from the subset will
        be ignored.

        When using a ``formatter`` string the dtypes must be compatible, otherwise a
        `ValueError` will be raised.

        When instantiating a Styler, default formatting can be applied be setting the
        ``pandas.options``:

          - ``styler.format.formatter``: default None.
          - ``styler.format.na_rep``: default None.
          - ``styler.format.precision``: default 6.
          - ``styler.format.decimal``: default ".".
          - ``styler.format.thousands``: default None.
          - ``styler.format.escape``: default None.

        .. warning::
           `Styler.format` is ignored when using the output format `Styler.to_excel`,
           since Excel and Python have inherrently different formatting structures.
           However, it is possible to use the `number-format` pseudo CSS attribute
           to force Excel permissible formatting. See examples.

        Examples
        --------
        Using ``na_rep`` and ``precision`` with the default ``formatter``

        >>> df = pd.DataFrame([[np.nan, 1.0, \'A\'], [2.0, np.nan, 3.0]])
        >>> df.style.format(na_rep=\'MISS\', precision=3)  # doctest: +SKIP
                0       1       2
        0    MISS   1.000       A
        1   2.000    MISS   3.000

        Using a ``formatter`` specification on consistent column dtypes

        >>> df.style.format(\'{:.2f}\', na_rep=\'MISS\', subset=[0,1])  # doctest: +SKIP
                0      1          2
        0    MISS   1.00          A
        1    2.00   MISS   3.000000

        Using the default ``formatter`` for unspecified columns

        >>> df.style.format({0: \'{:.2f}\', 1: \'£ {:.1f}\'}, na_rep=\'MISS\', precision=1)
        ...  # doctest: +SKIP
                 0      1     2
        0    MISS   £ 1.0     A
        1    2.00    MISS   3.0

        Multiple ``na_rep`` or ``precision`` specifications under the default
        ``formatter``.

        >>> (df.style.format(na_rep=\'MISS\', precision=1, subset=[0])
        ...     .format(na_rep=\'PASS\', precision=2, subset=[1, 2]))  # doctest: +SKIP
                0      1      2
        0    MISS   1.00      A
        1     2.0   PASS   3.00

        Using a callable ``formatter`` function.

        >>> func = lambda s: \'STRING\' if isinstance(s, str) else \'FLOAT\'
        >>> df.style.format({0: \'{:.1f}\', 2: func}, precision=4, na_rep=\'MISS\')
        ...  # doctest: +SKIP
                0        1        2
        0    MISS   1.0000   STRING
        1     2.0     MISS    FLOAT

        Using a ``formatter`` with HTML ``escape`` and ``na_rep``.

        >>> df = pd.DataFrame([[\'<div></div>\', \'"A&B"\', None]])
        >>> s = df.style.format(
        ...     \'<a href="a.com/{0}">{0}</a>\', escape="html", na_rep="NA"
        ...     )
        >>> s.to_html()  # doctest: +SKIP
        ...
        <td .. ><a href="a.com/&lt;div&gt;&lt;/div&gt;">&lt;div&gt;&lt;/div&gt;</a></td>
        <td .. ><a href="a.com/&#34;A&amp;B&#34;">&#34;A&amp;B&#34;</a></td>
        <td .. >NA</td>
        ...

        Using a ``formatter`` with ``escape`` in \'latex\' mode.

        >>> df = pd.DataFrame([["123"], ["~ ^"], ["$%#"]])
        >>> df.style.format("\\\\textbf{{{}}}", escape="latex").to_latex()
        ...  # doctest: +SKIP
        \\begin{tabular}{ll}
         & 0 \\\\\n        0 & \\textbf{123} \\\\\n        1 & \\textbf{\\textasciitilde \\space \\textasciicircum } \\\\\n        2 & \\textbf{\\$\\%\\#} \\\\\n        \\end{tabular}

        Applying ``escape`` in \'latex-math\' mode. In the example below
        we enter math mode using the character ``$``.

        >>> df = pd.DataFrame([[r"$\\sum_{i=1}^{10} a_i$ a~b $\\alpha \\\n        ...     = \\frac{\\beta}{\\zeta^2}$"], ["%#^ $ \\$x^2 $"]])
        >>> df.style.format(escape="latex-math").to_latex()
        ...  # doctest: +SKIP
        \\begin{tabular}{ll}
         & 0 \\\\\n        0 & $\\sum_{i=1}^{10} a_i$ a\\textasciitilde b $\\alpha = \\frac{\\beta}{\\zeta^2}$ \\\\\n        1 & \\%\\#\\textasciicircum \\space $ \\$x^2 $ \\\\\n        \\end{tabular}

        We can use the character ``\\(`` to enter math mode and the character ``\\)``
        to close math mode.

        >>> df = pd.DataFrame([[r"\\(\\sum_{i=1}^{10} a_i\\) a~b \\(\\alpha \\\n        ...     = \\frac{\\beta}{\\zeta^2}\\)"], ["%#^ \\( \\$x^2 \\)"]])
        >>> df.style.format(escape="latex-math").to_latex()
        ...  # doctest: +SKIP
        \\begin{tabular}{ll}
         & 0 \\\\\n        0 & \\(\\sum_{i=1}^{10} a_i\\) a\\textasciitilde b \\(\\alpha
        = \\frac{\\beta}{\\zeta^2}\\) \\\\\n        1 & \\%\\#\\textasciicircum \\space \\( \\$x^2 \\) \\\\\n        \\end{tabular}

        If we have in one DataFrame cell a combination of both shorthands
        for math formulas, the shorthand with the sign ``$`` will be applied.

        >>> df = pd.DataFrame([[r"\\( x^2 \\)  $x^2$"], \\\n        ...     [r"$\\frac{\\beta}{\\zeta}$ \\(\\frac{\\beta}{\\zeta}\\)"]])
        >>> df.style.format(escape="latex-math").to_latex()
        ...  # doctest: +SKIP
        \\begin{tabular}{ll}
         & 0 \\\\\n        0 & \\textbackslash ( x\\textasciicircum 2 \\textbackslash )  $x^2$ \\\\\n        1 & $\\frac{\\beta}{\\zeta}$ \\textbackslash (\\textbackslash
        frac\\{\\textbackslash beta\\}\\{\\textbackslash zeta\\}\\textbackslash ) \\\\\n        \\end{tabular}

        Pandas defines a `number-format` pseudo CSS attribute instead of the `.format`
        method to create `to_excel` permissible formatting. Note that semi-colons are
        CSS protected characters but used as separators in Excel\'s format string.
        Replace semi-colons with the section separator character (ASCII-245) when
        defining the formatting here.

        >>> df = pd.DataFrame({"A": [1, 0, -1]})
        >>> pseudo_css = "number-format: 0§[Red](0)§-§@;"
        >>> filename = "formatted_file.xlsx"
        >>> df.style.map(lambda v: pseudo_css).to_excel(filename) # doctest: +SKIP

        .. figure:: ../../_static/style/format_excel_css.png
        '''
    def format_index(self, formatter: ExtFormatter | None = None, axis: Axis = 0, level: Level | list[Level] | None = None, na_rep: str | None = None, precision: int | None = None, decimal: str = '.', thousands: str | None = None, escape: str | None = None, hyperlinks: str | None = None) -> StylerRenderer:
        '''
        Format the text display value of index labels or column headers.

        .. versionadded:: 1.4.0

        Parameters
        ----------
        formatter : str, callable, dict or None
            Object to define how values are displayed. See notes.
        axis : {0, "index", 1, "columns"}
            Whether to apply the formatter to the index or column headers.
        level : int, str, list
            The level(s) over which to apply the generic formatter.
        na_rep : str, optional
            Representation for missing values.
            If ``na_rep`` is None, no special formatting is applied.
        precision : int, optional
            Floating point precision to use for display purposes, if not determined by
            the specified ``formatter``.
        decimal : str, default "."
            Character used as decimal separator for floats, complex and integers.
        thousands : str, optional, default None
            Character used as thousands separator for floats, complex and integers.
        escape : str, optional
            Use \'html\' to replace the characters ``&``, ``<``, ``>``, ``\'``, and ``"``
            in cell display string with HTML-safe sequences.
            Use \'latex\' to replace the characters ``&``, ``%``, ``$``, ``#``, ``_``,
            ``{``, ``}``, ``~``, ``^``, and ``\\`` in the cell display string with
            LaTeX-safe sequences.
            Escaping is done before ``formatter``.
        hyperlinks : {"html", "latex"}, optional
            Convert string patterns containing https://, http://, ftp:// or www. to
            HTML <a> tags as clickable URL hyperlinks if "html", or LaTeX \\href
            commands if "latex".

        Returns
        -------
        Styler

        See Also
        --------
        Styler.format: Format the text display value of data cells.

        Notes
        -----
        This method assigns a formatting function, ``formatter``, to each level label
        in the DataFrame\'s index or column headers. If ``formatter`` is ``None``,
        then the default formatter is used.
        If a callable then that function should take a label value as input and return
        a displayable representation, such as a string. If ``formatter`` is
        given as a string this is assumed to be a valid Python format specification
        and is wrapped to a callable as ``string.format(x)``. If a ``dict`` is given,
        keys should correspond to MultiIndex level numbers or names, and values should
        be string or callable, as above.

        The default formatter currently expresses floats and complex numbers with the
        pandas display precision unless using the ``precision`` argument here. The
        default formatter does not adjust the representation of missing values unless
        the ``na_rep`` argument is used.

        The ``level`` argument defines which levels of a MultiIndex to apply the
        method to. If the ``formatter`` argument is given in dict form but does
        not include all levels within the level argument then these unspecified levels
        will have the default formatter applied. Any levels in the formatter dict
        specifically excluded from the level argument will be ignored.

        When using a ``formatter`` string the dtypes must be compatible, otherwise a
        `ValueError` will be raised.

        .. warning::
           `Styler.format_index` is ignored when using the output format
           `Styler.to_excel`, since Excel and Python have inherrently different
           formatting structures.
           However, it is possible to use the `number-format` pseudo CSS attribute
           to force Excel permissible formatting. See documentation for `Styler.format`.

        Examples
        --------
        Using ``na_rep`` and ``precision`` with the default ``formatter``

        >>> df = pd.DataFrame([[1, 2, 3]], columns=[2.0, np.nan, 4.0])
        >>> df.style.format_index(axis=1, na_rep=\'MISS\', precision=3)  # doctest: +SKIP
            2.000    MISS   4.000
        0       1       2       3

        Using a ``formatter`` specification on consistent dtypes in a level

        >>> df.style.format_index(\'{:.2f}\', axis=1, na_rep=\'MISS\')  # doctest: +SKIP
             2.00   MISS    4.00
        0       1      2       3

        Using the default ``formatter`` for unspecified levels

        >>> df = pd.DataFrame([[1, 2, 3]],
        ...     columns=pd.MultiIndex.from_arrays([["a", "a", "b"],[2, np.nan, 4]]))
        >>> df.style.format_index({0: lambda v: v.upper()}, axis=1, precision=1)
        ...  # doctest: +SKIP
                       A       B
              2.0    nan     4.0
        0       1      2       3

        Using a callable ``formatter`` function.

        >>> func = lambda s: \'STRING\' if isinstance(s, str) else \'FLOAT\'
        >>> df.style.format_index(func, axis=1, na_rep=\'MISS\')
        ...  # doctest: +SKIP
                  STRING  STRING
            FLOAT   MISS   FLOAT
        0       1      2       3

        Using a ``formatter`` with HTML ``escape`` and ``na_rep``.

        >>> df = pd.DataFrame([[1, 2, 3]], columns=[\'"A"\', \'A&B\', None])
        >>> s = df.style.format_index(\'$ {0}\', axis=1, escape="html", na_rep="NA")
        ...  # doctest: +SKIP
        <th .. >$ &#34;A&#34;</th>
        <th .. >$ A&amp;B</th>
        <th .. >NA</td>
        ...

        Using a ``formatter`` with LaTeX ``escape``.

        >>> df = pd.DataFrame([[1, 2, 3]], columns=["123", "~", "$%#"])
        >>> df.style.format_index("\\\\textbf{{{}}}", escape="latex", axis=1).to_latex()
        ...  # doctest: +SKIP
        \\begin{tabular}{lrrr}
        {} & {\\textbf{123}} & {\\textbf{\\textasciitilde }} & {\\textbf{\\$\\%\\#}} \\\\\n        0 & 1 & 2 & 3 \\\\\n        \\end{tabular}
        '''
    def relabel_index(self, labels: Sequence | Index, axis: Axis = 0, level: Level | list[Level] | None = None) -> StylerRenderer:
        '''
        Relabel the index, or column header, keys to display a set of specified values.

        .. versionadded:: 1.5.0

        Parameters
        ----------
        labels : list-like or Index
            New labels to display. Must have same length as the underlying values not
            hidden.
        axis : {"index", 0, "columns", 1}
            Apply to the index or columns.
        level : int, str, list, optional
            The level(s) over which to apply the new labels. If `None` will apply
            to all levels of an Index or MultiIndex which are not hidden.

        Returns
        -------
        Styler

        See Also
        --------
        Styler.format_index: Format the text display value of index or column headers.
        Styler.hide: Hide the index, column headers, or specified data from display.

        Notes
        -----
        As part of Styler, this method allows the display of an index to be
        completely user-specified without affecting the underlying DataFrame data,
        index, or column headers. This means that the flexibility of indexing is
        maintained whilst the final display is customisable.

        Since Styler is designed to be progressively constructed with method chaining,
        this method is adapted to react to the **currently specified hidden elements**.
        This is useful because it means one does not have to specify all the new
        labels if the majority of an index, or column headers, have already been hidden.
        The following produce equivalent display (note the length of ``labels`` in
        each case).

        .. code-block:: python

            # relabel first, then hide
            df = pd.DataFrame({"col": ["a", "b", "c"]})
            df.style.relabel_index(["A", "B", "C"]).hide([0,1])
            # hide first, then relabel
            df = pd.DataFrame({"col": ["a", "b", "c"]})
            df.style.hide([0,1]).relabel_index(["C"])

        This method should be used, rather than :meth:`Styler.format_index`, in one of
        the following cases (see examples):

          - A specified set of labels are required which are not a function of the
            underlying index keys.
          - The function of the underlying index keys requires a counter variable,
            such as those available upon enumeration.

        Examples
        --------
        Basic use

        >>> df = pd.DataFrame({"col": ["a", "b", "c"]})
        >>> df.style.relabel_index(["A", "B", "C"])  # doctest: +SKIP
             col
        A      a
        B      b
        C      c

        Chaining with pre-hidden elements

        >>> df.style.hide([0,1]).relabel_index(["C"])  # doctest: +SKIP
             col
        C      c

        Using a MultiIndex

        >>> midx = pd.MultiIndex.from_product([[0, 1], [0, 1], [0, 1]])
        >>> df = pd.DataFrame({"col": list(range(8))}, index=midx)
        >>> styler = df.style  # doctest: +SKIP
                  col
        0  0  0     0
              1     1
           1  0     2
              1     3
        1  0  0     4
              1     5
           1  0     6
              1     7
        >>> styler.hide((midx.get_level_values(0)==0)|(midx.get_level_values(1)==0))
        ...  # doctest: +SKIP
        >>> styler.hide(level=[0,1])  # doctest: +SKIP
        >>> styler.relabel_index(["binary6", "binary7"])  # doctest: +SKIP
                  col
        binary6     6
        binary7     7

        We can also achieve the above by indexing first and then re-labeling

        >>> styler = df.loc[[(1,1,0), (1,1,1)]].style
        >>> styler.hide(level=[0,1]).relabel_index(["binary6", "binary7"])
        ...  # doctest: +SKIP
                  col
        binary6     6
        binary7     7

        Defining a formatting function which uses an enumeration counter. Also note
        that the value of the index key is passed in the case of string labels so it
        can also be inserted into the label, using curly brackets (or double curly
        brackets if the string if pre-formatted),

        >>> df = pd.DataFrame({"samples": np.random.rand(10)})
        >>> styler = df.loc[np.random.randint(0,10,3)].style
        >>> styler.relabel_index([f"sample{i+1} ({{}})" for i in range(3)])
        ...  # doctest: +SKIP
                         samples
        sample1 (5)     0.315811
        sample2 (0)     0.495941
        sample3 (2)     0.067946
        '''

def _element(html_element: str, html_class: str | None, value: Any, is_visible: bool, **kwargs) -> dict:
    """
    Template to return container with information for a <td></td> or <th></th> element.
    """
def _get_trimming_maximums(rn, cn, max_elements, max_rows: Incomplete | None = None, max_cols: Incomplete | None = None, scaling_factor: float = 0.8) -> tuple[int, int]:
    """
    Recursively reduce the number of rows and columns to satisfy max elements.

    Parameters
    ----------
    rn, cn : int
        The number of input rows / columns
    max_elements : int
        The number of allowable elements
    max_rows, max_cols : int, optional
        Directly specify an initial maximum rows or columns before compression.
    scaling_factor : float
        Factor at which to reduce the number of rows / columns to fit.

    Returns
    -------
    rn, cn : tuple
        New rn and cn values that satisfy the max_elements constraint
    """
def _get_level_lengths(index: Index, sparsify: bool, max_index: int, hidden_elements: Sequence[int] | None = None):
    """
    Given an index, find the level length for each element.

    Parameters
    ----------
    index : Index
        Index or columns to determine lengths of each element
    sparsify : bool
        Whether to hide or show each distinct element in a MultiIndex
    max_index : int
        The maximum number of elements to analyse along the index due to trimming
    hidden_elements : sequence of int
        Index positions of elements hidden from display in the index affecting
        length

    Returns
    -------
    Dict :
        Result is a dictionary of (level, initial_position): span
    """
def _is_visible(idx_row, idx_col, lengths) -> bool:
    """
    Index -> {(idx_row, idx_col): bool}).
    """
def format_table_styles(styles: CSSStyles) -> CSSStyles:
    """
    looks for multiple CSS selectors and separates them:
    [{'selector': 'td, th', 'props': 'a:v;'}]
        ---> [{'selector': 'td', 'props': 'a:v;'},
              {'selector': 'th', 'props': 'a:v;'}]
    """
def _default_formatter(x: Any, precision: int, thousands: bool = False) -> Any:
    '''
    Format the display of a value

    Parameters
    ----------
    x : Any
        Input variable to be formatted
    precision : Int
        Floating point precision used if ``x`` is float or complex.
    thousands : bool, default False
        Whether to group digits with thousands separated with ",".

    Returns
    -------
    value : Any
        Matches input type, or string if input is float or complex or int with sep.
    '''
def _wrap_decimal_thousands(formatter: Callable, decimal: str, thousands: str | None) -> Callable:
    """
    Takes a string formatting function and wraps logic to deal with thousands and
    decimal parameters, in the case that they are non-standard and that the input
    is a (float, complex, int).
    """
def _str_escape(x, escape):
    """if escaping: only use on str, else return input"""
def _render_href(x, format):
    """uses regex to detect a common URL pattern and converts to href tag in format."""
def _maybe_wrap_formatter(formatter: BaseFormatter | None = None, na_rep: str | None = None, precision: int | None = None, decimal: str = '.', thousands: str | None = None, escape: str | None = None, hyperlinks: str | None = None) -> Callable:
    """
    Allows formatters to be expressed as str, callable or None, where None returns
    a default formatting function. wraps with na_rep, and precision where they are
    available.
    """
def non_reducing_slice(slice_: Subset):
    """
    Ensure that a slice doesn't reduce to a Series or Scalar.

    Any user-passed `subset` should have this called on it
    to make sure we're always working with DataFrames.
    """
def maybe_convert_css_to_tuples(style: CSSProperties) -> CSSList:
    """
    Convert css-string to sequence of tuples format if needed.
    'color:red; border:1px solid black;' -> [('color', 'red'),
                                             ('border','1px solid red')]
    """
def refactor_levels(level: Level | list[Level] | None, obj: Index) -> list[int]:
    """
    Returns a consistent levels arg for use in ``hide_index`` or ``hide_columns``.

    Parameters
    ----------
    level : int, str, list
        Original ``level`` arg supplied to above methods.
    obj:
        Either ``self.index`` or ``self.columns``

    Returns
    -------
    list : refactored arg with a list of levels to hide
    """

class Tooltips:
    '''
    An extension to ``Styler`` that allows for and manipulates tooltips on hover
    of ``<td>`` cells in the HTML result.

    Parameters
    ----------
    css_name: str, default "pd-t"
        Name of the CSS class that controls visualisation of tooltips.
    css_props: list-like, default; see Notes
        List of (attr, value) tuples defining properties of the CSS class.
    tooltips: DataFrame, default empty
        DataFrame of strings aligned with underlying Styler data for tooltip
        display.

    Notes
    -----
    The default properties for the tooltip CSS class are:

        - visibility: hidden
        - position: absolute
        - z-index: 1
        - background-color: black
        - color: white
        - transform: translate(-20px, -20px)

    Hidden visibility is a key prerequisite to the hover functionality, and should
    always be included in any manual properties specification.
    '''
    class_name: Incomplete
    class_properties: Incomplete
    tt_data: Incomplete
    table_styles: CSSStyles
    def __init__(self, css_props: CSSProperties = [('visibility', 'hidden'), ('position', 'absolute'), ('z-index', 1), ('background-color', 'black'), ('color', 'white'), ('transform', 'translate(-20px, -20px)')], css_name: str = 'pd-t', tooltips: DataFrame = ...) -> None: ...
    @property
    def _class_styles(self):
        """
        Combine the ``_Tooltips`` CSS class name and CSS properties to the format
        required to extend the underlying ``Styler`` `table_styles` to allow
        tooltips to render in HTML.

        Returns
        -------
        styles : List
        """
    def _pseudo_css(self, uuid: str, name: str, row: int, col: int, text: str):
        '''
        For every table data-cell that has a valid tooltip (not None, NaN or
        empty string) must create two pseudo CSS entries for the specific
        <td> element id which are added to overall table styles:
        an on hover visibility change and a content change
        dependent upon the user\'s chosen display string.

        For example:
            [{"selector": "T__row1_col1:hover .pd-t",
             "props": [("visibility", "visible")]},
            {"selector": "T__row1_col1 .pd-t::after",
             "props": [("content", "Some Valid Text String")]}]

        Parameters
        ----------
        uuid: str
            The uuid of the Styler instance
        name: str
            The css-name of the class used for styling tooltips
        row : int
            The row index of the specified tooltip string data
        col : int
            The col index of the specified tooltip string data
        text : str
            The textual content of the tooltip to be displayed in HTML.

        Returns
        -------
        pseudo_css : List
        '''
    def _translate(self, styler: StylerRenderer, d: dict):
        """
        Mutate the render dictionary to allow for tooltips:

        - Add ``<span>`` HTML element to each data cells ``display_value``. Ignores
          headers.
        - Add table level CSS styles to control pseudo classes.

        Parameters
        ----------
        styler_data : DataFrame
            Underlying ``Styler`` DataFrame used for reindexing.
        uuid : str
            The underlying ``Styler`` uuid for CSS id.
        d : dict
            The dictionary prior to final render

        Returns
        -------
        render_dict : Dict
        """

def _parse_latex_table_wrapping(table_styles: CSSStyles, caption: str | None) -> bool:
    """
    Indicate whether LaTeX {tabular} should be wrapped with a {table} environment.

    Parses the `table_styles` and detects any selectors which must be included outside
    of {tabular}, i.e. indicating that wrapping must occur, and therefore return True,
    or if a caption exists and requires similar.
    """
def _parse_latex_table_styles(table_styles: CSSStyles, selector: str) -> str | None:
    '''
    Return the first \'props\' \'value\' from ``tables_styles`` identified by ``selector``.

    Examples
    --------
    >>> table_styles = [{\'selector\': \'foo\', \'props\': [(\'attr\',\'value\')]},
    ...                 {\'selector\': \'bar\', \'props\': [(\'attr\', \'overwritten\')]},
    ...                 {\'selector\': \'bar\', \'props\': [(\'a1\', \'baz\'), (\'a2\', \'ignore\')]}]
    >>> _parse_latex_table_styles(table_styles, selector=\'bar\')
    \'baz\'

    Notes
    -----
    The replacement of "§" with ":" is to avoid the CSS problem where ":" has structural
    significance and cannot be used in LaTeX labels, but is often required by them.
    '''
def _parse_latex_cell_styles(latex_styles: CSSList, display_value: str, convert_css: bool = False) -> str:
    """
    Mutate the ``display_value`` string including LaTeX commands from ``latex_styles``.

    This method builds a recursive latex chain of commands based on the
    CSSList input, nested around ``display_value``.

    If a CSS style is given as ('<command>', '<options>') this is translated to
    '\\<command><options>{display_value}', and this value is treated as the
    display value for the next iteration.

    The most recent style forms the inner component, for example for styles:
    `[('c1', 'o1'), ('c2', 'o2')]` this returns: `\\c1o1{\\c2o2{display_value}}`

    Sometimes latex commands have to be wrapped with curly braces in different ways:
    We create some parsing flags to identify the different behaviours:

     - `--rwrap`        : `\\<command><options>{<display_value>}`
     - `--wrap`         : `{\\<command><options> <display_value>}`
     - `--nowrap`       : `\\<command><options> <display_value>`
     - `--lwrap`        : `{\\<command><options>} <display_value>`
     - `--dwrap`        : `{\\<command><options>}{<display_value>}`

    For example for styles:
    `[('c1', 'o1--wrap'), ('c2', 'o2')]` this returns: `{\\c1o1 \\c2o2{display_value}}
    """
def _parse_latex_header_span(cell: dict[str, Any], multirow_align: str, multicol_align: str, wrap: bool = False, convert_css: bool = False) -> str:
    '''
    Refactor the cell `display_value` if a \'colspan\' or \'rowspan\' attribute is present.

    \'rowspan\' and \'colspan\' do not occur simultaneouly. If they are detected then
    the `display_value` is altered to a LaTeX `multirow` or `multicol` command
    respectively, with the appropriate cell-span.

    ``wrap`` is used to enclose the `display_value` in braces which is needed for
    column headers using an siunitx package.

    Requires the package {multirow}, whereas multicol support is usually built in
    to the {tabular} environment.

    Examples
    --------
    >>> cell = {\'cellstyle\': \'\', \'display_value\':\'text\', \'attributes\': \'colspan="3"\'}
    >>> _parse_latex_header_span(cell, \'t\', \'c\')
    \'\\\\multicolumn{3}{c}{text}\'
    '''
def _parse_latex_options_strip(value: str | float, arg: str) -> str:
    """
    Strip a css_value which may have latex wrapping arguments, css comment identifiers,
    and whitespaces, to a valid string for latex options parsing.

    For example: 'red /* --wrap */  ' --> 'red'
    """
def _parse_latex_css_conversion(styles: CSSList) -> CSSList:
    """
    Convert CSS (attribute,value) pairs to equivalent LaTeX (command,options) pairs.

    Ignore conversion if tagged with `--latex` option, skipped if no conversion found.
    """
def _escape_latex(s: str) -> str:
    """
    Replace the characters ``&``, ``%``, ``$``, ``#``, ``_``, ``{``, ``}``,
    ``~``, ``^``, and ``\\`` in the string with LaTeX-safe sequences.

    Use this if you need to display text that might contain such characters in LaTeX.

    Parameters
    ----------
    s : str
        Input to be escaped

    Return
    ------
    str :
        Escaped string
    """
def _math_mode_with_dollar(s: str) -> str:
    """
    All characters in LaTeX math mode are preserved.

    The substrings in LaTeX math mode, which start with
    the character ``$`` and end with ``$``, are preserved
    without escaping. Otherwise regular LaTeX escaping applies.

    Parameters
    ----------
    s : str
        Input to be escaped

    Return
    ------
    str :
        Escaped string
    """
def _math_mode_with_parentheses(s: str) -> str:
    """
    All characters in LaTeX math mode are preserved.

    The substrings in LaTeX math mode, which start with
    the character ``\\(`` and end with ``\\)``, are preserved
    without escaping. Otherwise regular LaTeX escaping applies.

    Parameters
    ----------
    s : str
        Input to be escaped

    Return
    ------
    str :
        Escaped string
    """
def _escape_latex_math(s: str) -> str:
    """
    All characters in LaTeX math mode are preserved.

    The substrings in LaTeX math mode, which either are surrounded
    by two characters ``$`` or start with the character ``\\(`` and end with ``\\)``,
    are preserved without escaping. Otherwise regular LaTeX escaping applies.

    Parameters
    ----------
    s : str
        Input to be escaped

    Return
    ------
    str :
        Escaped string
    """
