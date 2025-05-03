import lib as lib
from pandas._libs.lib import is_list_like as is_list_like
from pandas.compat._optional import import_optional_dependency as import_optional_dependency
from pandas.core.dtypes.inference import is_file_like as is_file_like
from pandas.core.dtypes.missing import isna as isna
from pandas.core.indexes.base import Index as Index
from pandas.core.indexes.multi import MultiIndex as MultiIndex
from pandas.core.series import Series as Series
from pandas.errors import AbstractMethodError as AbstractMethodError, EmptyDataError as EmptyDataError
from pandas.io.common import file_exists as file_exists, get_handle as get_handle, is_fsspec_url as is_fsspec_url, is_url as is_url, stringify_path as stringify_path, validate_header_arg as validate_header_arg
from pandas.io.formats.printing import pprint_thing as pprint_thing
from pandas.io.parsers.readers import TextParser as TextParser
from pandas.util._decorators import doc as doc
from pandas.util._exceptions import find_stack_level as find_stack_level
from pandas.util._validators import check_dtype_backend as check_dtype_backend
from re import Pattern
from typing import Literal

TYPE_CHECKING: bool
_shared_docs: dict
def _remove_whitespace(s: str, regex: Pattern = ...) -> str:
    """
    Replace extra whitespace inside of a string with a single space.

    Parameters
    ----------
    s : str or unicode
        The string from which to remove extra whitespace.
    regex : re.Pattern
        The regular expression to use to remove extra whitespace.

    Returns
    -------
    subd : str or unicode
        `s` with all extra whitespace replaced with a single space.
    """
def _get_skiprows(skiprows: int | Sequence[int] | slice | None) -> int | Sequence[int]:
    """
    Get an iterator given an integer, slice or container.

    Parameters
    ----------
    skiprows : int, slice, container
        The iterator to use to skip rows; can also be a slice.

    Raises
    ------
    TypeError
        * If `skiprows` is not a slice, integer, or Container

    Returns
    -------
    it : iterable
        A proper iterator to use to skip rows of a DataFrame.
    """
def _read(obj: FilePath | BaseBuffer, encoding: str | None, storage_options: StorageOptions | None) -> str | bytes:
    """
    Try to read from a url, file or string.

    Parameters
    ----------
    obj : str, unicode, path object, or file-like object

    Returns
    -------
    raw_text : str
    """

class _HtmlFrameParser:
    def __init__(self, io: FilePath | ReadBuffer[str] | ReadBuffer[bytes], match: str | Pattern, attrs: dict[str, str] | None, encoding: str, displayed_only: bool, extract_links: Literal[None, 'header', 'footer', 'body', 'all'], storage_options: StorageOptions) -> None: ...
    def parse_tables(self):
        """
        Parse and return all tables from the DOM.

        Returns
        -------
        list of parsed (header, body, footer) tuples from tables.
        """
    def _attr_getter(self, obj, attr):
        '''
        Return the attribute value of an individual DOM node.

        Parameters
        ----------
        obj : node-like
            A DOM node.

        attr : str or unicode
            The attribute, such as "colspan"

        Returns
        -------
        str or unicode
            The attribute value.
        '''
    def _href_getter(self, obj) -> str | None:
        """
        Return a href if the DOM node contains a child <a> or None.

        Parameters
        ----------
        obj : node-like
            A DOM node.

        Returns
        -------
        href : str or unicode
            The href from the <a> child of the DOM node.
        """
    def _text_getter(self, obj):
        """
        Return the text of an individual DOM node.

        Parameters
        ----------
        obj : node-like
            A DOM node.

        Returns
        -------
        text : str or unicode
            The text from an individual DOM node.
        """
    def _parse_td(self, obj):
        """
        Return the td elements from a row element.

        Parameters
        ----------
        obj : node-like
            A DOM <tr> node.

        Returns
        -------
        list of node-like
            These are the elements of each row, i.e., the columns.
        """
    def _parse_thead_tr(self, table):
        """
        Return the list of thead row elements from the parsed table element.

        Parameters
        ----------
        table : a table element that contains zero or more thead elements.

        Returns
        -------
        list of node-like
            These are the <tr> row elements of a table.
        """
    def _parse_tbody_tr(self, table):
        """
        Return the list of tbody row elements from the parsed table element.

        HTML5 table bodies consist of either 0 or more <tbody> elements (which
        only contain <tr> elements) or 0 or more <tr> elements. This method
        checks for both structures.

        Parameters
        ----------
        table : a table element that contains row elements.

        Returns
        -------
        list of node-like
            These are the <tr> row elements of a table.
        """
    def _parse_tfoot_tr(self, table):
        """
        Return the list of tfoot row elements from the parsed table element.

        Parameters
        ----------
        table : a table element that contains row elements.

        Returns
        -------
        list of node-like
            These are the <tr> row elements of a table.
        """
    def _parse_tables(self, document, match, attrs):
        """
        Return all tables from the parsed DOM.

        Parameters
        ----------
        document : the DOM from which to parse the table element.

        match : str or regular expression
            The text to search for in the DOM tree.

        attrs : dict
            A dictionary of table attributes that can be used to disambiguate
            multiple tables on a page.

        Raises
        ------
        ValueError : `match` does not match any text in the document.

        Returns
        -------
        list of node-like
            HTML <table> elements to be parsed into raw data.
        """
    def _equals_tag(self, obj, tag) -> bool:
        """
        Return whether an individual DOM node matches a tag

        Parameters
        ----------
        obj : node-like
            A DOM node.

        tag : str
            Tag name to be checked for equality.

        Returns
        -------
        boolean
            Whether `obj`'s tag name is `tag`
        """
    def _build_doc(self):
        """
        Return a tree-like object that can be used to iterate over the DOM.

        Returns
        -------
        node-like
            The DOM from which to parse the table element.
        """
    def _parse_thead_tbody_tfoot(self, table_html):
        """
        Given a table, return parsed header, body, and foot.

        Parameters
        ----------
        table_html : node-like

        Returns
        -------
        tuple of (header, body, footer), each a list of list-of-text rows.

        Notes
        -----
        Header and body are lists-of-lists. Top level list is a list of
        rows. Each row is a list of str text.

        Logic: Use <thead>, <tbody>, <tfoot> elements to identify
               header, body, and footer, otherwise:
               - Put all rows into body
               - Move rows from top of body to header only if
                 all elements inside row are <th>
               - Move rows from bottom of body to footer only if
                 all elements inside row are <th>
        """
    def _expand_colspan_rowspan(self, rows, section: Literal['header', 'footer', 'body']):
        """
        Given a list of <tr>s, return a list of text rows.

        Parameters
        ----------
        rows : list of node-like
            List of <tr>s
        section : the section that the rows belong to (header, body or footer).

        Returns
        -------
        list of list
            Each returned row is a list of str text, or tuple (text, link)
            if extract_links is not None.

        Notes
        -----
        Any cell with ``rowspan`` or ``colspan`` will have its contents copied
        to subsequent cells.
        """
    def _handle_hidden_tables(self, tbl_list, attr_name: str):
        """
        Return list of tables, potentially removing hidden elements

        Parameters
        ----------
        tbl_list : list of node-like
            Type of list elements will vary depending upon parser used
        attr_name : str
            Name of the accessor for retrieving HTML attributes

        Returns
        -------
        list of node-like
            Return type matches `tbl_list`
        """

class _BeautifulSoupHtml5LibFrameParser(_HtmlFrameParser):
    def _parse_tables(self, document, match, attrs): ...
    def _href_getter(self, obj) -> str | None: ...
    def _text_getter(self, obj): ...
    def _equals_tag(self, obj, tag) -> bool: ...
    def _parse_td(self, row): ...
    def _parse_thead_tr(self, table): ...
    def _parse_tbody_tr(self, table): ...
    def _parse_tfoot_tr(self, table): ...
    def _setup_build_doc(self): ...
    def _build_doc(self): ...
def _build_xpath_expr(attrs) -> str:
    """
    Build an xpath expression to simulate bs4's ability to pass in kwargs to
    search for attributes when using the lxml parser.

    Parameters
    ----------
    attrs : dict
        A dict of HTML attributes. These are NOT checked for validity.

    Returns
    -------
    expr : unicode
        An XPath expression that checks for the given HTML attributes.
    """

_re_namespace: dict

class _LxmlFrameParser(_HtmlFrameParser):
    def _href_getter(self, obj) -> str | None: ...
    def _text_getter(self, obj): ...
    def _parse_td(self, row): ...
    def _parse_tables(self, document, match, kwargs): ...
    def _equals_tag(self, obj, tag) -> bool: ...
    def _build_doc(self):
        """
        Raises
        ------
        ValueError
            * If a URL that lxml cannot parse is passed.

        Exception
            * Any other ``Exception`` thrown. For example, trying to parse a
              URL that is syntactically correct on a machine with no internet
              connection will fail.

        See Also
        --------
        pandas.io.html._HtmlFrameParser._build_doc
        """
    def _parse_thead_tr(self, table): ...
    def _parse_tbody_tr(self, table): ...
    def _parse_tfoot_tr(self, table): ...
def _expand_elements(body) -> None: ...
def _data_to_frame(**kwargs): ...

_valid_parsers: dict
def _parser_dispatch(flavor: HTMLFlavors | None) -> type[_HtmlFrameParser]:
    '''
    Choose the parser based on the input flavor.

    Parameters
    ----------
    flavor : {{"lxml", "html5lib", "bs4"}} or None
        The type of parser to use. This must be a valid backend.

    Returns
    -------
    cls : _HtmlFrameParser subclass
        The parser class based on the requested input flavor.

    Raises
    ------
    ValueError
        * If `flavor` is not a valid backend.
    ImportError
        * If you do not have the requested `flavor`
    '''
def _print_as_set(s) -> str: ...
def _validate_flavor(flavor): ...
def _parse(flavor, io, match, attrs, encoding, displayed_only, extract_links, storage_options, **kwargs): ...
def read_html(io: FilePath | ReadBuffer[str], *, match: str | Pattern = ..., flavor: HTMLFlavors | Sequence[HTMLFlavors] | None, header: int | Sequence[int] | None, index_col: int | Sequence[int] | None, skiprows: int | Sequence[int] | slice | None, attrs: dict[str, str] | None, parse_dates: bool = ..., thousands: str | None = ..., encoding: str | None, decimal: str = ..., converters: dict | None, na_values: Iterable[object] | None, keep_default_na: bool = ..., displayed_only: bool = ..., extract_links: Literal[None, 'header', 'footer', 'body', 'all'], dtype_backend: DtypeBackend | lib.NoDefault = ..., storage_options: StorageOptions) -> list[DataFrame]:
    '''
    Read HTML tables into a ``list`` of ``DataFrame`` objects.

    Parameters
    ----------
    io : str, path object, or file-like object
        String, path object (implementing ``os.PathLike[str]``), or file-like
        object implementing a string ``read()`` function.
        The string can represent a URL or the HTML itself. Note that
        lxml only accepts the http, ftp and file url protocols. If you have a
        URL that starts with ``\'https\'`` you might try removing the ``\'s\'``.

        .. deprecated:: 2.1.0
            Passing html literal strings is deprecated.
            Wrap literal string/bytes input in ``io.StringIO``/``io.BytesIO`` instead.

    match : str or compiled regular expression, optional
        The set of tables containing text matching this regex or string will be
        returned. Unless the HTML is extremely simple you will probably need to
        pass a non-empty string here. Defaults to \'.+\' (match any non-empty
        string). The default value will return all tables contained on a page.
        This value is converted to a regular expression so that there is
        consistent behavior between Beautiful Soup and lxml.

    flavor : {"lxml", "html5lib", "bs4"} or list-like, optional
        The parsing engine (or list of parsing engines) to use. \'bs4\' and
        \'html5lib\' are synonymous with each other, they are both there for
        backwards compatibility. The default of ``None`` tries to use ``lxml``
        to parse and if that fails it falls back on ``bs4`` + ``html5lib``.

    header : int or list-like, optional
        The row (or list of rows for a :class:`~pandas.MultiIndex`) to use to
        make the columns headers.

    index_col : int or list-like, optional
        The column (or list of columns) to use to create the index.

    skiprows : int, list-like or slice, optional
        Number of rows to skip after parsing the column integer. 0-based. If a
        sequence of integers or a slice is given, will skip the rows indexed by
        that sequence.  Note that a single element sequence means \'skip the nth
        row\' whereas an integer means \'skip n rows\'.

    attrs : dict, optional
        This is a dictionary of attributes that you can pass to use to identify
        the table in the HTML. These are not checked for validity before being
        passed to lxml or Beautiful Soup. However, these attributes must be
        valid HTML table attributes to work correctly. For example, ::

            attrs = {\'id\': \'table\'}

        is a valid attribute dictionary because the \'id\' HTML tag attribute is
        a valid HTML attribute for *any* HTML tag as per `this document
        <https://html.spec.whatwg.org/multipage/dom.html#global-attributes>`__. ::

            attrs = {\'asdf\': \'table\'}

        is *not* a valid attribute dictionary because \'asdf\' is not a valid
        HTML attribute even if it is a valid XML attribute.  Valid HTML 4.01
        table attributes can be found `here
        <http://www.w3.org/TR/REC-html40/struct/tables.html#h-11.2>`__. A
        working draft of the HTML 5 spec can be found `here
        <https://html.spec.whatwg.org/multipage/tables.html>`__. It contains the
        latest information on table attributes for the modern web.

    parse_dates : bool, optional
        See :func:`~read_csv` for more details.

    thousands : str, optional
        Separator to use to parse thousands. Defaults to ``\',\'``.

    encoding : str, optional
        The encoding used to decode the web page. Defaults to ``None``.``None``
        preserves the previous encoding behavior, which depends on the
        underlying parser library (e.g., the parser library will try to use
        the encoding provided by the document).

    decimal : str, default \'.\'
        Character to recognize as decimal point (e.g. use \',\' for European
        data).

    converters : dict, default None
        Dict of functions for converting values in certain columns. Keys can
        either be integers or column labels, values are functions that take one
        input argument, the cell (not column) content, and return the
        transformed content.

    na_values : iterable, default None
        Custom NA values.

    keep_default_na : bool, default True
        If na_values are specified and keep_default_na is False the default NaN
        values are overridden, otherwise they\'re appended to.

    displayed_only : bool, default True
        Whether elements with "display: none" should be parsed.

    extract_links : {None, "all", "header", "body", "footer"}
        Table elements in the specified section(s) with <a> tags will have their
        href extracted.

        .. versionadded:: 1.5.0

    dtype_backend : {\'numpy_nullable\', \'pyarrow\'}, default \'numpy_nullable\'
        Back-end data type applied to the resultant :class:`DataFrame`
        (still experimental). Behaviour is as follows:

        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
          (default).
        * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype`
          DataFrame.

        .. versionadded:: 2.0

    storage_options : dict, optional
        Extra options that make sense for a particular storage connection, e.g.
        host, port, username, password, etc. For HTTP(S) URLs the key-value pairs
        are forwarded to ``urllib.request.Request`` as header options. For other
        URLs (e.g. starting with "s3://", and "gcs://") the key-value pairs are
        forwarded to ``fsspec.open``. Please see ``fsspec`` and ``urllib`` for more
        details, and for more examples on storage options refer `here
        <https://pandas.pydata.org/docs/user_guide/io.html?
        highlight=storage_options#reading-writing-remote-files>`_.

        .. versionadded:: 2.1.0

    Returns
    -------
    dfs
        A list of DataFrames.

    See Also
    --------
    read_csv : Read a comma-separated values (csv) file into DataFrame.

    Notes
    -----
    Before using this function you should read the :ref:`gotchas about the
    HTML parsing libraries <io.html.gotchas>`.

    Expect to do some cleanup after you call this function. For example, you
    might need to manually assign column names if the column names are
    converted to NaN when you pass the `header=0` argument. We try to assume as
    little as possible about the structure of the table and push the
    idiosyncrasies of the HTML contained in the table to the user.

    This function searches for ``<table>`` elements and only for ``<tr>``
    and ``<th>`` rows and ``<td>`` elements within each ``<tr>`` or ``<th>``
    element in the table. ``<td>`` stands for "table data". This function
    attempts to properly handle ``colspan`` and ``rowspan`` attributes.
    If the function has a ``<thead>`` argument, it is used to construct
    the header, otherwise the function attempts to find the header within
    the body (by putting rows with only ``<th>`` elements into the header).

    Similar to :func:`~read_csv` the `header` argument is applied
    **after** `skiprows` is applied.

    This function will *always* return a list of :class:`DataFrame` *or*
    it will fail, e.g., it will *not* return an empty list.

    Examples
    --------
    See the :ref:`read_html documentation in the IO section of the docs
    <io.read_html>` for some examples of reading in HTML tables.
    '''
