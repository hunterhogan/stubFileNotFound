import np
import npt
import pandas._config.config as config
import pandas._libs.lib as lib
import pandas._libs.tslibs.timezones as timezones
import pandas._libs.writers as libwriters
import pandas.core.common as com
import pandas.core.frame
import pandas.core.series
import re
from _typeshed import Incomplete
from datetime import tzinfo
from pandas._config import using_copy_on_write as using_copy_on_write, using_pyarrow_string_dtype as using_pyarrow_string_dtype
from pandas._config.config import get_option as get_option
from pandas._libs.algos import ensure_object as ensure_object
from pandas._libs.lib import is_list_like as is_list_like, is_string_array as is_string_array
from pandas._libs.properties import cache_readonly as cache_readonly
from pandas.compat._optional import import_optional_dependency as import_optional_dependency
from pandas.compat.pickle_compat import patch_pickle as patch_pickle
from pandas.core.arrays.categorical import Categorical as Categorical
from pandas.core.arrays.datetimes import DatetimeArray as DatetimeArray
from pandas.core.arrays.period import PeriodArray as PeriodArray
from pandas.core.computation.pytables import PyTablesExpr as PyTablesExpr, Term as Term, maybe_expression as maybe_expression
from pandas.core.construction import extract_array as extract_array
from pandas.core.dtypes.base import ExtensionDtype as ExtensionDtype
from pandas.core.dtypes.common import is_bool_dtype as is_bool_dtype, is_complex_dtype as is_complex_dtype, is_string_dtype as is_string_dtype, needs_i8_conversion as needs_i8_conversion
from pandas.core.dtypes.dtypes import CategoricalDtype as CategoricalDtype, DatetimeTZDtype as DatetimeTZDtype, PeriodDtype as PeriodDtype
from pandas.core.dtypes.missing import array_equivalent as array_equivalent, isna as isna
from pandas.core.frame import DataFrame as DataFrame
from pandas.core.indexes.base import Index as Index, ensure_index as ensure_index
from pandas.core.indexes.datetimes import DatetimeIndex as DatetimeIndex
from pandas.core.indexes.multi import MultiIndex as MultiIndex
from pandas.core.indexes.period import PeriodIndex as PeriodIndex
from pandas.core.indexes.range import RangeIndex as RangeIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex as TimedeltaIndex
from pandas.core.internals.array_manager import ArrayManager as ArrayManager
from pandas.core.internals.managers import BlockManager as BlockManager
from pandas.core.reshape.concat import concat as concat
from pandas.core.series import Series as Series
from pandas.errors import AttributeConflictWarning as AttributeConflictWarning, ClosedFileError as ClosedFileError, IncompatibilityWarning as IncompatibilityWarning, PerformanceWarning as PerformanceWarning, PossibleDataLossError as PossibleDataLossError
from pandas.io.common import stringify_path as stringify_path
from pandas.io.formats.printing import adjoin as adjoin, pprint_thing as pprint_thing
from pandas.util._exceptions import find_stack_level as find_stack_level
from typing import Any, ClassVar, Literal

TYPE_CHECKING: bool
_version: str
_default_encoding: str
def _ensure_decoded(s):
    """if we have bytes, decode them to unicode"""
def _ensure_encoding(encoding: str | None) -> str: ...
def _ensure_str(name):
    """
    Ensure that an index / column name is a str (python 3); otherwise they
    may be np.string dtype. Non-string dtypes are passed through unchanged.

    https://github.com/pandas-dev/pandas/issues/13492
    """
def _ensure_term(where, scope_level: int):
    """
    Ensure that the where is a Term or a list of Term.

    This makes sure that we are capturing the scope of variables that are
    passed create the terms here with a frame_level=2 (we are 2 levels down)
    """

incompatibility_doc: str
attribute_conflict_doc: str
performance_doc: str
_FORMAT_MAP: dict
_AXES_MAP: dict
dropna_doc: str
format_doc: str
_table_mod: None
_table_file_open_policy_is_strict: bool
def _tables(): ...
def to_hdf(path_or_buf: FilePath | HDFStore, key: str, value: DataFrame | Series, mode: str = ..., complevel: int | None, complib: str | None, append: bool = ..., format: str | None, index: bool = ..., min_itemsize: int | dict[str, int] | None, nan_rep, dropna: bool | None, data_columns: Literal[True] | list[str] | None, errors: str = ..., encoding: str = ...) -> None:
    """store this object, close it if we opened it"""
def read_hdf(path_or_buf: FilePath | HDFStore, key, mode: str = ..., errors: str = ..., where: str | list | None, start: int | None, stop: int | None, columns: list[str] | None, iterator: bool = ..., chunksize: int | None, **kwargs):
    '''
    Read from the store, close it if we opened it.

    Retrieve pandas object stored in file, optionally based on where
    criteria.

    .. warning::

       Pandas uses PyTables for reading and writing HDF5 files, which allows
       serializing object-dtype data with pickle when using the "fixed" format.
       Loading pickled data received from untrusted sources can be unsafe.

       See: https://docs.python.org/3/library/pickle.html for more.

    Parameters
    ----------
    path_or_buf : str, path object, pandas.HDFStore
        Any valid string path is acceptable. Only supports the local file system,
        remote URLs and file-like objects are not supported.

        If you want to pass in a path object, pandas accepts any
        ``os.PathLike``.

        Alternatively, pandas accepts an open :class:`pandas.HDFStore` object.

    key : object, optional
        The group identifier in the store. Can be omitted if the HDF file
        contains a single pandas object.
    mode : {\'r\', \'r+\', \'a\'}, default \'r\'
        Mode to use when opening the file. Ignored if path_or_buf is a
        :class:`pandas.HDFStore`. Default is \'r\'.
    errors : str, default \'strict\'
        Specifies how encoding and decoding errors are to be handled.
        See the errors argument for :func:`open` for a full list
        of options.
    where : list, optional
        A list of Term (or convertible) objects.
    start : int, optional
        Row number to start selection.
    stop  : int, optional
        Row number to stop selection.
    columns : list, optional
        A list of columns names to return.
    iterator : bool, optional
        Return an iterator object.
    chunksize : int, optional
        Number of rows to include in an iteration when using an iterator.
    **kwargs
        Additional keyword arguments passed to HDFStore.

    Returns
    -------
    object
        The selected object. Return type depends on the object stored.

    See Also
    --------
    DataFrame.to_hdf : Write a HDF file from a DataFrame.
    HDFStore : Low-level access to HDF files.

    Examples
    --------
    >>> df = pd.DataFrame([[1, 1.0, \'a\']], columns=[\'x\', \'y\', \'z\'])  # doctest: +SKIP
    >>> df.to_hdf(\'./store.h5\', \'data\')  # doctest: +SKIP
    >>> reread = pd.read_hdf(\'./store.h5\')  # doctest: +SKIP
    '''
def _is_metadata_of(group: Node, parent_group: Node) -> bool:
    """Check if a given group is a metadata group for a given parent_group."""

class HDFStore:
    def __init__(self, path, mode: str = ..., complevel: int | None, complib, fletcher32: bool = ..., **kwargs) -> None: ...
    def __fspath__(self) -> str: ...
    def __getitem__(self, key: str): ...
    def __setitem__(self, key: str, value) -> None: ...
    def __delitem__(self, key: str) -> None: ...
    def __getattr__(self, name: str):
        """allow attribute access to get stores"""
    def __contains__(self, key: str) -> bool:
        """
        check for existence of this key
        can match the exact pathname or the pathnm w/o the leading '/'
        """
    def __len__(self) -> int: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None: ...
    def keys(self, include: str = ...) -> list[str]:
        '''
        Return a list of keys corresponding to objects stored in HDFStore.

        Parameters
        ----------

        include : str, default \'pandas\'
                When kind equals \'pandas\' return pandas objects.
                When kind equals \'native\' return native HDF5 Table objects.

        Returns
        -------
        list
            List of ABSOLUTE path-names (e.g. have the leading \'/\').

        Raises
        ------
        raises ValueError if kind has an illegal value

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=[\'A\', \'B\'])
        >>> store = pd.HDFStore("store.h5", \'w\')  # doctest: +SKIP
        >>> store.put(\'data\', df)  # doctest: +SKIP
        >>> store.get(\'data\')  # doctest: +SKIP
        >>> print(store.keys())  # doctest: +SKIP
        [\'/data1\', \'/data2\']
        >>> store.close()  # doctest: +SKIP
        '''
    def __iter__(self) -> Iterator[str]: ...
    def items(self) -> Iterator[tuple[str, list]]:
        """
        iterate on key->group
        """
    def open(self, mode: str = ..., **kwargs) -> None:
        """
        Open the file in the specified mode

        Parameters
        ----------
        mode : {'a', 'w', 'r', 'r+'}, default 'a'
            See HDFStore docstring or tables.open_file for info about modes
        **kwargs
            These parameters will be passed to the PyTables open_file method.
        """
    def close(self) -> None:
        """
        Close the PyTables file handle
        """
    def flush(self, fsync: bool = ...) -> None:
        """
        Force all buffered modifications to be written to disk.

        Parameters
        ----------
        fsync : bool (default False)
          call ``os.fsync()`` on the file handle to force writing to disk.

        Notes
        -----
        Without ``fsync=True``, flushing may not guarantee that the OS writes
        to disk. With fsync, the operation will block until the OS claims the
        file has been written; however, other caching layers may still
        interfere.
        """
    def get(self, key: str):
        '''
        Retrieve pandas object stored in file.

        Parameters
        ----------
        key : str

        Returns
        -------
        object
            Same type as object stored in file.

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=[\'A\', \'B\'])
        >>> store = pd.HDFStore("store.h5", \'w\')  # doctest: +SKIP
        >>> store.put(\'data\', df)  # doctest: +SKIP
        >>> store.get(\'data\')  # doctest: +SKIP
        >>> store.close()  # doctest: +SKIP
        '''
    def select(self, key: str, where, start, stop, columns, iterator: bool = ..., chunksize: int | None, auto_close: bool = ...):
        '''
        Retrieve pandas object stored in file, optionally based on where criteria.

        .. warning::

           Pandas uses PyTables for reading and writing HDF5 files, which allows
           serializing object-dtype data with pickle when using the "fixed" format.
           Loading pickled data received from untrusted sources can be unsafe.

           See: https://docs.python.org/3/library/pickle.html for more.

        Parameters
        ----------
        key : str
            Object being retrieved from file.
        where : list or None
            List of Term (or convertible) objects, optional.
        start : int or None
            Row number to start selection.
        stop : int, default None
            Row number to stop selection.
        columns : list or None
            A list of columns that if not None, will limit the return columns.
        iterator : bool or False
            Returns an iterator.
        chunksize : int or None
            Number or rows to include in iteration, return an iterator.
        auto_close : bool or False
            Should automatically close the store when finished.

        Returns
        -------
        object
            Retrieved object from file.

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=[\'A\', \'B\'])
        >>> store = pd.HDFStore("store.h5", \'w\')  # doctest: +SKIP
        >>> store.put(\'data\', df)  # doctest: +SKIP
        >>> store.get(\'data\')  # doctest: +SKIP
        >>> print(store.keys())  # doctest: +SKIP
        [\'/data1\', \'/data2\']
        >>> store.select(\'/data1\')  # doctest: +SKIP
           A  B
        0  1  2
        1  3  4
        >>> store.select(\'/data1\', where=\'columns == A\')  # doctest: +SKIP
           A
        0  1
        1  3
        >>> store.close()  # doctest: +SKIP
        '''
    def select_as_coordinates(self, key: str, where, start: int | None, stop: int | None):
        '''
        return the selection as an Index

        .. warning::

           Pandas uses PyTables for reading and writing HDF5 files, which allows
           serializing object-dtype data with pickle when using the "fixed" format.
           Loading pickled data received from untrusted sources can be unsafe.

           See: https://docs.python.org/3/library/pickle.html for more.


        Parameters
        ----------
        key : str
        where : list of Term (or convertible) objects, optional
        start : integer (defaults to None), row number to start selection
        stop  : integer (defaults to None), row number to stop selection
        '''
    def select_column(self, key: str, column: str, start: int | None, stop: int | None):
        '''
        return a single column from the table. This is generally only useful to
        select an indexable

        .. warning::

           Pandas uses PyTables for reading and writing HDF5 files, which allows
           serializing object-dtype data with pickle when using the "fixed" format.
           Loading pickled data received from untrusted sources can be unsafe.

           See: https://docs.python.org/3/library/pickle.html for more.

        Parameters
        ----------
        key : str
        column : str
            The column of interest.
        start : int or None, default None
        stop : int or None, default None

        Raises
        ------
        raises KeyError if the column is not found (or key is not a valid
            store)
        raises ValueError if the column can not be extracted individually (it
            is part of a data block)

        '''
    def select_as_multiple(self, keys, where, selector, columns, start, stop, iterator: bool = ..., chunksize: int | None, auto_close: bool = ...):
        '''
        Retrieve pandas objects from multiple tables.

        .. warning::

           Pandas uses PyTables for reading and writing HDF5 files, which allows
           serializing object-dtype data with pickle when using the "fixed" format.
           Loading pickled data received from untrusted sources can be unsafe.

           See: https://docs.python.org/3/library/pickle.html for more.

        Parameters
        ----------
        keys : a list of the tables
        selector : the table to apply the where criteria (defaults to keys[0]
            if not supplied)
        columns : the columns I want back
        start : integer (defaults to None), row number to start selection
        stop  : integer (defaults to None), row number to stop selection
        iterator : bool, return an iterator, default False
        chunksize : nrows to include in iteration, return an iterator
        auto_close : bool, default False
            Should automatically close the store when finished.

        Raises
        ------
        raises KeyError if keys or selector is not found or keys is empty
        raises TypeError if keys is not a list or tuple
        raises ValueError if the tables are not ALL THE SAME DIMENSIONS
        '''
    def put(self, key: str, value: DataFrame | Series, format, index: bool = ..., append: bool = ..., complib, complevel: int | None, min_itemsize: int | dict[str, int] | None, nan_rep, data_columns: Literal[True] | list[str] | None, encoding, errors: str = ..., track_times: bool = ..., dropna: bool = ...) -> None:
        '''
        Store object in HDFStore.

        Parameters
        ----------
        key : str
        value : {Series, DataFrame}
        format : \'fixed(f)|table(t)\', default is \'fixed\'
            Format to use when storing object in HDFStore. Value can be one of:

            ``\'fixed\'``
                Fixed format.  Fast writing/reading. Not-appendable, nor searchable.
            ``\'table\'``
                Table format.  Write as a PyTables Table structure which may perform
                worse but allow more flexible operations like searching / selecting
                subsets of the data.
        index : bool, default True
            Write DataFrame index as a column.
        append : bool, default False
            This will force Table format, append the input data to the existing.
        data_columns : list of columns or True, default None
            List of columns to create as data columns, or True to use all columns.
            See `here
            <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#query-via-data-columns>`__.
        encoding : str, default None
            Provide an encoding for strings.
        track_times : bool, default True
            Parameter is propagated to \'create_table\' method of \'PyTables\'.
            If set to False it enables to have the same h5 files (same hashes)
            independent on creation time.
        dropna : bool, default False, optional
            Remove missing values.

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=[\'A\', \'B\'])
        >>> store = pd.HDFStore("store.h5", \'w\')  # doctest: +SKIP
        >>> store.put(\'data\', df)  # doctest: +SKIP
        '''
    def remove(self, key: str, where, start, stop) -> None:
        """
        Remove pandas object partially by specifying the where condition

        Parameters
        ----------
        key : str
            Node to remove or delete rows from
        where : list of Term (or convertible) objects, optional
        start : integer (defaults to None), row number to start selection
        stop  : integer (defaults to None), row number to stop selection

        Returns
        -------
        number of rows removed (or None if not a Table)

        Raises
        ------
        raises KeyError if key is not a valid store

        """
    def append(self, key: str, value: DataFrame | Series, format, axes, index: bool | list[str] = ..., append: bool = ..., complib, complevel: int | None, columns, min_itemsize: int | dict[str, int] | None, nan_rep, chunksize: int | None, expectedrows, dropna: bool | None, data_columns: Literal[True] | list[str] | None, encoding, errors: str = ...) -> None:
        '''
        Append to Table in file.

        Node must already exist and be Table format.

        Parameters
        ----------
        key : str
        value : {Series, DataFrame}
        format : \'table\' is the default
            Format to use when storing object in HDFStore.  Value can be one of:

            ``\'table\'``
                Table format. Write as a PyTables Table structure which may perform
                worse but allow more flexible operations like searching / selecting
                subsets of the data.
        index : bool, default True
            Write DataFrame index as a column.
        append       : bool, default True
            Append the input data to the existing.
        data_columns : list of columns, or True, default None
            List of columns to create as indexed data columns for on-disk
            queries, or True to use all columns. By default only the axes
            of the object are indexed. See `here
            <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#query-via-data-columns>`__.
        min_itemsize : dict of columns that specify minimum str sizes
        nan_rep      : str to use as str nan representation
        chunksize    : size to chunk the writing
        expectedrows : expected TOTAL row size of this table
        encoding     : default None, provide an encoding for str
        dropna : bool, default False, optional
            Do not write an ALL nan row to the store settable
            by the option \'io.hdf.dropna_table\'.

        Notes
        -----
        Does *not* check if data being appended overlaps with existing
        data in the table, so be careful

        Examples
        --------
        >>> df1 = pd.DataFrame([[1, 2], [3, 4]], columns=[\'A\', \'B\'])
        >>> store = pd.HDFStore("store.h5", \'w\')  # doctest: +SKIP
        >>> store.put(\'data\', df1, format=\'table\')  # doctest: +SKIP
        >>> df2 = pd.DataFrame([[5, 6], [7, 8]], columns=[\'A\', \'B\'])
        >>> store.append(\'data\', df2)  # doctest: +SKIP
        >>> store.close()  # doctest: +SKIP
           A  B
        0  1  2
        1  3  4
        0  5  6
        1  7  8
        '''
    def append_to_multiple(self, d: dict, value, selector, data_columns, axes, dropna: bool = ..., **kwargs) -> None:
        """
        Append to multiple tables

        Parameters
        ----------
        d : a dict of table_name to table_columns, None is acceptable as the
            values of one node (this will get all the remaining columns)
        value : a pandas object
        selector : a string that designates the indexable table; all of its
            columns will be designed as data_columns, unless data_columns is
            passed, in which case these are used
        data_columns : list of columns to create as data columns, or True to
            use all columns
        dropna : if evaluates to True, drop rows from all tables if any single
                 row in each table has all NaN. Default False.

        Notes
        -----
        axes parameter is currently not accepted

        """
    def create_table_index(self, key: str, columns, optlevel: int | None, kind: str | None) -> None:
        '''
        Create a pytables index on the table.

        Parameters
        ----------
        key : str
        columns : None, bool, or listlike[str]
            Indicate which columns to create an index on.

            * False : Do not create any indexes.
            * True : Create indexes on all columns.
            * None : Create indexes on all columns.
            * listlike : Create indexes on the given columns.

        optlevel : int or None, default None
            Optimization level, if None, pytables defaults to 6.
        kind : str or None, default None
            Kind of index, if None, pytables defaults to "medium".

        Raises
        ------
        TypeError: raises if the node is not a table
        '''
    def groups(self) -> list:
        '''
        Return a list of all the top-level nodes.

        Each node returned is not a pandas storage object.

        Returns
        -------
        list
            List of objects.

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=[\'A\', \'B\'])
        >>> store = pd.HDFStore("store.h5", \'w\')  # doctest: +SKIP
        >>> store.put(\'data\', df)  # doctest: +SKIP
        >>> print(store.groups())  # doctest: +SKIP
        >>> store.close()  # doctest: +SKIP
        [/data (Group) \'\'
          children := [\'axis0\' (Array), \'axis1\' (Array), \'block0_values\' (Array),
          \'block0_items\' (Array)]]
        '''
    def walk(self, where: str = ...) -> Iterator[tuple[str, list[str], list[str]]]:
        '''
        Walk the pytables group hierarchy for pandas objects.

        This generator will yield the group path, subgroups and pandas object
        names for each group.

        Any non-pandas PyTables objects that are not a group will be ignored.

        The `where` group itself is listed first (preorder), then each of its
        child groups (following an alphanumerical order) is also traversed,
        following the same procedure.

        Parameters
        ----------
        where : str, default "/"
            Group where to start walking.

        Yields
        ------
        path : str
            Full path to a group (without trailing \'/\').
        groups : list
            Names (strings) of the groups contained in `path`.
        leaves : list
            Names (strings) of the pandas objects contained in `path`.

        Examples
        --------
        >>> df1 = pd.DataFrame([[1, 2], [3, 4]], columns=[\'A\', \'B\'])
        >>> store = pd.HDFStore("store.h5", \'w\')  # doctest: +SKIP
        >>> store.put(\'data\', df1, format=\'table\')  # doctest: +SKIP
        >>> df2 = pd.DataFrame([[5, 6], [7, 8]], columns=[\'A\', \'B\'])
        >>> store.append(\'data\', df2)  # doctest: +SKIP
        >>> store.close()  # doctest: +SKIP
        >>> for group in store.walk():  # doctest: +SKIP
        ...     print(group)  # doctest: +SKIP
        >>> store.close()  # doctest: +SKIP
        '''
    def get_node(self, key: str) -> Node | None:
        """return the node with the key or None if it does not exist"""
    def get_storer(self, key: str) -> GenericFixed | Table:
        """return the storer object for a key, raise if not in the file"""
    def copy(self, file, mode: str = ..., propindexes: bool = ..., keys, complib, complevel: int | None, fletcher32: bool = ..., overwrite: bool = ...) -> HDFStore:
        """
        Copy the existing store to a new file, updating in place.

        Parameters
        ----------
        propindexes : bool, default True
            Restore indexes in copied file.
        keys : list, optional
            List of keys to include in the copy (defaults to all).
        overwrite : bool, default True
            Whether to overwrite (remove and replace) existing nodes in the new store.
        mode, complib, complevel, fletcher32 same as in HDFStore.__init__

        Returns
        -------
        open file handle of the new store
        """
    def info(self) -> str:
        '''
        Print detailed information on the store.

        Returns
        -------
        str

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=[\'A\', \'B\'])
        >>> store = pd.HDFStore("store.h5", \'w\')  # doctest: +SKIP
        >>> store.put(\'data\', df)  # doctest: +SKIP
        >>> print(store.info())  # doctest: +SKIP
        >>> store.close()  # doctest: +SKIP
        <class \'pandas.io.pytables.HDFStore\'>
        File path: store.h5
        /data    frame    (shape->[2,2])
        '''
    def _check_if_open(self) -> None: ...
    def _validate_format(self, format: str) -> str:
        """validate / deprecate formats"""
    def _create_storer(self, group, format, value: DataFrame | Series | None, encoding: str = ..., errors: str = ...) -> GenericFixed | Table:
        """return a suitable class to operate"""
    def _write_to_group(self, key: str, value: DataFrame | Series, format, axes, index: bool | list[str] = ..., append: bool = ..., complib, complevel: int | None, fletcher32, min_itemsize: int | dict[str, int] | None, chunksize: int | None, expectedrows, dropna: bool = ..., nan_rep, data_columns, encoding, errors: str = ..., track_times: bool = ...) -> None: ...
    def _read_group(self, group: Node): ...
    def _identify_group(self, key: str, append: bool) -> Node:
        """Identify HDF5 group based on key, delete/create group if needed."""
    def _create_nodes_and_group(self, key: str) -> Node:
        """Create nodes from key and return group name."""
    @property
    def root(self): ...
    @property
    def filename(self): ...
    @property
    def is_open(self): ...

class TableIterator:
    def __init__(self, store: HDFStore, s: GenericFixed | Table, func, where, nrows, start, stop, iterator: bool = ..., chunksize: int | None, auto_close: bool = ...) -> None: ...
    def __iter__(self) -> Iterator: ...
    def close(self) -> None: ...
    def get_result(self, coordinates: bool = ...): ...

class IndexCol:
    is_an_indexable: ClassVar[bool] = ...
    is_data_indexable: ClassVar[bool] = ...
    _info_fields: ClassVar[list] = ...
    def __init__(self, name: str, values, kind, typ, cname: str | None, axis, pos, freq, tz, index_name, ordered, table, meta, metadata) -> None: ...
    def set_pos(self, pos: int) -> None:
        """set the position of this column in the Table"""
    def __eq__(self, other: object) -> bool:
        """compare 2 col items"""
    def __ne__(self, other) -> bool: ...
    def convert(self, values: np.ndarray, nan_rep, encoding: str, errors: str) -> tuple[np.ndarray, np.ndarray] | tuple[Index, Index]:
        """
        Convert the data from this selection to the appropriate pandas type.
        """
    def take_data(self):
        """return the values"""
    def __iter__(self) -> Iterator: ...
    def maybe_set_size(self, min_itemsize) -> None:
        """
        maybe set a string col itemsize:
            min_itemsize can be an integer or a dict with this columns name
            with an integer size
        """
    def validate_names(self) -> None: ...
    def validate_and_set(self, handler: AppendableTable, append: bool) -> None: ...
    def validate_col(self, itemsize):
        """validate this column: return the compared against itemsize"""
    def validate_attr(self, append: bool) -> None: ...
    def update_info(self, info) -> None:
        """
        set/update the info for this indexable with the key/value
        if there is a conflict raise/warn as needed
        """
    def set_info(self, info) -> None:
        """set my state from the passed info"""
    def set_attr(self) -> None:
        """set the kind for this column"""
    def validate_metadata(self, handler: AppendableTable) -> None:
        """validate that kind=category does not change the categories"""
    def write_metadata(self, handler: AppendableTable) -> None:
        """set the meta data"""
    @property
    def itemsize(self): ...
    @property
    def kind_attr(self): ...
    @property
    def is_indexed(self): ...
    @property
    def attrs(self): ...
    @property
    def description(self): ...
    @property
    def col(self): ...
    @property
    def cvalues(self): ...

class GenericIndexCol(IndexCol):
    def convert(self, values: np.ndarray, nan_rep, encoding: str, errors: str) -> tuple[Index, Index]:
        """
        Convert the data from this selection to the appropriate pandas type.

        Parameters
        ----------
        values : np.ndarray
        nan_rep : str
        encoding : str
        errors : str
        """
    def set_attr(self) -> None: ...
    @property
    def is_indexed(self): ...

class DataCol(IndexCol):
    is_an_indexable: ClassVar[bool] = ...
    is_data_indexable: ClassVar[bool] = ...
    _info_fields: ClassVar[list] = ...
    def __init__(self, name: str, values, kind, typ, cname: str | None, pos, tz, ordered, table, meta, metadata, dtype: DtypeArg | None, data) -> None: ...
    def __eq__(self, other: object) -> bool:
        """compare 2 col items"""
    def set_data(self, data: ArrayLike) -> None: ...
    def take_data(self):
        """return the data"""
    @classmethod
    def _get_atom(cls, values: ArrayLike) -> Col:
        """
        Get an appropriately typed and shaped pytables.Col object for values.
        """
    @classmethod
    def get_atom_string(cls, shape, itemsize): ...
    @classmethod
    def get_atom_coltype(cls, kind: str) -> type[Col]:
        """return the PyTables column class for this column"""
    @classmethod
    def get_atom_data(cls, shape, kind: str) -> Col: ...
    @classmethod
    def get_atom_datetime64(cls, shape): ...
    @classmethod
    def get_atom_timedelta64(cls, shape): ...
    def validate_attr(self, append) -> None:
        """validate that we have the same order as the existing & same dtype"""
    def convert(self, values: np.ndarray, nan_rep, encoding: str, errors: str):
        """
        Convert the data from this selection to the appropriate pandas type.

        Parameters
        ----------
        values : np.ndarray
        nan_rep :
        encoding : str
        errors : str

        Returns
        -------
        index : listlike to become an Index
        data : ndarraylike to become a column
        """
    def set_attr(self) -> None:
        """set the data for this column"""
    @property
    def dtype_attr(self): ...
    @property
    def meta_attr(self): ...
    @property
    def shape(self): ...
    @property
    def cvalues(self): ...

class DataIndexableCol(DataCol):
    is_data_indexable: ClassVar[bool] = ...
    def validate_names(self) -> None: ...
    @classmethod
    def get_atom_string(cls, shape, itemsize): ...
    @classmethod
    def get_atom_data(cls, shape, kind: str) -> Col: ...
    @classmethod
    def get_atom_datetime64(cls, shape): ...
    @classmethod
    def get_atom_timedelta64(cls, shape): ...

class GenericDataIndexableCol(DataIndexableCol): ...

class Fixed:
    format_type: ClassVar[str] = ...
    is_table: ClassVar[bool] = ...
    def __init__(self, parent: HDFStore, group: Node, encoding: str | None = ..., errors: str = ...) -> None: ...
    def set_object_info(self) -> None:
        """set my pandas type & version"""
    def copy(self) -> Fixed: ...
    def set_attrs(self) -> None:
        """set our object attributes"""
    def get_attrs(self) -> None:
        """get our object attributes"""
    def validate(self, other) -> Literal[True] | None:
        """validate against an existing storable"""
    def validate_version(self, where) -> None:
        """are we trying to operate on an old version?"""
    def infer_axes(self) -> bool:
        """
        infer the axes of my storer
        return a boolean indicating if we have a valid storer or not
        """
    def read(self, where, columns, start: int | None, stop: int | None): ...
    def write(self, obj, **kwargs) -> None: ...
    def delete(self, where, start: int | None, stop: int | None) -> None:
        """
        support fully deleting the node in its entirety (only) - where
        specification must be None
        """
    @property
    def is_old_version(self): ...
    @property
    def version(self): ...
    @property
    def pandas_type(self): ...
    @property
    def shape(self): ...
    @property
    def pathname(self): ...
    @property
    def _handle(self): ...
    @property
    def _filters(self): ...
    @property
    def _complevel(self): ...
    @property
    def _fletcher32(self): ...
    @property
    def attrs(self): ...
    @property
    def storable(self): ...
    @property
    def is_exists(self): ...
    @property
    def nrows(self): ...

class GenericFixed(Fixed):
    _index_type_map: ClassVar[dict] = ...
    _reverse_index_map: ClassVar[dict] = ...
    attributes: ClassVar[list] = ...
    def _class_to_alias(self, cls) -> str: ...
    def _alias_to_class(self, alias): ...
    def _get_index_factory(self, attrs): ...
    def validate_read(self, columns, where) -> None:
        """
        raise if any keywords are passed which are not-None
        """
    def set_attrs(self) -> None:
        """set our object attributes"""
    def get_attrs(self) -> None:
        """retrieve our attributes"""
    def write(self, obj, **kwargs) -> None: ...
    def read_array(self, key: str, start: int | None, stop: int | None):
        """read an array for the specified node (off of group"""
    def read_index(self, key: str, start: int | None, stop: int | None) -> Index: ...
    def write_index(self, key: str, index: Index) -> None: ...
    def write_multi_index(self, key: str, index: MultiIndex) -> None: ...
    def read_multi_index(self, key: str, start: int | None, stop: int | None) -> MultiIndex: ...
    def read_index_node(self, node: Node, start: int | None, stop: int | None) -> Index: ...
    def write_array_empty(self, key: str, value: ArrayLike) -> None:
        """write a 0-len array"""
    def write_array(self, key: str, obj: AnyArrayLike, items: Index | None) -> None: ...
    @property
    def is_exists(self): ...

class SeriesFixed(GenericFixed):
    pandas_kind: ClassVar[str] = ...
    attributes: ClassVar[list] = ...
    def read(self, where, columns, start: int | None, stop: int | None) -> Series: ...
    def write(self, obj, **kwargs) -> None: ...
    @property
    def shape(self): ...

class BlockManagerFixed(GenericFixed):
    attributes: ClassVar[list] = ...
    def read(self, where, columns, start: int | None, stop: int | None) -> DataFrame: ...
    def write(self, obj, **kwargs) -> None: ...
    @property
    def shape(self): ...

class FrameFixed(BlockManagerFixed):
    pandas_kind: ClassVar[str] = ...
    obj_type: ClassVar[type[pandas.core.frame.DataFrame]] = ...

class Table(Fixed):
    pandas_kind: ClassVar[str] = ...
    format_type: ClassVar[str] = ...
    levels: ClassVar[int] = ...
    is_table: ClassVar[bool] = ...
    indexables: Incomplete
    def __init__(self, parent: HDFStore, group: Node, encoding: str | None, errors: str = ..., index_axes: list[IndexCol] | None, non_index_axes: list[tuple[AxisInt, Any]] | None, values_axes: list[DataCol] | None, data_columns: list | None, info: dict | None, nan_rep) -> None: ...
    def __getitem__(self, c: str):
        """return the axis for c"""
    def validate(self, other) -> None:
        """validate against an existing table"""
    def validate_multiindex(self, obj: DataFrame | Series) -> tuple[DataFrame, list[Hashable]]:
        """
        validate that we can store the multi-index; reset and return the
        new object
        """
    def queryables(self) -> dict[str, Any]:
        """return a dict of the kinds allowable columns for this object"""
    def index_cols(self):
        """return a list of my index cols"""
    def values_cols(self) -> list[str]:
        """return a list of my values cols"""
    def _get_metadata_path(self, key: str) -> str:
        """return the metadata pathname for this key"""
    def write_metadata(self, key: str, values: np.ndarray) -> None:
        """
        Write out a metadata array to the key as a fixed-format Series.

        Parameters
        ----------
        key : str
        values : ndarray
        """
    def read_metadata(self, key: str):
        """return the meta data array for this key"""
    def set_attrs(self) -> None:
        """set our table type & indexables"""
    def get_attrs(self) -> None:
        """retrieve our attributes"""
    def validate_version(self, where) -> None:
        """are we trying to operate on an old version?"""
    def validate_min_itemsize(self, min_itemsize) -> None:
        """
        validate the min_itemsize doesn't contain items that are not in the
        axes this needs data_columns to be defined
        """
    def create_index(self, columns, optlevel, kind: str | None) -> None:
        '''
        Create a pytables index on the specified columns.

        Parameters
        ----------
        columns : None, bool, or listlike[str]
            Indicate which columns to create an index on.

            * False : Do not create any indexes.
            * True : Create indexes on all columns.
            * None : Create indexes on all columns.
            * listlike : Create indexes on the given columns.

        optlevel : int or None, default None
            Optimization level, if None, pytables defaults to 6.
        kind : str or None, default None
            Kind of index, if None, pytables defaults to "medium".

        Raises
        ------
        TypeError if trying to create an index on a complex-type column.

        Notes
        -----
        Cannot index Time64Col or ComplexCol.
        Pytables must be >= 3.0.
        '''
    def _read_axes(self, where, start: int | None, stop: int | None) -> list[tuple[np.ndarray, np.ndarray] | tuple[Index, Index]]:
        """
        Create the axes sniffed from the table.

        Parameters
        ----------
        where : ???
        start : int or None, default None
        stop : int or None, default None

        Returns
        -------
        List[Tuple[index_values, column_values]]
        """
    @classmethod
    def get_object(cls, obj, transposed: bool):
        """return the data for this obj"""
    def validate_data_columns(self, data_columns, min_itemsize, non_index_axes):
        """
        take the input data_columns and min_itemize and create a data
        columns spec
        """
    def _create_axes(self, axes, obj: DataFrame, validate: bool = ..., nan_rep, data_columns, min_itemsize):
        """
        Create and return the axes.

        Parameters
        ----------
        axes: list or None
            The names or numbers of the axes to create.
        obj : DataFrame
            The object to create axes on.
        validate: bool, default True
            Whether to validate the obj against an existing object already written.
        nan_rep :
            A value to use for string column nan_rep.
        data_columns : List[str], True, or None, default None
            Specify the columns that we want to create to allow indexing on.

            * True : Use all available columns.
            * None : Use no columns.
            * List[str] : Use the specified columns.

        min_itemsize: Dict[str, int] or None, default None
            The min itemsize for a column in bytes.
        """
    @staticmethod
    def _get_blocks_and_items(frame: DataFrame, table_exists: bool, new_non_index_axes, values_axes, data_columns): ...
    def process_axes(self, obj, selection: Selection, columns) -> DataFrame:
        """process axes filters"""
    def create_description(self, complib, complevel: int | None, fletcher32: bool, expectedrows: int | None) -> dict[str, Any]:
        """create the description of the table from the axes & values"""
    def read_coordinates(self, where, start: int | None, stop: int | None):
        """
        select coordinates (row numbers) from a table; return the
        coordinates object
        """
    def read_column(self, column: str, where, start: int | None, stop: int | None):
        """
        return a single column from the table, generally only indexables
        are interesting
        """
    @property
    def table_type_short(self): ...
    @property
    def is_multi_index(self): ...
    @property
    def nrows_expected(self): ...
    @property
    def is_exists(self): ...
    @property
    def storable(self): ...
    @property
    def table(self): ...
    @property
    def dtype(self): ...
    @property
    def description(self): ...
    @property
    def axes(self): ...
    @property
    def ncols(self): ...
    @property
    def is_transposed(self): ...
    @property
    def data_orientation(self): ...

class WORMTable(Table):
    table_type: ClassVar[str] = ...
    def read(self, where, columns, start: int | None, stop: int | None):
        """
        read the indices and the indexing array, calculate offset rows and return
        """
    def write(self, obj, **kwargs) -> None:
        """
        write in a format that we can search later on (but cannot append
        to): write out the indices and the values using _write_array
        (e.g. a CArray) create an indexing table so that we can search
        """

class AppendableTable(Table):
    table_type: ClassVar[str] = ...
    def write(self, obj, axes, append: bool = ..., complib, complevel, fletcher32, min_itemsize, chunksize: int | None, expectedrows, dropna: bool = ..., nan_rep, data_columns, track_times: bool = ...) -> None: ...
    def write_data(self, chunksize: int | None, dropna: bool = ...) -> None:
        """
        we form the data into a 2-d including indexes,values,mask write chunk-by-chunk
        """
    def write_data_chunk(self, rows: np.ndarray, indexes: list[np.ndarray], mask: npt.NDArray[np.bool_] | None, values: list[np.ndarray]) -> None:
        """
        Parameters
        ----------
        rows : an empty memory space where we are putting the chunk
        indexes : an array of the indexes
        mask : an array of the masks
        values : an array of the values
        """
    def delete(self, where, start: int | None, stop: int | None): ...

class AppendableFrameTable(AppendableTable):
    pandas_kind: ClassVar[str] = ...
    table_type: ClassVar[str] = ...
    ndim: ClassVar[int] = ...
    obj_type: ClassVar[type[pandas.core.frame.DataFrame]] = ...
    @classmethod
    def get_object(cls, obj, transposed: bool):
        """these are written transposed"""
    def read(self, where, columns, start: int | None, stop: int | None): ...
    @property
    def is_transposed(self): ...

class AppendableSeriesTable(AppendableFrameTable):
    pandas_kind: ClassVar[str] = ...
    table_type: ClassVar[str] = ...
    ndim: ClassVar[int] = ...
    obj_type: ClassVar[type[pandas.core.series.Series]] = ...
    @classmethod
    def get_object(cls, obj, transposed: bool): ...
    def write(self, obj, data_columns, **kwargs) -> None:
        """we are going to write this as a frame table"""
    def read(self, where, columns, start: int | None, stop: int | None) -> Series: ...
    @property
    def is_transposed(self): ...

class AppendableMultiSeriesTable(AppendableSeriesTable):
    pandas_kind: ClassVar[str] = ...
    table_type: ClassVar[str] = ...
    def write(self, obj, **kwargs) -> None:
        """we are going to write this as a frame table"""

class GenericTable(AppendableFrameTable):
    pandas_kind: ClassVar[str] = ...
    table_type: ClassVar[str] = ...
    ndim: ClassVar[int] = ...
    obj_type: ClassVar[type[pandas.core.frame.DataFrame]] = ...
    indexables: Incomplete
    def get_attrs(self) -> None:
        """retrieve our attributes"""
    def write(self, **kwargs) -> None: ...
    @property
    def pandas_type(self): ...
    @property
    def storable(self): ...

class AppendableMultiFrameTable(AppendableFrameTable):
    table_type: ClassVar[str] = ...
    obj_type: ClassVar[type[pandas.core.frame.DataFrame]] = ...
    ndim: ClassVar[int] = ...
    _re_levels: ClassVar[re.Pattern] = ...
    def write(self, obj, data_columns, **kwargs) -> None: ...
    def read(self, where, columns, start: int | None, stop: int | None): ...
    @property
    def table_type_short(self): ...
def _reindex_axis(obj: DataFrame, axis: AxisInt, labels: Index, other) -> DataFrame: ...
def _get_tz(tz: tzinfo) -> str | tzinfo:
    """for a tz-aware type, return an encoded zone"""
def _set_tz(values: np.ndarray | Index, tz: str | tzinfo | None, coerce: bool = ...) -> np.ndarray | DatetimeIndex:
    """
    coerce the values to a DatetimeIndex if tz is set
    preserve the input shape if possible

    Parameters
    ----------
    values : ndarray or Index
    tz : str or tzinfo
    coerce : if we do not have a passed timezone, coerce to M8[ns] ndarray
    """
def _convert_index(name: str, index: Index, encoding: str, errors: str) -> IndexCol: ...
def _unconvert_index(data, kind: str, encoding: str, errors: str) -> np.ndarray | Index: ...
def _maybe_convert_for_string_atom(name: str, bvalues: ArrayLike, existing_col, min_itemsize, nan_rep, encoding, errors, columns: list[str]): ...
def _convert_string_array(data: np.ndarray, encoding: str, errors: str) -> np.ndarray:
    """
    Take a string-like that is object dtype and coerce to a fixed size string type.

    Parameters
    ----------
    data : np.ndarray[object]
    encoding : str
    errors : str
        Handler for encoding errors.

    Returns
    -------
    np.ndarray[fixed-length-string]
    """
def _unconvert_string_array(data: np.ndarray, nan_rep, encoding: str, errors: str) -> np.ndarray:
    """
    Inverse of _convert_string_array.

    Parameters
    ----------
    data : np.ndarray[fixed-length-string]
    nan_rep : the storage repr of NaN
    encoding : str
    errors : str
        Handler for encoding errors.

    Returns
    -------
    np.ndarray[object]
        Decoded data.
    """
def _maybe_convert(values: np.ndarray, val_kind: str, encoding: str, errors: str): ...
def _get_converter(kind: str, encoding: str, errors: str): ...
def _need_convert(kind: str) -> bool: ...
def _maybe_adjust_name(name: str, version: Sequence[int]) -> str:
    """
    Prior to 0.10.1, we named values blocks like: values_block_0 an the
    name values_0, adjust the given name if necessary.

    Parameters
    ----------
    name : str
    version : Tuple[int, int, int]

    Returns
    -------
    str
    """
def _dtype_to_kind(dtype_str: str) -> str:
    '''
    Find the "kind" string describing the given dtype name.
    '''
def _get_data_and_dtype_name(data: ArrayLike):
    """
    Convert the passed data into a storable form and a dtype string.
    """

class Selection:
    def __init__(self, table: Table, where, start: int | None, stop: int | None) -> None: ...
    def generate(self, where):
        """where can be a : dict,list,tuple,string"""
    def select(self):
        """
        generate the selection
        """
    def select_coords(self):
        """
        generate the selection
        """
