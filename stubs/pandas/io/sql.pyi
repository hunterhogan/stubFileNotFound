import _abc
import abc
import lib as lib
import np
import pandas.core.base
import pandas.core.common as com
from contextlib import ExitStack
from pandas._config import using_pyarrow_string_dtype as using_pyarrow_string_dtype
from pandas._config.config import get_option as get_option
from pandas._libs.lib import is_list_like as is_list_like
from pandas.compat._optional import import_optional_dependency as import_optional_dependency
from pandas.core.arrays.arrow.array import ArrowExtensionArray as ArrowExtensionArray
from pandas.core.base import PandasObject as PandasObject
from pandas.core.common import maybe_make_list as maybe_make_list
from pandas.core.dtypes.dtypes import ArrowDtype as ArrowDtype, DatetimeTZDtype as DatetimeTZDtype
from pandas.core.dtypes.inference import is_dict_like as is_dict_like
from pandas.core.dtypes.missing import isna as isna
from pandas.core.frame import DataFrame as DataFrame
from pandas.core.internals.construction import convert_object_array as convert_object_array
from pandas.core.series import Series as Series
from pandas.core.tools.datetimes import to_datetime as to_datetime
from pandas.errors import AbstractMethodError as AbstractMethodError, DatabaseError as DatabaseError
from pandas.util._exceptions import find_stack_level as find_stack_level
from pandas.util._validators import check_dtype_backend as check_dtype_backend
from typing import Any, Callable, ClassVar, Literal

TYPE_CHECKING: bool
def _process_parse_dates_argument(parse_dates):
    """Process parse_dates argument for read_sql functions"""
def _handle_date_column(col, utc: bool = ..., format: str | dict[str, Any] | None): ...
def _parse_date_columns(data_frame, parse_dates):
    """
    Force non-datetime columns to be read as such.
    Supports both string formatted and integer timestamp columns.
    """
def _convert_arrays_to_dataframe(data, columns, coerce_float: bool = ..., dtype_backend: DtypeBackend | Literal['numpy'] = ...) -> DataFrame: ...
def _wrap_result(data, columns, index_col, coerce_float: bool = ..., parse_dates, dtype: DtypeArg | None, dtype_backend: DtypeBackend | Literal['numpy'] = ...):
    """Wrap result set of a SQLAlchemy query in a DataFrame."""
def _wrap_result_adbc(df: DataFrame, *, index_col, parse_dates, dtype: DtypeArg | None, dtype_backend: DtypeBackend | Literal['numpy'] = ...) -> DataFrame:
    """Wrap result set of a SQLAlchemy query in a DataFrame."""
def execute(sql, con, params):
    """
    Execute the given SQL query using the provided connection object.

    Parameters
    ----------
    sql : string
        SQL query to be executed.
    con : SQLAlchemy connection or sqlite3 connection
        If a DBAPI2 object, only sqlite3 is supported.
    params : list or tuple, optional, default: None
        List of parameters to pass to execute method.

    Returns
    -------
    Results Iterable
    """
def read_sql_table(table_name: str, con, schema: str | None, index_col: str | list[str] | None, coerce_float: bool = ..., parse_dates: list[str] | dict[str, str] | None, columns: list[str] | None, chunksize: int | None, dtype_backend: DtypeBackend | lib.NoDefault = ...) -> DataFrame | Iterator[DataFrame]:
    '''
    Read SQL database table into a DataFrame.

    Given a table name and a SQLAlchemy connectable, returns a DataFrame.
    This function does not support DBAPI connections.

    Parameters
    ----------
    table_name : str
        Name of SQL table in database.
    con : SQLAlchemy connectable or str
        A database URI could be provided as str.
        SQLite DBAPI connection mode not supported.
    schema : str, default None
        Name of SQL schema in database to query (if database flavor
        supports this). Uses default schema if None (default).
    index_col : str or list of str, optional, default: None
        Column(s) to set as index(MultiIndex).
    coerce_float : bool, default True
        Attempts to convert values of non-string, non-numeric objects (like
        decimal.Decimal) to floating point. Can result in loss of Precision.
    parse_dates : list or dict, default None
        - List of column names to parse as dates.
        - Dict of ``{column_name: format string}`` where format string is
          strftime compatible in case of parsing string times or is one of
          (D, s, ns, ms, us) in case of parsing integer timestamps.
        - Dict of ``{column_name: arg dict}``, where the arg dict corresponds
          to the keyword arguments of :func:`pandas.to_datetime`
          Especially useful with databases without native Datetime support,
          such as SQLite.
    columns : list, default None
        List of column names to select from SQL table.
    chunksize : int, default None
        If specified, returns an iterator where `chunksize` is the number of
        rows to include in each chunk.
    dtype_backend : {\'numpy_nullable\', \'pyarrow\'}, default \'numpy_nullable\'
        Back-end data type applied to the resultant :class:`DataFrame`
        (still experimental). Behaviour is as follows:

        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
          (default).
        * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype`
          DataFrame.

        .. versionadded:: 2.0

    Returns
    -------
    DataFrame or Iterator[DataFrame]
        A SQL table is returned as two-dimensional data structure with labeled
        axes.

    See Also
    --------
    read_sql_query : Read SQL query into a DataFrame.
    read_sql : Read SQL query or database table into a DataFrame.

    Notes
    -----
    Any datetime values with time zone information will be converted to UTC.

    Examples
    --------
    >>> pd.read_sql_table(\'table_name\', \'postgres:///db_name\')  # doctest:+SKIP
    '''
def read_sql_query(sql, con, index_col: str | list[str] | None, coerce_float: bool = ..., params: list[Any] | Mapping[str, Any] | None, parse_dates: list[str] | dict[str, str] | None, chunksize: int | None, dtype: DtypeArg | None, dtype_backend: DtypeBackend | lib.NoDefault = ...) -> DataFrame | Iterator[DataFrame]:
    '''
    Read SQL query into a DataFrame.

    Returns a DataFrame corresponding to the result set of the query
    string. Optionally provide an `index_col` parameter to use one of the
    columns as the index, otherwise default integer index will be used.

    Parameters
    ----------
    sql : str SQL query or SQLAlchemy Selectable (select or text object)
        SQL query to be executed.
    con : SQLAlchemy connectable, str, or sqlite3 connection
        Using SQLAlchemy makes it possible to use any DB supported by that
        library. If a DBAPI2 object, only sqlite3 is supported.
    index_col : str or list of str, optional, default: None
        Column(s) to set as index(MultiIndex).
    coerce_float : bool, default True
        Attempts to convert values of non-string, non-numeric objects (like
        decimal.Decimal) to floating point. Useful for SQL result sets.
    params : list, tuple or mapping, optional, default: None
        List of parameters to pass to execute method.  The syntax used
        to pass parameters is database driver dependent. Check your
        database driver documentation for which of the five syntax styles,
        described in PEP 249\'s paramstyle, is supported.
        Eg. for psycopg2, uses %(name)s so use params={\'name\' : \'value\'}.
    parse_dates : list or dict, default: None
        - List of column names to parse as dates.
        - Dict of ``{column_name: format string}`` where format string is
          strftime compatible in case of parsing string times, or is one of
          (D, s, ns, ms, us) in case of parsing integer timestamps.
        - Dict of ``{column_name: arg dict}``, where the arg dict corresponds
          to the keyword arguments of :func:`pandas.to_datetime`
          Especially useful with databases without native Datetime support,
          such as SQLite.
    chunksize : int, default None
        If specified, return an iterator where `chunksize` is the number of
        rows to include in each chunk.
    dtype : Type name or dict of columns
        Data type for data or columns. E.g. np.float64 or
        {\'a\': np.float64, \'b\': np.int32, \'c\': \'Int64\'}.

        .. versionadded:: 1.3.0
    dtype_backend : {\'numpy_nullable\', \'pyarrow\'}, default \'numpy_nullable\'
        Back-end data type applied to the resultant :class:`DataFrame`
        (still experimental). Behaviour is as follows:

        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
          (default).
        * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype`
          DataFrame.

        .. versionadded:: 2.0

    Returns
    -------
    DataFrame or Iterator[DataFrame]

    See Also
    --------
    read_sql_table : Read SQL database table into a DataFrame.
    read_sql : Read SQL query or database table into a DataFrame.

    Notes
    -----
    Any datetime values with time zone information parsed via the `parse_dates`
    parameter will be converted to UTC.

    Examples
    --------
    >>> from sqlalchemy import create_engine  # doctest: +SKIP
    >>> engine = create_engine("sqlite:///database.db")  # doctest: +SKIP
    >>> with engine.connect() as conn, conn.begin():  # doctest: +SKIP
    ...     data = pd.read_sql_table("data", conn)  # doctest: +SKIP
    '''
def read_sql(sql, con, index_col: str | list[str] | None, coerce_float: bool = ..., params, parse_dates, columns: list[str] | None, chunksize: int | None, dtype_backend: DtypeBackend | lib.NoDefault = ..., dtype: DtypeArg | None) -> DataFrame | Iterator[DataFrame]:
    '''
    Read SQL query or database table into a DataFrame.

    This function is a convenience wrapper around ``read_sql_table`` and
    ``read_sql_query`` (for backward compatibility). It will delegate
    to the specific function depending on the provided input. A SQL query
    will be routed to ``read_sql_query``, while a database table name will
    be routed to ``read_sql_table``. Note that the delegated function might
    have more specific notes about their functionality not listed here.

    Parameters
    ----------
    sql : str or SQLAlchemy Selectable (select or text object)
        SQL query to be executed or a table name.
    con : ADBC Connection, SQLAlchemy connectable, str, or sqlite3 connection
        ADBC provides high performance I/O with native type support, where available.
        Using SQLAlchemy makes it possible to use any DB supported by that
        library. If a DBAPI2 object, only sqlite3 is supported. The user is responsible
        for engine disposal and connection closure for the ADBC connection and
        SQLAlchemy connectable; str connections are closed automatically. See
        `here <https://docs.sqlalchemy.org/en/20/core/connections.html>`_.
    index_col : str or list of str, optional, default: None
        Column(s) to set as index(MultiIndex).
    coerce_float : bool, default True
        Attempts to convert values of non-string, non-numeric objects (like
        decimal.Decimal) to floating point, useful for SQL result sets.
    params : list, tuple or dict, optional, default: None
        List of parameters to pass to execute method.  The syntax used
        to pass parameters is database driver dependent. Check your
        database driver documentation for which of the five syntax styles,
        described in PEP 249\'s paramstyle, is supported.
        Eg. for psycopg2, uses %(name)s so use params={\'name\' : \'value\'}.
    parse_dates : list or dict, default: None
        - List of column names to parse as dates.
        - Dict of ``{column_name: format string}`` where format string is
          strftime compatible in case of parsing string times, or is one of
          (D, s, ns, ms, us) in case of parsing integer timestamps.
        - Dict of ``{column_name: arg dict}``, where the arg dict corresponds
          to the keyword arguments of :func:`pandas.to_datetime`
          Especially useful with databases without native Datetime support,
          such as SQLite.
    columns : list, default: None
        List of column names to select from SQL table (only used when reading
        a table).
    chunksize : int, default None
        If specified, return an iterator where `chunksize` is the
        number of rows to include in each chunk.
    dtype_backend : {\'numpy_nullable\', \'pyarrow\'}, default \'numpy_nullable\'
        Back-end data type applied to the resultant :class:`DataFrame`
        (still experimental). Behaviour is as follows:

        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
          (default).
        * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype`
          DataFrame.

        .. versionadded:: 2.0
    dtype : Type name or dict of columns
        Data type for data or columns. E.g. np.float64 or
        {\'a\': np.float64, \'b\': np.int32, \'c\': \'Int64\'}.
        The argument is ignored if a table is passed instead of a query.

        .. versionadded:: 2.0.0

    Returns
    -------
    DataFrame or Iterator[DataFrame]

    See Also
    --------
    read_sql_table : Read SQL database table into a DataFrame.
    read_sql_query : Read SQL query into a DataFrame.

    Examples
    --------
    Read data from SQL via either a SQL query or a SQL tablename.
    When using a SQLite database only SQL queries are accepted,
    providing only the SQL tablename will result in an error.

    >>> from sqlite3 import connect
    >>> conn = connect(\':memory:\')
    >>> df = pd.DataFrame(data=[[0, \'10/11/12\'], [1, \'12/11/10\']],
    ...                   columns=[\'int_column\', \'date_column\'])
    >>> df.to_sql(name=\'test_data\', con=conn)
    2

    >>> pd.read_sql(\'SELECT int_column, date_column FROM test_data\', conn)
       int_column date_column
    0           0    10/11/12
    1           1    12/11/10

    >>> pd.read_sql(\'test_data\', \'postgres:///db_name\')  # doctest:+SKIP

    Apply date parsing to columns through the ``parse_dates`` argument
    The ``parse_dates`` argument calls ``pd.to_datetime`` on the provided columns.
    Custom argument values for applying ``pd.to_datetime`` on a column are specified
    via a dictionary format:

    >>> pd.read_sql(\'SELECT int_column, date_column FROM test_data\',
    ...             conn,
    ...             parse_dates={"date_column": {"format": "%d/%m/%y"}})
       int_column date_column
    0           0  2012-11-10
    1           1  2010-11-12

    .. versionadded:: 2.2.0

       pandas now supports reading via ADBC drivers

    >>> from adbc_driver_postgresql import dbapi  # doctest:+SKIP
    >>> with dbapi.connect(\'postgres:///db_name\') as conn:  # doctest:+SKIP
    ...     pd.read_sql(\'SELECT int_column FROM test_data\', conn)
       int_column
    0           0
    1           1
    '''
def to_sql(frame, name: str, con, schema: str | None, if_exists: Literal['fail', 'replace', 'append'] = ..., index: bool = ..., index_label: IndexLabel | None, chunksize: int | None, dtype: DtypeArg | None, method: Literal['multi'] | Callable | None, engine: str = ..., **engine_kwargs) -> int | None:
    """
    Write records stored in a DataFrame to a SQL database.

    Parameters
    ----------
    frame : DataFrame, Series
    name : str
        Name of SQL table.
    con : ADBC Connection, SQLAlchemy connectable, str, or sqlite3 connection
        or sqlite3 DBAPI2 connection
        ADBC provides high performance I/O with native type support, where available.
        Using SQLAlchemy makes it possible to use any DB supported by that
        library.
        If a DBAPI2 object, only sqlite3 is supported.
    schema : str, optional
        Name of SQL schema in database to write to (if database flavor
        supports this). If None, use default schema (default).
    if_exists : {'fail', 'replace', 'append'}, default 'fail'
        - fail: If table exists, do nothing.
        - replace: If table exists, drop it, recreate it, and insert data.
        - append: If table exists, insert data. Create if does not exist.
    index : bool, default True
        Write DataFrame index as a column.
    index_label : str or sequence, optional
        Column label for index column(s). If None is given (default) and
        `index` is True, then the index names are used.
        A sequence should be given if the DataFrame uses MultiIndex.
    chunksize : int, optional
        Specify the number of rows in each batch to be written at a time.
        By default, all rows will be written at once.
    dtype : dict or scalar, optional
        Specifying the datatype for columns. If a dictionary is used, the
        keys should be the column names and the values should be the
        SQLAlchemy types or strings for the sqlite3 fallback mode. If a
        scalar is provided, it will be applied to all columns.
    method : {None, 'multi', callable}, optional
        Controls the SQL insertion clause used:

        - None : Uses standard SQL ``INSERT`` clause (one per row).
        - ``'multi'``: Pass multiple values in a single ``INSERT`` clause.
        - callable with signature ``(pd_table, conn, keys, data_iter) -> int | None``.

        Details and a sample callable implementation can be found in the
        section :ref:`insert method <io.sql.method>`.
    engine : {'auto', 'sqlalchemy'}, default 'auto'
        SQL engine library to use. If 'auto', then the option
        ``io.sql.engine`` is used. The default ``io.sql.engine``
        behavior is 'sqlalchemy'

        .. versionadded:: 1.3.0

    **engine_kwargs
        Any additional kwargs are passed to the engine.

    Returns
    -------
    None or int
        Number of rows affected by to_sql. None is returned if the callable
        passed into ``method`` does not return an integer number of rows.

        .. versionadded:: 1.4.0

    Notes
    -----
    The returned rows affected is the sum of the ``rowcount`` attribute of ``sqlite3.Cursor``
    or SQLAlchemy connectable. If using ADBC the returned rows are the result
    of ``Cursor.adbc_ingest``. The returned value may not reflect the exact number of written
    rows as stipulated in the
    `sqlite3 <https://docs.python.org/3/library/sqlite3.html#sqlite3.Cursor.rowcount>`__ or
    `SQLAlchemy <https://docs.sqlalchemy.org/en/14/core/connections.html#sqlalchemy.engine.BaseCursorResult.rowcount>`__
    """
def has_table(table_name: str, con, schema: str | None) -> bool:
    """
    Check if DataBase has named table.

    Parameters
    ----------
    table_name: string
        Name of SQL table.
    con: ADBC Connection, SQLAlchemy connectable, str, or sqlite3 connection
        ADBC provides high performance I/O with native type support, where available.
        Using SQLAlchemy makes it possible to use any DB supported by that
        library.
        If a DBAPI2 object, only sqlite3 is supported.
    schema : string, default None
        Name of SQL schema in database to write to (if database flavor supports
        this). If None, use default schema (default).

    Returns
    -------
    boolean
    """
def table_exists(table_name: str, con, schema: str | None) -> bool:
    """
    Check if DataBase has named table.

    Parameters
    ----------
    table_name: string
        Name of SQL table.
    con: ADBC Connection, SQLAlchemy connectable, str, or sqlite3 connection
        ADBC provides high performance I/O with native type support, where available.
        Using SQLAlchemy makes it possible to use any DB supported by that
        library.
        If a DBAPI2 object, only sqlite3 is supported.
    schema : string, default None
        Name of SQL schema in database to write to (if database flavor supports
        this). If None, use default schema (default).

    Returns
    -------
    boolean
    """
def pandasSQL_builder(con, schema: str | None, need_transaction: bool = ...) -> PandasSQL:
    """
    Convenience function to return the correct PandasSQL subclass based on the
    provided parameters.  Also creates a sqlalchemy connection and transaction
    if necessary.
    """

class SQLTable(pandas.core.base.PandasObject):
    def __init__(self, name: str, pandas_sql_engine, frame, index: bool | str | list[str] | None = ..., if_exists: Literal['fail', 'replace', 'append'] = ..., prefix: str = ..., index_label, schema, keys, dtype: DtypeArg | None) -> None: ...
    def exists(self): ...
    def sql_schema(self) -> str: ...
    def _execute_create(self) -> None: ...
    def create(self) -> None: ...
    def _execute_insert(self, conn, keys: list[str], data_iter) -> int:
        """
        Execute SQL statement inserting data

        Parameters
        ----------
        conn : sqlalchemy.engine.Engine or sqlalchemy.engine.Connection
        keys : list of str
           Column names
        data_iter : generator of list
           Each item contains a list of values to be inserted
        """
    def _execute_insert_multi(self, conn, keys: list[str], data_iter) -> int:
        """
        Alternative to _execute_insert for DBs support multi-value INSERT.

        Note: multi-value insert is usually faster for analytics DBs
        and tables containing a few columns
        but performance degrades quickly with increase of columns.

        """
    def insert_data(self) -> tuple[list[str], list[np.ndarray]]: ...
    def insert(self, chunksize: int | None, method: Literal['multi'] | Callable | None) -> int | None: ...
    def _query_iterator(self, result, exit_stack: ExitStack, chunksize: int | None, columns, coerce_float: bool = ..., parse_dates, dtype_backend: DtypeBackend | Literal['numpy'] = ...):
        """Return generator through chunked result set."""
    def read(self, exit_stack: ExitStack, coerce_float: bool = ..., parse_dates, columns, chunksize: int | None, dtype_backend: DtypeBackend | Literal['numpy'] = ...) -> DataFrame | Iterator[DataFrame]: ...
    def _index_name(self, index, index_label): ...
    def _get_column_names_and_types(self, dtype_mapper): ...
    def _create_table_setup(self): ...
    def _harmonize_columns(self, parse_dates, dtype_backend: DtypeBackend | Literal['numpy'] = ...) -> None:
        """
        Make the DataFrame's column types align with the SQL table
        column types.
        Need to work around limited NA value support. Floats are always
        fine, ints must always be floats if there are Null values.
        Booleans are hard because converting bool column with None replaces
        all Nones with false. Therefore only convert bool if there are no
        NA values.
        Datetimes should already be converted to np.datetime64 if supported,
        but here we also force conversion if required.
        """
    def _sqlalchemy_type(self, col: Index | Series): ...
    def _get_dtype(self, sqltype): ...

class PandasSQL(pandas.core.base.PandasObject, abc.ABC):
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    def __enter__(self) -> Self: ...
    def __exit__(self, *args) -> None: ...
    def read_table(self, table_name: str, index_col: str | list[str] | None, coerce_float: bool = ..., parse_dates, columns, schema: str | None, chunksize: int | None, dtype_backend: DtypeBackend | Literal['numpy'] = ...) -> DataFrame | Iterator[DataFrame]: ...
    def read_query(self, sql: str, index_col: str | list[str] | None, coerce_float: bool = ..., parse_dates, params, chunksize: int | None, dtype: DtypeArg | None, dtype_backend: DtypeBackend | Literal['numpy'] = ...) -> DataFrame | Iterator[DataFrame]: ...
    def to_sql(self, frame, name: str, if_exists: Literal['fail', 'replace', 'append'] = ..., index: bool = ..., index_label, schema, chunksize: int | None, dtype: DtypeArg | None, method: Literal['multi'] | Callable | None, engine: str = ..., **engine_kwargs) -> int | None: ...
    def execute(self, sql: str | Select | TextClause, params): ...
    def has_table(self, name: str, schema: str | None) -> bool: ...
    def _create_sql_schema(self, frame: DataFrame, table_name: str, keys: list[str] | None, dtype: DtypeArg | None, schema: str | None) -> str: ...

class BaseEngine:
    def insert_records(self, table: SQLTable, con, frame, name: str, index: bool | str | list[str] | None = ..., schema, chunksize: int | None, method, **engine_kwargs) -> int | None:
        """
        Inserts data into already-prepared table
        """

class SQLAlchemyEngine(BaseEngine):
    def __init__(self) -> None: ...
    def insert_records(self, table: SQLTable, con, frame, name: str, index: bool | str | list[str] | None = ..., schema, chunksize: int | None, method, **engine_kwargs) -> int | None: ...
def get_engine(engine: str) -> BaseEngine:
    """return our implementation"""

class SQLDatabase(PandasSQL):
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    def __init__(self, con, schema: str | None, need_transaction: bool = ...) -> None: ...
    def __exit__(self, *args) -> None: ...
    def run_transaction(self, *args, **kwds): ...
    def execute(self, sql: str | Select | TextClause, params):
        """Simple passthrough to SQLAlchemy connectable"""
    def read_table(self, table_name: str, index_col: str | list[str] | None, coerce_float: bool = ..., parse_dates, columns, schema: str | None, chunksize: int | None, dtype_backend: DtypeBackend | Literal['numpy'] = ...) -> DataFrame | Iterator[DataFrame]:
        '''
        Read SQL database table into a DataFrame.

        Parameters
        ----------
        table_name : str
            Name of SQL table in database.
        index_col : string, optional, default: None
            Column to set as index.
        coerce_float : bool, default True
            Attempts to convert values of non-string, non-numeric objects
            (like decimal.Decimal) to floating point. This can result in
            loss of precision.
        parse_dates : list or dict, default: None
            - List of column names to parse as dates.
            - Dict of ``{column_name: format string}`` where format string is
              strftime compatible in case of parsing string times, or is one of
              (D, s, ns, ms, us) in case of parsing integer timestamps.
            - Dict of ``{column_name: arg}``, where the arg corresponds
              to the keyword arguments of :func:`pandas.to_datetime`.
              Especially useful with databases without native Datetime support,
              such as SQLite.
        columns : list, default: None
            List of column names to select from SQL table.
        schema : string, default None
            Name of SQL schema in database to query (if database flavor
            supports this).  If specified, this overwrites the default
            schema of the SQL database object.
        chunksize : int, default None
            If specified, return an iterator where `chunksize` is the number
            of rows to include in each chunk.
        dtype_backend : {\'numpy_nullable\', \'pyarrow\'}, default \'numpy_nullable\'
            Back-end data type applied to the resultant :class:`DataFrame`
            (still experimental). Behaviour is as follows:

            * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
              (default).
            * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype`
              DataFrame.

            .. versionadded:: 2.0

        Returns
        -------
        DataFrame

        See Also
        --------
        pandas.read_sql_table
        SQLDatabase.read_query

        '''
    @staticmethod
    def _query_iterator(result, exit_stack: ExitStack, chunksize: int, columns, index_col, coerce_float: bool = ..., parse_dates, dtype: DtypeArg | None, dtype_backend: DtypeBackend | Literal['numpy'] = ...):
        """Return generator through chunked result set"""
    def read_query(self, sql: str, index_col: str | list[str] | None, coerce_float: bool = ..., parse_dates, params, chunksize: int | None, dtype: DtypeArg | None, dtype_backend: DtypeBackend | Literal['numpy'] = ...) -> DataFrame | Iterator[DataFrame]:
        """
        Read SQL query into a DataFrame.

        Parameters
        ----------
        sql : str
            SQL query to be executed.
        index_col : string, optional, default: None
            Column name to use as index for the returned DataFrame object.
        coerce_float : bool, default True
            Attempt to convert values of non-string, non-numeric objects (like
            decimal.Decimal) to floating point, useful for SQL result sets.
        params : list, tuple or dict, optional, default: None
            List of parameters to pass to execute method.  The syntax used
            to pass parameters is database driver dependent. Check your
            database driver documentation for which of the five syntax styles,
            described in PEP 249's paramstyle, is supported.
            Eg. for psycopg2, uses %(name)s so use params={'name' : 'value'}
        parse_dates : list or dict, default: None
            - List of column names to parse as dates.
            - Dict of ``{column_name: format string}`` where format string is
              strftime compatible in case of parsing string times, or is one of
              (D, s, ns, ms, us) in case of parsing integer timestamps.
            - Dict of ``{column_name: arg dict}``, where the arg dict
              corresponds to the keyword arguments of
              :func:`pandas.to_datetime` Especially useful with databases
              without native Datetime support, such as SQLite.
        chunksize : int, default None
            If specified, return an iterator where `chunksize` is the number
            of rows to include in each chunk.
        dtype : Type name or dict of columns
            Data type for data or columns. E.g. np.float64 or
            {'a': np.float64, 'b': np.int32, 'c': 'Int64'}

            .. versionadded:: 1.3.0

        Returns
        -------
        DataFrame

        See Also
        --------
        read_sql_table : Read SQL database table into a DataFrame.
        read_sql

        """
    def read_sql(self, sql: str, index_col: str | list[str] | None, coerce_float: bool = ..., parse_dates, params, chunksize: int | None, dtype: DtypeArg | None, dtype_backend: DtypeBackend | Literal['numpy'] = ...) -> DataFrame | Iterator[DataFrame]:
        """
        Read SQL query into a DataFrame.

        Parameters
        ----------
        sql : str
            SQL query to be executed.
        index_col : string, optional, default: None
            Column name to use as index for the returned DataFrame object.
        coerce_float : bool, default True
            Attempt to convert values of non-string, non-numeric objects (like
            decimal.Decimal) to floating point, useful for SQL result sets.
        params : list, tuple or dict, optional, default: None
            List of parameters to pass to execute method.  The syntax used
            to pass parameters is database driver dependent. Check your
            database driver documentation for which of the five syntax styles,
            described in PEP 249's paramstyle, is supported.
            Eg. for psycopg2, uses %(name)s so use params={'name' : 'value'}
        parse_dates : list or dict, default: None
            - List of column names to parse as dates.
            - Dict of ``{column_name: format string}`` where format string is
              strftime compatible in case of parsing string times, or is one of
              (D, s, ns, ms, us) in case of parsing integer timestamps.
            - Dict of ``{column_name: arg dict}``, where the arg dict
              corresponds to the keyword arguments of
              :func:`pandas.to_datetime` Especially useful with databases
              without native Datetime support, such as SQLite.
        chunksize : int, default None
            If specified, return an iterator where `chunksize` is the number
            of rows to include in each chunk.
        dtype : Type name or dict of columns
            Data type for data or columns. E.g. np.float64 or
            {'a': np.float64, 'b': np.int32, 'c': 'Int64'}

            .. versionadded:: 1.3.0

        Returns
        -------
        DataFrame

        See Also
        --------
        read_sql_table : Read SQL database table into a DataFrame.
        read_sql

        """
    def prep_table(self, frame, name: str, if_exists: Literal['fail', 'replace', 'append'] = ..., index: bool | str | list[str] | None = ..., index_label, schema, dtype: DtypeArg | None) -> SQLTable:
        """
        Prepares table in the database for data insertion. Creates it if needed, etc.
        """
    def check_case_sensitive(self, name: str, schema: str | None) -> None:
        """
        Checks table name for issues with case-sensitivity.
        Method is called after data is inserted.
        """
    def to_sql(self, frame, name: str, if_exists: Literal['fail', 'replace', 'append'] = ..., index: bool = ..., index_label, schema: str | None, chunksize: int | None, dtype: DtypeArg | None, method: Literal['multi'] | Callable | None, engine: str = ..., **engine_kwargs) -> int | None:
        """
        Write records stored in a DataFrame to a SQL database.

        Parameters
        ----------
        frame : DataFrame
        name : string
            Name of SQL table.
        if_exists : {'fail', 'replace', 'append'}, default 'fail'
            - fail: If table exists, do nothing.
            - replace: If table exists, drop it, recreate it, and insert data.
            - append: If table exists, insert data. Create if does not exist.
        index : boolean, default True
            Write DataFrame index as a column.
        index_label : string or sequence, default None
            Column label for index column(s). If None is given (default) and
            `index` is True, then the index names are used.
            A sequence should be given if the DataFrame uses MultiIndex.
        schema : string, default None
            Name of SQL schema in database to write to (if database flavor
            supports this). If specified, this overwrites the default
            schema of the SQLDatabase object.
        chunksize : int, default None
            If not None, then rows will be written in batches of this size at a
            time.  If None, all rows will be written at once.
        dtype : single type or dict of column name to SQL type, default None
            Optional specifying the datatype for columns. The SQL type should
            be a SQLAlchemy type. If all columns are of the same type, one
            single value can be used.
        method : {None', 'multi', callable}, default None
            Controls the SQL insertion clause used:

            * None : Uses standard SQL ``INSERT`` clause (one per row).
            * 'multi': Pass multiple values in a single ``INSERT`` clause.
            * callable with signature ``(pd_table, conn, keys, data_iter)``.

            Details and a sample callable implementation can be found in the
            section :ref:`insert method <io.sql.method>`.
        engine : {'auto', 'sqlalchemy'}, default 'auto'
            SQL engine library to use. If 'auto', then the option
            ``io.sql.engine`` is used. The default ``io.sql.engine``
            behavior is 'sqlalchemy'

            .. versionadded:: 1.3.0

        **engine_kwargs
            Any additional kwargs are passed to the engine.
        """
    def has_table(self, name: str, schema: str | None) -> bool: ...
    def get_table(self, table_name: str, schema: str | None) -> Table: ...
    def drop_table(self, table_name: str, schema: str | None) -> None: ...
    def _create_sql_schema(self, frame: DataFrame, table_name: str, keys: list[str] | None, dtype: DtypeArg | None, schema: str | None) -> str: ...
    @property
    def tables(self): ...

class ADBCDatabase(PandasSQL):
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    def __init__(self, con) -> None: ...
    def run_transaction(self, *args, **kwds): ...
    def execute(self, sql: str | Select | TextClause, params): ...
    def read_table(self, table_name: str, index_col: str | list[str] | None, coerce_float: bool = ..., parse_dates, columns, schema: str | None, chunksize: int | None, dtype_backend: DtypeBackend | Literal['numpy'] = ...) -> DataFrame | Iterator[DataFrame]:
        '''
        Read SQL database table into a DataFrame.

        Parameters
        ----------
        table_name : str
            Name of SQL table in database.
        coerce_float : bool, default True
            Raises NotImplementedError
        parse_dates : list or dict, default: None
            - List of column names to parse as dates.
            - Dict of ``{column_name: format string}`` where format string is
              strftime compatible in case of parsing string times, or is one of
              (D, s, ns, ms, us) in case of parsing integer timestamps.
            - Dict of ``{column_name: arg}``, where the arg corresponds
              to the keyword arguments of :func:`pandas.to_datetime`.
              Especially useful with databases without native Datetime support,
              such as SQLite.
        columns : list, default: None
            List of column names to select from SQL table.
        schema : string, default None
            Name of SQL schema in database to query (if database flavor
            supports this).  If specified, this overwrites the default
            schema of the SQL database object.
        chunksize : int, default None
            Raises NotImplementedError
        dtype_backend : {\'numpy_nullable\', \'pyarrow\'}, default \'numpy_nullable\'
            Back-end data type applied to the resultant :class:`DataFrame`
            (still experimental). Behaviour is as follows:

            * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
              (default).
            * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype`
              DataFrame.

            .. versionadded:: 2.0

        Returns
        -------
        DataFrame

        See Also
        --------
        pandas.read_sql_table
        SQLDatabase.read_query

        '''
    def read_query(self, sql: str, index_col: str | list[str] | None, coerce_float: bool = ..., parse_dates, params, chunksize: int | None, dtype: DtypeArg | None, dtype_backend: DtypeBackend | Literal['numpy'] = ...) -> DataFrame | Iterator[DataFrame]:
        """
        Read SQL query into a DataFrame.

        Parameters
        ----------
        sql : str
            SQL query to be executed.
        index_col : string, optional, default: None
            Column name to use as index for the returned DataFrame object.
        coerce_float : bool, default True
            Raises NotImplementedError
        params : list, tuple or dict, optional, default: None
            Raises NotImplementedError
        parse_dates : list or dict, default: None
            - List of column names to parse as dates.
            - Dict of ``{column_name: format string}`` where format string is
              strftime compatible in case of parsing string times, or is one of
              (D, s, ns, ms, us) in case of parsing integer timestamps.
            - Dict of ``{column_name: arg dict}``, where the arg dict
              corresponds to the keyword arguments of
              :func:`pandas.to_datetime` Especially useful with databases
              without native Datetime support, such as SQLite.
        chunksize : int, default None
            Raises NotImplementedError
        dtype : Type name or dict of columns
            Data type for data or columns. E.g. np.float64 or
            {'a': np.float64, 'b': np.int32, 'c': 'Int64'}

            .. versionadded:: 1.3.0

        Returns
        -------
        DataFrame

        See Also
        --------
        read_sql_table : Read SQL database table into a DataFrame.
        read_sql

        """
    def read_sql(self, sql: str, index_col: str | list[str] | None, coerce_float: bool = ..., parse_dates, params, chunksize: int | None, dtype: DtypeArg | None, dtype_backend: DtypeBackend | Literal['numpy'] = ...) -> DataFrame | Iterator[DataFrame]:
        """
        Read SQL query into a DataFrame.

        Parameters
        ----------
        sql : str
            SQL query to be executed.
        index_col : string, optional, default: None
            Column name to use as index for the returned DataFrame object.
        coerce_float : bool, default True
            Raises NotImplementedError
        params : list, tuple or dict, optional, default: None
            Raises NotImplementedError
        parse_dates : list or dict, default: None
            - List of column names to parse as dates.
            - Dict of ``{column_name: format string}`` where format string is
              strftime compatible in case of parsing string times, or is one of
              (D, s, ns, ms, us) in case of parsing integer timestamps.
            - Dict of ``{column_name: arg dict}``, where the arg dict
              corresponds to the keyword arguments of
              :func:`pandas.to_datetime` Especially useful with databases
              without native Datetime support, such as SQLite.
        chunksize : int, default None
            Raises NotImplementedError
        dtype : Type name or dict of columns
            Data type for data or columns. E.g. np.float64 or
            {'a': np.float64, 'b': np.int32, 'c': 'Int64'}

            .. versionadded:: 1.3.0

        Returns
        -------
        DataFrame

        See Also
        --------
        read_sql_table : Read SQL database table into a DataFrame.
        read_sql

        """
    def to_sql(self, frame, name: str, if_exists: Literal['fail', 'replace', 'append'] = ..., index: bool = ..., index_label, schema: str | None, chunksize: int | None, dtype: DtypeArg | None, method: Literal['multi'] | Callable | None, engine: str = ..., **engine_kwargs) -> int | None:
        """
        Write records stored in a DataFrame to a SQL database.

        Parameters
        ----------
        frame : DataFrame
        name : string
            Name of SQL table.
        if_exists : {'fail', 'replace', 'append'}, default 'fail'
            - fail: If table exists, do nothing.
            - replace: If table exists, drop it, recreate it, and insert data.
            - append: If table exists, insert data. Create if does not exist.
        index : boolean, default True
            Write DataFrame index as a column.
        index_label : string or sequence, default None
            Raises NotImplementedError
        schema : string, default None
            Name of SQL schema in database to write to (if database flavor
            supports this). If specified, this overwrites the default
            schema of the SQLDatabase object.
        chunksize : int, default None
            Raises NotImplementedError
        dtype : single type or dict of column name to SQL type, default None
            Raises NotImplementedError
        method : {None', 'multi', callable}, default None
            Raises NotImplementedError
        engine : {'auto', 'sqlalchemy'}, default 'auto'
            Raises NotImplementedError if not set to 'auto'
        """
    def has_table(self, name: str, schema: str | None) -> bool: ...
    def _create_sql_schema(self, frame: DataFrame, table_name: str, keys: list[str] | None, dtype: DtypeArg | None, schema: str | None) -> str: ...
_SQL_TYPES: dict
def _get_unicode_name(name: object): ...
def _get_valid_sqlite_name(name: object): ...

class SQLiteTable(SQLTable):
    def __init__(self, *args, **kwargs) -> None: ...
    def _register_date_adapters(self) -> None: ...
    def sql_schema(self) -> str: ...
    def _execute_create(self) -> None: ...
    def insert_statement(self, *, num_rows: int) -> str: ...
    def _execute_insert(self, conn, keys, data_iter) -> int: ...
    def _execute_insert_multi(self, conn, keys, data_iter) -> int: ...
    def _create_table_setup(self):
        """
        Return a list of SQL statements that creates a table reflecting the
        structure of a DataFrame.  The first entry will be a CREATE TABLE
        statement while the rest will be CREATE INDEX statements.
        """
    def _sql_type_name(self, col): ...

class SQLiteDatabase(PandasSQL):
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    def __init__(self, con) -> None: ...
    def run_transaction(self, *args, **kwds): ...
    def execute(self, sql: str | Select | TextClause, params): ...
    @staticmethod
    def _query_iterator(cursor, chunksize: int, columns, index_col, coerce_float: bool = ..., parse_dates, dtype: DtypeArg | None, dtype_backend: DtypeBackend | Literal['numpy'] = ...):
        """Return generator through chunked result set"""
    def read_query(self, sql, index_col, coerce_float: bool = ..., parse_dates, params, chunksize: int | None, dtype: DtypeArg | None, dtype_backend: DtypeBackend | Literal['numpy'] = ...) -> DataFrame | Iterator[DataFrame]: ...
    def _fetchall_as_list(self, cur): ...
    def to_sql(self, frame, name: str, if_exists: str = ..., index: bool = ..., index_label, schema, chunksize: int | None, dtype: DtypeArg | None, method: Literal['multi'] | Callable | None, engine: str = ..., **engine_kwargs) -> int | None:
        """
        Write records stored in a DataFrame to a SQL database.

        Parameters
        ----------
        frame: DataFrame
        name: string
            Name of SQL table.
        if_exists: {'fail', 'replace', 'append'}, default 'fail'
            fail: If table exists, do nothing.
            replace: If table exists, drop it, recreate it, and insert data.
            append: If table exists, insert data. Create if it does not exist.
        index : bool, default True
            Write DataFrame index as a column
        index_label : string or sequence, default None
            Column label for index column(s). If None is given (default) and
            `index` is True, then the index names are used.
            A sequence should be given if the DataFrame uses MultiIndex.
        schema : string, default None
            Ignored parameter included for compatibility with SQLAlchemy
            version of ``to_sql``.
        chunksize : int, default None
            If not None, then rows will be written in batches of this
            size at a time. If None, all rows will be written at once.
        dtype : single type or dict of column name to SQL type, default None
            Optional specifying the datatype for columns. The SQL type should
            be a string. If all columns are of the same type, one single value
            can be used.
        method : {None, 'multi', callable}, default None
            Controls the SQL insertion clause used:

            * None : Uses standard SQL ``INSERT`` clause (one per row).
            * 'multi': Pass multiple values in a single ``INSERT`` clause.
            * callable with signature ``(pd_table, conn, keys, data_iter)``.

            Details and a sample callable implementation can be found in the
            section :ref:`insert method <io.sql.method>`.
        """
    def has_table(self, name: str, schema: str | None) -> bool: ...
    def get_table(self, table_name: str, schema: str | None) -> None: ...
    def drop_table(self, name: str, schema: str | None) -> None: ...
    def _create_sql_schema(self, frame, table_name: str, keys, dtype: DtypeArg | None, schema: str | None) -> str: ...
def get_schema(frame, name: str, keys, con, dtype: DtypeArg | None, schema: str | None) -> str:
    """
    Get the SQL db table schema for the given frame.

    Parameters
    ----------
    frame : DataFrame
    name : str
        name of SQL table
    keys : string or sequence, default: None
        columns to use a primary key
    con: ADBC Connection, SQLAlchemy connectable, sqlite3 connection, default: None
        ADBC provides high performance I/O with native type support, where available.
        Using SQLAlchemy makes it possible to use any DB supported by that
        library
        If a DBAPI2 object, only sqlite3 is supported.
    dtype : dict of column name to SQL type, default None
        Optional specifying the datatype for columns. The SQL type should
        be a SQLAlchemy type, or a string for sqlite3 fallback connection.
    schema: str, default: None
        Optional specifying the schema to be used in creating the table.
    """
