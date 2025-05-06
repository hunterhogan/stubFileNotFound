import abc
import numpy as np
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterator, Mapping
from contextlib import ExitStack
from pandas import Index as Index, get_option as get_option
from pandas._libs import lib as lib
from pandas._typing import DateTimeErrorChoices as DateTimeErrorChoices, DtypeArg as DtypeArg, DtypeBackend as DtypeBackend, IndexLabel as IndexLabel, Self as Self
from pandas.core.api import DataFrame as DataFrame, Series as Series
from pandas.core.base import PandasObject as PandasObject
from pandas.core.dtypes.common import is_dict_like as is_dict_like, is_list_like as is_list_like
from pandas.core.dtypes.dtypes import ArrowDtype as ArrowDtype, DatetimeTZDtype as DatetimeTZDtype
from pandas.errors import AbstractMethodError as AbstractMethodError, DatabaseError as DatabaseError
from sqlalchemy import Table
from sqlalchemy.sql.expression import Select as Select, TextClause as TextClause
from typing import Any, Literal, overload

from collections.abc import Callable

def _process_parse_dates_argument(parse_dates):
    """Process parse_dates argument for read_sql functions"""
def _handle_date_column(col, utc: bool = False, format: str | dict[str, Any] | None = None): ...
def _parse_date_columns(data_frame, parse_dates):
    """
    Force non-datetime columns to be read as such.
    Supports both string formatted and integer timestamp columns.
    """
def _convert_arrays_to_dataframe(data, columns, coerce_float: bool = True, dtype_backend: DtypeBackend | Literal['numpy'] = 'numpy') -> DataFrame: ...
def _wrap_result(data, columns, index_col: Incomplete | None = None, coerce_float: bool = True, parse_dates: Incomplete | None = None, dtype: DtypeArg | None = None, dtype_backend: DtypeBackend | Literal['numpy'] = 'numpy'):
    """Wrap result set of a SQLAlchemy query in a DataFrame."""
def _wrap_result_adbc(df: DataFrame, *, index_col: Incomplete | None = None, parse_dates: Incomplete | None = None, dtype: DtypeArg | None = None, dtype_backend: DtypeBackend | Literal['numpy'] = 'numpy') -> DataFrame:
    """Wrap result set of a SQLAlchemy query in a DataFrame."""
def execute(sql, con, params: Incomplete | None = None):
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
@overload
def read_sql_table(table_name: str, con, schema=..., index_col: str | list[str] | None = ..., coerce_float=..., parse_dates: list[str] | dict[str, str] | None = ..., columns: list[str] | None = ..., chunksize: None = None, dtype_backend: DtypeBackend | lib.NoDefault = ...) -> DataFrame: ...
@overload
def read_sql_table(table_name: str, con, schema=..., index_col: str | list[str] | None = ..., coerce_float=..., parse_dates: list[str] | dict[str, str] | None = ..., columns: list[str] | None = ..., chunksize: int = ..., dtype_backend: DtypeBackend | lib.NoDefault = ...) -> Iterator[DataFrame]: ...
@overload
def read_sql_query(sql, con, index_col: str | list[str] | None = ..., coerce_float=..., params: list[Any] | Mapping[str, Any] | None = ..., parse_dates: list[str] | dict[str, str] | None = ..., chunksize: None = None, dtype: DtypeArg | None = ..., dtype_backend: DtypeBackend | lib.NoDefault = ...) -> DataFrame: ...
@overload
def read_sql_query(sql, con, index_col: str | list[str] | None = ..., coerce_float=..., params: list[Any] | Mapping[str, Any] | None = ..., parse_dates: list[str] | dict[str, str] | None = ..., chunksize: int = ..., dtype: DtypeArg | None = ..., dtype_backend: DtypeBackend | lib.NoDefault = ...) -> Iterator[DataFrame]: ...
@overload
def read_sql(sql, con, index_col: str | list[str] | None = ..., coerce_float=..., params=..., parse_dates=..., columns: list[str] = ..., chunksize: None = None, dtype_backend: DtypeBackend | lib.NoDefault = ..., dtype: DtypeArg | None = None) -> DataFrame: ...
@overload
def read_sql(sql, con, index_col: str | list[str] | None = ..., coerce_float=..., params=..., parse_dates=..., columns: list[str] = ..., chunksize: int = ..., dtype_backend: DtypeBackend | lib.NoDefault = ..., dtype: DtypeArg | None = None) -> Iterator[DataFrame]: ...
def to_sql(frame, name: str, con, schema: str | None = None, if_exists: Literal['fail', 'replace', 'append'] = 'fail', index: bool = True, index_label: IndexLabel | None = None, chunksize: int | None = None, dtype: DtypeArg | None = None, method: Literal['multi'] | Callable | None = None, engine: str = 'auto', **engine_kwargs) -> int | None:
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
def has_table(table_name: str, con, schema: str | None = None) -> bool:
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
table_exists = has_table

def pandasSQL_builder(con, schema: str | None = None, need_transaction: bool = False) -> PandasSQL:
    """
    Convenience function to return the correct PandasSQL subclass based on the
    provided parameters.  Also creates a sqlalchemy connection and transaction
    if necessary.
    """

class SQLTable(PandasObject):
    """
    For mapping Pandas tables to SQL tables.
    Uses fact that table is reflected by SQLAlchemy to
    do better type conversions.
    Also holds various flags needed to avoid having to
    pass them between functions all the time.
    """
    name: Incomplete
    pd_sql: Incomplete
    prefix: Incomplete
    frame: Incomplete
    index: Incomplete
    schema: Incomplete
    if_exists: Incomplete
    keys: Incomplete
    dtype: Incomplete
    table: Incomplete
    def __init__(self, name: str, pandas_sql_engine, frame: Incomplete | None = None, index: bool | str | list[str] | None = True, if_exists: Literal['fail', 'replace', 'append'] = 'fail', prefix: str = 'pandas', index_label: Incomplete | None = None, schema: Incomplete | None = None, keys: Incomplete | None = None, dtype: DtypeArg | None = None) -> None: ...
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
    def insert(self, chunksize: int | None = None, method: Literal['multi'] | Callable | None = None) -> int | None: ...
    def _query_iterator(self, result, exit_stack: ExitStack, chunksize: int | None, columns, coerce_float: bool = True, parse_dates: Incomplete | None = None, dtype_backend: DtypeBackend | Literal['numpy'] = 'numpy'):
        """Return generator through chunked result set."""
    def read(self, exit_stack: ExitStack, coerce_float: bool = True, parse_dates: Incomplete | None = None, columns: Incomplete | None = None, chunksize: int | None = None, dtype_backend: DtypeBackend | Literal['numpy'] = 'numpy') -> DataFrame | Iterator[DataFrame]: ...
    def _index_name(self, index, index_label): ...
    def _get_column_names_and_types(self, dtype_mapper): ...
    def _create_table_setup(self): ...
    def _harmonize_columns(self, parse_dates: Incomplete | None = None, dtype_backend: DtypeBackend | Literal['numpy'] = 'numpy') -> None:
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

class PandasSQL(PandasObject, ABC, metaclass=abc.ABCMeta):
    """
    Subclasses Should define read_query and to_sql.
    """
    def __enter__(self) -> Self: ...
    def __exit__(self, *args) -> None: ...
    def read_table(self, table_name: str, index_col: str | list[str] | None = None, coerce_float: bool = True, parse_dates: Incomplete | None = None, columns: Incomplete | None = None, schema: str | None = None, chunksize: int | None = None, dtype_backend: DtypeBackend | Literal['numpy'] = 'numpy') -> DataFrame | Iterator[DataFrame]: ...
    @abstractmethod
    def read_query(self, sql: str, index_col: str | list[str] | None = None, coerce_float: bool = True, parse_dates: Incomplete | None = None, params: Incomplete | None = None, chunksize: int | None = None, dtype: DtypeArg | None = None, dtype_backend: DtypeBackend | Literal['numpy'] = 'numpy') -> DataFrame | Iterator[DataFrame]: ...
    @abstractmethod
    def to_sql(self, frame, name: str, if_exists: Literal['fail', 'replace', 'append'] = 'fail', index: bool = True, index_label: Incomplete | None = None, schema: Incomplete | None = None, chunksize: int | None = None, dtype: DtypeArg | None = None, method: Literal['multi'] | Callable | None = None, engine: str = 'auto', **engine_kwargs) -> int | None: ...
    @abstractmethod
    def execute(self, sql: str | Select | TextClause, params: Incomplete | None = None): ...
    @abstractmethod
    def has_table(self, name: str, schema: str | None = None) -> bool: ...
    @abstractmethod
    def _create_sql_schema(self, frame: DataFrame, table_name: str, keys: list[str] | None = None, dtype: DtypeArg | None = None, schema: str | None = None) -> str: ...

class BaseEngine:
    def insert_records(self, table: SQLTable, con, frame, name: str, index: bool | str | list[str] | None = True, schema: Incomplete | None = None, chunksize: int | None = None, method: Incomplete | None = None, **engine_kwargs) -> int | None:
        """
        Inserts data into already-prepared table
        """

class SQLAlchemyEngine(BaseEngine):
    def __init__(self) -> None: ...
    def insert_records(self, table: SQLTable, con, frame, name: str, index: bool | str | list[str] | None = True, schema: Incomplete | None = None, chunksize: int | None = None, method: Incomplete | None = None, **engine_kwargs) -> int | None: ...

def get_engine(engine: str) -> BaseEngine:
    """return our implementation"""

class SQLDatabase(PandasSQL):
    """
    This class enables conversion between DataFrame and SQL databases
    using SQLAlchemy to handle DataBase abstraction.

    Parameters
    ----------
    con : SQLAlchemy Connectable or URI string.
        Connectable to connect with the database. Using SQLAlchemy makes it
        possible to use any DB supported by that library.
    schema : string, default None
        Name of SQL schema in database to write to (if database flavor
        supports this). If None, use default schema (default).
    need_transaction : bool, default False
        If True, SQLDatabase will create a transaction.

    """
    exit_stack: Incomplete
    con: Incomplete
    meta: Incomplete
    returns_generator: bool
    def __init__(self, con, schema: str | None = None, need_transaction: bool = False) -> None: ...
    def __exit__(self, *args) -> None: ...
    def run_transaction(self) -> Generator[Incomplete]: ...
    def execute(self, sql: str | Select | TextClause, params: Incomplete | None = None):
        """Simple passthrough to SQLAlchemy connectable"""
    def read_table(self, table_name: str, index_col: str | list[str] | None = None, coerce_float: bool = True, parse_dates: Incomplete | None = None, columns: Incomplete | None = None, schema: str | None = None, chunksize: int | None = None, dtype_backend: DtypeBackend | Literal['numpy'] = 'numpy') -> DataFrame | Iterator[DataFrame]:
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
    def _query_iterator(result, exit_stack: ExitStack, chunksize: int, columns, index_col: Incomplete | None = None, coerce_float: bool = True, parse_dates: Incomplete | None = None, dtype: DtypeArg | None = None, dtype_backend: DtypeBackend | Literal['numpy'] = 'numpy'):
        """Return generator through chunked result set"""
    def read_query(self, sql: str, index_col: str | list[str] | None = None, coerce_float: bool = True, parse_dates: Incomplete | None = None, params: Incomplete | None = None, chunksize: int | None = None, dtype: DtypeArg | None = None, dtype_backend: DtypeBackend | Literal['numpy'] = 'numpy') -> DataFrame | Iterator[DataFrame]:
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
    read_sql = read_query
    def prep_table(self, frame, name: str, if_exists: Literal['fail', 'replace', 'append'] = 'fail', index: bool | str | list[str] | None = True, index_label: Incomplete | None = None, schema: Incomplete | None = None, dtype: DtypeArg | None = None) -> SQLTable:
        """
        Prepares table in the database for data insertion. Creates it if needed, etc.
        """
    def check_case_sensitive(self, name: str, schema: str | None) -> None:
        """
        Checks table name for issues with case-sensitivity.
        Method is called after data is inserted.
        """
    def to_sql(self, frame, name: str, if_exists: Literal['fail', 'replace', 'append'] = 'fail', index: bool = True, index_label: Incomplete | None = None, schema: str | None = None, chunksize: int | None = None, dtype: DtypeArg | None = None, method: Literal['multi'] | Callable | None = None, engine: str = 'auto', **engine_kwargs) -> int | None:
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
    @property
    def tables(self): ...
    def has_table(self, name: str, schema: str | None = None) -> bool: ...
    def get_table(self, table_name: str, schema: str | None = None) -> Table: ...
    def drop_table(self, table_name: str, schema: str | None = None) -> None: ...
    def _create_sql_schema(self, frame: DataFrame, table_name: str, keys: list[str] | None = None, dtype: DtypeArg | None = None, schema: str | None = None) -> str: ...

class ADBCDatabase(PandasSQL):
    """
    This class enables conversion between DataFrame and SQL databases
    using ADBC to handle DataBase abstraction.

    Parameters
    ----------
    con : adbc_driver_manager.dbapi.Connection
    """
    con: Incomplete
    def __init__(self, con) -> None: ...
    def run_transaction(self) -> Generator[Incomplete]: ...
    def execute(self, sql: str | Select | TextClause, params: Incomplete | None = None): ...
    def read_table(self, table_name: str, index_col: str | list[str] | None = None, coerce_float: bool = True, parse_dates: Incomplete | None = None, columns: Incomplete | None = None, schema: str | None = None, chunksize: int | None = None, dtype_backend: DtypeBackend | Literal['numpy'] = 'numpy') -> DataFrame | Iterator[DataFrame]:
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
    def read_query(self, sql: str, index_col: str | list[str] | None = None, coerce_float: bool = True, parse_dates: Incomplete | None = None, params: Incomplete | None = None, chunksize: int | None = None, dtype: DtypeArg | None = None, dtype_backend: DtypeBackend | Literal['numpy'] = 'numpy') -> DataFrame | Iterator[DataFrame]:
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
    read_sql = read_query
    def to_sql(self, frame, name: str, if_exists: Literal['fail', 'replace', 'append'] = 'fail', index: bool = True, index_label: Incomplete | None = None, schema: str | None = None, chunksize: int | None = None, dtype: DtypeArg | None = None, method: Literal['multi'] | Callable | None = None, engine: str = 'auto', **engine_kwargs) -> int | None:
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
    def has_table(self, name: str, schema: str | None = None) -> bool: ...
    def _create_sql_schema(self, frame: DataFrame, table_name: str, keys: list[str] | None = None, dtype: DtypeArg | None = None, schema: str | None = None) -> str: ...

_SQL_TYPES: Incomplete

def _get_unicode_name(name: object): ...
def _get_valid_sqlite_name(name: object): ...

class SQLiteTable(SQLTable):
    """
    Patch the SQLTable for fallback support.
    Instead of a table variable just use the Create Table statement.
    """
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
    """
    Version of SQLDatabase to support SQLite connections (fallback without
    SQLAlchemy). This should only be used internally.

    Parameters
    ----------
    con : sqlite connection object

    """
    con: Incomplete
    def __init__(self, con) -> None: ...
    def run_transaction(self) -> Generator[Incomplete]: ...
    def execute(self, sql: str | Select | TextClause, params: Incomplete | None = None): ...
    @staticmethod
    def _query_iterator(cursor, chunksize: int, columns, index_col: Incomplete | None = None, coerce_float: bool = True, parse_dates: Incomplete | None = None, dtype: DtypeArg | None = None, dtype_backend: DtypeBackend | Literal['numpy'] = 'numpy'):
        """Return generator through chunked result set"""
    def read_query(self, sql, index_col: Incomplete | None = None, coerce_float: bool = True, parse_dates: Incomplete | None = None, params: Incomplete | None = None, chunksize: int | None = None, dtype: DtypeArg | None = None, dtype_backend: DtypeBackend | Literal['numpy'] = 'numpy') -> DataFrame | Iterator[DataFrame]: ...
    def _fetchall_as_list(self, cur): ...
    def to_sql(self, frame, name: str, if_exists: str = 'fail', index: bool = True, index_label: Incomplete | None = None, schema: Incomplete | None = None, chunksize: int | None = None, dtype: DtypeArg | None = None, method: Literal['multi'] | Callable | None = None, engine: str = 'auto', **engine_kwargs) -> int | None:
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
    def has_table(self, name: str, schema: str | None = None) -> bool: ...
    def get_table(self, table_name: str, schema: str | None = None) -> None: ...
    def drop_table(self, name: str, schema: str | None = None) -> None: ...
    def _create_sql_schema(self, frame, table_name: str, keys: Incomplete | None = None, dtype: DtypeArg | None = None, schema: str | None = None) -> str: ...

def get_schema(frame, name: str, keys: Incomplete | None = None, con: Incomplete | None = None, dtype: DtypeArg | None = None, schema: str | None = None) -> str:
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
