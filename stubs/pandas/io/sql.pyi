from collections.abc import (
    Callable,
    Generator,
    Iterable,
    Mapping,
)
import sqlite3
from typing import (
    Any,
    Literal,
    overload,
)

from pandas.core.frame import DataFrame
import sqlalchemy.engine
from sqlalchemy.orm import FromStatement
import sqlalchemy.sql.expression
from typing_extensions import TypeAlias

from pandas._libs.lib import _NoDefaultDoNotUse
from pandas._typing import (
    DtypeArg,
    DtypeBackend,
    Scalar,
    npt,
)

_SQLConnection: TypeAlias = str | sqlalchemy.engine.Connectable | sqlite3.Connection

_SQLStatement: TypeAlias = (
    str
    | sqlalchemy.sql.expression.Selectable
    | sqlalchemy.sql.expression.TextClause
    | sqlalchemy.sql.Select
    | FromStatement
    | sqlalchemy.sql.expression.UpdateBase
)

@overload
def read_sql_table(
    table_name: str,
    con: _SQLConnection,
    schema: str | None = None,
    index_col: str | list[str] | None = None,
    coerce_float: bool = True,
    parse_dates: list[str] | dict[str, str] | dict[str, dict[str, Any]] | None = None,
    columns: list[str] | None = None,
    *,
    chunksize: int,
    dtype_backend: DtypeBackend | _NoDefaultDoNotUse = ...,
) -> Generator[DataFrame, None, None]: ...
@overload
def read_sql_table(
    table_name: str,
    con: _SQLConnection,
    schema: str | None = None,
    index_col: str | list[str] | None = None,
    coerce_float: bool = True,
    parse_dates: list[str] | dict[str, str] | dict[str, dict[str, Any]] | None = None,
    columns: list[str] | None = None,
    chunksize: None = None,
    dtype_backend: DtypeBackend | _NoDefaultDoNotUse = ...,
) -> DataFrame: ...
@overload
def read_sql_query(
    sql: _SQLStatement,
    con: _SQLConnection,
    index_col: str | list[str] | None = None,
    coerce_float: bool = True,
    params: (
        list[Scalar]
        | tuple[Scalar, ...]
        | tuple[tuple[Scalar, ...], ...]
        | Mapping[str, Scalar]
        | Mapping[str, tuple[Scalar, ...]]
        | None
    ) = None,
    parse_dates: list[str] | dict[str, str] | dict[str, dict[str, Any]] | None = None,
    *,
    chunksize: int,
    dtype: DtypeArg | None = None,
    dtype_backend: DtypeBackend | _NoDefaultDoNotUse = ...,
) -> Generator[DataFrame, None, None]: ...
@overload
def read_sql_query(
    sql: _SQLStatement,
    con: _SQLConnection,
    index_col: str | list[str] | None = None,
    coerce_float: bool = True,
    params: (
        list[Scalar]
        | tuple[Scalar, ...]
        | tuple[tuple[Scalar, ...], ...]
        | Mapping[str, Scalar]
        | Mapping[str, tuple[Scalar, ...]]
        | None
    ) = None,
    parse_dates: list[str] | dict[str, str] | dict[str, dict[str, Any]] | None = None,
    chunksize: None = None,
    dtype: DtypeArg | None = None,
    dtype_backend: DtypeBackend | _NoDefaultDoNotUse = ...,
) -> DataFrame: ...
@overload
def read_sql(
    sql: _SQLStatement,
    con: _SQLConnection,
    index_col: str | list[str] | None = None,
    coerce_float: bool = True,
    params: (
        list[Scalar]
        | tuple[Scalar, ...]
        | tuple[tuple[Scalar, ...], ...]
        | Mapping[str, Scalar]
        | Mapping[str, tuple[Scalar, ...]]
        | None
    ) = None,
    parse_dates: list[str] | dict[str, str] | dict[str, dict[str, Any]] | None = None,
    columns: list[str] = None,
    *,
    chunksize: int,
    dtype: DtypeArg | None = None,
    dtype_backend: DtypeBackend | _NoDefaultDoNotUse = ...,
) -> Generator[DataFrame, None, None]: ...
@overload
def read_sql(
    sql: _SQLStatement,
    con: _SQLConnection,
    index_col: str | list[str] | None = None,
    coerce_float: bool = True,
    params: (
        list[Scalar]
        | tuple[Scalar, ...]
        | tuple[tuple[Scalar, ...], ...]
        | Mapping[str, Scalar]
        | Mapping[str, tuple[Scalar, ...]]
        | None
    ) = None,
    parse_dates: list[str] | dict[str, str] | dict[str, dict[str, Any]] | None = None,
    columns: list[str] = None,
    chunksize: None = None,
    dtype: DtypeArg | None = None,
    dtype_backend: DtypeBackend | _NoDefaultDoNotUse = ...,
) -> DataFrame: ...

class PandasSQL:
    def to_sql(
        self,
        frame: DataFrame,
        name: str,
        if_exists: Literal["fail", "replace", "append"] = 'fail',
        index: bool = True,
        index_label: Any=None,
        schema: str | None = None,
        chunksize: Any=None,
        dtype: DtypeArg | None = None,
        method: (
            Literal["multi"]
            | Callable[[SQLTable, Any, list[str], Iterable[Any]], int | None]
            | None
        ) = None,
        engine: str = 'auto',
        **engine_kwargs: dict[str, Any] | None,
    ) -> int | None: ...

class SQLTable:
    name: str
    pd_sql: PandasSQL  # pandas SQL interface
    prefix: str
    frame: DataFrame | None
    index: list[str]
    schema: str
    if_exists: Literal["fail", "replace", "append"]
    keys: list[str]
    dtype: DtypeArg | None
    table: Any  # sqlalchemy.Table
    def __init__(
        self,
        name: str,
        pandas_sql_engine: PandasSQL,
        frame: DataFrame | None = None,
        index: bool | str | list[str] | None = True,
        if_exists: Literal["fail", "replace", "append"] = 'fail',
        prefix: str = 'pandas',
        index_label: str | list[str] | None = None,
        schema: str | None = None,
        keys: str | list[str] | None = None,
        dtype: DtypeArg | None = None,
    ) -> None: ...
    def exists(self) -> bool: ...
    def sql_schema(self) -> str: ...
    def create(self) -> None: ...
    def insert_data(self) -> tuple[list[str], list[npt.NDArray]]: ...
    def insert(
        self, chunksize: int | None = None, method: str | None = None
    ) -> int | None: ...
    def read(
        self,
        coerce_float: bool = True,
        parse_dates: bool | list[str] | None = None,
        columns: list[str] | None = None,
        chunksize: int | None = None,
    ) -> DataFrame | Generator[DataFrame, None, None]: ...
