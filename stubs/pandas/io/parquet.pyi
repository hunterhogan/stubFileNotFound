import lib as lib
import pandas as pd
from pandas._config import using_pyarrow_string_dtype as using_pyarrow_string_dtype
from pandas._config.config import _get_option as _get_option, get_option as get_option
from pandas.compat._optional import import_optional_dependency as import_optional_dependency
from pandas.core.frame import DataFrame as DataFrame
from pandas.errors import AbstractMethodError as AbstractMethodError
from pandas.io._util import arrow_string_types_mapper as arrow_string_types_mapper
from pandas.io.common import IOHandles as IOHandles, get_handle as get_handle, is_fsspec_url as is_fsspec_url, is_url as is_url, stringify_path as stringify_path
from pandas.util._decorators import doc as doc
from pandas.util._exceptions import find_stack_level as find_stack_level
from pandas.util._validators import check_dtype_backend as check_dtype_backend
from typing import Any, Literal

TYPE_CHECKING: bool
_shared_docs: dict
def get_engine(engine: str) -> BaseImpl:
    """return our implementation"""
def _get_path_or_handle(path: FilePath | ReadBuffer[bytes] | WriteBuffer[bytes], fs: Any, storage_options: StorageOptions | None, mode: str = ..., is_dir: bool = ...) -> tuple[FilePath | ReadBuffer[bytes] | WriteBuffer[bytes], IOHandles[bytes] | None, Any]:
    """File handling for PyArrow."""

class BaseImpl:
    @staticmethod
    def validate_dataframe(df: DataFrame) -> None: ...
    def write(self, df: DataFrame, path, compression, **kwargs): ...
    def read(self, path, columns, **kwargs) -> DataFrame: ...

class PyArrowImpl(BaseImpl):
    def __init__(self) -> None: ...
    def write(self, df: DataFrame, path: FilePath | WriteBuffer[bytes], compression: str | None = ..., index: bool | None, storage_options: StorageOptions | None, partition_cols: list[str] | None, filesystem, **kwargs) -> None: ...
    def read(self, path, columns, filters, use_nullable_dtypes: bool = ..., dtype_backend: DtypeBackend | lib.NoDefault = ..., storage_options: StorageOptions | None, filesystem, **kwargs) -> DataFrame: ...

class FastParquetImpl(BaseImpl):
    def __init__(self) -> None: ...
    def write(self, df: DataFrame, path, compression: Literal['snappy', 'gzip', 'brotli'] | None = ..., index, partition_cols, storage_options: StorageOptions | None, filesystem, **kwargs) -> None: ...
    def read(self, path, columns, filters, storage_options: StorageOptions | None, filesystem, **kwargs) -> DataFrame: ...
def to_parquet(df: DataFrame, path: FilePath | WriteBuffer[bytes] | None, engine: str = ..., compression: str | None = ..., index: bool | None, storage_options: StorageOptions | None, partition_cols: list[str] | None, filesystem: Any, **kwargs) -> bytes | None:
    '''
    Write a DataFrame to the parquet format.

    Parameters
    ----------
    df : DataFrame
    path : str, path object, file-like object, or None, default None
        String, path object (implementing ``os.PathLike[str]``), or file-like
        object implementing a binary ``write()`` function. If None, the result is
        returned as bytes. If a string, it will be used as Root Directory path
        when writing a partitioned dataset. The engine fastparquet does not
        accept file-like objects.
    engine : {\'auto\', \'pyarrow\', \'fastparquet\'}, default \'auto\'
        Parquet library to use. If \'auto\', then the option
        ``io.parquet.engine`` is used. The default ``io.parquet.engine``
        behavior is to try \'pyarrow\', falling back to \'fastparquet\' if
        \'pyarrow\' is unavailable.

        When using the ``\'pyarrow\'`` engine and no storage options are provided
        and a filesystem is implemented by both ``pyarrow.fs`` and ``fsspec``
        (e.g. "s3://"), then the ``pyarrow.fs`` filesystem is attempted first.
        Use the filesystem keyword with an instantiated fsspec filesystem
        if you wish to use its implementation.
    compression : {\'snappy\', \'gzip\', \'brotli\', \'lz4\', \'zstd\', None},
        default \'snappy\'. Name of the compression to use. Use ``None``
        for no compression.
    index : bool, default None
        If ``True``, include the dataframe\'s index(es) in the file output. If
        ``False``, they will not be written to the file.
        If ``None``, similar to ``True`` the dataframe\'s index(es)
        will be saved. However, instead of being saved as values,
        the RangeIndex will be stored as a range in the metadata so it
        doesn\'t require much space and is faster. Other indexes will
        be included as columns in the file output.
    partition_cols : str or list, optional, default None
        Column names by which to partition the dataset.
        Columns are partitioned in the order they are given.
        Must be None if path is not a string.
    storage_options : dict, optional
        Extra options that make sense for a particular storage connection, e.g.
        host, port, username, password, etc. For HTTP(S) URLs the key-value pairs
        are forwarded to ``urllib.request.Request`` as header options. For other
        URLs (e.g. starting with "s3://", and "gcs://") the key-value pairs are
        forwarded to ``fsspec.open``. Please see ``fsspec`` and ``urllib`` for more
        details, and for more examples on storage options refer `here
        <https://pandas.pydata.org/docs/user_guide/io.html?
        highlight=storage_options#reading-writing-remote-files>`_.

    filesystem : fsspec or pyarrow filesystem, default None
        Filesystem object to use when reading the parquet file. Only implemented
        for ``engine="pyarrow"``.

        .. versionadded:: 2.1.0

    kwargs
        Additional keyword arguments passed to the engine

    Returns
    -------
    bytes if no path argument is provided else None
    '''
def read_parquet(path: FilePath | ReadBuffer[bytes], engine: str = ..., columns: list[str] | None, storage_options: StorageOptions | None, use_nullable_dtypes: bool | lib.NoDefault = ..., dtype_backend: DtypeBackend | lib.NoDefault = ..., filesystem: Any, filters: list[tuple] | list[list[tuple]] | None, **kwargs) -> DataFrame:
    '''
    Load a parquet object from the file path, returning a DataFrame.

    Parameters
    ----------
    path : str, path object or file-like object
        String, path object (implementing ``os.PathLike[str]``), or file-like
        object implementing a binary ``read()`` function.
        The string could be a URL. Valid URL schemes include http, ftp, s3,
        gs, and file. For file URLs, a host is expected. A local file could be:
        ``file://localhost/path/to/table.parquet``.
        A file URL can also be a path to a directory that contains multiple
        partitioned parquet files. Both pyarrow and fastparquet support
        paths to directories as well as file URLs. A directory path could be:
        ``file://localhost/path/to/tables`` or ``s3://bucket/partition_dir``.
    engine : {\'auto\', \'pyarrow\', \'fastparquet\'}, default \'auto\'
        Parquet library to use. If \'auto\', then the option
        ``io.parquet.engine`` is used. The default ``io.parquet.engine``
        behavior is to try \'pyarrow\', falling back to \'fastparquet\' if
        \'pyarrow\' is unavailable.

        When using the ``\'pyarrow\'`` engine and no storage options are provided
        and a filesystem is implemented by both ``pyarrow.fs`` and ``fsspec``
        (e.g. "s3://"), then the ``pyarrow.fs`` filesystem is attempted first.
        Use the filesystem keyword with an instantiated fsspec filesystem
        if you wish to use its implementation.
    columns : list, default=None
        If not None, only these columns will be read from the file.
    storage_options : dict, optional
        Extra options that make sense for a particular storage connection, e.g.
        host, port, username, password, etc. For HTTP(S) URLs the key-value pairs
        are forwarded to ``urllib.request.Request`` as header options. For other
        URLs (e.g. starting with "s3://", and "gcs://") the key-value pairs are
        forwarded to ``fsspec.open``. Please see ``fsspec`` and ``urllib`` for more
        details, and for more examples on storage options refer `here
        <https://pandas.pydata.org/docs/user_guide/io.html?
        highlight=storage_options#reading-writing-remote-files>`_.

        .. versionadded:: 1.3.0

    use_nullable_dtypes : bool, default False
        If True, use dtypes that use ``pd.NA`` as missing value indicator
        for the resulting DataFrame. (only applicable for the ``pyarrow``
        engine)
        As new dtypes are added that support ``pd.NA`` in the future, the
        output with this option will change to use those dtypes.
        Note: this is an experimental option, and behaviour (e.g. additional
        support dtypes) may change without notice.

        .. deprecated:: 2.0

    dtype_backend : {\'numpy_nullable\', \'pyarrow\'}, default \'numpy_nullable\'
        Back-end data type applied to the resultant :class:`DataFrame`
        (still experimental). Behaviour is as follows:

        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
          (default).
        * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype`
          DataFrame.

        .. versionadded:: 2.0

    filesystem : fsspec or pyarrow filesystem, default None
        Filesystem object to use when reading the parquet file. Only implemented
        for ``engine="pyarrow"``.

        .. versionadded:: 2.1.0

    filters : List[Tuple] or List[List[Tuple]], default None
        To filter out data.
        Filter syntax: [[(column, op, val), ...],...]
        where op is [==, =, >, >=, <, <=, !=, in, not in]
        The innermost tuples are transposed into a set of filters applied
        through an `AND` operation.
        The outer list combines these sets of filters through an `OR`
        operation.
        A single list of tuples can also be used, meaning that no `OR`
        operation between set of filters is to be conducted.

        Using this argument will NOT result in row-wise filtering of the final
        partitions unless ``engine="pyarrow"`` is also specified.  For
        other engines, filtering is only performed at the partition level, that is,
        to prevent the loading of some row-groups and/or files.

        .. versionadded:: 2.1.0

    **kwargs
        Any additional kwargs are passed to the engine.

    Returns
    -------
    DataFrame

    See Also
    --------
    DataFrame.to_parquet : Create a parquet object that serializes a DataFrame.

    Examples
    --------
    >>> original_df = pd.DataFrame(
    ...     {"foo": range(5), "bar": range(5, 10)}
    ...    )
    >>> original_df
       foo  bar
    0    0    5
    1    1    6
    2    2    7
    3    3    8
    4    4    9
    >>> df_parquet_bytes = original_df.to_parquet()
    >>> from io import BytesIO
    >>> restored_df = pd.read_parquet(BytesIO(df_parquet_bytes))
    >>> restored_df
       foo  bar
    0    0    5
    1    1    6
    2    2    7
    3    3    8
    4    4    9
    >>> restored_df.equals(original_df)
    True
    >>> restored_bar = pd.read_parquet(BytesIO(df_parquet_bytes), columns=["bar"])
    >>> restored_bar
        bar
    0    5
    1    6
    2    7
    3    8
    4    9
    >>> restored_bar.equals(original_df[[\'bar\']])
    True

    The function uses `kwargs` that are passed directly to the engine.
    In the following example, we use the `filters` argument of the pyarrow
    engine to filter the rows of the DataFrame.

    Since `pyarrow` is the default engine, we can omit the `engine` argument.
    Note that the `filters` argument is implemented by the `pyarrow` engine,
    which can benefit from multithreading and also potentially be more
    economical in terms of memory.

    >>> sel = [("foo", ">", 2)]
    >>> restored_part = pd.read_parquet(BytesIO(df_parquet_bytes), filters=sel)
    >>> restored_part
        foo  bar
    0    3    8
    1    4    9
    '''
