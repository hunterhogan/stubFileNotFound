import lib as lib
import pandas as pd
from pandas._config import using_pyarrow_string_dtype as using_pyarrow_string_dtype
from pandas.compat._optional import import_optional_dependency as import_optional_dependency
from pandas.core.frame import DataFrame as DataFrame
from pandas.io._util import arrow_string_types_mapper as arrow_string_types_mapper
from pandas.io.common import get_handle as get_handle
from pandas.util._decorators import doc as doc
from pandas.util._validators import check_dtype_backend as check_dtype_backend
from typing import Any

TYPE_CHECKING: bool
_shared_docs: dict
def to_feather(df: DataFrame, path: FilePath | WriteBuffer[bytes], storage_options: StorageOptions | None, **kwargs: Any) -> None:
    '''
    Write a DataFrame to the binary Feather format.

    Parameters
    ----------
    df : DataFrame
    path : str, path object, or file-like object
    storage_options : dict, optional
        Extra options that make sense for a particular storage connection, e.g.
        host, port, username, password, etc. For HTTP(S) URLs the key-value pairs
        are forwarded to ``urllib.request.Request`` as header options. For other
        URLs (e.g. starting with "s3://", and "gcs://") the key-value pairs are
        forwarded to ``fsspec.open``. Please see ``fsspec`` and ``urllib`` for more
        details, and for more examples on storage options refer `here
        <https://pandas.pydata.org/docs/user_guide/io.html?
        highlight=storage_options#reading-writing-remote-files>`_.
    **kwargs :
        Additional keywords passed to `pyarrow.feather.write_feather`.

    '''
def read_feather(path: FilePath | ReadBuffer[bytes], columns: Sequence[Hashable] | None, use_threads: bool = ..., storage_options: StorageOptions | None, dtype_backend: DtypeBackend | lib.NoDefault = ...) -> DataFrame:
    '''
    Load a feather-format object from the file path.

    Parameters
    ----------
    path : str, path object, or file-like object
        String, path object (implementing ``os.PathLike[str]``), or file-like
        object implementing a binary ``read()`` function. The string could be a URL.
        Valid URL schemes include http, ftp, s3, and file. For file URLs, a host is
        expected. A local file could be: ``file://localhost/path/to/table.feather``.
    columns : sequence, default None
        If not provided, all columns are read.
    use_threads : bool, default True
        Whether to parallelize reading using multiple threads.
    storage_options : dict, optional
        Extra options that make sense for a particular storage connection, e.g.
        host, port, username, password, etc. For HTTP(S) URLs the key-value pairs
        are forwarded to ``urllib.request.Request`` as header options. For other
        URLs (e.g. starting with "s3://", and "gcs://") the key-value pairs are
        forwarded to ``fsspec.open``. Please see ``fsspec`` and ``urllib`` for more
        details, and for more examples on storage options refer `here
        <https://pandas.pydata.org/docs/user_guide/io.html?
        highlight=storage_options#reading-writing-remote-files>`_.

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
    type of object stored in file

    Examples
    --------
    >>> df = pd.read_feather("path/to/file.feather")  # doctest: +SKIP
    '''
