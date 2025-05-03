from collections.abc import Hashable, Sequence
from pandas._config import using_pyarrow_string_dtype as using_pyarrow_string_dtype
from pandas._libs import lib as lib
from pandas._typing import DtypeBackend as DtypeBackend, FilePath as FilePath, ReadBuffer as ReadBuffer, StorageOptions as StorageOptions, WriteBuffer as WriteBuffer
from pandas.compat._optional import import_optional_dependency as import_optional_dependency
from pandas.core.api import DataFrame as DataFrame
from pandas.core.shared_docs import _shared_docs as _shared_docs
from pandas.io._util import arrow_string_types_mapper as arrow_string_types_mapper
from pandas.io.common import get_handle as get_handle
from pandas.util._decorators import doc as doc
from pandas.util._validators import check_dtype_backend as check_dtype_backend
from typing import Any

def to_feather(df: DataFrame, path: FilePath | WriteBuffer[bytes], storage_options: StorageOptions | None = None, **kwargs: Any) -> None:
    """
    Write a DataFrame to the binary Feather format.

    Parameters
    ----------
    df : DataFrame
    path : str, path object, or file-like object
    {storage_options}
    **kwargs :
        Additional keywords passed to `pyarrow.feather.write_feather`.

    """
def read_feather(path: FilePath | ReadBuffer[bytes], columns: Sequence[Hashable] | None = None, use_threads: bool = True, storage_options: StorageOptions | None = None, dtype_backend: DtypeBackend | lib.NoDefault = ...) -> DataFrame:
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
    {storage_options}

    dtype_backend : {{\'numpy_nullable\', \'pyarrow\'}}, default \'numpy_nullable\'
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
