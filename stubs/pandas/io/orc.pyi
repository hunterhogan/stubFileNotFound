from fsspec import AbstractFileSystem  # pyright: ignore[reportMissingTypeStubs]
from pandas import DataFrame
from pandas._libs.lib import NoDefault
from pandas._typing import DtypeBackend, FilePath, HashableT, ReadBuffer
from pyarrow.fs import FileSystem
from typing import Any

def read_orc(
    path: FilePath | ReadBuffer[bytes],
    columns: list[HashableT] | None = None,
    dtype_backend: DtypeBackend | NoDefault = "numpy_nullable",
    filesystem: FileSystem | AbstractFileSystem | None = None,
    **kwargs: Any,
) -> DataFrame: ...
