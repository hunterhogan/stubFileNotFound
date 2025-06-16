from typing import Any

from pandas import DataFrame

from pandas._typing import (
    FilePath,
    ParquetEngine,
    ReadBuffer,
    StorageOptions,
)

def read_parquet(
    path: FilePath | ReadBuffer[bytes],
    engine: ParquetEngine = 'auto',
    columns: list[str] | None = None,
    storage_options: StorageOptions = None,
    **kwargs: Any,
) -> DataFrame: ...
