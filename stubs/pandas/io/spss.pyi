from pandas._libs.lib import NoDefault
from pandas._typing import DtypeBackend, FilePath, HashableT
from pandas.core.frame import DataFrame
from typing import Any

def read_spss(
    path: FilePath,
    usecols: list[HashableT] | None = None,
    convert_categoricals: bool = True,
    dtype_backend: DtypeBackend | NoDefault = "numpy_nullable",
    **kwargs: Any,
) -> DataFrame: ...
