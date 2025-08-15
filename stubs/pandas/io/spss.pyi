from pandas._libs.lib import _NoDefaultDoNotUse
from pandas._typing import DtypeBackend, FilePath, HashableT
from pandas.core.frame import DataFrame

def read_spss(
    path: FilePath,
    usecols: list[HashableT] | None = None,
    convert_categoricals: bool = True,
    dtype_backend: DtypeBackend | _NoDefaultDoNotUse = "numpy_nullable",
) -> DataFrame: ...
