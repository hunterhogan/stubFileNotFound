from typing import (
    Literal,
    TypeAlias,
    overload,
)

import numpy as np
import pandas as pd

from pandas._libs.lib import _NoDefaultDoNotUse
from pandas._typing import (
    DtypeBackend,
    RaiseCoerce,
    Scalar,
    npt,
)
from typing import Any

_Downcast: TypeAlias = Literal["integer", "signed", "unsigned", "float"] | None

@overload
def to_numeric(
    arg: Scalar,
    errors: Literal["raise", "coerce"] = 'raise',
    downcast: _Downcast = None,
    dtype_backend: DtypeBackend | _NoDefaultDoNotUse = ...,
) -> float: ...
@overload
def to_numeric(
    arg: list[Any] | tuple[Any, ...] | np.ndarray[Any, Any],
    errors: RaiseCoerce = 'raise',
    downcast: _Downcast = None,
    dtype_backend: DtypeBackend | _NoDefaultDoNotUse = ...,
) -> npt.NDArray[Any]: ...
@overload
def to_numeric(
    arg: pd.Series,
    errors: RaiseCoerce = 'raise',
    downcast: _Downcast = None,
    dtype_backend: DtypeBackend | _NoDefaultDoNotUse = ...,
) -> pd.Series: ...
