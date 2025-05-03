import npt
import pandas._libs.lib as lib
from pandas.core.dtypes.cast import np_can_hold_element as np_can_hold_element
from pandas.core.dtypes.common import is_numeric_dtype as is_numeric_dtype
from pandas.errors import LossySetitemError as LossySetitemError
from typing import Any

TYPE_CHECKING: bool
def to_numpy_dtype_inference(arr: ArrayLike, dtype: npt.DTypeLike | None, na_value, hasna: bool) -> tuple[npt.DTypeLike, Any]: ...
