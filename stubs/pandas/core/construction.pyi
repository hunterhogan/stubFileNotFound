from collections.abc import Sequence

import numpy as np
from pandas.core.arrays.base import ExtensionArray

from pandas.core.dtypes.dtypes import ExtensionDtype

def array(
    data: Sequence[object],
    dtype: str | np.dtype | ExtensionDtype | None = None,
    copy: bool = True,
) -> ExtensionArray: ...
def extract_array(obj, extract_numpy: bool = False): ...
def sanitize_array(
    data, index, dtype=None, copy: bool = False, raise_cast_failure: bool = ...
): ...
def is_empty_data(data) -> bool: ...
