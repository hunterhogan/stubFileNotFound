import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from pandas.core.arrays.base import (
    ExtensionArray,
    ExtensionOpsMixin,
)

from pandas.core.dtypes.dtypes import ExtensionDtype
from typing import Any

class PandasDtype(ExtensionDtype):
    @property
    def numpy_dtype(self) -> np.dtype: ...
    @property
    def itemsize(self) -> int: ...

class PandasArray(ExtensionArray, ExtensionOpsMixin, NDArrayOperatorsMixin):
    def __array_ufunc__(self, ufunc: Any, method: Any, *inputs, **kwargs): ...
