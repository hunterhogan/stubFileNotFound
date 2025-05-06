import numpy as np
import pyarrow
from _typeshed import Incomplete
from collections.abc import Mapping
from pandas._libs import lib as lib
from pandas._typing import Dtype as Dtype, DtypeObj as DtypeObj, Self as Self, npt as npt
from pandas.core.arrays.masked import BaseMaskedArray as BaseMaskedArray, BaseMaskedDtype as BaseMaskedDtype
from pandas.core.dtypes.common import is_integer_dtype as is_integer_dtype, is_string_dtype as is_string_dtype, pandas_dtype as pandas_dtype
from pandas.errors import AbstractMethodError as AbstractMethodError
from pandas.util._decorators import cache_readonly as cache_readonly
from typing import Any

from collections.abc import Callable

class NumericDtype(BaseMaskedDtype):
    _default_np_dtype: np.dtype
    _checker: Callable[[Any], bool]
    def __repr__(self) -> str: ...
    def is_signed_integer(self) -> bool: ...
    def is_unsigned_integer(self) -> bool: ...
    @property
    def _is_numeric(self) -> bool: ...
    def __from_arrow__(self, array: pyarrow.Array | pyarrow.ChunkedArray) -> BaseMaskedArray:
        """
        Construct IntegerArray/FloatingArray from pyarrow Array/ChunkedArray.
        """
    @classmethod
    def _get_dtype_mapping(cls) -> Mapping[np.dtype, NumericDtype]: ...
    @classmethod
    def _standardize_dtype(cls, dtype: NumericDtype | str | np.dtype) -> NumericDtype:
        """
        Convert a string representation or a numpy dtype to NumericDtype.
        """
    @classmethod
    def _safe_cast(cls, values: np.ndarray, dtype: np.dtype, copy: bool) -> np.ndarray:
        '''
        Safely cast the values to the given dtype.

        "safe" in this context means the casting is lossless.
        '''

def _coerce_to_data_and_mask(values, dtype, copy: bool, dtype_cls: type[NumericDtype], default_dtype: np.dtype): ...

class NumericArray(BaseMaskedArray):
    """
    Base class for IntegerArray and FloatingArray.
    """
    _dtype_cls: type[NumericDtype]
    def __init__(self, values: np.ndarray, mask: npt.NDArray[np.bool_], copy: bool = False) -> None: ...
    def dtype(self) -> NumericDtype: ...
    @classmethod
    def _coerce_to_array(cls, value, *, dtype: DtypeObj, copy: bool = False) -> tuple[np.ndarray, np.ndarray]: ...
    @classmethod
    def _from_sequence_of_strings(cls, strings, *, dtype: Dtype | None = None, copy: bool = False) -> Self: ...
    _HANDLED_TYPES: Incomplete
