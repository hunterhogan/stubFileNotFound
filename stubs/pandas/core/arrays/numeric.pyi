import np
import npt
import pandas._libs.lib as lib
import pandas._libs.missing as libmissing
import pandas.core.arrays.masked
import pandas.core.dtypes.dtypes
import pyarrow
from _typeshed import Incomplete
from pandas._libs.properties import cache_readonly as cache_readonly
from pandas.core.arrays.masked import BaseMaskedArray as BaseMaskedArray
from pandas.core.dtypes.common import is_integer_dtype as is_integer_dtype, is_string_dtype as is_string_dtype, pandas_dtype as pandas_dtype
from pandas.core.dtypes.dtypes import BaseMaskedDtype as BaseMaskedDtype
from pandas.errors import AbstractMethodError as AbstractMethodError
from typing import ClassVar

TYPE_CHECKING: bool

class NumericDtype(pandas.core.dtypes.dtypes.BaseMaskedDtype):
    is_signed_integer: Incomplete
    is_unsigned_integer: Incomplete
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
    @property
    def _is_numeric(self): ...
def _coerce_to_data_and_mask(values, dtype, copy: bool, dtype_cls: type[NumericDtype], default_dtype: np.dtype): ...

class NumericArray(pandas.core.arrays.masked.BaseMaskedArray):
    _HANDLED_TYPES: ClassVar[tuple] = ...
    dtype: Incomplete
    def __init__(self, values: np.ndarray, mask: npt.NDArray[np.bool_], copy: bool = ...) -> None: ...
    @classmethod
    def _coerce_to_array(cls, value, *, dtype: DtypeObj, copy: bool = ...) -> tuple[np.ndarray, np.ndarray]: ...
    @classmethod
    def _from_sequence_of_strings(cls, strings, *, dtype: Dtype | None, copy: bool = ...) -> Self: ...
