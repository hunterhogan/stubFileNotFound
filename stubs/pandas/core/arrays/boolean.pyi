import np
import npt
import pandas._libs.lib as lib
import pandas._libs.missing as libmissing
import pandas.core.array_algos.masked_accumulations as masked_accumulations
import pandas.core.arrays.masked
import pandas.core.dtypes.dtypes
import pandas.core.ops as ops
import pyarrow
from pandas._libs.lib import is_list_like as is_list_like
from pandas.core.arrays.masked import BaseMaskedArray as BaseMaskedArray
from pandas.core.dtypes.base import register_extension_dtype as register_extension_dtype
from pandas.core.dtypes.dtypes import BaseMaskedDtype as BaseMaskedDtype
from pandas.core.dtypes.missing import isna as isna
from typing import ClassVar as _ClassVar

TYPE_CHECKING: bool

class BooleanDtype(pandas.core.dtypes.dtypes.BaseMaskedDtype):
    name: _ClassVar[str] = ...
    @classmethod
    def construct_array_type(cls) -> type_t[BooleanArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
    def __from_arrow__(self, array: pyarrow.Array | pyarrow.ChunkedArray) -> BooleanArray:
        """
        Construct BooleanArray from pyarrow Array/ChunkedArray.
        """
    @property
    def type(self): ...
    @property
    def kind(self): ...
    @property
    def numpy_dtype(self): ...
    @property
    def _is_boolean(self): ...
    @property
    def _is_numeric(self): ...
def coerce_to_array(values, mask, copy: bool = ...) -> tuple[np.ndarray, np.ndarray]:
    """
    Coerce the input values array to numpy arrays with a mask.

    Parameters
    ----------
    values : 1D list-like
    mask : bool 1D array, optional
    copy : bool, default False
        if True, copy the input

    Returns
    -------
    tuple of (values, mask)
    """

class BooleanArray(pandas.core.arrays.masked.BaseMaskedArray):
    _internal_fill_value: _ClassVar[bool] = ...
    _truthy_value: _ClassVar[bool] = ...
    _falsey_value: _ClassVar[bool] = ...
    _TRUE_VALUES: _ClassVar[set] = ...
    _FALSE_VALUES: _ClassVar[set] = ...
    _HANDLED_TYPES: _ClassVar[tuple] = ...
    @classmethod
    def _simple_new(cls, values: np.ndarray, mask: npt.NDArray[np.bool_]) -> Self: ...
    def __init__(self, values: np.ndarray, mask: np.ndarray, copy: bool = ...) -> None: ...
    @classmethod
    def _from_sequence_of_strings(cls, strings: list[str], *, dtype: Dtype | None, copy: bool = ..., true_values: list[str] | None, false_values: list[str] | None) -> BooleanArray: ...
    @classmethod
    def _coerce_to_array(cls, value, *, dtype: DtypeObj, copy: bool = ...) -> tuple[np.ndarray, np.ndarray]: ...
    def _logical_method(self, other, op): ...
    def _accumulate(self, name: str, *, skipna: bool = ..., **kwargs) -> BaseMaskedArray: ...
    @property
    def dtype(self): ...
