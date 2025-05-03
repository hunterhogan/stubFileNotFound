import numpy as np
import pyarrow
from _typeshed import Incomplete
from pandas._libs import lib as lib
from pandas._typing import Dtype as Dtype, DtypeObj as DtypeObj, Self as Self, npt as npt, type_t as type_t
from pandas.core import ops as ops
from pandas.core.array_algos import masked_accumulations as masked_accumulations
from pandas.core.arrays.masked import BaseMaskedArray as BaseMaskedArray, BaseMaskedDtype as BaseMaskedDtype
from pandas.core.dtypes.common import is_list_like as is_list_like
from pandas.core.dtypes.dtypes import register_extension_dtype as register_extension_dtype
from pandas.core.dtypes.missing import isna as isna
from typing import ClassVar

class BooleanDtype(BaseMaskedDtype):
    """
    Extension dtype for boolean data.

    .. warning::

       BooleanDtype is considered experimental. The implementation and
       parts of the API may change without warning.

    Attributes
    ----------
    None

    Methods
    -------
    None

    Examples
    --------
    >>> pd.BooleanDtype()
    BooleanDtype
    """
    name: ClassVar[str]
    @property
    def type(self) -> type: ...
    @property
    def kind(self) -> str: ...
    @property
    def numpy_dtype(self) -> np.dtype: ...
    @classmethod
    def construct_array_type(cls) -> type_t[BooleanArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
    def __repr__(self) -> str: ...
    @property
    def _is_boolean(self) -> bool: ...
    @property
    def _is_numeric(self) -> bool: ...
    def __from_arrow__(self, array: pyarrow.Array | pyarrow.ChunkedArray) -> BooleanArray:
        """
        Construct BooleanArray from pyarrow Array/ChunkedArray.
        """

def coerce_to_array(values, mask: Incomplete | None = None, copy: bool = False) -> tuple[np.ndarray, np.ndarray]:
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

class BooleanArray(BaseMaskedArray):
    '''
    Array of boolean (True/False) data with missing values.

    This is a pandas Extension array for boolean data, under the hood
    represented by 2 numpy arrays: a boolean array with the data and
    a boolean array with the mask (True indicating missing).

    BooleanArray implements Kleene logic (sometimes called three-value
    logic) for logical operations. See :ref:`boolean.kleene` for more.

    To construct an BooleanArray from generic array-like input, use
    :func:`pandas.array` specifying ``dtype="boolean"`` (see examples
    below).

    .. warning::

       BooleanArray is considered experimental. The implementation and
       parts of the API may change without warning.

    Parameters
    ----------
    values : numpy.ndarray
        A 1-d boolean-dtype array with the data.
    mask : numpy.ndarray
        A 1-d boolean-dtype array indicating missing values (True
        indicates missing).
    copy : bool, default False
        Whether to copy the `values` and `mask` arrays.

    Attributes
    ----------
    None

    Methods
    -------
    None

    Returns
    -------
    BooleanArray

    Examples
    --------
    Create an BooleanArray with :func:`pandas.array`:

    >>> pd.array([True, False, None], dtype="boolean")
    <BooleanArray>
    [True, False, <NA>]
    Length: 3, dtype: boolean
    '''
    _internal_fill_value: bool
    _truthy_value: bool
    _falsey_value: bool
    _TRUE_VALUES: Incomplete
    _FALSE_VALUES: Incomplete
    @classmethod
    def _simple_new(cls, values: np.ndarray, mask: npt.NDArray[np.bool_]) -> Self: ...
    _dtype: Incomplete
    def __init__(self, values: np.ndarray, mask: np.ndarray, copy: bool = False) -> None: ...
    @property
    def dtype(self) -> BooleanDtype: ...
    @classmethod
    def _from_sequence_of_strings(cls, strings: list[str], *, dtype: Dtype | None = None, copy: bool = False, true_values: list[str] | None = None, false_values: list[str] | None = None) -> BooleanArray: ...
    _HANDLED_TYPES: Incomplete
    @classmethod
    def _coerce_to_array(cls, value, *, dtype: DtypeObj, copy: bool = False) -> tuple[np.ndarray, np.ndarray]: ...
    def _logical_method(self, other, op): ...
    def _accumulate(self, name: str, *, skipna: bool = True, **kwargs) -> BaseMaskedArray: ...
