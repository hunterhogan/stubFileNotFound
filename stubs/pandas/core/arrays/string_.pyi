import _abc
import np
import npt
import pandas._libs.lib as lib
import pandas._libs.missing
import pandas._libs.missing as libmissing
import pandas.compat.numpy.function as nv
import pandas.core.array_algos.masked_reductions as masked_reductions
import pandas.core.arrays.base
import pandas.core.arrays.numpy_
import pandas.core.dtypes.base
import pandas.core.ops as ops
import pyarrow
from pandas._config.config import get_option as get_option
from pandas._libs.arrays import NDArrayBacked as NDArrayBacked
from pandas._libs.lib import ensure_string_array as ensure_string_array
from pandas.core.arrays.base import ExtensionArray as ExtensionArray
from pandas.core.arrays.floating import FloatingArray as FloatingArray, FloatingDtype as FloatingDtype
from pandas.core.arrays.integer import IntegerArray as IntegerArray, IntegerDtype as IntegerDtype
from pandas.core.arrays.numpy_ import NumpyExtensionArray as NumpyExtensionArray
from pandas.core.construction import extract_array as extract_array
from pandas.core.dtypes.base import ExtensionDtype as ExtensionDtype, StorageExtensionDtype as StorageExtensionDtype, register_extension_dtype as register_extension_dtype
from pandas.core.dtypes.common import is_bool_dtype as is_bool_dtype, is_integer_dtype as is_integer_dtype, is_object_dtype as is_object_dtype, is_string_dtype as is_string_dtype, pandas_dtype as pandas_dtype
from pandas.core.dtypes.inference import is_array_like as is_array_like
from pandas.core.dtypes.missing import isna as isna
from pandas.core.indexers.utils import check_array_indexer as check_array_indexer
from pandas.util._decorators import doc as doc
from typing import ClassVar as _ClassVar, Literal

TYPE_CHECKING: bool
pa_version_under10p1: bool

class StringDtype(pandas.core.dtypes.base.StorageExtensionDtype):
    name: _ClassVar[str] = ...
    _metadata: _ClassVar[tuple] = ...
    def __init__(self, storage) -> None: ...
    @classmethod
    def construct_from_string(cls, string) -> Self:
        """
        Construct a StringDtype from a string.

        Parameters
        ----------
        string : str
            The type of the name. The storage type will be taking from `string`.
            Valid options and their storage types are

            ========================== ==============================================
            string                     result storage
            ========================== ==============================================
            ``'string'``               pd.options.mode.string_storage, default python
            ``'string[python]'``       python
            ``'string[pyarrow]'``      pyarrow
            ========================== ==============================================

        Returns
        -------
        StringDtype

        Raise
        -----
        TypeError
            If the string is not a valid option.
        """
    def construct_array_type(self) -> type_t[BaseStringArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
    def __from_arrow__(self, array: pyarrow.Array | pyarrow.ChunkedArray) -> BaseStringArray:
        """
        Construct StringArray from pyarrow Array/ChunkedArray.
        """
    @property
    def na_value(self): ...
    @property
    def type(self): ...

class BaseStringArray(pandas.core.arrays.base.ExtensionArray):
    def tolist(self):
        """
        Return a list of the values.

        These are each a scalar type, which is a Python scalar
        (for str, int, float) or a pandas scalar
        (for Timestamp/Timedelta/Interval/Period)

        Returns
        -------
        list

        Examples
        --------
        >>> arr = pd.array([1, 2, 3])
        >>> arr.tolist()
        [1, 2, 3]
        """
    @classmethod
    def _from_scalars(cls, scalars, dtype: DtypeObj) -> Self: ...

class StringArray(BaseStringArray, pandas.core.arrays.numpy_.NumpyExtensionArray):
    _typ: _ClassVar[str] = ...
    _str_na_value: _ClassVar[pandas._libs.missing.NAType] = ...
    __abstractmethods__: _ClassVar[frozenset] = ...
    _abc_impl: _ClassVar[_abc._abc_data] = ...
    def __init__(self, values, copy: bool = ...) -> None: ...
    def _validate(self):
        """Validate that we only store NA or strings."""
    @classmethod
    def _from_sequence(cls, scalars, *, dtype: Dtype | None, copy: bool = ...): ...
    @classmethod
    def _from_sequence_of_strings(cls, strings, *, dtype: Dtype | None, copy: bool = ...): ...
    @classmethod
    def _empty(cls, shape, dtype) -> StringArray: ...
    def __arrow_array__(self, type):
        """
        Convert myself into a pyarrow Array.
        """
    def _values_for_factorize(self): ...
    def __setitem__(self, key, value) -> None: ...
    def _putmask(self, mask: npt.NDArray[np.bool_], value) -> None: ...
    def astype(self, dtype, copy: bool = ...): ...
    def _reduce(self, name: str, *, skipna: bool = ..., axis: AxisInt | None = ..., **kwargs): ...
    def min(self, axis, skipna: bool = ..., **kwargs) -> Scalar: ...
    def max(self, axis, skipna: bool = ..., **kwargs) -> Scalar: ...
    def value_counts(self, dropna: bool = ...) -> Series: ...
    def memory_usage(self, deep: bool = ...) -> int: ...
    def searchsorted(self, value: NumpyValueArrayLike | ExtensionArray, side: Literal['left', 'right'] = ..., sorter: NumpySorter | None) -> npt.NDArray[np.intp] | np.intp:
        """
        Find indices where elements should be inserted to maintain order.

        Find the indices into a sorted array `self` (a) such that, if the
        corresponding elements in `value` were inserted before the indices,
        the order of `self` would be preserved.

        Assuming that `self` is sorted:

        ======  ================================
        `side`  returned index `i` satisfies
        ======  ================================
        left    ``self[i-1] < value <= self[i]``
        right   ``self[i-1] <= value < self[i]``
        ======  ================================

        Parameters
        ----------
        value : array-like, list or scalar
            Value(s) to insert into `self`.
        side : {'left', 'right'}, optional
            If 'left', the index of the first suitable location found is given.
            If 'right', return the last such index.  If there is no suitable
            index, return either 0 or N (where N is the length of `self`).
        sorter : 1-D array-like, optional
            Optional array of integer indices that sort array a into ascending
            order. They are typically the result of argsort.

        Returns
        -------
        array of ints or int
            If value is array-like, array of insertion points.
            If value is scalar, a single integer.

        See Also
        --------
        numpy.searchsorted : Similar method from NumPy.

        Examples
        --------
        >>> arr = pd.array([1, 2, 3, 5])
        >>> arr.searchsorted([4])
        array([3])
        """
    def _cmp_method(self, other, op): ...
    def _arith_method(self, other, op): ...
    def _str_map(self, f, na_value, dtype: Dtype | None, convert: bool = ...): ...
