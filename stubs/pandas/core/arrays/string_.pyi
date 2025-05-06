import numpy as np
import pyarrow
from _typeshed import Incomplete
from pandas import Series as Series
from pandas._libs import lib as lib, missing as libmissing
from pandas._typing import AxisInt as AxisInt, Dtype as Dtype, DtypeObj as DtypeObj, NumpySorter as NumpySorter, NumpyValueArrayLike as NumpyValueArrayLike, Scalar as Scalar, Self as Self, npt as npt, type_t as type_t
from pandas.core.arrays.base import ExtensionArray as ExtensionArray
from pandas.core.arrays.floating import FloatingArray as FloatingArray, FloatingDtype as FloatingDtype
from pandas.core.arrays.integer import IntegerArray as IntegerArray, IntegerDtype as IntegerDtype
from pandas.core.arrays.numpy_ import NumpyExtensionArray as NumpyExtensionArray
from pandas.core.dtypes.base import ExtensionDtype as ExtensionDtype, StorageExtensionDtype as StorageExtensionDtype, register_extension_dtype as register_extension_dtype
from pandas.core.dtypes.common import is_array_like as is_array_like, is_bool_dtype as is_bool_dtype, is_integer_dtype as is_integer_dtype, is_object_dtype as is_object_dtype, is_string_dtype as is_string_dtype, pandas_dtype as pandas_dtype
from typing import ClassVar, Literal

class StringDtype(StorageExtensionDtype):
    '''
    Extension dtype for string data.

    .. warning::

       StringDtype is considered experimental. The implementation and
       parts of the API may change without warning.

    Parameters
    ----------
    storage : {"python", "pyarrow", "pyarrow_numpy"}, optional
        If not given, the value of ``pd.options.mode.string_storage``.

    Attributes
    ----------
    None

    Methods
    -------
    None

    Examples
    --------
    >>> pd.StringDtype()
    string[python]

    >>> pd.StringDtype(storage="pyarrow")
    string[pyarrow]
    '''
    name: ClassVar[str]
    @property
    def na_value(self) -> libmissing.NAType | float: ...
    _metadata: Incomplete
    storage: Incomplete
    def __init__(self, storage: Incomplete | None = None) -> None: ...
    @property
    def type(self) -> type[str]: ...
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

class BaseStringArray(ExtensionArray):
    """
    Mixin class for StringArray, ArrowStringArray.
    """
    def tolist(self): ...
    @classmethod
    def _from_scalars(cls, scalars, dtype: DtypeObj) -> Self: ...

class StringArray(BaseStringArray, NumpyExtensionArray):
    '''
    Extension array for string data.

    .. warning::

       StringArray is considered experimental. The implementation and
       parts of the API may change without warning.

    Parameters
    ----------
    values : array-like
        The array of data.

        .. warning::

           Currently, this expects an object-dtype ndarray
           where the elements are Python strings
           or nan-likes (``None``, ``np.nan``, ``NA``).
           This may change without warning in the future. Use
           :meth:`pandas.array` with ``dtype="string"`` for a stable way of
           creating a `StringArray` from any sequence.

        .. versionchanged:: 1.5.0

           StringArray now accepts array-likes containing
           nan-likes(``None``, ``np.nan``) for the ``values`` parameter
           in addition to strings and :attr:`pandas.NA`

    copy : bool, default False
        Whether to copy the array of data.

    Attributes
    ----------
    None

    Methods
    -------
    None

    See Also
    --------
    :func:`pandas.array`
        The recommended function for creating a StringArray.
    Series.str
        The string methods are available on Series backed by
        a StringArray.

    Notes
    -----
    StringArray returns a BooleanArray for comparison methods.

    Examples
    --------
    >>> pd.array([\'This is\', \'some text\', None, \'data.\'], dtype="string")
    <StringArray>
    [\'This is\', \'some text\', <NA>, \'data.\']
    Length: 4, dtype: string

    Unlike arrays instantiated with ``dtype="object"``, ``StringArray``
    will convert the values to strings.

    >>> pd.array([\'1\', 1], dtype="object")
    <NumpyExtensionArray>
    [\'1\', 1]
    Length: 2, dtype: object
    >>> pd.array([\'1\', 1], dtype="string")
    <StringArray>
    [\'1\', \'1\']
    Length: 2, dtype: string

    However, instantiating StringArrays directly with non-strings will raise an error.

    For comparison methods, `StringArray` returns a :class:`pandas.BooleanArray`:

    >>> pd.array(["a", None, "c"], dtype="string") == "a"
    <BooleanArray>
    [True, <NA>, False]
    Length: 3, dtype: boolean
    '''
    _typ: str
    def __init__(self, values, copy: bool = False) -> None: ...
    def _validate(self) -> None:
        """Validate that we only store NA or strings."""
    @classmethod
    def _from_sequence(cls, scalars, *, dtype: Dtype | None = None, copy: bool = False): ...
    @classmethod
    def _from_sequence_of_strings(cls, strings, *, dtype: Dtype | None = None, copy: bool = False): ...
    @classmethod
    def _empty(cls, shape, dtype) -> StringArray: ...
    def __arrow_array__(self, type: Incomplete | None = None):
        """
        Convert myself into a pyarrow Array.
        """
    def _values_for_factorize(self): ...
    def __setitem__(self, key, value) -> None: ...
    def _putmask(self, mask: npt.NDArray[np.bool_], value) -> None: ...
    def astype(self, dtype, copy: bool = True): ...
    def _reduce(self, name: str, *, skipna: bool = True, axis: AxisInt | None = 0, **kwargs): ...
    def min(self, axis: Incomplete | None = None, skipna: bool = True, **kwargs) -> Scalar: ...
    def max(self, axis: Incomplete | None = None, skipna: bool = True, **kwargs) -> Scalar: ...
    def value_counts(self, dropna: bool = True) -> Series: ...
    def memory_usage(self, deep: bool = False) -> int: ...
    def searchsorted(self, value: NumpyValueArrayLike | ExtensionArray, side: Literal['left', 'right'] = 'left', sorter: NumpySorter | None = None) -> npt.NDArray[np.intp] | np.intp: ...
    def _cmp_method(self, other, op): ...
    _arith_method = _cmp_method
    _str_na_value: Incomplete
    def _str_map(self, f, na_value: Incomplete | None = None, dtype: Dtype | None = None, convert: bool = True): ...
