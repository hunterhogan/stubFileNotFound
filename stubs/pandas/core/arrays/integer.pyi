import np
import numpy
import numpy.dtypes
import pandas.core.arrays.numeric
from pandas.core.arrays.numeric import NumericArray as NumericArray, NumericDtype as NumericDtype
from pandas.core.dtypes.base import register_extension_dtype as register_extension_dtype
from pandas.core.dtypes.common import is_integer_dtype as is_integer_dtype
from typing import ClassVar as _ClassVar

class IntegerDtype(pandas.core.arrays.numeric.NumericDtype):
    _default_np_dtype: _ClassVar[numpy.dtypes.Int64DType] = ...
    def _checker(self, arr_or_dtype) -> bool:
        """
        Check whether the provided array or dtype is of an integer dtype.

        Unlike in `is_any_int_dtype`, timedelta64 instances will return False.

        The nullable Integer dtypes (e.g. pandas.Int64Dtype) are also considered
        as integer by this function.

        Parameters
        ----------
        arr_or_dtype : array-like or dtype
            The array or dtype to check.

        Returns
        -------
        boolean
            Whether or not the array or dtype is of an integer dtype and
            not an instance of timedelta64.

        Examples
        --------
        >>> from pandas.api.types import is_integer_dtype
        >>> is_integer_dtype(str)
        False
        >>> is_integer_dtype(int)
        True
        >>> is_integer_dtype(float)
        False
        >>> is_integer_dtype(np.uint64)
        True
        >>> is_integer_dtype('int8')
        True
        >>> is_integer_dtype('Int8')
        True
        >>> is_integer_dtype(pd.Int8Dtype)
        True
        >>> is_integer_dtype(np.datetime64)
        False
        >>> is_integer_dtype(np.timedelta64)
        False
        >>> is_integer_dtype(np.array(['a', 'b']))
        False
        >>> is_integer_dtype(pd.Series([1, 2]))
        True
        >>> is_integer_dtype(np.array([], dtype=np.timedelta64))
        False
        >>> is_integer_dtype(pd.Index([1, 2.]))  # float
        False
        """
    @classmethod
    def construct_array_type(cls) -> type[IntegerArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
    @classmethod
    def _get_dtype_mapping(cls) -> dict[np.dtype, IntegerDtype]: ...
    @classmethod
    def _safe_cast(cls, values: np.ndarray, dtype: np.dtype, copy: bool) -> np.ndarray:
        '''
        Safely cast the values to the given dtype.

        "safe" in this context means the casting is lossless. e.g. if \'values\'
        has a floating dtype, each value must be an integer.
        '''

class IntegerArray(pandas.core.arrays.numeric.NumericArray):
    class _dtype_cls(pandas.core.arrays.numeric.NumericDtype):
        _default_np_dtype: _ClassVar[numpy.dtypes.Int64DType] = ...
        def _checker(self, arr_or_dtype) -> bool:
            """
            Check whether the provided array or dtype is of an integer dtype.

            Unlike in `is_any_int_dtype`, timedelta64 instances will return False.

            The nullable Integer dtypes (e.g. pandas.Int64Dtype) are also considered
            as integer by this function.

            Parameters
            ----------
            arr_or_dtype : array-like or dtype
                The array or dtype to check.

            Returns
            -------
            boolean
                Whether or not the array or dtype is of an integer dtype and
                not an instance of timedelta64.

            Examples
            --------
            >>> from pandas.api.types import is_integer_dtype
            >>> is_integer_dtype(str)
            False
            >>> is_integer_dtype(int)
            True
            >>> is_integer_dtype(float)
            False
            >>> is_integer_dtype(np.uint64)
            True
            >>> is_integer_dtype('int8')
            True
            >>> is_integer_dtype('Int8')
            True
            >>> is_integer_dtype(pd.Int8Dtype)
            True
            >>> is_integer_dtype(np.datetime64)
            False
            >>> is_integer_dtype(np.timedelta64)
            False
            >>> is_integer_dtype(np.array(['a', 'b']))
            False
            >>> is_integer_dtype(pd.Series([1, 2]))
            True
            >>> is_integer_dtype(np.array([], dtype=np.timedelta64))
            False
            >>> is_integer_dtype(pd.Index([1, 2.]))  # float
            False
            """
        @classmethod
        def construct_array_type(cls) -> type[IntegerArray]:
            """
            Return the array type associated with this dtype.

            Returns
            -------
            type
            """
        @classmethod
        def _get_dtype_mapping(cls) -> dict[np.dtype, IntegerDtype]: ...
        @classmethod
        def _safe_cast(cls, values: np.ndarray, dtype: np.dtype, copy: bool) -> np.ndarray:
            '''
            Safely cast the values to the given dtype.

            "safe" in this context means the casting is lossless. e.g. if \'values\'
            has a floating dtype, each value must be an integer.
            '''
    _internal_fill_value: _ClassVar[int] = ...
    _truthy_value: _ClassVar[int] = ...
    _falsey_value: _ClassVar[int] = ...
_dtype_docstring: str

class Int8Dtype(IntegerDtype):
    type: _ClassVar[type[numpy.int8]] = ...
    name: _ClassVar[str] = ...

class Int16Dtype(IntegerDtype):
    type: _ClassVar[type[numpy.int16]] = ...
    name: _ClassVar[str] = ...

class Int32Dtype(IntegerDtype):
    type: _ClassVar[type[numpy.int32]] = ...
    name: _ClassVar[str] = ...

class Int64Dtype(IntegerDtype):
    type: _ClassVar[type[numpy.int64]] = ...
    name: _ClassVar[str] = ...

class UInt8Dtype(IntegerDtype):
    type: _ClassVar[type[numpy.uint8]] = ...
    name: _ClassVar[str] = ...

class UInt16Dtype(IntegerDtype):
    type: _ClassVar[type[numpy.uint16]] = ...
    name: _ClassVar[str] = ...

class UInt32Dtype(IntegerDtype):
    type: _ClassVar[type[numpy.uint32]] = ...
    name: _ClassVar[str] = ...

class UInt64Dtype(IntegerDtype):
    type: _ClassVar[type[numpy.uint64]] = ...
    name: _ClassVar[str] = ...
NUMPY_INT_TO_DTYPE: dict
