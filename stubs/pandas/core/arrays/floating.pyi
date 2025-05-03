import np
import numpy
import numpy.dtypes
import pandas.core.arrays.numeric
from pandas.core.arrays.numeric import NumericArray as NumericArray, NumericDtype as NumericDtype
from pandas.core.dtypes.base import register_extension_dtype as register_extension_dtype
from pandas.core.dtypes.common import is_float_dtype as is_float_dtype
from typing import ClassVar as _ClassVar

class FloatingDtype(pandas.core.arrays.numeric.NumericDtype):
    _default_np_dtype: _ClassVar[numpy.dtypes.Float64DType] = ...
    def _checker(self, arr_or_dtype) -> bool:
        """
        Check whether the provided array or dtype is of a float dtype.

        Parameters
        ----------
        arr_or_dtype : array-like or dtype
            The array or dtype to check.

        Returns
        -------
        boolean
            Whether or not the array or dtype is of a float dtype.

        Examples
        --------
        >>> from pandas.api.types import is_float_dtype
        >>> is_float_dtype(str)
        False
        >>> is_float_dtype(int)
        False
        >>> is_float_dtype(float)
        True
        >>> is_float_dtype(np.array(['a', 'b']))
        False
        >>> is_float_dtype(pd.Series([1, 2]))
        False
        >>> is_float_dtype(pd.Index([1, 2.]))
        True
        """
    @classmethod
    def construct_array_type(cls) -> type[FloatingArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
    @classmethod
    def _get_dtype_mapping(cls) -> dict[np.dtype, FloatingDtype]: ...
    @classmethod
    def _safe_cast(cls, values: np.ndarray, dtype: np.dtype, copy: bool) -> np.ndarray:
        '''
        Safely cast the values to the given dtype.

        "safe" in this context means the casting is lossless.
        '''

class FloatingArray(pandas.core.arrays.numeric.NumericArray):
    class _dtype_cls(pandas.core.arrays.numeric.NumericDtype):
        _default_np_dtype: _ClassVar[numpy.dtypes.Float64DType] = ...
        def _checker(self, arr_or_dtype) -> bool:
            """
            Check whether the provided array or dtype is of a float dtype.

            Parameters
            ----------
            arr_or_dtype : array-like or dtype
                The array or dtype to check.

            Returns
            -------
            boolean
                Whether or not the array or dtype is of a float dtype.

            Examples
            --------
            >>> from pandas.api.types import is_float_dtype
            >>> is_float_dtype(str)
            False
            >>> is_float_dtype(int)
            False
            >>> is_float_dtype(float)
            True
            >>> is_float_dtype(np.array(['a', 'b']))
            False
            >>> is_float_dtype(pd.Series([1, 2]))
            False
            >>> is_float_dtype(pd.Index([1, 2.]))
            True
            """
        @classmethod
        def construct_array_type(cls) -> type[FloatingArray]:
            """
            Return the array type associated with this dtype.

            Returns
            -------
            type
            """
        @classmethod
        def _get_dtype_mapping(cls) -> dict[np.dtype, FloatingDtype]: ...
        @classmethod
        def _safe_cast(cls, values: np.ndarray, dtype: np.dtype, copy: bool) -> np.ndarray:
            '''
            Safely cast the values to the given dtype.

            "safe" in this context means the casting is lossless.
            '''
    _internal_fill_value: _ClassVar[float] = ...
    _truthy_value: _ClassVar[float] = ...
    _falsey_value: _ClassVar[float] = ...
_dtype_docstring: str

class Float32Dtype(FloatingDtype):
    type: _ClassVar[type[numpy.float32]] = ...
    name: _ClassVar[str] = ...

class Float64Dtype(FloatingDtype):
    type: _ClassVar[type[numpy.float64]] = ...
    name: _ClassVar[str] = ...
NUMPY_FLOAT_TO_DTYPE: dict
