import numpy as np
from _typeshed import Incomplete
from pandas.core.arrays.numeric import NumericArray as NumericArray, NumericDtype as NumericDtype
from pandas.core.dtypes.base import register_extension_dtype as register_extension_dtype
from pandas.core.dtypes.common import is_float_dtype as is_float_dtype
from typing import ClassVar

class FloatingDtype(NumericDtype):
    """
    An ExtensionDtype to hold a single size of floating dtype.

    These specific implementations are subclasses of the non-public
    FloatingDtype. For example we have Float32Dtype to represent float32.

    The attributes name & type are set when these subclasses are created.
    """
    _default_np_dtype: Incomplete
    _checker = is_float_dtype
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

class FloatingArray(NumericArray):
    '''
    Array of floating (optional missing) values.

    .. warning::

       FloatingArray is currently experimental, and its API or internal
       implementation may change without warning. Especially the behaviour
       regarding NaN (distinct from NA missing values) is subject to change.

    We represent a FloatingArray with 2 numpy arrays:

    - data: contains a numpy float array of the appropriate dtype
    - mask: a boolean array holding a mask on the data, True is missing

    To construct an FloatingArray from generic array-like input, use
    :func:`pandas.array` with one of the float dtypes (see examples).

    See :ref:`integer_na` for more.

    Parameters
    ----------
    values : numpy.ndarray
        A 1-d float-dtype array.
    mask : numpy.ndarray
        A 1-d boolean-dtype array indicating missing values.
    copy : bool, default False
        Whether to copy the `values` and `mask`.

    Attributes
    ----------
    None

    Methods
    -------
    None

    Returns
    -------
    FloatingArray

    Examples
    --------
    Create an FloatingArray with :func:`pandas.array`:

    >>> pd.array([0.1, None, 0.3], dtype=pd.Float32Dtype())
    <FloatingArray>
    [0.1, <NA>, 0.3]
    Length: 3, dtype: Float32

    String aliases for the dtypes are also available. They are capitalized.

    >>> pd.array([0.1, None, 0.3], dtype="Float32")
    <FloatingArray>
    [0.1, <NA>, 0.3]
    Length: 3, dtype: Float32
    '''
    _dtype_cls = FloatingDtype
    _internal_fill_value: Incomplete
    _truthy_value: float
    _falsey_value: float

_dtype_docstring: str

class Float32Dtype(FloatingDtype):
    type: Incomplete
    name: ClassVar[str]
    __doc__: Incomplete

class Float64Dtype(FloatingDtype):
    type = np.float64
    name: ClassVar[str]
    __doc__: Incomplete

NUMPY_FLOAT_TO_DTYPE: dict[np.dtype, FloatingDtype]
