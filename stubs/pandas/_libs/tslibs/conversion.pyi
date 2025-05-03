import _cython_3_0_11
import numpy.dtypes
from _typeshed import Incomplete
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime as OutOfBoundsDatetime
from typing import ClassVar

DT64NS_DTYPE: numpy.dtypes.DateTime64DType
TD64NS_DTYPE: numpy.dtypes.TimeDelta64DType
__pyx_capi__: dict
__reduce_cython__: _cython_3_0_11.cython_function_or_method
__setstate_cython__: _cython_3_0_11.cython_function_or_method
__test__: dict
cast_from_unit_vectorized: _cython_3_0_11.cython_function_or_method
localize_pydatetime: _cython_3_0_11.cython_function_or_method

class _TSObject:
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    creso: Incomplete
    dts: Incomplete
    fold: Incomplete
    tzinfo: Incomplete
    value: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __reduce__(self): ...
