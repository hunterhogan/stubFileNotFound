import _cython_3_0_11
import _strptime
from typing import ClassVar

_CACHE_MAX_SIZE: int
_TimeRE_cache: TimeRE
__pyx_capi__: dict
__reduce_cython__: _cython_3_0_11.cython_function_or_method
__setstate_cython__: _cython_3_0_11.cython_function_or_method
__test__: dict
_array_strptime_object_fallback: _cython_3_0_11.cython_function_or_method
_regex_cache: dict
_test_format_is_iso: _cython_3_0_11.cython_function_or_method
array_strptime: _cython_3_0_11.cython_function_or_method

class DatetimeParseState:
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __reduce__(self): ...

class TimeRE(_strptime.TimeRE):
    def __init__(self, *args, **kwargs) -> None:
        """
        Create keys/values.

        Order of execution is important for dependency reasons.
        """
    def __getitem__(self, index): ...
