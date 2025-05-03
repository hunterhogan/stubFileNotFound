import _cython_3_0_11
import datetime

__pyx_unpickle_ABCTimestamp: _cython_3_0_11.cython_function_or_method
__test__: dict

class ABCTimestamp(datetime.datetime):
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __reduce_cython__(self, *args, **kwargs): ...
    def __setstate_cython__(self, *args, **kwargs): ...
