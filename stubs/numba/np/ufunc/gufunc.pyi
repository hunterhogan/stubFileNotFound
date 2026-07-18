from _typeshed import Incomplete
from numba import typeof as typeof
from numba.core import errors as errors, serialize as serialize, types as types
from numba.core.typing import npydecl as npydecl
from numba.core.typing.templates import AbstractTemplate as AbstractTemplate, signature as signature
from numba.np.numpy_support import ufunc_find_matching_loop as ufunc_find_matching_loop
from numba.np.ufunc.sigparse import parse_signature as parse_signature
from numba.np.ufunc.ufunc_base import UfuncBase as UfuncBase, UfuncLowererBase as UfuncLowererBase
from numba.np.ufunc.ufuncbuilder import GUFuncBuilder as GUFuncBuilder

def make_gufunc_kernel(_dufunc): ...

class GUFuncLowerer(UfuncLowererBase):
    """Callable class responsible for lowering calls to a specific gufunc.
    """
    def __init__(self, gufunc) -> None: ...

class GUFunc(serialize.ReduceMixin, UfuncBase):
    """
    Dynamic generalized universal function (GUFunc)
    intended to act like a normal Numpy gufunc, but capable
    of call-time (just-in-time) compilation of fast loops
    specialized to inputs.
    """
    ufunc: Incomplete
    gufunc_builder: Incomplete
    __doc__: Incomplete
    def __init__(self, py_func, signature, identity=None, cache=None, is_dynamic: bool = False, targetoptions=None, writable_args=()) -> None: ...
    def add(self, fty) -> None: ...
    def build_ufunc(self): ...
    def expected_ndims(self): ...
    def match_signature(self, ewise_types, sig): ...
    @property
    def is_dynamic(self): ...
    def __call__(self, *args, **kwargs): ...
