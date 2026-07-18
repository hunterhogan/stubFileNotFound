from _typeshed import Incomplete
from numba import jit as jit, typeof as typeof
from numba.core import cgutils as cgutils, errors as errors, serialize as serialize, sigutils as sigutils, types as types
from numba.core.compiler_lock import global_compiler_lock as global_compiler_lock
from numba.core.extending import intrinsic as intrinsic, is_jitted as is_jitted, overload_attribute as overload_attribute, overload_method as overload_method, register_jitable as register_jitable
from numba.core.typing import npydecl as npydecl
from numba.core.typing.templates import AbstractTemplate as AbstractTemplate, signature as signature
from numba.cpython.unsafe.tuple import tuple_setitem as tuple_setitem
from numba.np import numpy_support as numpy_support
from numba.np.ufunc import _internal, ufuncbuilder as ufuncbuilder
from numba.np.ufunc.ufunc_base import UfuncBase as UfuncBase, UfuncLowererBase as UfuncLowererBase
from numba.parfors import array_analysis as array_analysis

class UfuncAtIterator:
    ufunc: Incomplete
    a: Incomplete
    a_ty: Incomplete
    indices: Incomplete
    indices_ty: Incomplete
    b: Incomplete
    b_ty: Incomplete
    def __init__(self, ufunc, a, a_ty, indices, indices_ty, b=None, b_ty=None) -> None: ...
    def run(self, context, builder) -> None: ...
    def need_advanced_indexing(self): ...

def make_dufunc_kernel(_dufunc): ...

class DUFuncLowerer(UfuncLowererBase):
    """Callable class responsible for lowering calls to a specific DUFunc.
    """
    def __init__(self, dufunc) -> None: ...

class DUFunc(serialize.ReduceMixin, _internal._DUFunc, UfuncBase):
    """
    Dynamic universal function (DUFunc) intended to act like a normal
    Numpy ufunc, but capable of call-time (just-in-time) compilation
    of fast loops specialized to inputs.
    """
    def __init__(self, py_func, identity=None, cache: bool = False, targetoptions=None) -> None: ...
    def build_ufunc(self):
        """
        For compatibility with the various *UFuncBuilder classes.
        """
    @property
    def targetoptions(self): ...
    @property
    def nin(self): ...
    @property
    def nout(self): ...
    @property
    def nargs(self): ...
    @property
    def ntypes(self): ...
    @property
    def types(self): ...
    @property
    def identity(self): ...
    @property
    def signature(self): ...
    def disable_compile(self) -> None:
        """
        Disable the compilation of new signatures at call time.
        """
    def add(self, sig):
        """
        Compile the DUFunc for the given signature.
        """
    def __call__(self, *args, **kws):
        """
        Allow any argument that has overridden __array_ufunc__ (NEP-18)
        to take control of DUFunc.__call__.
        """
    def match_signature(self, ewise_types, sig): ...
    def at(self, a, indices, b=None): ...
    def find_ewise_function(self, ewise_types):
        """
        Given a tuple of element-wise argument types, find a matching
        signature in the dispatcher.

        Return a 2-tuple containing the matching signature, and
        compilation result.  Will return two None's if no matching
        signature was found.
        """
