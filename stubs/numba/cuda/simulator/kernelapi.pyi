from .vector_types import vector_types as vector_types
from _typeshed import Incomplete
from collections.abc import Generator
from numba.core import types as types
from numba.np import numpy_support as numpy_support

class Dim3:
    """
    Used to implement thread/block indices/dimensions
    """
    x: Incomplete
    y: Incomplete
    z: Incomplete
    def __init__(self, x, y, z) -> None: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __iter__(self): ...

class GridGroup:
    """
    Used to implement the grid group.
    """
    def sync(self) -> None: ...

class FakeCUDACg:
    """
    CUDA Cooperative Groups
    """
    def this_grid(self): ...

class FakeCUDALocal:
    """
    CUDA Local arrays
    """
    def array(self, shape, dtype): ...

class FakeCUDAConst:
    """
    CUDA Const arrays
    """
    def array_like(self, ary): ...

class FakeCUDAShared:
    """
    CUDA Shared arrays.

    Limitations: assumes that only one call to cuda.shared.array is on a line,
    and that that line is only executed once per thread. i.e.::

        a = cuda.shared.array(...); b = cuda.shared.array(...)

    will erroneously alias a and b, and::

        for i in range(10):
            sharedarrs[i] = cuda.shared.array(...)

    will alias all arrays created at that point (though it is not certain that
    this would be supported by Numba anyway).
    """
    _allocations: Incomplete
    _dynshared_size: Incomplete
    _dynshared: Incomplete
    def __init__(self, dynshared_size) -> None: ...
    def array(self, shape, dtype): ...

addlock: Incomplete
sublock: Incomplete
andlock: Incomplete
orlock: Incomplete
xorlock: Incomplete
maxlock: Incomplete
minlock: Incomplete
compare_and_swaplock: Incomplete
caslock: Incomplete
inclock: Incomplete
declock: Incomplete
exchlock: Incomplete

class FakeCUDAAtomic:
    def add(self, array, index, val): ...
    def sub(self, array, index, val): ...
    def and_(self, array, index, val): ...
    def or_(self, array, index, val): ...
    def xor(self, array, index, val): ...
    def inc(self, array, index, val): ...
    def dec(self, array, index, val): ...
    def exch(self, array, index, val): ...
    def max(self, array, index, val): ...
    def min(self, array, index, val): ...
    def nanmax(self, array, index, val): ...
    def nanmin(self, array, index, val): ...
    def compare_and_swap(self, array, old, val): ...
    def cas(self, array, index, old, val): ...

class FakeCUDAFp16:
    def hadd(self, a, b): ...
    def hsub(self, a, b): ...
    def hmul(self, a, b): ...
    def hdiv(self, a, b): ...
    def hfma(self, a, b, c): ...
    def hneg(self, a): ...
    def habs(self, a): ...
    def hsin(self, x): ...
    def hcos(self, x): ...
    def hlog(self, x): ...
    def hlog2(self, x): ...
    def hlog10(self, x): ...
    def hexp(self, x): ...
    def hexp2(self, x): ...
    def hexp10(self, x): ...
    def hsqrt(self, x): ...
    def hrsqrt(self, x): ...
    def hceil(self, x): ...
    def hfloor(self, x): ...
    def hrcp(self, x): ...
    def htrunc(self, x): ...
    def hrint(self, x): ...
    def heq(self, a, b): ...
    def hne(self, a, b): ...
    def hge(self, a, b): ...
    def hgt(self, a, b): ...
    def hle(self, a, b): ...
    def hlt(self, a, b): ...
    def hmax(self, a, b): ...
    def hmin(self, a, b): ...

class FakeCUDAModule:
    """
    An instance of this class will be injected into the __globals__ for an
    executing function in order to implement calls to cuda.*. This will fail to
    work correctly if the user code does::

        from numba import cuda as something_else

    In other words, the CUDA module must be called cuda.
    """
    gridDim: Incomplete
    blockDim: Incomplete
    _cg: Incomplete
    _local: Incomplete
    _shared: Incomplete
    _const: Incomplete
    _atomic: Incomplete
    _fp16: Incomplete
    def __init__(self, grid_dim, block_dim, dynshared_size) -> None: ...
    @property
    def cg(self): ...
    @property
    def local(self): ...
    @property
    def shared(self): ...
    @property
    def const(self): ...
    @property
    def atomic(self): ...
    @property
    def fp16(self): ...
    @property
    def threadIdx(self): ...
    @property
    def blockIdx(self): ...
    @property
    def warpsize(self): ...
    @property
    def laneid(self): ...
    def syncthreads(self) -> None: ...
    def threadfence(self) -> None: ...
    def threadfence_block(self) -> None: ...
    def threadfence_system(self) -> None: ...
    def syncthreads_count(self, val): ...
    def syncthreads_and(self, val): ...
    def syncthreads_or(self, val): ...
    def popc(self, val): ...
    def fma(self, a, b, c): ...
    def cbrt(self, a): ...
    def brev(self, val): ...
    def clz(self, val): ...
    def ffs(self, val): ...
    def selp(self, a, b, c): ...
    def grid(self, n): ...
    def gridsize(self, n): ...

def swapped_cuda_module(fn, fake_cuda_module) -> Generator[None]: ...
