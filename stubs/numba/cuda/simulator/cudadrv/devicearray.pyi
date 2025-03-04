from _typeshed import Incomplete
from collections.abc import Generator
from numba.np.numpy_support import numpy_version as numpy_version

DeviceRecord: Incomplete
from_record_like: Incomplete
errmsg_contiguous_buffer: str

class FakeShape(tuple):
    """
    The FakeShape class is used to provide a shape which does not allow negative
    indexing, similar to the shape in CUDA Python. (Numpy shape arrays allow
    negative indexing)
    """
    def __getitem__(self, k): ...

class FakeWithinKernelCUDAArray:
    """
    Created to emulate the behavior of arrays within kernels, where either
    array.item or array['item'] is valid (that is, give all structured
    arrays `numpy.recarray`-like semantics). This behaviour does not follow
    the semantics of Python and NumPy with non-jitted code, and will be
    deprecated and removed.
    """
    def __init__(self, item) -> None: ...
    def __wrap_if_fake(self, item): ...
    def __getattr__(self, attrname): ...
    def __setattr__(self, nm, val) -> None: ...
    def __getitem__(self, idx): ...
    def __setitem__(self, idx, val) -> None: ...
    def __len__(self) -> int: ...
    def __array_ufunc__(self, ufunc, method, *args, **kwargs): ...

class FakeCUDAArray:
    """
    Implements the interface of a DeviceArray/DeviceRecord, but mostly just
    wraps a NumPy array.
    """
    __cuda_ndarray__: bool
    _ary: Incomplete
    stream: Incomplete
    def __init__(self, ary, stream: int = 0) -> None: ...
    @property
    def alloc_size(self): ...
    @property
    def nbytes(self): ...
    def __getattr__(self, attrname): ...
    def bind(self, stream: int = 0): ...
    @property
    def T(self): ...
    def transpose(self, axes: Incomplete | None = None): ...
    def __getitem__(self, idx): ...
    def __setitem__(self, idx, val) -> None: ...
    def copy_to_host(self, ary: Incomplete | None = None, stream: int = 0): ...
    def copy_to_device(self, ary, stream: int = 0) -> None:
        """
        Copy from the provided array into this array.

        This may be less forgiving than the CUDA Python implementation, which
        will copy data up to the length of the smallest of the two arrays,
        whereas this expects the size of the arrays to be equal.
        """
    @property
    def shape(self): ...
    def ravel(self, *args, **kwargs): ...
    def reshape(self, *args, **kwargs): ...
    def view(self, *args, **kwargs): ...
    def is_c_contiguous(self): ...
    def is_f_contiguous(self): ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __len__(self) -> int: ...
    def __eq__(self, other): ...
    def __ne__(self, other): ...
    def __lt__(self, other): ...
    def __le__(self, other): ...
    def __gt__(self, other): ...
    def __ge__(self, other): ...
    def __add__(self, other): ...
    def __sub__(self, other): ...
    def __mul__(self, other): ...
    def __floordiv__(self, other): ...
    def __truediv__(self, other): ...
    def __mod__(self, other): ...
    def __pow__(self, other): ...
    def split(self, section, stream: int = 0): ...

def array_core(ary):
    """
    Extract the repeated core of a broadcast array.

    Broadcast arrays are by definition non-contiguous due to repeated
    dimensions, i.e., dimensions with stride 0. In order to ascertain memory
    contiguity and copy the underlying data from such arrays, we must create
    a view without the repeated dimensions.

    """
def is_contiguous(ary):
    """
    Returns True iff `ary` is C-style contiguous while ignoring
    broadcasted and 1-sized dimensions.
    As opposed to array_core(), it does not call require_context(),
    which can be quite expensive.
    """
def sentry_contiguous(ary) -> None: ...
def check_array_compatibility(ary1, ary2) -> None: ...
def to_device(ary, stream: int = 0, copy: bool = True, to: Incomplete | None = None): ...
def pinned(arg) -> Generator[None]: ...
def mapped_array(*args, **kwargs): ...
def pinned_array(shape, dtype=..., strides: Incomplete | None = None, order: str = 'C'): ...
def managed_array(shape, dtype=..., strides: Incomplete | None = None, order: str = 'C'): ...
def device_array(*args, **kwargs): ...
def _contiguous_strides_like_array(ary):
    """
    Given an array, compute strides for a new contiguous array of the same
    shape.
    """
def _order_like_array(ary): ...
def device_array_like(ary, stream: int = 0): ...
def pinned_array_like(ary): ...
def auto_device(ary, stream: int = 0, copy: bool = True): ...
def is_cuda_ndarray(obj):
    """Check if an object is a CUDA ndarray"""
def verify_cuda_ndarray_interface(obj) -> None:
    """Verify the CUDA ndarray interface for an obj"""
def require_cuda_ndarray(obj) -> None:
    """Raises ValueError is is_cuda_ndarray(obj) evaluates False"""
