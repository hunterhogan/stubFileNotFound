from numba import literal_unroll as literal_unroll
from numba.core import errors as errors, types as types
from numba.core.extending import overload as overload
from numba.np.numpy_support import as_dtype as as_dtype, from_dtype as from_dtype, type_can_asarray as type_can_asarray

def roots_impl(p): ...
def polyutils_trimseq(seq): ...
def polyutils_as_series(alist, trim: bool = True): ...
def _get_list_type(l): ...
def _poly_result_dtype(*args): ...
def numpy_polyadd(c1, c2): ...
def numpy_polysub(c1, c2): ...
def numpy_polymul(c1, c2): ...
def poly_polyval(x, c, tensor: bool = True): ...
def poly_polyint(c, m: int = 1): ...
def numpy_polydiv(c1, c2): ...
