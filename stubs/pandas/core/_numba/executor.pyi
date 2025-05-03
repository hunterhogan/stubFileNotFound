import functools
import np
from pandas.compat._optional import import_optional_dependency as import_optional_dependency
from typing import Callable

TYPE_CHECKING: bool
generate_apply_looper: functools._lru_cache_wrapper
make_looper: functools._lru_cache_wrapper
default_dtype_mapping: dict

float_dtype_mapping: dict
identity_dtype_mapping: dict
def generate_shared_aggregator(func: Callable[..., Scalar], dtype_mapping: dict[np.dtype, np.dtype], is_grouped_kernel: bool, nopython: bool, nogil: bool, parallel: bool):
    """
    Generate a Numba function that loops over the columns 2D object and applies
    a 1D numba kernel over each column.

    Parameters
    ----------
    func : function
        aggregation function to be applied to each column
    dtype_mapping: dict or None
        If not None, maps a dtype to a result dtype.
        Otherwise, will fall back to default mapping.
    is_grouped_kernel: bool, default False
        Whether func operates using the group labels (True)
        or using starts/ends arrays

        If true, you also need to pass the number of groups to this function
    nopython : bool
        nopython to be passed into numba.jit
    nogil : bool
        nogil to be passed into numba.jit
    parallel : bool
        parallel to be passed into numba.jit

    Returns
    -------
    Numba function
    """
