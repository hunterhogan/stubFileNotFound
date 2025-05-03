import functools
from pandas.compat._optional import import_optional_dependency as import_optional_dependency
from pandas.core.util.numba_ import jit_user_function as jit_user_function
from pandas.errors import NumbaUtilError as NumbaUtilError
from typing import Callable

TYPE_CHECKING: bool
def validate_udf(func: Callable) -> None:
    """
    Validate user defined function for ops when using Numba with groupby ops.

    The first signature arguments should include:

    def f(values, index, ...):
        ...

    Parameters
    ----------
    func : function, default False
        user defined function

    Returns
    -------
    None

    Raises
    ------
    NumbaUtilError
    """

generate_numba_agg_func: functools._lru_cache_wrapper
generate_numba_transform_func: functools._lru_cache_wrapper
