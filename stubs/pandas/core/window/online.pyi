from pandas.compat._optional import import_optional_dependency as import_optional_dependency

TYPE_CHECKING: bool
def generate_online_numba_ewma_func(nopython: bool, nogil: bool, parallel: bool):
    """
    Generate a numba jitted groupby ewma function specified by values
    from engine_kwargs.

    Parameters
    ----------
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

class EWMMeanState:
    def __init__(self, com, adjust, ignore_na, axis, shape) -> None: ...
    def run_ewm(self, weighted_avg, deltas, min_periods, ewm_func): ...
    def reset(self) -> None: ...
