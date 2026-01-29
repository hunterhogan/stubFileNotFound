import numpy as np
from numpy.typing import NDArray
from numba import guvectorize
import numba

def _resample_loop(x: NDArray[np.floating],
                  t_out: NDArray[np.floating],
                  interp_win: NDArray[np.floating],
                  interp_delta: NDArray[np.floating],
                  num_table: int,
                  scale: float,
                  y: NDArray[np.floating]) -> None:
    """Core resampling loop implementation.

    Parameters
    ----------
    x : np.ndarray
        Input signal
    t_out : np.ndarray
        Output time points
    interp_win : np.ndarray
        Interpolation window
    interp_delta : np.ndarray
        Interpolation window delta values
    num_table : int
        Precision of the filter
    scale : float
        Scaling factor for resampling
    y : np.ndarray
        Output buffer
    """
    ...

# JIT-compiled parallel version of _resample_loop
_resample_loop_p = ...

# JIT-compiled sequential version of _resample_loop
_resample_loop_s = ...

@guvectorize((numba.float32[:, :, :], numba.float32[:, :], numba.float32[:], numba.float32[:], numba.int32, numba.float32, numba.float32[:, :]), "(n),(m),(p),(p),(),()->(m)", nopython=True)
def resample_f_p(x: NDArray[np.floating],
                t_out: NDArray[np.floating],
                interp_win: NDArray[np.floating],
                interp_delta: NDArray[np.floating],
                num_table: int,
                scale: float,
                y: NDArray[np.floating]) -> None:
    """Parallelized gufunc implementation of resampling.

    Parameters
    ----------
    x : np.ndarray
        Input signal
    t_out : np.ndarray
        Output time points
    interp_win : np.ndarray
        Interpolation window
    interp_delta : np.ndarray
        Interpolation window delta values
    num_table : int
        Precision of the filter
    scale : float
        Scaling factor for resampling
    y : np.ndarray
        Output buffer (modified in place)
    """
    ...
@guvectorize((numba.float32[:, :, :], numba.float32[:, :], numba.float32[:], numba.float32[:], numba.int32, numba.float32, numba.float32[:, :]), "(n),(m),(p),(p),(),()->(m)", nopython=True)
def resample_f_s(x: NDArray[np.floating],
                t_out: NDArray[np.floating],
                interp_win: NDArray[np.floating],
                interp_delta: NDArray[np.floating],
                num_table: int,
                scale: float,
                y: NDArray[np.floating]) -> None:
    """Sequential gufunc implementation of resampling.

    Parameters
    ----------
    x : np.ndarray
        Input signal
    t_out : np.ndarray
        Output time points
    interp_win : np.ndarray
        Interpolation window
    interp_delta : np.ndarray
        Interpolation window delta values
    num_table : int
        Precision of the filter
    scale : float
        Scaling factor for resampling
    y : np.ndarray
        Output buffer (modified in place)
    """
    ...
