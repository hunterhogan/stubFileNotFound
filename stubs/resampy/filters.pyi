
from collections.abc import Callable
import numpy as np
from numpy.typing import NDArray

__all__ = ['get_filter', 'clear_cache', 'sinc_window']

"""Filter construction and loading."""

# Dictionary to cache loaded filters
FILTER_CACHE: dict[str, tuple[NDArray[np.floating], int, float]] = {}

# List of filter functions available
FILTER_FUNCTIONS = ['sinc_window']

def sinc_window(num_zeros: int = 64,
                precision: int = 9,
                window: Callable | None = None,
                rolloff: float = 0.945) -> tuple[NDArray[np.floating], int, float]:
    """Construct a windowed sinc interpolation filter

    Parameters
    ----------
    num_zeros : int > 0
        The number of zero-crossings to retain in the sinc filter
    precision : int > 0
        The number of filter coefficients to retain for each zero-crossing
    window : callable
        The window function.  By default, uses a Hann window.
    rolloff : float > 0
        The roll-off frequency (as a fraction of nyquist)

    Returns
    -------
    interp_window: np.ndarray [shape=(num_zeros * num_table + 1)]
        The interpolation window (right-hand side)
    num_bits: int
        The number of bits of precision to use in the filter table
    rolloff : float > 0
        The roll-off frequency of the filter, as a fraction of Nyquist

    Raises
    ------
    TypeError
        if `window` is not callable or `None`
    ValueError
        if `num_zeros < 1`, `precision < 1`,
        or `rolloff` is outside the range `(0, 1]`.
    """
    ...

def get_filter(name_or_function: str | Callable[..., tuple[NDArray[np.floating], int, float]],
               **kwargs) -> tuple[NDArray[np.floating], int, float]:
    """Retrieve a window given its name or function handle.

    Parameters
    ----------
    name_or_function : str or callable
        If a function, returns `name_or_function(**kwargs)`.

        If a string, and it matches the name of one of the defined
        filter functions, the corresponding function is called with `**kwargs`.

        If a string, and it matches the name of a pre-computed filter,
        the corresponding filter is retrieved, and kwargs is ignored.

        Valid pre-computed filter names are:
            - 'kaiser_fast'
            - 'kaiser_best'

    **kwargs
        Additional keyword arguments passed to `name_or_function` (if callable)

    Returns
    -------
    half_window : np.ndarray
        The right wing of the interpolation filter
    precision : int > 0
        The number of samples between zero-crossings of the filter
    rolloff : float > 0
        The roll-off frequency of the filter as a fraction of Nyquist

    Raises
    ------
    NotImplementedError
        If `name_or_function` cannot be found as a filter.
    """
    ...

def load_filter(filter_name: str) -> tuple[NDArray[np.floating], int, float]:
    """Retrieve a pre-computed filter.

    Parameters
    ----------
    filter_name : str
        The key of the filter, e.g., 'kaiser_fast'

    Returns
    -------
    half_window : np.ndarray
        The right wing of the interpolation filter
    precision : int > 0
        The number of samples between zero-crossings of the filter
    rolloff : float > 0
        The roll-off frequency of the filter, as a fraction of Nyquist
    """
    ...

def clear_cache() -> None:
    """Clear the filter cache.

    Calling this function will ensure that packaged filters are reloaded
    upon the next usage.
    """
    ...
