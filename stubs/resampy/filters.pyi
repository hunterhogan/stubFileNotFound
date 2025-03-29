from _typeshed import Incomplete

__all__ = ['get_filter', 'clear_cache', 'sinc_window']


import sys

"""Filter construction and loading.
--------------------------------

`resampy` provides two pre-computed resampling filters which are tuned for either
high-quality or fast calculation.  These filters are constructed by the `create_filters.py`
script.

    - `kaiser_best` :
        > Parameters for kaiser_best:
        >   ----------------------------------------
        >       beta        = 12.9846
        >       roll        = 0.917347
        >       # zeros     = 50
        >       precision   = 13
        >       attenuation = -120.0
        >   ----------------------------------------

    - `kaiser_fast` :
        > Parameters for kaiser_fast:
        > ----------------------------------------
        >     beta        = 9.90322
        >     roll        = 0.868212
        >     # zeros     = 24
        >     precision   = 9
        >     attenuation = -93.0
        > ----------------------------------------


These filters can be used by calling `resample` as follows:

    >>> # High-quality
    >>> resampy.resample(x, sr_orig, sr_new, filter='kaiser_best')  # doctest: +SKIP
    >>> # Fast calculation
    >>> resampy.resample(x, sr_orig, sr_new, filter='kaiser_fast')  # doctest: +SKIP


It is also possible to construct custom filters as follows:

    >>> resampy.resample(x, sr_orig, sr_new, filter='sinc_window',
    ...                  **kwargs)                                  # doctest: +SKIP

where ``**kwargs`` are additional parameters to `sinc_window`.

"""
if sys.version_info < (3, 9):
    ...
else:
    ...
FILTER_FUNCTIONS = ...
FILTER_CACHE = ...
__all__ = ['get_filter', 'clear_cache'] + FILTER_FUNCTIONS
def sinc_window(num_zeros: int = 64, precision: int = 9, window: Incomplete | None = None, rolloff: float = 0.945):
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

    Examples
    --------
    >>> import scipy, scipy.signal
    >>> import resampy
    >>> np.set_printoptions(threshold=5, suppress=False)
    >>> # A filter with 10 zero-crossings, 32 samples per crossing, and a
    >>> # Hann window for tapering.
    >>> halfwin, prec, rolloff = resampy.filters.sinc_window(num_zeros=10, precision=5,
    ...                                                      window=scipy.signal.hann)
    >>> halfwin
    array([  9.450e-01,   9.436e-01, ...,  -7.455e-07,  -0.000e+00])
    >>> prec
    32
    >>> rolloff
    0.945

    >>> # Or using sinc-window filter construction directly in resample
    >>> y = resampy.resample(x, sr_orig, sr_new, filter='sinc_window',
    ...                      num_zeros=10, precision=5,
    ...                      window=scipy.signal.hann)              # doctest: +SKIP
    """
def get_filter(name_or_function, **kwargs):
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

def load_filter(filter_name):
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

def clear_cache(): # -> None:
    """Clear the filter cache.

    Calling this function will ensure that packaged filters are reloaded
    upon the next usage.
    """
    ...
