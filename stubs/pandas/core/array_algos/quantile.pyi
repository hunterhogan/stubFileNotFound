import numpy as np
from pandas._typing import ArrayLike as ArrayLike, Scalar as Scalar, npt as npt
from pandas.core.dtypes.missing import isna as isna, na_value_for_dtype as na_value_for_dtype

def quantile_compat(values: ArrayLike, qs: npt.NDArray[np.float64], interpolation: str) -> ArrayLike:
    """
    Compute the quantiles of the given values for each quantile in `qs`.

    Parameters
    ----------
    values : np.ndarray or ExtensionArray
    qs : np.ndarray[float64]
    interpolation : str

    Returns
    -------
    np.ndarray or ExtensionArray
    """
def quantile_with_mask(values: np.ndarray, mask: npt.NDArray[np.bool_], fill_value, qs: npt.NDArray[np.float64], interpolation: str) -> np.ndarray:
    """
    Compute the quantiles of the given values for each quantile in `qs`.

    Parameters
    ----------
    values : np.ndarray
        For ExtensionArray, this is _values_for_factorize()[0]
    mask : np.ndarray[bool]
        mask = isna(values)
        For ExtensionArray, this is computed before calling _value_for_factorize
    fill_value : Scalar
        The value to interpret fill NA entries with
        For ExtensionArray, this is _values_for_factorize()[1]
    qs : np.ndarray[float64]
    interpolation : str
        Type of interpolation

    Returns
    -------
    np.ndarray

    Notes
    -----
    Assumes values is already 2D.  For ExtensionArray this means np.atleast_2d
    has been called on _values_for_factorize()[0]

    Quantile is computed along axis=1.
    """
def _nanpercentile_1d(values: np.ndarray, mask: npt.NDArray[np.bool_], qs: npt.NDArray[np.float64], na_value: Scalar, interpolation: str) -> Scalar | np.ndarray:
    """
    Wrapper for np.percentile that skips missing values, specialized to
    1-dimensional case.

    Parameters
    ----------
    values : array over which to find quantiles
    mask : ndarray[bool]
        locations in values that should be considered missing
    qs : np.ndarray[float64] of quantile indices to find
    na_value : scalar
        value to return for empty or all-null values
    interpolation : str

    Returns
    -------
    quantiles : scalar or array
    """
def _nanpercentile(values: np.ndarray, qs: npt.NDArray[np.float64], *, na_value, mask: npt.NDArray[np.bool_], interpolation: str):
    """
    Wrapper for np.percentile that skips missing values.

    Parameters
    ----------
    values : np.ndarray[ndim=2]  over which to find quantiles
    qs : np.ndarray[float64] of quantile indices to find
    na_value : scalar
        value to return for empty or all-null values
    mask : np.ndarray[bool]
        locations in values that should be considered missing
    interpolation : str

    Returns
    -------
    quantiles : scalar or array
    """
