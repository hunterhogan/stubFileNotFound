import np
import numpy.dtypes
import pandas._libs.lib as lib
from pandas._libs.tslibs.timedeltas import array_to_timedelta64 as array_to_timedelta64
from pandas.core.dtypes.base import ExtensionDtype as ExtensionDtype
from pandas.core.dtypes.common import is_object_dtype as is_object_dtype, is_string_dtype as is_string_dtype, pandas_dtype as pandas_dtype
from pandas.core.dtypes.dtypes import NumpyEADtype as NumpyEADtype
from pandas.errors import IntCastingNaNError as IntCastingNaNError

TYPE_CHECKING: bool
_dtype_obj: numpy.dtypes.ObjectDType
def _astype_nansafe(arr: np.ndarray, dtype: DtypeObj, copy: bool = ..., skipna: bool = ...) -> ArrayLike:
    """
    Cast the elements of an array to a given dtype a nan-safe manner.

    Parameters
    ----------
    arr : ndarray
    dtype : np.dtype or ExtensionDtype
    copy : bool, default True
        If False, a view will be attempted but may fail, if
        e.g. the item sizes don't align.
    skipna: bool, default False
        Whether or not we should skip NaN when casting as a string-type.

    Raises
    ------
    ValueError
        The dtype was a datetime64/timedelta64 dtype, but it had no unit.
    """
def _astype_float_to_int_nansafe(values: np.ndarray, dtype: np.dtype, copy: bool) -> np.ndarray:
    """
    astype with a check preventing converting NaN to an meaningless integer value.
    """
def astype_array(values: ArrayLike, dtype: DtypeObj, copy: bool = ...) -> ArrayLike:
    """
    Cast array (ndarray or ExtensionArray) to the new dtype.

    Parameters
    ----------
    values : ndarray or ExtensionArray
    dtype : dtype object
    copy : bool, default False
        copy if indicated

    Returns
    -------
    ndarray or ExtensionArray
    """
def astype_array_safe(values: ArrayLike, dtype, copy: bool = ..., errors: IgnoreRaise = ...) -> ArrayLike:
    """
    Cast array (ndarray or ExtensionArray) to the new dtype.

    This basically is the implementation for DataFrame/Series.astype and
    includes all custom logic for pandas (NaN-safety, converting str to object,
    not allowing )

    Parameters
    ----------
    values : ndarray or ExtensionArray
    dtype : str, dtype convertible
    copy : bool, default False
        copy if indicated
    errors : str, {'raise', 'ignore'}, default 'raise'
        - ``raise`` : allow exceptions to be raised
        - ``ignore`` : suppress exceptions. On error return original object

    Returns
    -------
    ndarray or ExtensionArray
    """
def astype_is_view(dtype: DtypeObj, new_dtype: DtypeObj) -> bool:
    """Checks if astype avoided copying the data.

    Parameters
    ----------
    dtype : Original dtype
    new_dtype : target dtype

    Returns
    -------
    True if new data is a view or not guaranteed to be a copy, False otherwise
    """
