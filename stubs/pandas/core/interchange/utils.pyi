import pandas as pd
from _typeshed import Incomplete
from pandas._typing import DtypeObj as DtypeObj
from pandas.core.dtypes.dtypes import ArrowDtype as ArrowDtype, CategoricalDtype as CategoricalDtype, DatetimeTZDtype as DatetimeTZDtype

PYARROW_CTYPES: Incomplete

class ArrowCTypes:
    """
    Enum for Apache Arrow C type format strings.

    The Arrow C data interface:
    https://arrow.apache.org/docs/format/CDataInterface.html#data-type-description-format-strings
    """
    NULL: str
    BOOL: str
    INT8: str
    UINT8: str
    INT16: str
    UINT16: str
    INT32: str
    UINT32: str
    INT64: str
    UINT64: str
    FLOAT16: str
    FLOAT32: str
    FLOAT64: str
    STRING: str
    LARGE_STRING: str
    DATE32: str
    DATE64: str
    TIMESTAMP: str
    TIME: str

class Endianness:
    """Enum indicating the byte-order of a data-type."""
    LITTLE: str
    BIG: str
    NATIVE: str
    NA: str

def dtype_to_arrow_c_fmt(dtype: DtypeObj) -> str:
    """
    Represent pandas `dtype` as a format string in Apache Arrow C notation.

    Parameters
    ----------
    dtype : np.dtype
        Datatype of pandas DataFrame to represent.

    Returns
    -------
    str
        Format string in Apache Arrow C notation of the given `dtype`.
    """
def maybe_rechunk(series: pd.Series, *, allow_copy: bool) -> pd.Series | None:
    """
    Rechunk a multi-chunk pyarrow array into a single-chunk array, if necessary.

    - Returns `None` if the input series is not backed by a multi-chunk pyarrow array
      (and so doesn't need rechunking)
    - Returns a single-chunk-backed-Series if the input is backed by a multi-chunk
      pyarrow array and `allow_copy` is `True`.
    - Raises a `RuntimeError` if `allow_copy` is `False` and input is a
      based by a multi-chunk pyarrow array.
    """
