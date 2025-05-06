from collections.abc import Sequence
from pandas import DataFrame as DataFrame
from pandas._libs import lib as lib
from pandas._typing import DtypeBackend as DtypeBackend
from pathlib import Path

def read_spss(path: str | Path, usecols: Sequence[str] | None = None, convert_categoricals: bool = True, dtype_backend: DtypeBackend | lib.NoDefault = ...) -> DataFrame:
    '''
    Load an SPSS file from the file path, returning a DataFrame.

    Parameters
    ----------
    path : str or Path
        File path.
    usecols : list-like, optional
        Return a subset of the columns. If None, return all columns.
    convert_categoricals : bool, default is True
        Convert categorical columns into pd.Categorical.
    dtype_backend : {\'numpy_nullable\', \'pyarrow\'}, default \'numpy_nullable\'
        Back-end data type applied to the resultant :class:`DataFrame`
        (still experimental). Behaviour is as follows:

        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
          (default).
        * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype`
          DataFrame.

        .. versionadded:: 2.0

    Returns
    -------
    DataFrame

    Examples
    --------
    >>> df = pd.read_spss("spss_data.sav")  # doctest: +SKIP
    '''
