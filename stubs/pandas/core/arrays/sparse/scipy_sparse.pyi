import numpy as np
import scipy.sparse
from collections.abc import Iterable
from pandas._libs import lib as lib
from pandas._typing import IndexLabel as IndexLabel, npt as npt
from pandas.core.algorithms import factorize as factorize
from pandas.core.dtypes.missing import notna as notna
from pandas.core.indexes.api import MultiIndex as MultiIndex
from pandas.core.series import Series as Series

def _check_is_partition(parts: Iterable, whole: Iterable): ...
def _levels_to_axis(ss, levels: tuple[int] | list[int], valid_ilocs: npt.NDArray[np.intp], sort_labels: bool = False) -> tuple[npt.NDArray[np.intp], list[IndexLabel]]:
    """
    For a MultiIndexed sparse Series `ss`, return `ax_coords` and `ax_labels`,
    where `ax_coords` are the coordinates along one of the two axes of the
    destination sparse matrix, and `ax_labels` are the labels from `ss`' Index
    which correspond to these coordinates.

    Parameters
    ----------
    ss : Series
    levels : tuple/list
    valid_ilocs : numpy.ndarray
        Array of integer positions of valid values for the sparse matrix in ss.
    sort_labels : bool, default False
        Sort the axis labels before forming the sparse matrix. When `levels`
        refers to a single level, set to True for a faster execution.

    Returns
    -------
    ax_coords : numpy.ndarray (axis coordinates)
    ax_labels : list (axis labels)
    """
def _to_ijv(ss, row_levels: tuple[int] | list[int] = (0,), column_levels: tuple[int] | list[int] = (1,), sort_labels: bool = False) -> tuple[np.ndarray, npt.NDArray[np.intp], npt.NDArray[np.intp], list[IndexLabel], list[IndexLabel]]:
    """
    For an arbitrary MultiIndexed sparse Series return (v, i, j, ilabels,
    jlabels) where (v, (i, j)) is suitable for passing to scipy.sparse.coo
    constructor, and ilabels and jlabels are the row and column labels
    respectively.

    Parameters
    ----------
    ss : Series
    row_levels : tuple/list
    column_levels : tuple/list
    sort_labels : bool, default False
        Sort the row and column labels before forming the sparse matrix.
        When `row_levels` and/or `column_levels` refer to a single level,
        set to `True` for a faster execution.

    Returns
    -------
    values : numpy.ndarray
        Valid values to populate a sparse matrix, extracted from
        ss.
    i_coords : numpy.ndarray (row coordinates of the values)
    j_coords : numpy.ndarray (column coordinates of the values)
    i_labels : list (row labels)
    j_labels : list (column labels)
    """
def sparse_series_to_coo(ss: Series, row_levels: Iterable[int] = (0,), column_levels: Iterable[int] = (1,), sort_labels: bool = False) -> tuple[scipy.sparse.coo_matrix, list[IndexLabel], list[IndexLabel]]:
    """
    Convert a sparse Series to a scipy.sparse.coo_matrix using index
    levels row_levels, column_levels as the row and column
    labels respectively. Returns the sparse_matrix, row and column labels.
    """
def coo_to_sparse_series(A: scipy.sparse.coo_matrix, dense_index: bool = False) -> Series:
    """
    Convert a scipy.sparse.coo_matrix to a Series with type sparse.

    Parameters
    ----------
    A : scipy.sparse.coo_matrix
    dense_index : bool, default False

    Returns
    -------
    Series

    Raises
    ------
    TypeError if A is not a coo_matrix
    """
