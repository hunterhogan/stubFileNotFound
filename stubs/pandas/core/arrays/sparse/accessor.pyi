import pandas.core.accessor
from _typeshed import Incomplete
from pandas.compat._optional import import_optional_dependency as import_optional_dependency
from pandas.core.accessor import PandasDelegate as PandasDelegate, delegate_names as delegate_names
from pandas.core.arrays.sparse.array import SparseArray as SparseArray
from pandas.core.dtypes.cast import find_common_type as find_common_type
from pandas.core.dtypes.dtypes import SparseDtype as SparseDtype
from typing import ClassVar

TYPE_CHECKING: bool

class BaseAccessor:
    _validation_msg: ClassVar[str] = ...
    def __init__(self, data) -> None: ...
    def _validate(self, data): ...

class SparseAccessor(BaseAccessor, pandas.core.accessor.PandasDelegate):
    npoints: Incomplete
    density: Incomplete
    fill_value: Incomplete
    sp_values: Incomplete
    def _validate(self, data): ...
    def _delegate_property_get(self, name: str, *args, **kwargs): ...
    def _delegate_method(self, name: str, *args, **kwargs): ...
    @classmethod
    def from_coo(cls, A, dense_index: bool = ...) -> Series:
        """
        Create a Series with sparse values from a scipy.sparse.coo_matrix.

        Parameters
        ----------
        A : scipy.sparse.coo_matrix
        dense_index : bool, default False
            If False (default), the index consists of only the
            coords of the non-null entries of the original coo_matrix.
            If True, the index consists of the full sorted
            (row, col) coordinates of the coo_matrix.

        Returns
        -------
        s : Series
            A Series with sparse values.

        Examples
        --------
        >>> from scipy import sparse

        >>> A = sparse.coo_matrix(
        ...     ([3.0, 1.0, 2.0], ([1, 0, 0], [0, 2, 3])), shape=(3, 4)
        ... )
        >>> A
        <COOrdinate sparse matrix of dtype 'float64'
            with 3 stored elements and shape (3, 4)>

        >>> A.todense()
        matrix([[0., 0., 1., 2.],
        [3., 0., 0., 0.],
        [0., 0., 0., 0.]])

        >>> ss = pd.Series.sparse.from_coo(A)
        >>> ss
        0  2    1.0
           3    2.0
        1  0    3.0
        dtype: Sparse[float64, nan]
        """
    def to_coo(self, row_levels: tuple = ..., column_levels: tuple = ..., sort_labels: bool = ...):
        '''
        Create a scipy.sparse.coo_matrix from a Series with MultiIndex.

        Use row_levels and column_levels to determine the row and column
        coordinates respectively. row_levels and column_levels are the names
        (labels) or numbers of the levels. {row_levels, column_levels} must be
        a partition of the MultiIndex level names (or numbers).

        Parameters
        ----------
        row_levels : tuple/list
        column_levels : tuple/list
        sort_labels : bool, default False
            Sort the row and column labels before forming the sparse matrix.
            When `row_levels` and/or `column_levels` refer to a single level,
            set to `True` for a faster execution.

        Returns
        -------
        y : scipy.sparse.coo_matrix
        rows : list (row labels)
        columns : list (column labels)

        Examples
        --------
        >>> s = pd.Series([3.0, np.nan, 1.0, 3.0, np.nan, np.nan])
        >>> s.index = pd.MultiIndex.from_tuples(
        ...     [
        ...         (1, 2, "a", 0),
        ...         (1, 2, "a", 1),
        ...         (1, 1, "b", 0),
        ...         (1, 1, "b", 1),
        ...         (2, 1, "b", 0),
        ...         (2, 1, "b", 1)
        ...     ],
        ...     names=["A", "B", "C", "D"],
        ... )
        >>> s
        A  B  C  D
        1  2  a  0    3.0
                 1    NaN
           1  b  0    1.0
                 1    3.0
        2  1  b  0    NaN
                 1    NaN
        dtype: float64

        >>> ss = s.astype("Sparse")
        >>> ss
        A  B  C  D
        1  2  a  0    3.0
                 1    NaN
           1  b  0    1.0
                 1    3.0
        2  1  b  0    NaN
                 1    NaN
        dtype: Sparse[float64, nan]

        >>> A, rows, columns = ss.sparse.to_coo(
        ...     row_levels=["A", "B"], column_levels=["C", "D"], sort_labels=True
        ... )
        >>> A
        <COOrdinate sparse matrix of dtype \'float64\'
            with 3 stored elements and shape (3, 4)>
        >>> A.todense()
        matrix([[0., 0., 1., 3.],
        [3., 0., 0., 0.],
        [0., 0., 0., 0.]])

        >>> rows
        [(1, 1), (1, 2), (2, 1)]
        >>> columns
        [(\'a\', 0), (\'a\', 1), (\'b\', 0), (\'b\', 1)]
        '''
    def to_dense(self) -> Series:
        """
        Convert a Series from sparse values to dense.

        Returns
        -------
        Series:
            A Series with the same values, stored as a dense array.

        Examples
        --------
        >>> series = pd.Series(pd.arrays.SparseArray([0, 1, 0]))
        >>> series
        0    0
        1    1
        2    0
        dtype: Sparse[int64, 0]

        >>> series.sparse.to_dense()
        0    0
        1    1
        2    0
        dtype: int64
        """

class SparseFrameAccessor(BaseAccessor, pandas.core.accessor.PandasDelegate):
    def _validate(self, data): ...
    @classmethod
    def from_spmatrix(cls, data, index, columns) -> DataFrame:
        """
        Create a new DataFrame from a scipy sparse matrix.

        Parameters
        ----------
        data : scipy.sparse.spmatrix
            Must be convertible to csc format.
        index, columns : Index, optional
            Row and column labels to use for the resulting DataFrame.
            Defaults to a RangeIndex.

        Returns
        -------
        DataFrame
            Each column of the DataFrame is stored as a
            :class:`arrays.SparseArray`.

        Examples
        --------
        >>> import scipy.sparse
        >>> mat = scipy.sparse.eye(3, dtype=float)
        >>> pd.DataFrame.sparse.from_spmatrix(mat)
             0    1    2
        0  1.0    0    0
        1    0  1.0    0
        2    0    0  1.0
        """
    def to_dense(self) -> DataFrame:
        '''
        Convert a DataFrame with sparse values to dense.

        Returns
        -------
        DataFrame
            A DataFrame with the same values stored as dense arrays.

        Examples
        --------
        >>> df = pd.DataFrame({"A": pd.arrays.SparseArray([0, 1, 0])})
        >>> df.sparse.to_dense()
           A
        0  0
        1  1
        2  0
        '''
    def to_coo(self):
        '''
        Return the contents of the frame as a sparse SciPy COO matrix.

        Returns
        -------
        scipy.sparse.spmatrix
            If the caller is heterogeneous and contains booleans or objects,
            the result will be of dtype=object. See Notes.

        Notes
        -----
        The dtype will be the lowest-common-denominator type (implicit
        upcasting); that is to say if the dtypes (even of numeric types)
        are mixed, the one that accommodates all will be chosen.

        e.g. If the dtypes are float16 and float32, dtype will be upcast to
        float32. By numpy.find_common_type convention, mixing int64 and
        and uint64 will result in a float64 dtype.

        Examples
        --------
        >>> df = pd.DataFrame({"A": pd.arrays.SparseArray([0, 1, 0, 1])})
        >>> df.sparse.to_coo()
        <COOrdinate sparse matrix of dtype \'int64\'
            with 2 stored elements and shape (4, 1)>
        '''
    @staticmethod
    def _prep_index(data, index, columns): ...
    @property
    def density(self): ...
