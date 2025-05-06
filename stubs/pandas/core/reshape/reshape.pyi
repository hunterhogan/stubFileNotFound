import numpy as np
from _typeshed import Incomplete
from pandas._typing import ArrayLike as ArrayLike, Level as Level, npt as npt
from pandas.core.algorithms import factorize as factorize, unique as unique
from pandas.core.arrays import ExtensionArray as ExtensionArray
from pandas.core.dtypes.cast import find_common_type as find_common_type, maybe_promote as maybe_promote
from pandas.core.dtypes.common import ensure_platform_int as ensure_platform_int, is_1d_only_ea_dtype as is_1d_only_ea_dtype, is_integer as is_integer, needs_i8_conversion as needs_i8_conversion
from pandas.core.frame import DataFrame as DataFrame
from pandas.core.indexes.api import Index as Index, MultiIndex as MultiIndex, RangeIndex as RangeIndex
from pandas.core.series import Series as Series
from pandas.core.sorting import compress_group_index as compress_group_index, decons_obs_group_ids as decons_obs_group_ids, get_compressed_ids as get_compressed_ids, get_group_index as get_group_index, get_group_index_sorter as get_group_index_sorter

class _Unstacker:
    '''
    Helper class to unstack data / pivot with multi-level index

    Parameters
    ----------
    index : MultiIndex
    level : int or str, default last level
        Level to "unstack". Accepts a name for the level.
    fill_value : scalar, optional
        Default value to fill in missing values if subgroups do not have the
        same set of labels. By default, missing values will be replaced with
        the default fill value for that data type, NaN for float, NaT for
        datetimelike, etc. For integer types, by default data will converted to
        float and missing values will be set to NaN.
    constructor : object
        Pandas ``DataFrame`` or subclass used to create unstacked
        response.  If None, DataFrame will be used.

    Examples
    --------
    >>> index = pd.MultiIndex.from_tuples([(\'one\', \'a\'), (\'one\', \'b\'),
    ...                                    (\'two\', \'a\'), (\'two\', \'b\')])
    >>> s = pd.Series(np.arange(1, 5, dtype=np.int64), index=index)
    >>> s
    one  a    1
         b    2
    two  a    3
         b    4
    dtype: int64

    >>> s.unstack(level=-1)
         a  b
    one  1  2
    two  3  4

    >>> s.unstack(level=0)
       one  two
    a    1    3
    b    2    4

    Returns
    -------
    unstacked : DataFrame
    '''
    constructor: Incomplete
    sort: Incomplete
    index: Incomplete
    level: Incomplete
    lift: Incomplete
    new_index_levels: Incomplete
    new_index_names: Incomplete
    removed_name: Incomplete
    removed_level: Incomplete
    removed_level_full: Incomplete
    def __init__(self, index: MultiIndex, level: Level, constructor, sort: bool = True) -> None: ...
    def _indexer_and_to_sort(self) -> tuple[npt.NDArray[np.intp], list[np.ndarray]]: ...
    def sorted_labels(self) -> list[np.ndarray]: ...
    def _make_sorted_values(self, values: np.ndarray) -> np.ndarray: ...
    full_shape: Incomplete
    group_index: Incomplete
    mask: Incomplete
    compressor: Incomplete
    def _make_selectors(self) -> None: ...
    def mask_all(self) -> bool: ...
    def arange_result(self) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.bool_]]: ...
    def get_result(self, values, value_columns, fill_value) -> DataFrame: ...
    def get_new_values(self, values, fill_value: Incomplete | None = None): ...
    def get_new_columns(self, value_columns: Index | None): ...
    def _repeater(self) -> np.ndarray: ...
    def new_index(self) -> MultiIndex: ...

def _unstack_multiple(data: Series | DataFrame, clocs, fill_value: Incomplete | None = None, sort: bool = True): ...
def unstack(obj: Series | DataFrame, level, fill_value: Incomplete | None = None, sort: bool = True): ...
def _unstack_frame(obj: DataFrame, level, fill_value: Incomplete | None = None, sort: bool = True) -> DataFrame: ...
def _unstack_extension_series(series: Series, level, fill_value, sort: bool) -> DataFrame:
    """
    Unstack an ExtensionArray-backed Series.

    The ExtensionDtype is preserved.

    Parameters
    ----------
    series : Series
        A Series with an ExtensionArray for values
    level : Any
        The level name or number.
    fill_value : Any
        The user-level (not physical storage) fill value to use for
        missing values introduced by the reshape. Passed to
        ``series.values.take``.
    sort : bool
        Whether to sort the resulting MuliIndex levels

    Returns
    -------
    DataFrame
        Each column of the DataFrame will have the same dtype as
        the input Series.
    """
def stack(frame: DataFrame, level: int = -1, dropna: bool = True, sort: bool = True):
    """
    Convert DataFrame to Series with multi-level Index. Columns become the
    second level of the resulting hierarchical index

    Returns
    -------
    stacked : Series or DataFrame
    """
def stack_multiple(frame: DataFrame, level, dropna: bool = True, sort: bool = True): ...
def _stack_multi_column_index(columns: MultiIndex) -> MultiIndex:
    """Creates a MultiIndex from the first N-1 levels of this MultiIndex."""
def _stack_multi_columns(frame: DataFrame, level_num: int = -1, dropna: bool = True, sort: bool = True) -> DataFrame: ...
def _reorder_for_extension_array_stack(arr: ExtensionArray, n_rows: int, n_columns: int) -> ExtensionArray:
    """
    Re-orders the values when stacking multiple extension-arrays.

    The indirect stacking method used for EAs requires a followup
    take to get the order correct.

    Parameters
    ----------
    arr : ExtensionArray
    n_rows, n_columns : int
        The number of rows and columns in the original DataFrame.

    Returns
    -------
    taken : ExtensionArray
        The original `arr` with elements re-ordered appropriately

    Examples
    --------
    >>> arr = np.array(['a', 'b', 'c', 'd', 'e', 'f'])
    >>> _reorder_for_extension_array_stack(arr, 2, 3)
    array(['a', 'c', 'e', 'b', 'd', 'f'], dtype='<U1')

    >>> _reorder_for_extension_array_stack(arr, 3, 2)
    array(['a', 'd', 'b', 'e', 'c', 'f'], dtype='<U1')
    """
def stack_v3(frame: DataFrame, level: list[int]) -> Series | DataFrame: ...
