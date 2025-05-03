import np
import pandas._libs.reshape as libreshape
import pandas.core.algorithms as algos
from _typeshed import Incomplete
from pandas._libs.algos import ensure_platform_int as ensure_platform_int
from pandas._libs.lib import is_integer as is_integer
from pandas._libs.properties import cache_readonly as cache_readonly
from pandas.core.algorithms import factorize as factorize, unique as unique
from pandas.core.arrays.categorical import factorize_from_iterable as factorize_from_iterable
from pandas.core.construction import ensure_wrapped_if_datetimelike as ensure_wrapped_if_datetimelike
from pandas.core.dtypes.base import ExtensionDtype as ExtensionDtype
from pandas.core.dtypes.cast import find_common_type as find_common_type, maybe_promote as maybe_promote
from pandas.core.dtypes.common import is_1d_only_ea_dtype as is_1d_only_ea_dtype, needs_i8_conversion as needs_i8_conversion
from pandas.core.dtypes.missing import notna as notna
from pandas.core.frame import DataFrame as DataFrame
from pandas.core.indexes.base import Index as Index
from pandas.core.indexes.multi import MultiIndex as MultiIndex
from pandas.core.indexes.range import RangeIndex as RangeIndex
from pandas.core.reshape.concat import concat as concat
from pandas.core.series import Series as Series
from pandas.core.sorting import compress_group_index as compress_group_index, decons_obs_group_ids as decons_obs_group_ids, get_compressed_ids as get_compressed_ids, get_group_index as get_group_index, get_group_index_sorter as get_group_index_sorter
from pandas.errors import PerformanceWarning as PerformanceWarning
from pandas.util._exceptions import find_stack_level as find_stack_level

TYPE_CHECKING: bool

class _Unstacker:
    _indexer_and_to_sort: Incomplete
    sorted_labels: Incomplete
    mask_all: Incomplete
    arange_result: Incomplete
    _repeater: Incomplete
    new_index: Incomplete
    def __init__(self, index: MultiIndex, level: Level, constructor, sort: bool = ...) -> None: ...
    def _make_sorted_values(self, values: np.ndarray) -> np.ndarray: ...
    def _make_selectors(self): ...
    def get_result(self, values, value_columns, fill_value) -> DataFrame: ...
    def get_new_values(self, values, fill_value): ...
    def get_new_columns(self, value_columns: Index | None): ...
def _unstack_multiple(data: Series | DataFrame, clocs, fill_value, sort: bool = ...): ...
def unstack(obj: Series | DataFrame, level, fill_value, sort: bool = ...): ...
def _unstack_frame(obj: DataFrame, level, fill_value, sort: bool = ...) -> DataFrame: ...
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
def stack(frame: DataFrame, level: int = ..., dropna: bool = ..., sort: bool = ...):
    """
    Convert DataFrame to Series with multi-level Index. Columns become the
    second level of the resulting hierarchical index

    Returns
    -------
    stacked : Series or DataFrame
    """
def stack_multiple(frame: DataFrame, level, dropna: bool = ..., sort: bool = ...): ...
def _stack_multi_column_index(columns: MultiIndex) -> MultiIndex:
    """Creates a MultiIndex from the first N-1 levels of this MultiIndex."""
def _stack_multi_columns(frame: DataFrame, level_num: int = ..., dropna: bool = ..., sort: bool = ...) -> DataFrame: ...
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
