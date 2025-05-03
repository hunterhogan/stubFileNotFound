import numpy as np
from _typeshed import Incomplete
from collections.abc import Hashable, Iterable, Sequence
from pandas import MultiIndex as MultiIndex, Series as Series
from pandas._libs import algos as algos, hashtable as hashtable, lib as lib
from pandas._libs.hashtable import unique_label_indices as unique_label_indices
from pandas._typing import ArrayLike as ArrayLike, AxisInt as AxisInt, IndexKeyFunc as IndexKeyFunc, Level as Level, NaPosition as NaPosition, Shape as Shape, SortKind as SortKind, npt as npt
from pandas.core.arrays import ExtensionArray as ExtensionArray
from pandas.core.construction import extract_array as extract_array
from pandas.core.dtypes.common import ensure_int64 as ensure_int64, ensure_platform_int as ensure_platform_int
from pandas.core.dtypes.generic import ABCMultiIndex as ABCMultiIndex, ABCRangeIndex as ABCRangeIndex
from pandas.core.dtypes.missing import isna as isna
from pandas.core.indexes.base import Index as Index
from typing import Callable

def get_indexer_indexer(target: Index, level: Level | list[Level] | None, ascending: list[bool] | bool, kind: SortKind, na_position: NaPosition, sort_remaining: bool, key: IndexKeyFunc) -> npt.NDArray[np.intp] | None:
    """
    Helper method that return the indexer according to input parameters for
    the sort_index method of DataFrame and Series.

    Parameters
    ----------
    target : Index
    level : int or level name or list of ints or list of level names
    ascending : bool or list of bools, default True
    kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}
    na_position : {'first', 'last'}
    sort_remaining : bool
    key : callable, optional

    Returns
    -------
    Optional[ndarray[intp]]
        The indexer for the new index.
    """
def get_group_index(labels, shape: Shape, sort: bool, xnull: bool) -> npt.NDArray[np.int64]:
    """
    For the particular label_list, gets the offsets into the hypothetical list
    representing the totally ordered cartesian product of all possible label
    combinations, *as long as* this space fits within int64 bounds;
    otherwise, though group indices identify unique combinations of
    labels, they cannot be deconstructed.
    - If `sort`, rank of returned ids preserve lexical ranks of labels.
      i.e. returned id's can be used to do lexical sort on labels;
    - If `xnull` nulls (-1 labels) are passed through.

    Parameters
    ----------
    labels : sequence of arrays
        Integers identifying levels at each location
    shape : tuple[int, ...]
        Number of unique levels at each location
    sort : bool
        If the ranks of returned ids should match lexical ranks of labels
    xnull : bool
        If true nulls are excluded. i.e. -1 values in the labels are
        passed through.

    Returns
    -------
    An array of type int64 where two elements are equal if their corresponding
    labels are equal at all location.

    Notes
    -----
    The length of `labels` and `shape` must be identical.
    """
def get_compressed_ids(labels, sizes: Shape) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.int64]]:
    """
    Group_index is offsets into cartesian product of all possible labels. This
    space can be huge, so this function compresses it, by computing offsets
    (comp_ids) into the list of unique labels (obs_group_ids).

    Parameters
    ----------
    labels : list of label arrays
    sizes : tuple[int] of size of the levels

    Returns
    -------
    np.ndarray[np.intp]
        comp_ids
    np.ndarray[np.int64]
        obs_group_ids
    """
def is_int64_overflow_possible(shape: Shape) -> bool: ...
def _decons_group_index(comp_labels: npt.NDArray[np.intp], shape: Shape) -> list[npt.NDArray[np.intp]]: ...
def decons_obs_group_ids(comp_ids: npt.NDArray[np.intp], obs_ids: npt.NDArray[np.intp], shape: Shape, labels: Sequence[npt.NDArray[np.signedinteger]], xnull: bool) -> list[npt.NDArray[np.intp]]:
    """
    Reconstruct labels from observed group ids.

    Parameters
    ----------
    comp_ids : np.ndarray[np.intp]
    obs_ids: np.ndarray[np.intp]
    shape : tuple[int]
    labels : Sequence[np.ndarray[np.signedinteger]]
    xnull : bool
        If nulls are excluded; i.e. -1 labels are passed through.
    """
def lexsort_indexer(keys: Sequence[ArrayLike | Index | Series], orders: Incomplete | None = None, na_position: str = 'last', key: Callable | None = None, codes_given: bool = False) -> npt.NDArray[np.intp]:
    '''
    Performs lexical sorting on a set of keys

    Parameters
    ----------
    keys : Sequence[ArrayLike | Index | Series]
        Sequence of arrays to be sorted by the indexer
        Sequence[Series] is only if key is not None.
    orders : bool or list of booleans, optional
        Determines the sorting order for each element in keys. If a list,
        it must be the same length as keys. This determines whether the
        corresponding element in keys should be sorted in ascending
        (True) or descending (False) order. if bool, applied to all
        elements as above. if None, defaults to True.
    na_position : {\'first\', \'last\'}, default \'last\'
        Determines placement of NA elements in the sorted list ("last" or "first")
    key : Callable, optional
        Callable key function applied to every element in keys before sorting
    codes_given: bool, False
        Avoid categorical materialization if codes are already provided.

    Returns
    -------
    np.ndarray[np.intp]
    '''
def nargsort(items: ArrayLike | Index | Series, kind: SortKind = 'quicksort', ascending: bool = True, na_position: str = 'last', key: Callable | None = None, mask: npt.NDArray[np.bool_] | None = None) -> npt.NDArray[np.intp]:
    """
    Intended to be a drop-in replacement for np.argsort which handles NaNs.

    Adds ascending, na_position, and key parameters.

    (GH #6399, #5231, #27237)

    Parameters
    ----------
    items : np.ndarray, ExtensionArray, Index, or Series
    kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, default 'quicksort'
    ascending : bool, default True
    na_position : {'first', 'last'}, default 'last'
    key : Optional[Callable], default None
    mask : Optional[np.ndarray[bool]], default None
        Passed when called by ExtensionArray.argsort.

    Returns
    -------
    np.ndarray[np.intp]
    """
def nargminmax(values: ExtensionArray, method: str, axis: AxisInt = 0):
    '''
    Implementation of np.argmin/argmax but for ExtensionArray and which
    handles missing values.

    Parameters
    ----------
    values : ExtensionArray
    method : {"argmax", "argmin"}
    axis : int, default 0

    Returns
    -------
    int
    '''
def _nanargminmax(values: np.ndarray, mask: npt.NDArray[np.bool_], func) -> int:
    """
    See nanargminmax.__doc__.
    """
def _ensure_key_mapped_multiindex(index: MultiIndex, key: Callable, level: Incomplete | None = None) -> MultiIndex:
    """
    Returns a new MultiIndex in which key has been applied
    to all levels specified in level (or all levels if level
    is None). Used for key sorting for MultiIndex.

    Parameters
    ----------
    index : MultiIndex
        Index to which to apply the key function on the
        specified levels.
    key : Callable
        Function that takes an Index and returns an Index of
        the same shape. This key is applied to each level
        separately. The name of the level can be used to
        distinguish different levels for application.
    level : list-like, int or str, default None
        Level or list of levels to apply the key function to.
        If None, key function is applied to all levels. Other
        levels are left unchanged.

    Returns
    -------
    labels : MultiIndex
        Resulting MultiIndex with modified levels.
    """
def ensure_key_mapped(values: ArrayLike | Index | Series, key: Callable | None, levels: Incomplete | None = None) -> ArrayLike | Index | Series:
    """
    Applies a callable key function to the values function and checks
    that the resulting value has the same shape. Can be called on Index
    subclasses, Series, DataFrames, or ndarrays.

    Parameters
    ----------
    values : Series, DataFrame, Index subclass, or ndarray
    key : Optional[Callable], key to be called on the values array
    levels : Optional[List], if values is a MultiIndex, list of levels to
    apply the key to.
    """
def get_flattened_list(comp_ids: npt.NDArray[np.intp], ngroups: int, levels: Iterable[Index], labels: Iterable[np.ndarray]) -> list[tuple]:
    """Map compressed group id -> key tuple."""
def get_indexer_dict(label_list: list[np.ndarray], keys: list[Index]) -> dict[Hashable, npt.NDArray[np.intp]]:
    """
    Returns
    -------
    dict:
        Labels mapped to indexers.
    """
def get_group_index_sorter(group_index: npt.NDArray[np.intp], ngroups: int | None = None) -> npt.NDArray[np.intp]:
    """
    algos.groupsort_indexer implements `counting sort` and it is at least
    O(ngroups), where
        ngroups = prod(shape)
        shape = map(len, keys)
    that is, linear in the number of combinations (cartesian product) of unique
    values of groupby keys. This can be huge when doing multi-key groupby.
    np.argsort(kind='mergesort') is O(count x log(count)) where count is the
    length of the data-frame;
    Both algorithms are `stable` sort and that is necessary for correctness of
    groupby operations. e.g. consider:
        df.groupby(key)[col].transform('first')

    Parameters
    ----------
    group_index : np.ndarray[np.intp]
        signed integer dtype
    ngroups : int or None, default None

    Returns
    -------
    np.ndarray[np.intp]
    """
def compress_group_index(group_index: npt.NDArray[np.int64], sort: bool = True) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """
    Group_index is offsets into cartesian product of all possible labels. This
    space can be huge, so this function compresses it, by computing offsets
    (comp_ids) into the list of unique labels (obs_group_ids).
    """
def _reorder_by_uniques(uniques: npt.NDArray[np.int64], labels: npt.NDArray[np.intp]) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.intp]]:
    """
    Parameters
    ----------
    uniques : np.ndarray[np.int64]
    labels : np.ndarray[np.intp]

    Returns
    -------
    np.ndarray[np.int64]
    np.ndarray[np.intp]
    """
