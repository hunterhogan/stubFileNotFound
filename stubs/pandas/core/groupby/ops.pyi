import numpy as np
from _typeshed import Incomplete
from collections.abc import Hashable, Iterator, Sequence
from pandas._libs import NaT as NaT, lib as lib
from pandas._typing import ArrayLike as ArrayLike, AxisInt as AxisInt, NDFrameT as NDFrameT, Shape as Shape, npt as npt
from pandas.core.dtypes.cast import maybe_cast_pointwise_result as maybe_cast_pointwise_result, maybe_downcast_to_dtype as maybe_downcast_to_dtype
from pandas.core.dtypes.common import ensure_float64 as ensure_float64, ensure_int64 as ensure_int64, ensure_platform_int as ensure_platform_int, ensure_uint64 as ensure_uint64, is_1d_only_ea_dtype as is_1d_only_ea_dtype
from pandas.core.dtypes.missing import isna as isna, maybe_fill as maybe_fill
from pandas.core.frame import DataFrame as DataFrame
from pandas.core.generic import NDFrame as NDFrame
from pandas.core.groupby import grouper as grouper
from pandas.core.indexes.api import CategoricalIndex as CategoricalIndex, Index as Index, MultiIndex as MultiIndex, ensure_index as ensure_index
from pandas.core.series import Series as Series
from pandas.core.sorting import compress_group_index as compress_group_index, decons_obs_group_ids as decons_obs_group_ids, get_flattened_list as get_flattened_list, get_group_index as get_group_index, get_group_index_sorter as get_group_index_sorter, get_indexer_dict as get_indexer_dict
from pandas.errors import AbstractMethodError as AbstractMethodError
from pandas.util._decorators import cache_readonly as cache_readonly
from typing import Callable, Generic

def check_result_array(obj, dtype) -> None: ...
def extract_result(res):
    """
    Extract the result object, it might be a 0-dim ndarray
    or a len-1 0-dim, or a scalar
    """

class WrappedCythonOp:
    '''
    Dispatch logic for functions defined in _libs.groupby

    Parameters
    ----------
    kind: str
        Whether the operation is an aggregate or transform.
    how: str
        Operation name, e.g. "mean".
    has_dropped_na: bool
        True precisely when dropna=True and the grouper contains a null value.
    '''
    cast_blocklist: Incomplete
    kind: Incomplete
    how: Incomplete
    has_dropped_na: Incomplete
    def __init__(self, kind: str, how: str, has_dropped_na: bool) -> None: ...
    _CYTHON_FUNCTIONS: dict[str, dict]
    _cython_arity: Incomplete
    @classmethod
    def get_kind_from_how(cls, how: str) -> str: ...
    @classmethod
    def _get_cython_function(cls, kind: str, how: str, dtype: np.dtype, is_numeric: bool): ...
    def _get_cython_vals(self, values: np.ndarray) -> np.ndarray:
        """
        Cast numeric dtypes to float64 for functions that only support that.

        Parameters
        ----------
        values : np.ndarray

        Returns
        -------
        values : np.ndarray
        """
    def _get_output_shape(self, ngroups: int, values: np.ndarray) -> Shape: ...
    def _get_out_dtype(self, dtype: np.dtype) -> np.dtype: ...
    def _get_result_dtype(self, dtype: np.dtype) -> np.dtype:
        """
        Get the desired dtype of a result based on the
        input dtype and how it was computed.

        Parameters
        ----------
        dtype : np.dtype

        Returns
        -------
        np.dtype
            The desired dtype of the result.
        """
    def _cython_op_ndim_compat(self, values: np.ndarray, *, min_count: int, ngroups: int, comp_ids: np.ndarray, mask: npt.NDArray[np.bool_] | None = None, result_mask: npt.NDArray[np.bool_] | None = None, **kwargs) -> np.ndarray: ...
    def _call_cython_op(self, values: np.ndarray, *, min_count: int, ngroups: int, comp_ids: np.ndarray, mask: npt.NDArray[np.bool_] | None, result_mask: npt.NDArray[np.bool_] | None, **kwargs) -> np.ndarray: ...
    def _validate_axis(self, axis: AxisInt, values: ArrayLike) -> None: ...
    def cython_operation(self, *, values: ArrayLike, axis: AxisInt, min_count: int = -1, comp_ids: np.ndarray, ngroups: int, **kwargs) -> ArrayLike:
        """
        Call our cython function, with appropriate pre- and post- processing.
        """

class BaseGrouper:
    """
    This is an internal Grouper class, which actually holds
    the generated groups

    Parameters
    ----------
    axis : Index
    groupings : Sequence[Grouping]
        all the grouping instances to handle in this grouper
        for example for grouper list to groupby, need to pass the list
    sort : bool, default True
        whether this grouper will give sorted result or not

    """
    axis: Index
    _groupings: list[grouper.Grouping]
    _sort: Incomplete
    dropna: Incomplete
    def __init__(self, axis: Index, groupings: Sequence[grouper.Grouping], sort: bool = True, dropna: bool = True) -> None: ...
    @property
    def groupings(self) -> list[grouper.Grouping]: ...
    @property
    def shape(self) -> Shape: ...
    def __iter__(self) -> Iterator[Hashable]: ...
    @property
    def nkeys(self) -> int: ...
    def get_iterator(self, data: NDFrameT, axis: AxisInt = 0) -> Iterator[tuple[Hashable, NDFrameT]]:
        """
        Groupby iterator

        Returns
        -------
        Generator yielding sequence of (name, subsetted object)
        for each group
        """
    def _get_splitter(self, data: NDFrame, axis: AxisInt = 0) -> DataSplitter:
        """
        Returns
        -------
        Generator yielding subsetted objects
        """
    def group_keys_seq(self): ...
    def indices(self) -> dict[Hashable, npt.NDArray[np.intp]]:
        """dict {group name -> group indices}"""
    def result_ilocs(self) -> npt.NDArray[np.intp]:
        """
        Get the original integer locations of result_index in the input.
        """
    @property
    def codes(self) -> list[npt.NDArray[np.signedinteger]]: ...
    @property
    def levels(self) -> list[Index]: ...
    @property
    def names(self) -> list[Hashable]: ...
    def size(self) -> Series:
        """
        Compute group sizes.
        """
    def groups(self) -> dict[Hashable, np.ndarray]:
        """dict {group name -> group labels}"""
    def is_monotonic(self) -> bool: ...
    def has_dropped_na(self) -> bool:
        """
        Whether grouper has null value(s) that are dropped.
        """
    def group_info(self) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], int]: ...
    def codes_info(self) -> npt.NDArray[np.intp]: ...
    def _get_compressed_codes(self) -> tuple[npt.NDArray[np.signedinteger], npt.NDArray[np.intp]]: ...
    def ngroups(self) -> int: ...
    @property
    def reconstructed_codes(self) -> list[npt.NDArray[np.intp]]: ...
    def result_index(self) -> Index: ...
    def get_group_levels(self) -> list[ArrayLike]: ...
    def _cython_operation(self, kind: str, values, how: str, axis: AxisInt, min_count: int = -1, **kwargs) -> ArrayLike:
        """
        Returns the values of a cython operation.
        """
    def agg_series(self, obj: Series, func: Callable, preserve_dtype: bool = False) -> ArrayLike:
        """
        Parameters
        ----------
        obj : Series
        func : function taking a Series and returning a scalar-like
        preserve_dtype : bool
            Whether the aggregation is known to be dtype-preserving.

        Returns
        -------
        np.ndarray or ExtensionArray
        """
    def _aggregate_series_pure_python(self, obj: Series, func: Callable) -> npt.NDArray[np.object_]: ...
    def apply_groupwise(self, f: Callable, data: DataFrame | Series, axis: AxisInt = 0) -> tuple[list, bool]: ...
    def _sort_idx(self) -> npt.NDArray[np.intp]: ...
    def _sorted_ids(self) -> npt.NDArray[np.intp]: ...

class BinGrouper(BaseGrouper):
    """
    This is an internal Grouper class

    Parameters
    ----------
    bins : the split index of binlabels to group the item of axis
    binlabels : the label list
    indexer : np.ndarray[np.intp], optional
        the indexer created by Grouper
        some groupers (TimeGrouper) will sort its axis and its
        group_info is also sorted, so need the indexer to reorder

    Examples
    --------
    bins: [2, 4, 6, 8, 10]
    binlabels: DatetimeIndex(['2005-01-01', '2005-01-03',
        '2005-01-05', '2005-01-07', '2005-01-09'],
        dtype='datetime64[ns]', freq='2D')

    the group_info, which contains the label of each item in grouped
    axis, the index of label in label list, group number, is

    (array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]), array([0, 1, 2, 3, 4]), 5)

    means that, the grouped axis has 10 items, can be grouped into 5
    labels, the first and second items belong to the first label, the
    third and forth items belong to the second label, and so on

    """
    bins: npt.NDArray[np.int64]
    binlabels: Index
    indexer: Incomplete
    def __init__(self, bins, binlabels, indexer: Incomplete | None = None) -> None: ...
    def groups(self):
        """dict {group name -> group labels}"""
    @property
    def nkeys(self) -> int: ...
    def codes_info(self) -> npt.NDArray[np.intp]: ...
    def get_iterator(self, data: NDFrame, axis: AxisInt = 0):
        """
        Groupby iterator

        Returns
        -------
        Generator yielding sequence of (name, subsetted object)
        for each group
        """
    def indices(self): ...
    def group_info(self) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], int]: ...
    def reconstructed_codes(self) -> list[np.ndarray]: ...
    def result_index(self) -> Index: ...
    @property
    def levels(self) -> list[Index]: ...
    @property
    def names(self) -> list[Hashable]: ...
    @property
    def groupings(self) -> list[grouper.Grouping]: ...

def _is_indexed_like(obj, axes, axis: AxisInt) -> bool: ...

class DataSplitter(Generic[NDFrameT]):
    data: Incomplete
    labels: Incomplete
    ngroups: Incomplete
    _slabels: Incomplete
    _sort_idx: Incomplete
    axis: Incomplete
    def __init__(self, data: NDFrameT, labels: npt.NDArray[np.intp], ngroups: int, *, sort_idx: npt.NDArray[np.intp], sorted_ids: npt.NDArray[np.intp], axis: AxisInt = 0) -> None: ...
    def __iter__(self) -> Iterator: ...
    def _sorted_data(self) -> NDFrameT: ...
    def _chop(self, sdata, slice_obj: slice) -> NDFrame: ...

class SeriesSplitter(DataSplitter):
    def _chop(self, sdata: Series, slice_obj: slice) -> Series: ...

class FrameSplitter(DataSplitter):
    def _chop(self, sdata: DataFrame, slice_obj: slice) -> DataFrame: ...

def _get_splitter(data: NDFrame, labels: npt.NDArray[np.intp], ngroups: int, *, sort_idx: npt.NDArray[np.intp], sorted_ids: npt.NDArray[np.intp], axis: AxisInt = 0) -> DataSplitter: ...
