import grouper as grouper
import np
import npt
import pandas._libs.groupby as libgroupby
import pandas._libs.lib as lib
import typing
from _typeshed import Incomplete
from builtins import AxisInt, Shape
from pandas._libs.algos import ensure_float64 as ensure_float64, ensure_int64 as ensure_int64, ensure_platform_int as ensure_platform_int, ensure_uint64 as ensure_uint64
from pandas._libs.properties import cache_readonly as cache_readonly
from pandas._libs.tslibs.nattype import NaT as NaT
from pandas._typing import NDFrameT as NDFrameT
from pandas.core.dtypes.cast import maybe_cast_pointwise_result as maybe_cast_pointwise_result, maybe_downcast_to_dtype as maybe_downcast_to_dtype
from pandas.core.dtypes.common import is_1d_only_ea_dtype as is_1d_only_ea_dtype
from pandas.core.dtypes.missing import isna as isna, maybe_fill as maybe_fill
from pandas.core.frame import DataFrame as DataFrame
from pandas.core.indexes.base import Index as Index, ensure_index as ensure_index
from pandas.core.indexes.category import CategoricalIndex as CategoricalIndex
from pandas.core.indexes.multi import MultiIndex as MultiIndex
from pandas.core.series import Series as Series
from pandas.core.sorting import compress_group_index as compress_group_index, decons_obs_group_ids as decons_obs_group_ids, get_flattened_list as get_flattened_list, get_group_index as get_group_index, get_group_index_sorter as get_group_index_sorter, get_indexer_dict as get_indexer_dict
from pandas.errors import AbstractMethodError as AbstractMethodError
from typing import ArrayLike, Callable, ClassVar

TYPE_CHECKING: bool
npt: None
def check_result_array(obj, dtype) -> None: ...
def extract_result(res):
    """
    Extract the result object, it might be a 0-dim ndarray
    or a len-1 0-dim, or a scalar
    """

class WrappedCythonOp:
    cast_blocklist: ClassVar[frozenset] = ...
    _CYTHON_FUNCTIONS: ClassVar[dict] = ...
    _cython_arity: ClassVar[dict] = ...
    def __init__(self, kind: str, how: str, has_dropped_na: bool) -> None: ...
    @classmethod
    def get_kind_from_how(cls, how: str) -> str: ...
    @classmethod
    def _get_cython_function(cls, *args, **kwargs): ...
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
    def _cython_op_ndim_compat(self, values: np.ndarray, *, min_count: int, ngroups: int, comp_ids: np.ndarray, mask: npt.NDArray[np.bool_] | None, result_mask: npt.NDArray[np.bool_] | None, **kwargs) -> np.ndarray: ...
    def _call_cython_op(self, values: np.ndarray, *, min_count: int, ngroups: int, comp_ids: np.ndarray, mask: npt.NDArray[np.bool_] | None, result_mask: npt.NDArray[np.bool_] | None, **kwargs) -> np.ndarray: ...
    def _validate_axis(self, axis: AxisInt, values: ArrayLike) -> None: ...
    def cython_operation(self, *, values: ArrayLike, axis: AxisInt, min_count: int = ..., comp_ids: np.ndarray, ngroups: int, **kwargs) -> ArrayLike:
        """
        Call our cython function, with appropriate pre- and post- processing.
        """

class BaseGrouper:
    group_keys_seq: Incomplete
    indices: Incomplete
    groups: Incomplete
    is_monotonic: Incomplete
    has_dropped_na: Incomplete
    group_info: Incomplete
    codes_info: Incomplete
    ngroups: Incomplete
    result_index: Incomplete
    _sort_idx: Incomplete
    _sorted_ids: Incomplete
    def __init__(self, axis: Index, groupings: Sequence[grouper.Grouping], sort: bool = ..., dropna: bool = ...) -> None: ...
    def __iter__(self) -> Iterator[Hashable]: ...
    def get_iterator(self, data: NDFrameT, axis: AxisInt = ...) -> Iterator[tuple[Hashable, NDFrameT]]:
        """
        Groupby iterator

        Returns
        -------
        Generator yielding sequence of (name, subsetted object)
        for each group
        """
    def _get_splitter(self, data: NDFrame, axis: AxisInt = ...) -> DataSplitter:
        """
        Returns
        -------
        Generator yielding subsetted objects
        """
    def result_ilocs(self) -> npt.NDArray[np.intp]:
        """
        Get the original integer locations of result_index in the input.
        """
    def size(self) -> Series:
        """
        Compute group sizes.
        """
    def _get_compressed_codes(self) -> tuple[npt.NDArray[np.signedinteger], npt.NDArray[np.intp]]: ...
    def get_group_levels(self) -> list[ArrayLike]: ...
    def _cython_operation(self, kind: str, values, how: str, axis: AxisInt, min_count: int = ..., **kwargs) -> ArrayLike:
        """
        Returns the values of a cython operation.
        """
    def agg_series(self, obj: Series, func: Callable, preserve_dtype: bool = ...) -> ArrayLike:
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
    def apply_groupwise(self, f: Callable, data: DataFrame | Series, axis: AxisInt = ...) -> tuple[list, bool]: ...
    @property
    def groupings(self): ...
    @property
    def shape(self): ...
    @property
    def nkeys(self): ...
    @property
    def codes(self): ...
    @property
    def levels(self): ...
    @property
    def names(self): ...
    @property
    def reconstructed_codes(self): ...

class BinGrouper(BaseGrouper):
    groups: Incomplete
    codes_info: Incomplete
    indices: Incomplete
    group_info: Incomplete
    reconstructed_codes: Incomplete
    result_index: Incomplete
    def __init__(self, bins, binlabels, indexer) -> None: ...
    def get_iterator(self, data: NDFrame, axis: AxisInt = ...):
        """
        Groupby iterator

        Returns
        -------
        Generator yielding sequence of (name, subsetted object)
        for each group
        """
    @property
    def nkeys(self): ...
    @property
    def levels(self): ...
    @property
    def names(self): ...
    @property
    def groupings(self): ...
def _is_indexed_like(obj, axes, axis: AxisInt) -> bool: ...

class DataSplitter(typing.Generic):
    __orig_bases__: ClassVar[tuple] = ...
    __parameters__: ClassVar[tuple] = ...
    _sorted_data: Incomplete
    def __init__(self, data: NDFrameT, labels: npt.NDArray[np.intp], ngroups: int, *, sort_idx: npt.NDArray[np.intp], sorted_ids: npt.NDArray[np.intp], axis: AxisInt = ...) -> None: ...
    def __iter__(self) -> Iterator: ...
    def _chop(self, sdata, slice_obj: slice) -> NDFrame: ...

class SeriesSplitter(DataSplitter):
    __parameters__: ClassVar[tuple] = ...
    def _chop(self, sdata: Series, slice_obj: slice) -> Series: ...

class FrameSplitter(DataSplitter):
    __parameters__: ClassVar[tuple] = ...
    def _chop(self, sdata: DataFrame, slice_obj: slice) -> DataFrame: ...
def _get_splitter(data: NDFrame, labels: npt.NDArray[np.intp], ngroups: int, *, sort_idx: npt.NDArray[np.intp], sorted_ids: npt.NDArray[np.intp], axis: AxisInt = ...) -> DataSplitter: ...
