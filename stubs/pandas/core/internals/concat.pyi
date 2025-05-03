import numpy as np
from _typeshed import Incomplete
from collections.abc import Sequence
from pandas import Index as Index
from pandas._libs import NaT as NaT, lib as lib
from pandas._libs.missing import NA as NA
from pandas._typing import ArrayLike as ArrayLike, AxisInt as AxisInt, DtypeObj as DtypeObj, Manager2D as Manager2D, Shape as Shape
from pandas.core.construction import ensure_wrapped_if_datetimelike as ensure_wrapped_if_datetimelike
from pandas.core.dtypes.cast import ensure_dtype_can_hold_na as ensure_dtype_can_hold_na, find_common_type as find_common_type
from pandas.core.dtypes.common import is_1d_only_ea_dtype as is_1d_only_ea_dtype, is_scalar as is_scalar, needs_i8_conversion as needs_i8_conversion
from pandas.core.dtypes.concat import concat_compat as concat_compat
from pandas.core.dtypes.dtypes import ExtensionDtype as ExtensionDtype, SparseDtype as SparseDtype
from pandas.core.dtypes.missing import is_valid_na_for_dtype as is_valid_na_for_dtype, isna as isna, isna_all as isna_all
from pandas.core.internals.array_manager import ArrayManager as ArrayManager
from pandas.core.internals.blocks import Block as Block, BlockPlacement as BlockPlacement, ensure_block_shape as ensure_block_shape, new_block_2d as new_block_2d
from pandas.core.internals.managers import BlockManager as BlockManager, make_na_array as make_na_array
from pandas.util._decorators import cache_readonly as cache_readonly
from pandas.util._exceptions import find_stack_level as find_stack_level

def _concatenate_array_managers(mgrs: list[ArrayManager], axes: list[Index], concat_axis: AxisInt) -> Manager2D:
    """
    Concatenate array managers into one.

    Parameters
    ----------
    mgrs_indexers : list of (ArrayManager, {axis: indexer,...}) tuples
    axes : list of Index
    concat_axis : int

    Returns
    -------
    ArrayManager
    """
def concatenate_managers(mgrs_indexers, axes: list[Index], concat_axis: AxisInt, copy: bool) -> Manager2D:
    """
    Concatenate block managers into one.

    Parameters
    ----------
    mgrs_indexers : list of (BlockManager, {axis: indexer,...}) tuples
    axes : list of Index
    concat_axis : int
    copy : bool

    Returns
    -------
    BlockManager
    """
def _maybe_reindex_columns_na_proxy(axes: list[Index], mgrs_indexers: list[tuple[BlockManager, dict[int, np.ndarray]]], needs_copy: bool) -> list[BlockManager]:
    """
    Reindex along columns so that all of the BlockManagers being concatenated
    have matching columns.

    Columns added in this reindexing have dtype=np.void, indicating they
    should be ignored when choosing a column's final dtype.
    """
def _is_homogeneous_mgr(mgr: BlockManager, first_dtype: DtypeObj) -> bool:
    """
    Check if this Manager can be treated as a single ndarray.
    """
def _concat_homogeneous_fastpath(mgrs_indexers, shape: Shape, first_dtype: np.dtype) -> Block:
    """
    With single-Block managers with homogeneous dtypes (that can already hold nan),
    we avoid [...]
    """
def _get_combined_plan(mgrs: list[BlockManager]) -> list[tuple[BlockPlacement, list[JoinUnit]]]: ...
def _get_block_for_concat_plan(mgr: BlockManager, bp: BlockPlacement, blkno: int, *, max_len: int) -> Block: ...

class JoinUnit:
    block: Incomplete
    def __init__(self, block: Block) -> None: ...
    def __repr__(self) -> str: ...
    def _is_valid_na_for(self, dtype: DtypeObj) -> bool:
        """
        Check that we are all-NA of a type/dtype that is compatible with this dtype.
        Augments `self.is_na` with an additional check of the type of NA values.
        """
    def is_na(self) -> bool: ...
    def is_na_after_size_and_isna_all_deprecation(self) -> bool:
        """
        Will self.is_na be True after values.size == 0 deprecation and isna_all
        deprecation are enforced?
        """
    def get_reindexed_values(self, empty_dtype: DtypeObj, upcasted_na) -> ArrayLike: ...

def _concatenate_join_units(join_units: list[JoinUnit], copy: bool) -> ArrayLike:
    """
    Concatenate values from several join units along axis=1.
    """
def _dtype_to_na_value(dtype: DtypeObj, has_none_blocks: bool):
    """
    Find the NA value to go with this dtype.
    """
def _get_empty_dtype(join_units: Sequence[JoinUnit]) -> tuple[DtypeObj, DtypeObj]:
    """
    Return dtype and N/A values to use when concatenating specified units.

    Returned N/A value may be None which means there was no casting involved.

    Returns
    -------
    dtype
    """
def _is_uniform_join_units(join_units: list[JoinUnit]) -> bool:
    """
    Check if the join units consist of blocks of uniform type that can
    be concatenated using Block.concat_same_type instead of the generic
    _concatenate_join_units (which uses `concat_compat`).

    """
