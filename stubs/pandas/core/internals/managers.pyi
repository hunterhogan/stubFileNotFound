import np
import npt
import pandas._libs.internals
import pandas._libs.internals as libinternals
import pandas._libs.lib
import pandas._libs.lib as lib
import pandas.core.algorithms as algos
import pandas.core.internals.base
from _typeshed import Incomplete
from collections.abc import Hashable, Sequence
from pandas._config import using_copy_on_write as using_copy_on_write, warn_copy_on_write as warn_copy_on_write
from pandas._libs.algos import ensure_platform_int as ensure_platform_int
from pandas._libs.internals import BlockPlacement as BlockPlacement, BlockValuesRefs as BlockValuesRefs
from pandas._libs.lib import is_list_like as is_list_like
from pandas._libs.properties import cache_readonly as cache_readonly
from pandas._libs.tslibs.timestamps import Timestamp as Timestamp
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray as NDArrayBackedExtensionArray
from pandas.core.arrays.arrow.array import ArrowExtensionArray as ArrowExtensionArray
from pandas.core.arrays.datetimes import DatetimeArray as DatetimeArray
from pandas.core.arrays.string_arrow import ArrowStringArray as ArrowStringArray
from pandas.core.construction import ensure_wrapped_if_datetimelike as ensure_wrapped_if_datetimelike, extract_array as extract_array
from pandas.core.dtypes.base import ExtensionDtype as ExtensionDtype
from pandas.core.dtypes.cast import infer_dtype_from_scalar as infer_dtype_from_scalar
from pandas.core.dtypes.common import is_1d_only_ea_dtype as is_1d_only_ea_dtype
from pandas.core.dtypes.dtypes import DatetimeTZDtype as DatetimeTZDtype
from pandas.core.dtypes.generic import ABCDataFrame as ABCDataFrame, ABCSeries as ABCSeries
from pandas.core.dtypes.missing import array_equals as array_equals, isna as isna
from pandas.core.indexers.utils import maybe_convert_indices as maybe_convert_indices
from pandas.core.indexes.base import Index as Index, ensure_index as ensure_index
from pandas.core.internals.base import DataManager as DataManager, SingleDataManager as SingleDataManager, ensure_np_dtype as ensure_np_dtype, interleaved_dtype as interleaved_dtype
from pandas.core.internals.blocks import Block as Block, NumpyBlock as NumpyBlock, ensure_block_shape as ensure_block_shape, extend_blocks as extend_blocks, get_block_type as get_block_type, maybe_coerce_values as maybe_coerce_values, new_block as new_block, new_block_2d as new_block_2d
from pandas.core.internals.ops import blockwise_all as blockwise_all, operate_blockwise as operate_blockwise
from pandas.errors import PerformanceWarning as PerformanceWarning
from pandas.util._exceptions import find_stack_level as find_stack_level
from typing import Callable, ClassVar, Literal

TYPE_CHECKING: bool
COW_WARNING_GENERAL_MSG: str
COW_WARNING_SETITEM_MSG: str

class BaseBlockManager(pandas.core.internals.base.DataManager):
    def __init__(self, blocks, axes, verify_integrity: bool = ...) -> None: ...
    @classmethod
    def from_blocks(cls, blocks: list[Block], axes: list[Index]) -> Self: ...
    def make_empty(self, axes) -> Self:
        """return an empty BlockManager with the items axis of len 0"""
    def __nonzero__(self) -> bool: ...
    def __bool__(self) -> bool: ...
    def _normalize_axis(self, axis: AxisInt) -> int: ...
    def set_axis(self, axis: AxisInt, new_labels: Index) -> None: ...
    def _has_no_reference(self, i: int) -> bool:
        """
        Check for column `i` if it has references.
        (whether it references another array or is itself being referenced)
        Returns True if the column has no references.
        """
    def _has_no_reference_block(self, blkno: int) -> bool:
        """
        Check for block `i` if it has references.
        (whether it references another array or is itself being referenced)
        Returns True if the block has no references.
        """
    def add_references(self, mgr: BaseBlockManager) -> None:
        """
        Adds the references from one manager to another. We assume that both
        managers have the same block structure.
        """
    def references_same_values(self, mgr: BaseBlockManager, blkno: int) -> bool:
        """
        Checks if two blocks from two different block managers reference the
        same underlying values.
        """
    def get_dtypes(self) -> npt.NDArray[np.object_]: ...
    def apply(self, f, align_keys: list[str] | None, **kwargs) -> Self:
        """
        Iterate over the blocks, collect and create a new BlockManager.

        Parameters
        ----------
        f : str or callable
            Name of the Block method to apply.
        align_keys: List[str] or None, default None
        **kwargs
            Keywords to pass to `f`

        Returns
        -------
        BlockManager
        """
    def apply_with_block(self, f, align_keys: list[str] | None, **kwargs) -> Self:
        """
        Iterate over the blocks, collect and create a new BlockManager.

        Parameters
        ----------
        f : str or callable
            Name of the Block method to apply.
        align_keys: List[str] or None, default None
        **kwargs
            Keywords to pass to `f`

        Returns
        -------
        BlockManager
        """
    def setitem(self, indexer, value, warn: bool = ...) -> Self:
        """
        Set values with indexer.

        For SingleBlockManager, this backs s[indexer] = value
        """
    def diff(self, n: int) -> Self: ...
    def astype(self, dtype, copy: bool | None = ..., errors: str = ...) -> Self: ...
    def convert(self, copy: bool | None) -> Self: ...
    def convert_dtypes(self, **kwargs): ...
    def get_values_for_csv(self, *, float_format, date_format, decimal, na_rep: str = ..., quoting) -> Self:
        """
        Convert values to native types (strings / python objects) that are used
        in formatting (repr / csv).
        """
    def _get_data_subset(self, predicate: Callable) -> Self: ...
    def get_bool_data(self) -> Self:
        """
        Select blocks that are bool-dtype and columns from object-dtype blocks
        that are all-bool.
        """
    def get_numeric_data(self) -> Self: ...
    def _combine(self, blocks: list[Block], index: Index | None) -> Self:
        """return a new manager with the blocks"""
    def copy(self, deep: bool | None | Literal['all'] = ...) -> Self:
        """
        Make deep or shallow copy of BlockManager

        Parameters
        ----------
        deep : bool, string or None, default True
            If False or None, return a shallow copy (do not copy data)
            If 'all', copy data and a deep copy of the index

        Returns
        -------
        BlockManager
        """
    def consolidate(self) -> Self:
        """
        Join together blocks having same dtype

        Returns
        -------
        y : BlockManager
        """
    def reindex_indexer(self, new_axis: Index, indexer: npt.NDArray[np.intp] | None, axis: AxisInt, fill_value, allow_dups: bool = ..., copy: bool | None = ..., only_slice: bool = ..., *, use_na_proxy: bool = ...) -> Self:
        """
        Parameters
        ----------
        new_axis : Index
        indexer : ndarray[intp] or None
        axis : int
        fill_value : object, default None
        allow_dups : bool, default False
        copy : bool or None, default True
            If None, regard as False to get shallow copy.
        only_slice : bool, default False
            Whether to take views, not copies, along columns.
        use_na_proxy : bool, default False
            Whether to use a np.void ndarray for newly introduced columns.

        pandas-indexer with -1's only.
        """
    def _slice_take_blocks_ax0(self, slice_or_indexer: slice | np.ndarray, fill_value: pandas._libs.lib._NoDefault = ..., only_slice: bool = ..., *, use_na_proxy: bool = ..., ref_inplace_op: bool = ...) -> list[Block]:
        """
        Slice/take blocks along axis=0.

        Overloaded for SingleBlock

        Parameters
        ----------
        slice_or_indexer : slice or np.ndarray[int64]
        fill_value : scalar, default lib.no_default
        only_slice : bool, default False
            If True, we always return views on existing arrays, never copies.
            This is used when called from ops.blockwise.operate_blockwise.
        use_na_proxy : bool, default False
            Whether to use a np.void ndarray for newly introduced columns.
        ref_inplace_op: bool, default False
            Don't track refs if True because we operate inplace

        Returns
        -------
        new_blocks : list of Block
        """
    def _make_na_block(self, placement: BlockPlacement, fill_value, use_na_proxy: bool = ...) -> Block: ...
    def take(self, indexer: npt.NDArray[np.intp], axis: AxisInt = ..., verify: bool = ...) -> Self:
        """
        Take items along any axis.

        indexer : np.ndarray[np.intp]
        axis : int, default 1
        verify : bool, default True
            Check that all entries are between 0 and len(self) - 1, inclusive.
            Pass verify=False if this check has been done by the caller.

        Returns
        -------
        BlockManager
        """
    @property
    def ndim(self): ...
    @property
    def blknos(self): ...
    @property
    def blklocs(self): ...
    @property
    def is_single_block(self): ...
    @property
    def items(self): ...
    @property
    def arrays(self): ...
    @property
    def any_extension_types(self): ...
    @property
    def is_view(self): ...
    @property
    def nblocks(self): ...

class BlockManager(pandas._libs.internals.BlockManager, BaseBlockManager):
    ndim: ClassVar[int] = ...
    def __init__(self, blocks: Sequence[Block], axes: Sequence[Index], verify_integrity: bool = ...) -> None: ...
    def _verify_integrity(self) -> None: ...
    @classmethod
    def from_blocks(cls, blocks: list[Block], axes: list[Index]) -> Self:
        """
        Constructor for BlockManager and SingleBlockManager with same signature.
        """
    def fast_xs(self, loc: int) -> SingleBlockManager:
        """
        Return the array corresponding to `frame.iloc[loc]`.

        Parameters
        ----------
        loc : int

        Returns
        -------
        np.ndarray or ExtensionArray
        """
    def iget(self, i: int, track_ref: bool = ...) -> SingleBlockManager:
        """
        Return the data as a SingleBlockManager.
        """
    def iget_values(self, i: int) -> ArrayLike:
        """
        Return the data for column i as the values (ndarray or ExtensionArray).

        Warning! The returned array is a view but doesn't handle Copy-on-Write,
        so this should be used with caution.
        """
    def iset(self, loc: int | slice | np.ndarray, value: ArrayLike, inplace: bool = ..., refs: BlockValuesRefs | None) -> None:
        """
        Set new item in-place. Does not consolidate. Adds new Block if not
        contained in the current set of items
        """
    def _iset_split_block(self, blkno_l: int, blk_locs: np.ndarray | list[int], value: ArrayLike | None, refs: BlockValuesRefs | None) -> None:
        """Removes columns from a block by splitting the block.

        Avoids copying the whole block through slicing and updates the manager
        after determinint the new block structure. Optionally adds a new block,
        otherwise has to be done by the caller.

        Parameters
        ----------
        blkno_l: The block number to operate on, relevant for updating the manager
        blk_locs: The locations of our block that should be deleted.
        value: The value to set as a replacement.
        refs: The reference tracking object of the value to set.
        """
    def _iset_single(self, loc: int, value: ArrayLike, inplace: bool, blkno: int, blk: Block, refs: BlockValuesRefs | None) -> None:
        """
        Fastpath for iset when we are only setting a single position and
        the Block currently in that position is itself single-column.

        In this case we can swap out the entire Block and blklocs and blknos
        are unaffected.
        """
    def column_setitem(self, loc: int, idx: int | slice | np.ndarray, value, inplace_only: bool = ...) -> None:
        '''
        Set values ("setitem") into a single column (not setting the full column).

        This is a method on the BlockManager level, to avoid creating an
        intermediate Series at the DataFrame level (`s = df[loc]; s[idx] = value`)
        '''
    def insert(self, loc: int, item: Hashable, value: ArrayLike, refs) -> None:
        """
        Insert item at selected position.

        Parameters
        ----------
        loc : int
        item : hashable
        value : np.ndarray or ExtensionArray
        refs : The reference tracking object of the value to set.
        """
    def _insert_update_mgr_locs(self, loc) -> None:
        """
        When inserting a new Block at location 'loc', we increment
        all of the mgr_locs of blocks above that by one.
        """
    def _insert_update_blklocs_and_blknos(self, loc) -> None:
        """
        When inserting a new Block at location 'loc', we update our
        _blklocs and _blknos.
        """
    def idelete(self, indexer) -> BlockManager:
        """
        Delete selected locations, returning a new BlockManager.
        """
    def grouped_reduce(self, func: Callable) -> Self:
        """
        Apply grouped reduction function blockwise, returning a new BlockManager.

        Parameters
        ----------
        func : grouped reduction function

        Returns
        -------
        BlockManager
        """
    def reduce(self, func: Callable) -> Self:
        """
        Apply reduction function blockwise, returning a single-row BlockManager.

        Parameters
        ----------
        func : reduction function

        Returns
        -------
        BlockManager
        """
    def operate_blockwise(self, other: BlockManager, array_op) -> BlockManager:
        """
        Apply array_op blockwise with another (aligned) BlockManager.
        """
    def _equal_values(self: BlockManager, other: BlockManager) -> bool:
        """
        Used in .equals defined in base class. Only check the column values
        assuming shape and indexes have already been checked.
        """
    def quantile(self, *, qs: Index, interpolation: QuantileInterpolation = ...) -> Self:
        """
        Iterate over blocks applying quantile reduction.
        This routine is intended for reduction type operations and
        will do inference on the generated blocks.

        Parameters
        ----------
        interpolation : type of interpolation, default 'linear'
        qs : list of the quantiles to be computed

        Returns
        -------
        BlockManager
        """
    def unstack(self, unstacker, fill_value) -> BlockManager:
        """
        Return a BlockManager with all blocks unstacked.

        Parameters
        ----------
        unstacker : reshape._Unstacker
        fill_value : Any
            fill_value for newly introduced missing values.

        Returns
        -------
        unstacked : BlockManager
        """
    def to_dict(self) -> dict[str, Self]:
        """
        Return a dict of str(dtype) -> BlockManager

        Returns
        -------
        values : a dict of dtype -> BlockManager
        """
    def as_array(self, dtype: np.dtype | None, copy: bool = ..., na_value: object = ...) -> np.ndarray:
        """
        Convert the blockmanager data into an numpy array.

        Parameters
        ----------
        dtype : np.dtype or None, default None
            Data type of the return array.
        copy : bool, default False
            If True then guarantee that a copy is returned. A value of
            False does not guarantee that the underlying data is not
            copied.
        na_value : object, default lib.no_default
            Value to be used as the missing value sentinel.

        Returns
        -------
        arr : ndarray
        """
    def _interleave(self, dtype: np.dtype | None, na_value: object = ...) -> np.ndarray:
        """
        Return ndarray from blocks with specified item order
        Items must be contained in the blocks
        """
    def is_consolidated(self) -> bool:
        """
        Return True if more than one block with the same dtype
        """
    def _consolidate_check(self) -> None: ...
    def _consolidate_inplace(self) -> None: ...
    @classmethod
    def concat_horizontal(cls, mgrs: list[Self], axes: list[Index]) -> Self:
        """
        Concatenate uniformly-indexed BlockManagers horizontally.
        """
    @classmethod
    def concat_vertical(cls, mgrs: list[Self], axes: list[Index]) -> Self:
        """
        Concatenate uniformly-indexed BlockManagers vertically.
        """
    @property
    def column_arrays(self): ...

class SingleBlockManager(BaseBlockManager, pandas.core.internals.base.SingleDataManager):
    _is_consolidated: ClassVar[bool] = ...
    _known_consolidated: ClassVar[bool] = ...
    is_single_block: ClassVar[bool] = ...
    _block: Incomplete
    def __init__(self, block: Block, axis: Index, verify_integrity: bool = ...) -> None: ...
    @classmethod
    def from_blocks(cls, blocks: list[Block], axes: list[Index]) -> Self:
        """
        Constructor for BlockManager and SingleBlockManager with same signature.
        """
    @classmethod
    def from_array(cls, array: ArrayLike, index: Index, refs: BlockValuesRefs | None) -> SingleBlockManager:
        """
        Constructor for if we have an array that is not yet a Block.
        """
    def to_2d_mgr(self, columns: Index) -> BlockManager:
        """
        Manager analogue of Series.to_frame
        """
    def _has_no_reference(self, i: int = ...) -> bool:
        """
        Check for column `i` if it has references.
        (whether it references another array or is itself being referenced)
        Returns True if the column has no references.
        """
    def _post_setstate(self) -> None: ...
    def get_rows_with_mask(self, indexer: npt.NDArray[np.bool_]) -> Self: ...
    def get_slice(self, slobj: slice, axis: AxisInt = ...) -> SingleBlockManager: ...
    def get_dtypes(self) -> npt.NDArray[np.object_]: ...
    def external_values(self):
        """The array that Series.values returns"""
    def internal_values(self):
        """The array that Series._values returns"""
    def array_values(self) -> ExtensionArray:
        """The array that Series.array returns"""
    def get_numeric_data(self) -> Self: ...
    def setitem_inplace(self, indexer, value, warn: bool = ...) -> None:
        """
        Set values with indexer.

        For Single[Block/Array]Manager, this backs s[indexer] = value

        This is an inplace version of `setitem()`, mutating the manager/values
        in place, not returning a new Manager (and Block), and thus never changing
        the dtype.
        """
    def idelete(self, indexer) -> SingleBlockManager:
        """
        Delete single location from SingleBlockManager.

        Ensures that self.blocks doesn't become empty.
        """
    def fast_xs(self, loc):
        """
        fast path for getting a cross-section
        return a view of the data
        """
    def set_values(self, values: ArrayLike) -> None:
        """
        Set the values of the single block in place.

        Use at your own risk! This does not check if the passed values are
        valid for the current Block/SingleBlockManager (length, dtype, etc),
        and this does not properly keep track of references.
        """
    def _equal_values(self, other: Self) -> bool:
        """
        Used in .equals defined in base class. Only check the column values
        assuming shape and indexes have already been checked.
        """
    @property
    def ndim(self): ...
    @property
    def _blknos(self): ...
    @property
    def _blklocs(self): ...
    @property
    def index(self): ...
    @property
    def dtype(self): ...
    @property
    def _can_hold_na(self): ...
def create_block_manager_from_blocks(blocks: list[Block], axes: list[Index], consolidate: bool = ..., verify_integrity: bool = ...) -> BlockManager: ...
def create_block_manager_from_column_arrays(arrays: list[ArrayLike], axes: list[Index], consolidate: bool, refs: list) -> BlockManager: ...
def raise_construction_error(tot_items: int, block_shape: Shape, axes: list[Index], e: ValueError | None):
    """raise a helpful message about our construction"""
def _grouping_func(tup: tuple[int, ArrayLike]) -> tuple[int, DtypeObj]: ...
def _form_blocks(arrays: list[ArrayLike], consolidate: bool, refs: list) -> list[Block]: ...
def _tuples_to_blocks_no_consolidate(tuples, refs) -> list[Block]: ...
def _stack_arrays(tuples, dtype: np.dtype): ...
def _consolidate(blocks: tuple[Block, ...]) -> tuple[Block, ...]:
    """
    Merge blocks having same dtype, exclude non-consolidating blocks
    """
def _merge_blocks(blocks: list[Block], dtype: DtypeObj, can_consolidate: bool) -> tuple[list[Block], bool]: ...
def _fast_count_smallints(arr: npt.NDArray[np.intp]):
    """Faster version of set(arr) for sequences of small numbers."""
def _preprocess_slice_or_indexer(slice_or_indexer: slice | np.ndarray, length: int, allow_fill: bool): ...
def make_na_array(dtype: DtypeObj, shape: Shape, fill_value) -> ArrayLike: ...
