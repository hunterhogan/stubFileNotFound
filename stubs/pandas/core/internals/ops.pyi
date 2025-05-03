from collections.abc import Iterator
from pandas._libs.internals import BlockPlacement as BlockPlacement
from pandas._typing import ArrayLike as ArrayLike
from pandas.core.dtypes.common import is_1d_only_ea_dtype as is_1d_only_ea_dtype
from pandas.core.internals.blocks import Block as Block
from pandas.core.internals.managers import BlockManager as BlockManager
from typing import NamedTuple

class BlockPairInfo(NamedTuple):
    lvals: ArrayLike
    rvals: ArrayLike
    locs: BlockPlacement
    left_ea: bool
    right_ea: bool
    rblk: Block

def _iter_block_pairs(left: BlockManager, right: BlockManager) -> Iterator[BlockPairInfo]: ...
def operate_blockwise(left: BlockManager, right: BlockManager, array_op) -> BlockManager: ...
def _reset_block_mgr_locs(nbs: list[Block], locs) -> None:
    """
    Reset mgr_locs to correspond to our original DataFrame.
    """
def _get_same_shape_values(lblk: Block, rblk: Block, left_ea: bool, right_ea: bool) -> tuple[ArrayLike, ArrayLike]:
    """
    Slice lblk.values to align with rblk.  Squeeze if we have EAs.
    """
def blockwise_all(left: BlockManager, right: BlockManager, op) -> bool:
    """
    Blockwise `all` reduction.
    """
