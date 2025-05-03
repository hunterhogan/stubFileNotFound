from pandas._libs.internals import BlockPlacement as BlockPlacement
from pandas.core.arrays.datetimes import DatetimeArray as DatetimeArray
from pandas.core.construction import extract_array as extract_array
from pandas.core.dtypes.common import pandas_dtype as pandas_dtype
from pandas.core.dtypes.dtypes import DatetimeTZDtype as DatetimeTZDtype, PeriodDtype as PeriodDtype
from pandas.core.internals.blocks import check_ndim as check_ndim, ensure_block_shape as ensure_block_shape, extract_pandas_array as extract_pandas_array, get_block_type as get_block_type, maybe_coerce_values as maybe_coerce_values

TYPE_CHECKING: bool
def make_block(values, placement, klass, ndim, dtype: Dtype | None) -> Block:
    """
    This is a pseudo-public analogue to blocks.new_block.

    We ask that downstream libraries use this rather than any fully-internal
    APIs, including but not limited to:

    - core.internals.blocks.make_block
    - Block.make_block
    - Block.make_block_same_class
    - Block.__init__
    """
def maybe_infer_ndim(values, placement: BlockPlacement, ndim: int | None) -> int:
    """
    If `ndim` is not provided, infer it from placement and values.
    """
def __getattr__(name: str): ...
