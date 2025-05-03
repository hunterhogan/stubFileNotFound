import np
import npt
import numpy.dtypes
import pandas._libs.internals
import pandas._libs.internals as libinternals
import pandas._libs.lib
import pandas._libs.lib as lib
import pandas.core.algorithms as algos
import pandas.core.base
import pandas.core.common as com
import pandas.core.computation.expressions as expressions
import pandas.core.missing as missing
from _typeshed import Incomplete
from builtins import AxisInt
from pandas._config import using_copy_on_write as using_copy_on_write, warn_copy_on_write as warn_copy_on_write
from pandas._config.config import get_option as get_option
from pandas._libs.internals import BlockPlacement as BlockPlacement, BlockValuesRefs as BlockValuesRefs
from pandas._libs.lib import is_list_like as is_list_like, is_scalar as is_scalar
from pandas._libs.missing import NA as NA
from pandas._libs.properties import cache_readonly as cache_readonly
from pandas._libs.tslibs.nattype import NaT as NaT
from pandas._typing import F as F
from pandas.core.array_algos.putmask import extract_bool_array as extract_bool_array, putmask_inplace as putmask_inplace, putmask_without_repeat as putmask_without_repeat, setitem_datetimelike_compat as setitem_datetimelike_compat, validate_putmask as validate_putmask
from pandas.core.array_algos.quantile import quantile_compat as quantile_compat
from pandas.core.array_algos.replace import compare_or_regex_search as compare_or_regex_search, replace_regex as replace_regex, should_use_regex as should_use_regex
from pandas.core.array_algos.transforms import shift as shift
from pandas.core.arrays.base import ExtensionArray as ExtensionArray
from pandas.core.arrays.categorical import Categorical as Categorical
from pandas.core.arrays.datetimes import DatetimeArray as DatetimeArray
from pandas.core.arrays.interval import IntervalArray as IntervalArray
from pandas.core.arrays.numpy_ import NumpyExtensionArray as NumpyExtensionArray
from pandas.core.arrays.period import PeriodArray as PeriodArray
from pandas.core.arrays.timedeltas import TimedeltaArray as TimedeltaArray
from pandas.core.base import PandasObject as PandasObject
from pandas.core.construction import ensure_wrapped_if_datetimelike as ensure_wrapped_if_datetimelike, extract_array as extract_array
from pandas.core.dtypes.astype import astype_array_safe as astype_array_safe, astype_is_view as astype_is_view
from pandas.core.dtypes.base import ExtensionDtype as ExtensionDtype
from pandas.core.dtypes.cast import can_hold_element as can_hold_element, convert_dtypes as convert_dtypes, find_result_type as find_result_type, maybe_downcast_to_dtype as maybe_downcast_to_dtype, np_can_hold_element as np_can_hold_element
from pandas.core.dtypes.common import is_1d_only_ea_dtype as is_1d_only_ea_dtype, is_float_dtype as is_float_dtype, is_integer_dtype as is_integer_dtype, is_string_dtype as is_string_dtype
from pandas.core.dtypes.dtypes import DatetimeTZDtype as DatetimeTZDtype, IntervalDtype as IntervalDtype, NumpyEADtype as NumpyEADtype, PeriodDtype as PeriodDtype
from pandas.core.dtypes.generic import ABCDataFrame as ABCDataFrame, ABCIndex as ABCIndex, ABCNumpyExtensionArray as ABCNumpyExtensionArray, ABCSeries as ABCSeries
from pandas.core.dtypes.missing import is_valid_na_for_dtype as is_valid_na_for_dtype, isna as isna, na_value_for_dtype as na_value_for_dtype
from pandas.core.indexers.utils import check_setitem_lengths as check_setitem_lengths
from pandas.core.indexes.base import get_values_for_csv as get_values_for_csv
from pandas.errors import AbstractMethodError as AbstractMethodError, LossySetitemError as LossySetitemError
from pandas.util._exceptions import find_stack_level as find_stack_level
from pandas.util._validators import validate_bool_kwarg as validate_bool_kwarg
from typing import Any, ArrayLike, ClassVar, DtypeBackend, DtypeObj, FillnaOptions, IgnoreRaise, InterpolateOptions, Literal, QuantileInterpolation

TYPE_CHECKING: bool
Self: None
npt: None
_dtype_obj: numpy.dtypes.ObjectDType
COW_WARNING_GENERAL_MSG: str
COW_WARNING_SETITEM_MSG: str
def maybe_split(meth: F) -> F:
    """
    If we have a multi-column block, split and operate block-wise.  Otherwise
    use the original method.
    """

class Block(pandas.core.base.PandasObject, pandas._libs.internals.Block):
    is_numeric: ClassVar[bool] = ...
    _validate_ndim: Incomplete
    is_object: Incomplete
    is_extension: Incomplete
    _can_consolidate: Incomplete
    _consolidate_key: Incomplete
    _can_hold_na: Incomplete
    fill_value: Incomplete
    mgr_locs: Incomplete
    dtype: Incomplete
    def external_values(self): ...
    def _standardize_fill_value(self, value): ...
    def make_block(self, values, placement: BlockPlacement | None, refs: BlockValuesRefs | None) -> Block:
        """
        Create a new block, with type inference propagate any values that are
        not specified
        """
    def make_block_same_class(self, values, placement: BlockPlacement | None, refs: BlockValuesRefs | None) -> Self:
        """Wrap given values in a block of same type as self."""
    def __len__(self) -> int: ...
    def slice_block_columns(self, slc: slice) -> Self:
        """
        Perform __getitem__-like, return result as block.
        """
    def take_block_columns(self, indices: npt.NDArray[np.intp]) -> Self:
        """
        Perform __getitem__-like, return result as block.

        Only supports slices that preserve dimensionality.
        """
    def getitem_block_columns(self, slicer: slice, new_mgr_locs: BlockPlacement, ref_inplace_op: bool = ...) -> Self:
        """
        Perform __getitem__-like, return result as block.

        Only supports slices that preserve dimensionality.
        """
    def _can_hold_element(self, element: Any) -> bool:
        """require the same dtype as ourselves"""
    def should_store(self, value: ArrayLike) -> bool:
        """
        Should we set self.values[indexer] = value inplace or do we need to cast?

        Parameters
        ----------
        value : np.ndarray or ExtensionArray

        Returns
        -------
        bool
        """
    def apply(self, func, **kwargs) -> list[Block]:
        """
        apply the function to my values; return a block if we are not
        one
        """
    def reduce(self, func) -> list[Block]: ...
    def _split_op_result(self, result: ArrayLike) -> list[Block]: ...
    def _split(self) -> list[Block]:
        """
        Split a block into a list of single-column blocks.
        """
    def split_and_operate(self, func, *args, **kwargs) -> list[Block]:
        """
        Split the block and apply func column-by-column.

        Parameters
        ----------
        func : Block method
        *args
        **kwargs

        Returns
        -------
        List[Block]
        """
    def coerce_to_target_dtype(self, other, warn_on_upcast: bool = ...) -> Block:
        """
        coerce the current block to a dtype compat for other
        we will return a block, possibly object, and not raise

        we can also safely try to coerce to the same dtype
        and will receive the same block
        """
    def _maybe_downcast(self, blocks: list[Block], downcast, using_cow: bool, caller: str) -> list[Block]: ...
    def _downcast_2d(self, *args, **kwargs) -> list[Block]:
        """
        downcast specialized to 2D case post-validation.

        Refactored to allow use of maybe_split.
        """
    def convert(self, *, copy: bool = ..., using_cow: bool = ...) -> list[Block]:
        """
        Attempt to coerce any object types to better types. Return a copy
        of the block (if copy = True).
        """
    def convert_dtypes(self, copy: bool, using_cow: bool, infer_objects: bool = ..., convert_string: bool = ..., convert_integer: bool = ..., convert_boolean: bool = ..., convert_floating: bool = ..., dtype_backend: DtypeBackend = ...) -> list[Block]: ...
    def astype(self, dtype: DtypeObj, copy: bool = ..., errors: IgnoreRaise = ..., using_cow: bool = ..., squeeze: bool = ...) -> Block:
        """
        Coerce to the new dtype.

        Parameters
        ----------
        dtype : np.dtype or ExtensionDtype
        copy : bool, default False
            copy if indicated
        errors : str, {'raise', 'ignore'}, default 'raise'
            - ``raise`` : allow exceptions to be raised
            - ``ignore`` : suppress exceptions. On error return original object
        using_cow: bool, default False
            Signaling if copy on write copy logic is used.
        squeeze : bool, default False
            squeeze values to ndim=1 if only one column is given

        Returns
        -------
        Block
        """
    def get_values_for_csv(self, *, float_format, date_format, decimal, na_rep: str = ..., quoting) -> Block:
        """convert to our native types format"""
    def copy(self, deep: bool = ...) -> Self:
        """copy constructor"""
    def _maybe_copy(self, using_cow: bool, inplace: bool) -> Self: ...
    def _get_refs_and_copy(self, using_cow: bool, inplace: bool): ...
    def replace(self, to_replace, value, inplace: bool = ..., mask: npt.NDArray[np.bool_] | None, using_cow: bool = ..., already_warned) -> list[Block]:
        """
        replace the to_replace value with value, possible to create new
        blocks here this is just a call to putmask.
        """
    def _replace_regex(self, to_replace, value, inplace: bool = ..., mask, using_cow: bool = ..., already_warned) -> list[Block]:
        """
        Replace elements by the given value.

        Parameters
        ----------
        to_replace : object or pattern
            Scalar to replace or regular expression to match.
        value : object
            Replacement object.
        inplace : bool, default False
            Perform inplace modification.
        mask : array-like of bool, optional
            True indicate corresponding element is ignored.
        using_cow: bool, default False
            Specifying if copy on write is enabled.

        Returns
        -------
        List[Block]
        """
    def replace_list(self, src_list: Iterable[Any], dest_list: Sequence[Any], inplace: bool = ..., regex: bool = ..., using_cow: bool = ..., already_warned) -> list[Block]:
        """
        See BlockManager.replace_list docstring.
        """
    def _replace_coerce(self, to_replace, value, mask: npt.NDArray[np.bool_], inplace: bool = ..., regex: bool = ..., using_cow: bool = ...) -> list[Block]:
        """
        Replace value corresponding to the given boolean array with another
        value.

        Parameters
        ----------
        to_replace : object or pattern
            Scalar to replace or regular expression to match.
        value : object
            Replacement object.
        mask : np.ndarray[bool]
            True indicate corresponding element is ignored.
        inplace : bool, default True
            Perform inplace modification.
        regex : bool, default False
            If true, perform regular expression substitution.

        Returns
        -------
        List[Block]
        """
    def _maybe_squeeze_arg(self, arg: np.ndarray) -> np.ndarray:
        """
        For compatibility with 1D-only ExtensionArrays.
        """
    def _unwrap_setitem_indexer(self, indexer):
        """
        For compatibility with 1D-only ExtensionArrays.
        """
    def iget(self, i: int | tuple[int, int] | tuple[slice, int]) -> np.ndarray: ...
    def _slice(self, slicer: slice | npt.NDArray[np.bool_] | npt.NDArray[np.intp]) -> ArrayLike:
        """return a slice of my values"""
    def set_inplace(self, locs, values: ArrayLike, copy: bool = ...) -> None:
        """
        Modify block values in-place with new item value.

        If copy=True, first copy the underlying values in place before modifying
        (for Copy-on-Write).

        Notes
        -----
        `set_inplace` never creates a new array or new Block, whereas `setitem`
        _may_ create a new array and always creates a new Block.

        Caller is responsible for checking values.dtype == self.dtype.
        """
    def take_nd(self, indexer: npt.NDArray[np.intp], axis: AxisInt, new_mgr_locs: BlockPlacement | None, fill_value: pandas._libs.lib._NoDefault = ...) -> Block:
        """
        Take values according to indexer and return them as a block.
        """
    def _unstack(self, unstacker, fill_value, new_placement: npt.NDArray[np.intp], needs_masking: npt.NDArray[np.bool_]):
        """
        Return a list of unstacked blocks of self

        Parameters
        ----------
        unstacker : reshape._Unstacker
        fill_value : int
            Only used in ExtensionBlock._unstack
        new_placement : np.ndarray[np.intp]
        allow_fill : bool
        needs_masking : np.ndarray[bool]

        Returns
        -------
        blocks : list of Block
            New blocks of unstacked values.
        mask : array-like of bool
            The mask of columns of `blocks` we should keep.
        """
    def setitem(self, indexer, value, using_cow: bool = ...) -> Block:
        """
        Attempt self.values[indexer] = value, possibly creating a new array.

        Parameters
        ----------
        indexer : tuple, list-like, array-like, slice, int
            The subset of self.values to set
        value : object
            The value being set
        using_cow: bool, default False
            Signaling if CoW is used.

        Returns
        -------
        Block

        Notes
        -----
        `indexer` is a direct slice/positional indexer. `value` must
        be a compatible shape.
        """
    def putmask(self, mask, new, using_cow: bool = ..., already_warned) -> list[Block]:
        """
        putmask the data to the block; it is possible that we may create a
        new dtype of block

        Return the resulting block(s).

        Parameters
        ----------
        mask : np.ndarray[bool], SparseArray[bool], or BooleanArray
        new : a ndarray/object
        using_cow: bool, default False

        Returns
        -------
        List[Block]
        """
    def where(self, other, cond, _downcast: str | bool = ..., using_cow: bool = ...) -> list[Block]:
        '''
        evaluate the block; return result block(s) from the result

        Parameters
        ----------
        other : a ndarray/object
        cond : np.ndarray[bool], SparseArray[bool], or BooleanArray
        _downcast : str or None, default "infer"
            Private because we only specify it when calling from fillna.

        Returns
        -------
        List[Block]
        '''
    def fillna(self, value, limit: int | None, inplace: bool = ..., downcast, using_cow: bool = ..., already_warned) -> list[Block]:
        """
        fillna on the block with the value. If we fail, then convert to
        block to hold objects instead and try again
        """
    def pad_or_backfill(self, *, method: FillnaOptions, axis: AxisInt = ..., inplace: bool = ..., limit: int | None, limit_area: Literal['inside', 'outside'] | None, downcast: Literal['infer'] | None, using_cow: bool = ..., already_warned) -> list[Block]: ...
    def interpolate(self, *, method: InterpolateOptions, index: Index, inplace: bool = ..., limit: int | None, limit_direction: Literal['forward', 'backward', 'both'] = ..., limit_area: Literal['inside', 'outside'] | None, downcast: Literal['infer'] | None, using_cow: bool = ..., already_warned, **kwargs) -> list[Block]: ...
    def diff(self, n: int) -> list[Block]:
        """return block for the diff of the values"""
    def shift(self, periods: int, fill_value: Any) -> list[Block]:
        """shift the block by periods, possibly upcast"""
    def quantile(self, qs: Index, interpolation: QuantileInterpolation = ...) -> Block:
        """
        compute the quantiles of the

        Parameters
        ----------
        qs : Index
            The quantiles to be computed in float64.
        interpolation : str, default 'linear'
            Type of interpolation.

        Returns
        -------
        Block
        """
    def round(self, decimals: int, using_cow: bool = ...) -> Self:
        """
        Rounds the values.
        If the block is not of an integer or float dtype, nothing happens.
        This is consistent with DataFrame.round behavivor.
        (Note: Series.round would raise)

        Parameters
        ----------
        decimals: int,
            Number of decimal places to round to.
            Caller is responsible for validating this
        using_cow: bool,
            Whether Copy on Write is enabled right now
        """
    def delete(self, loc) -> list[Block]:
        """Deletes the locs from the block.

        We split the block to avoid copying the underlying data. We create new
        blocks for every connected segment of the initial block that is not deleted.
        The new blocks point to the initial array.
        """
    def get_values(self, dtype: DtypeObj | None) -> np.ndarray:
        """
        return an internal format, currently just the ndarray
        this is often overridden to handle to_dense like operations
        """
    @property
    def is_bool(self): ...
    @property
    def shape(self): ...
    @property
    def is_view(self): ...
    @property
    def array_values(self): ...

class EABackedBlock(Block):
    array_values: Incomplete
    def shift(self, periods: int, fill_value: Any) -> list[Block]:
        """
        Shift the block by `periods`.

        Dispatches to underlying ExtensionArray and re-boxes in an
        ExtensionBlock.
        """
    def setitem(self, indexer, value, using_cow: bool = ...):
        """
        Attempt self.values[indexer] = value, possibly creating a new array.

        This differs from Block.setitem by not allowing setitem to change
        the dtype of the Block.

        Parameters
        ----------
        indexer : tuple, list-like, array-like, slice, int
            The subset of self.values to set
        value : object
            The value being set
        using_cow: bool, default False
            Signaling if CoW is used.

        Returns
        -------
        Block

        Notes
        -----
        `indexer` is a direct slice/positional indexer. `value` must
        be a compatible shape.
        """
    def where(self, other, cond, _downcast: str | bool = ..., using_cow: bool = ...) -> list[Block]: ...
    def putmask(self, mask, new, using_cow: bool = ..., already_warned) -> list[Block]:
        """
        See Block.putmask.__doc__
        """
    def delete(self, loc) -> list[Block]: ...
    def get_values(self, dtype: DtypeObj | None) -> np.ndarray:
        """
        return object dtype as boxed values, such as Timestamps/Timedelta
        """
    def pad_or_backfill(self, *, method: FillnaOptions, axis: AxisInt = ..., inplace: bool = ..., limit: int | None, limit_area: Literal['inside', 'outside'] | None, downcast: Literal['infer'] | None, using_cow: bool = ..., already_warned) -> list[Block]: ...

class ExtensionBlock(EABackedBlock):
    shape: Incomplete
    is_numeric: Incomplete
    def fillna(self, value, limit: int | None, inplace: bool = ..., downcast, using_cow: bool = ..., already_warned) -> list[Block]: ...
    def iget(self, i: int | tuple[int, int] | tuple[slice, int]): ...
    def set_inplace(self, locs, values: ArrayLike, copy: bool = ...) -> None: ...
    def _maybe_squeeze_arg(self, arg):
        """
        If necessary, squeeze a (N, 1) ndarray to (N,)
        """
    def _unwrap_setitem_indexer(self, indexer):
        """
        Adapt a 2D-indexer to our 1D values.

        This is intended for 'setitem', not 'iget' or '_slice'.
        """
    def _slice(self, slicer: slice | npt.NDArray[np.bool_] | npt.NDArray[np.intp]) -> ExtensionArray:
        """
        Return a slice of my values.

        Parameters
        ----------
        slicer : slice, ndarray[int], or ndarray[bool]
            Valid (non-reducing) indexer for self.values.

        Returns
        -------
        ExtensionArray
        """
    def slice_block_rows(self, slicer: slice) -> Self:
        """
        Perform __getitem__-like specialized to slicing along index.
        """
    def _unstack(self, unstacker, fill_value, new_placement: npt.NDArray[np.intp], needs_masking: npt.NDArray[np.bool_]): ...
    @property
    def is_view(self): ...

class NumpyBlock(Block):
    is_numeric: Incomplete
    def get_values(self, dtype: DtypeObj | None) -> np.ndarray: ...
    @property
    def is_view(self): ...
    @property
    def array_values(self): ...

class NumericBlock(NumpyBlock): ...
class ObjectBlock(NumpyBlock): ...

class NDArrayBackedExtensionBlock(EABackedBlock):
    @property
    def is_view(self): ...

class DatetimeLikeBlock(NDArrayBackedExtensionBlock):
    is_numeric: ClassVar[bool] = ...

class DatetimeTZBlock(DatetimeLikeBlock): ...
def maybe_coerce_values(values: ArrayLike) -> ArrayLike:
    """
    Input validation for values passed to __init__. Ensure that
    any datetime64/timedelta64 dtypes are in nanoseconds.  Ensure
    that we do not have string dtypes.

    Parameters
    ----------
    values : np.ndarray or ExtensionArray

    Returns
    -------
    values : np.ndarray or ExtensionArray
    """
def get_block_type(dtype: DtypeObj) -> type[Block]:
    """
    Find the appropriate Block subclass to use for the given values and dtype.

    Parameters
    ----------
    dtype : numpy or pandas dtype

    Returns
    -------
    cls : class, subclass of Block
    """
def new_block_2d(values: ArrayLike, placement: BlockPlacement, refs: BlockValuesRefs | None): ...
def new_block(values, placement: BlockPlacement, *, ndim: int, refs: BlockValuesRefs | None) -> Block: ...
def check_ndim(values, placement: BlockPlacement, ndim: int) -> None:
    """
    ndim inference and validation.

    Validates that values.ndim and ndim are consistent.
    Validates that len(values) and len(placement) are consistent.

    Parameters
    ----------
    values : array-like
    placement : BlockPlacement
    ndim : int

    Raises
    ------
    ValueError : the number of dimensions do not match
    """
def extract_pandas_array(values: ArrayLike, dtype: DtypeObj | None, ndim: int) -> tuple[ArrayLike, DtypeObj | None]:
    """
    Ensure that we don't allow NumpyExtensionArray / NumpyEADtype in internals.
    """
def extend_blocks(result, blocks) -> list[Block]:
    """return a new extended blocks, given the result"""
def ensure_block_shape(values: ArrayLike, ndim: int = ...) -> ArrayLike:
    """
    Reshape if possible to have values.ndim == ndim.
    """
def external_values(values: ArrayLike) -> ArrayLike:
    """
    The array that Series.values returns (public attribute).

    This has some historical constraints, and is overridden in block
    subclasses to return the correct array (e.g. period returns
    object ndarray and datetimetz a datetime64[ns] ndarray instead of
    proper extension array).
    """
