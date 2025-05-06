import numpy as np
from _typeshed import Incomplete
from collections.abc import Iterable, Sequence
from pandas._config import get_option as get_option, using_copy_on_write as using_copy_on_write, warn_copy_on_write as warn_copy_on_write
from pandas._libs import NaT as NaT, internals as libinternals, lib as lib
from pandas._libs.internals import BlockPlacement as BlockPlacement, BlockValuesRefs as BlockValuesRefs
from pandas._typing import ArrayLike as ArrayLike, AxisInt as AxisInt, DtypeBackend as DtypeBackend, DtypeObj as DtypeObj, F as F, FillnaOptions as FillnaOptions, IgnoreRaise as IgnoreRaise, InterpolateOptions as InterpolateOptions, QuantileInterpolation as QuantileInterpolation, Self as Self, Shape as Shape, npt as npt
from pandas.core.api import Index as Index
from pandas.core.array_algos.putmask import extract_bool_array as extract_bool_array, putmask_inplace as putmask_inplace, putmask_without_repeat as putmask_without_repeat, setitem_datetimelike_compat as setitem_datetimelike_compat, validate_putmask as validate_putmask
from pandas.core.array_algos.replace import compare_or_regex_search as compare_or_regex_search, replace_regex as replace_regex, should_use_regex as should_use_regex
from pandas.core.arrays import Categorical as Categorical, DatetimeArray as DatetimeArray, ExtensionArray as ExtensionArray, IntervalArray as IntervalArray, NumpyExtensionArray as NumpyExtensionArray, PeriodArray as PeriodArray, TimedeltaArray as TimedeltaArray
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray as NDArrayBackedExtensionArray
from pandas.core.base import PandasObject as PandasObject
from pandas.core.construction import ensure_wrapped_if_datetimelike as ensure_wrapped_if_datetimelike, extract_array as extract_array
from pandas.core.dtypes.astype import astype_array_safe as astype_array_safe, astype_is_view as astype_is_view
from pandas.core.dtypes.cast import LossySetitemError as LossySetitemError, can_hold_element as can_hold_element, convert_dtypes as convert_dtypes, find_result_type as find_result_type, maybe_downcast_to_dtype as maybe_downcast_to_dtype, np_can_hold_element as np_can_hold_element
from pandas.core.dtypes.common import is_1d_only_ea_dtype as is_1d_only_ea_dtype, is_float_dtype as is_float_dtype, is_integer_dtype as is_integer_dtype, is_list_like as is_list_like, is_scalar as is_scalar, is_string_dtype as is_string_dtype
from pandas.core.dtypes.dtypes import DatetimeTZDtype as DatetimeTZDtype, ExtensionDtype as ExtensionDtype, IntervalDtype as IntervalDtype, NumpyEADtype as NumpyEADtype, PeriodDtype as PeriodDtype
from pandas.core.dtypes.generic import ABCDataFrame as ABCDataFrame, ABCIndex as ABCIndex, ABCNumpyExtensionArray as ABCNumpyExtensionArray, ABCSeries as ABCSeries
from pandas.core.dtypes.missing import is_valid_na_for_dtype as is_valid_na_for_dtype, isna as isna, na_value_for_dtype as na_value_for_dtype
from typing import Any, Literal

from collections.abc import Callable

_dtype_obj: Incomplete
COW_WARNING_GENERAL_MSG: str
COW_WARNING_SETITEM_MSG: str

def maybe_split(meth: F) -> F:
    """
    If we have a multi-column block, split and operate block-wise.  Otherwise
    use the original method.
    """

class Block(PandasObject, libinternals.Block):
    """
    Canonical n-dimensional unit of homogeneous dtype contained in a pandas
    data structure

    Index-ignorant; let the container take care of that
    """
    values: np.ndarray | ExtensionArray
    ndim: int
    refs: BlockValuesRefs
    __init__: Callable
    __slots__: Incomplete
    is_numeric: bool
    def _validate_ndim(self) -> bool:
        """
        We validate dimension for blocks that can hold 2D values, which for now
        means numpy dtypes or DatetimeTZDtype.
        """
    def is_object(self) -> bool: ...
    def is_extension(self) -> bool: ...
    def _can_consolidate(self) -> bool: ...
    def _consolidate_key(self): ...
    def _can_hold_na(self) -> bool:
        """
        Can we store NA values in this Block?
        """
    @property
    def is_bool(self) -> bool:
        """
        We can be bool if a) we are bool dtype or b) object dtype with bool objects.
        """
    def external_values(self): ...
    def fill_value(self): ...
    def _standardize_fill_value(self, value): ...
    @property
    def mgr_locs(self) -> BlockPlacement: ...
    _mgr_locs: Incomplete
    @mgr_locs.setter
    def mgr_locs(self, new_mgr_locs: BlockPlacement) -> None: ...
    def make_block(self, values, placement: BlockPlacement | None = None, refs: BlockValuesRefs | None = None) -> Block:
        """
        Create a new block, with type inference propagate any values that are
        not specified
        """
    def make_block_same_class(self, values, placement: BlockPlacement | None = None, refs: BlockValuesRefs | None = None) -> Self:
        """Wrap given values in a block of same type as self."""
    def __repr__(self) -> str: ...
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
    def getitem_block_columns(self, slicer: slice, new_mgr_locs: BlockPlacement, ref_inplace_op: bool = False) -> Self:
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
    def coerce_to_target_dtype(self, other, warn_on_upcast: bool = False) -> Block:
        """
        coerce the current block to a dtype compat for other
        we will return a block, possibly object, and not raise

        we can also safely try to coerce to the same dtype
        and will receive the same block
        """
    def _maybe_downcast(self, blocks: list[Block], downcast, using_cow: bool, caller: str) -> list[Block]: ...
    def _downcast_2d(self, dtype, using_cow: bool = False) -> list[Block]:
        """
        downcast specialized to 2D case post-validation.

        Refactored to allow use of maybe_split.
        """
    def convert(self, *, copy: bool = True, using_cow: bool = False) -> list[Block]:
        """
        Attempt to coerce any object types to better types. Return a copy
        of the block (if copy = True).
        """
    def convert_dtypes(self, copy: bool, using_cow: bool, infer_objects: bool = True, convert_string: bool = True, convert_integer: bool = True, convert_boolean: bool = True, convert_floating: bool = True, dtype_backend: DtypeBackend = 'numpy_nullable') -> list[Block]: ...
    def dtype(self) -> DtypeObj: ...
    def astype(self, dtype: DtypeObj, copy: bool = False, errors: IgnoreRaise = 'raise', using_cow: bool = False, squeeze: bool = False) -> Block:
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
    def get_values_for_csv(self, *, float_format, date_format, decimal, na_rep: str = 'nan', quoting: Incomplete | None = None) -> Block:
        """convert to our native types format"""
    def copy(self, deep: bool = True) -> Self:
        """copy constructor"""
    def _maybe_copy(self, using_cow: bool, inplace: bool) -> Self: ...
    def _get_refs_and_copy(self, using_cow: bool, inplace: bool): ...
    def replace(self, to_replace, value, inplace: bool = False, mask: npt.NDArray[np.bool_] | None = None, using_cow: bool = False, already_warned: Incomplete | None = None) -> list[Block]:
        """
        replace the to_replace value with value, possible to create new
        blocks here this is just a call to putmask.
        """
    def _replace_regex(self, to_replace, value, inplace: bool = False, mask: Incomplete | None = None, using_cow: bool = False, already_warned: Incomplete | None = None) -> list[Block]:
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
    def replace_list(self, src_list: Iterable[Any], dest_list: Sequence[Any], inplace: bool = False, regex: bool = False, using_cow: bool = False, already_warned: Incomplete | None = None) -> list[Block]:
        """
        See BlockManager.replace_list docstring.
        """
    def _replace_coerce(self, to_replace, value, mask: npt.NDArray[np.bool_], inplace: bool = True, regex: bool = False, using_cow: bool = False) -> list[Block]:
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
    @property
    def shape(self) -> Shape: ...
    def iget(self, i: int | tuple[int, int] | tuple[slice, int]) -> np.ndarray: ...
    def _slice(self, slicer: slice | npt.NDArray[np.bool_] | npt.NDArray[np.intp]) -> ArrayLike:
        """return a slice of my values"""
    def set_inplace(self, locs, values: ArrayLike, copy: bool = False) -> None:
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
    def take_nd(self, indexer: npt.NDArray[np.intp], axis: AxisInt, new_mgr_locs: BlockPlacement | None = None, fill_value=...) -> Block:
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
    def setitem(self, indexer, value, using_cow: bool = False) -> Block:
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
    def putmask(self, mask, new, using_cow: bool = False, already_warned: Incomplete | None = None) -> list[Block]:
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
    def where(self, other, cond, _downcast: str | bool = 'infer', using_cow: bool = False) -> list[Block]:
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
    def fillna(self, value, limit: int | None = None, inplace: bool = False, downcast: Incomplete | None = None, using_cow: bool = False, already_warned: Incomplete | None = None) -> list[Block]:
        """
        fillna on the block with the value. If we fail, then convert to
        block to hold objects instead and try again
        """
    def pad_or_backfill(self, *, method: FillnaOptions, axis: AxisInt = 0, inplace: bool = False, limit: int | None = None, limit_area: Literal['inside', 'outside'] | None = None, downcast: Literal['infer'] | None = None, using_cow: bool = False, already_warned: Incomplete | None = None) -> list[Block]: ...
    def interpolate(self, *, method: InterpolateOptions, index: Index, inplace: bool = False, limit: int | None = None, limit_direction: Literal['forward', 'backward', 'both'] = 'forward', limit_area: Literal['inside', 'outside'] | None = None, downcast: Literal['infer'] | None = None, using_cow: bool = False, already_warned: Incomplete | None = None, **kwargs) -> list[Block]: ...
    def diff(self, n: int) -> list[Block]:
        """return block for the diff of the values"""
    def shift(self, periods: int, fill_value: Any = None) -> list[Block]:
        """shift the block by periods, possibly upcast"""
    def quantile(self, qs: Index, interpolation: QuantileInterpolation = 'linear') -> Block:
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
    def round(self, decimals: int, using_cow: bool = False) -> Self:
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
    @property
    def is_view(self) -> bool:
        """return a boolean if I am possibly a view"""
    @property
    def array_values(self) -> ExtensionArray:
        """
        The array that Series.array returns. Always an ExtensionArray.
        """
    def get_values(self, dtype: DtypeObj | None = None) -> np.ndarray:
        """
        return an internal format, currently just the ndarray
        this is often overridden to handle to_dense like operations
        """

class EABackedBlock(Block):
    """
    Mixin for Block subclasses backed by ExtensionArray.
    """
    values: ExtensionArray
    def shift(self, periods: int, fill_value: Any = None) -> list[Block]:
        """
        Shift the block by `periods`.

        Dispatches to underlying ExtensionArray and re-boxes in an
        ExtensionBlock.
        """
    def setitem(self, indexer, value, using_cow: bool = False):
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
    def where(self, other, cond, _downcast: str | bool = 'infer', using_cow: bool = False) -> list[Block]: ...
    def putmask(self, mask, new, using_cow: bool = False, already_warned: Incomplete | None = None) -> list[Block]:
        """
        See Block.putmask.__doc__
        """
    def delete(self, loc) -> list[Block]: ...
    def array_values(self) -> ExtensionArray: ...
    def get_values(self, dtype: DtypeObj | None = None) -> np.ndarray:
        """
        return object dtype as boxed values, such as Timestamps/Timedelta
        """
    def pad_or_backfill(self, *, method: FillnaOptions, axis: AxisInt = 0, inplace: bool = False, limit: int | None = None, limit_area: Literal['inside', 'outside'] | None = None, downcast: Literal['infer'] | None = None, using_cow: bool = False, already_warned: Incomplete | None = None) -> list[Block]: ...

class ExtensionBlock(EABackedBlock):
    """
    Block for holding extension types.

    Notes
    -----
    This holds all 3rd-party extension array types. It's also the immediate
    parent class for our internal extension types' blocks.

    ExtensionArrays are limited to 1-D.
    """
    values: ExtensionArray
    def fillna(self, value, limit: int | None = None, inplace: bool = False, downcast: Incomplete | None = None, using_cow: bool = False, already_warned: Incomplete | None = None) -> list[Block]: ...
    def shape(self) -> Shape: ...
    def iget(self, i: int | tuple[int, int] | tuple[slice, int]): ...
    def set_inplace(self, locs, values: ArrayLike, copy: bool = False) -> None: ...
    def _maybe_squeeze_arg(self, arg):
        """
        If necessary, squeeze a (N, 1) ndarray to (N,)
        """
    def _unwrap_setitem_indexer(self, indexer):
        """
        Adapt a 2D-indexer to our 1D values.

        This is intended for 'setitem', not 'iget' or '_slice'.
        """
    @property
    def is_view(self) -> bool:
        """Extension arrays are never treated as views."""
    def is_numeric(self) -> bool: ...
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

class NumpyBlock(Block):
    values: np.ndarray
    __slots__: Incomplete
    @property
    def is_view(self) -> bool:
        """return a boolean if I am possibly a view"""
    @property
    def array_values(self) -> ExtensionArray: ...
    def get_values(self, dtype: DtypeObj | None = None) -> np.ndarray: ...
    def is_numeric(self) -> bool: ...

class NumericBlock(NumpyBlock):
    __slots__: Incomplete

class ObjectBlock(NumpyBlock):
    __slots__: Incomplete

class NDArrayBackedExtensionBlock(EABackedBlock):
    """
    Block backed by an NDArrayBackedExtensionArray
    """
    values: NDArrayBackedExtensionArray
    @property
    def is_view(self) -> bool:
        """return a boolean if I am possibly a view"""

class DatetimeLikeBlock(NDArrayBackedExtensionBlock):
    """Block for datetime64[ns], timedelta64[ns]."""
    __slots__: Incomplete
    is_numeric: bool
    values: DatetimeArray | TimedeltaArray

class DatetimeTZBlock(DatetimeLikeBlock):
    """implement a datetime64 block with a tz attribute"""
    values: DatetimeArray
    __slots__: Incomplete

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
def new_block_2d(values: ArrayLike, placement: BlockPlacement, refs: BlockValuesRefs | None = None): ...
def new_block(values, placement: BlockPlacement, *, ndim: int, refs: BlockValuesRefs | None = None) -> Block: ...
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
def extend_blocks(result, blocks: Incomplete | None = None) -> list[Block]:
    """return a new extended blocks, given the result"""
def ensure_block_shape(values: ArrayLike, ndim: int = 1) -> ArrayLike:
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
