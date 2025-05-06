import numpy as np
from _typeshed import Incomplete
from collections.abc import Hashable
from pandas._libs import NaT as NaT, lib as lib
from pandas._typing import ArrayLike as ArrayLike, AxisInt as AxisInt, DtypeObj as DtypeObj, QuantileInterpolation as QuantileInterpolation, Self as Self, npt as npt
from pandas.core.arrays import DatetimeArray as DatetimeArray, ExtensionArray as ExtensionArray, NumpyExtensionArray as NumpyExtensionArray, TimedeltaArray as TimedeltaArray
from pandas.core.construction import ensure_wrapped_if_datetimelike as ensure_wrapped_if_datetimelike, extract_array as extract_array, sanitize_array as sanitize_array
from pandas.core.dtypes.astype import astype_array as astype_array, astype_array_safe as astype_array_safe
from pandas.core.dtypes.cast import ensure_dtype_can_hold_na as ensure_dtype_can_hold_na, find_common_type as find_common_type, infer_dtype_from_scalar as infer_dtype_from_scalar, np_find_common_type as np_find_common_type
from pandas.core.dtypes.common import ensure_platform_int as ensure_platform_int, is_datetime64_ns_dtype as is_datetime64_ns_dtype, is_integer as is_integer, is_numeric_dtype as is_numeric_dtype, is_object_dtype as is_object_dtype, is_timedelta64_ns_dtype as is_timedelta64_ns_dtype
from pandas.core.dtypes.generic import ABCDataFrame as ABCDataFrame, ABCSeries as ABCSeries
from pandas.core.dtypes.missing import array_equals as array_equals, isna as isna, na_value_for_dtype as na_value_for_dtype
from pandas.core.indexers import maybe_convert_indices as maybe_convert_indices, validate_indices as validate_indices
from pandas.core.indexes.api import Index as Index, ensure_index as ensure_index
from pandas.core.internals.base import DataManager as DataManager, SingleDataManager as SingleDataManager, ensure_np_dtype as ensure_np_dtype, interleaved_dtype as interleaved_dtype
from pandas.core.internals.blocks import BlockPlacement as BlockPlacement, ensure_block_shape as ensure_block_shape, external_values as external_values, extract_pandas_array as extract_pandas_array, maybe_coerce_values as maybe_coerce_values, new_block as new_block
from typing import Literal

from collections.abc import Callable

class BaseArrayManager(DataManager):
    """
    Core internal data structure to implement DataFrame and Series.

    Alternative to the BlockManager, storing a list of 1D arrays instead of
    Blocks.

    This is *not* a public API class

    Parameters
    ----------
    arrays : Sequence of arrays
    axes : Sequence of Index
    verify_integrity : bool, default True

    """
    __slots__: Incomplete
    arrays: list[np.ndarray | ExtensionArray]
    _axes: list[Index]
    def __init__(self, arrays: list[np.ndarray | ExtensionArray], axes: list[Index], verify_integrity: bool = True) -> None: ...
    def make_empty(self, axes: Incomplete | None = None) -> Self:
        """Return an empty ArrayManager with the items axis of len 0 (no columns)"""
    @property
    def items(self) -> Index: ...
    @property
    def axes(self) -> list[Index]:
        """Axes is BlockManager-compatible order (columns, rows)"""
    @property
    def shape_proper(self) -> tuple[int, ...]: ...
    @staticmethod
    def _normalize_axis(axis: AxisInt) -> int: ...
    def set_axis(self, axis: AxisInt, new_labels: Index) -> None: ...
    def get_dtypes(self) -> npt.NDArray[np.object_]: ...
    def add_references(self, mgr: BaseArrayManager) -> None:
        """
        Only implemented on the BlockManager level
        """
    def __getstate__(self): ...
    def __setstate__(self, state) -> None: ...
    def __repr__(self) -> str: ...
    def apply(self, f, align_keys: list[str] | None = None, **kwargs) -> Self:
        """
        Iterate over the arrays, collect and create a new ArrayManager.

        Parameters
        ----------
        f : str or callable
            Name of the Array method to apply.
        align_keys: List[str] or None, default None
        **kwargs
            Keywords to pass to `f`

        Returns
        -------
        ArrayManager
        """
    def apply_with_block(self, f, align_keys: Incomplete | None = None, **kwargs) -> Self: ...
    def setitem(self, indexer, value, warn: bool = True) -> Self: ...
    def diff(self, n: int) -> Self: ...
    def astype(self, dtype, copy: bool | None = False, errors: str = 'raise') -> Self: ...
    def convert(self, copy: bool | None) -> Self: ...
    def get_values_for_csv(self, *, float_format, date_format, decimal, na_rep: str = 'nan', quoting: Incomplete | None = None) -> Self: ...
    @property
    def any_extension_types(self) -> bool:
        """Whether any of the blocks in this manager are extension blocks"""
    @property
    def is_view(self) -> bool:
        """return a boolean if we are a single block and are a view"""
    @property
    def is_single_block(self) -> bool: ...
    def _get_data_subset(self, predicate: Callable) -> Self: ...
    def get_bool_data(self, copy: bool = False) -> Self:
        """
        Select columns that are bool-dtype and object-dtype columns that are all-bool.

        Parameters
        ----------
        copy : bool, default False
            Whether to copy the blocks
        """
    def get_numeric_data(self, copy: bool = False) -> Self:
        """
        Select columns that have a numeric dtype.

        Parameters
        ----------
        copy : bool, default False
            Whether to copy the blocks
        """
    def copy(self, deep: bool | Literal['all'] | None = True) -> Self:
        """
        Make deep or shallow copy of ArrayManager

        Parameters
        ----------
        deep : bool or string, default True
            If False, return shallow copy (do not copy data)
            If 'all', copy data and a deep copy of the index

        Returns
        -------
        BlockManager
        """
    def reindex_indexer(self, new_axis, indexer, axis: AxisInt, fill_value: Incomplete | None = None, allow_dups: bool = False, copy: bool | None = True, only_slice: bool = False, use_na_proxy: bool = False) -> Self: ...
    def _reindex_indexer(self, new_axis, indexer: npt.NDArray[np.intp] | None, axis: AxisInt, fill_value: Incomplete | None = None, allow_dups: bool = False, copy: bool | None = True, use_na_proxy: bool = False) -> Self:
        """
        Parameters
        ----------
        new_axis : Index
        indexer : ndarray[intp] or None
        axis : int
        fill_value : object, default None
        allow_dups : bool, default False
        copy : bool, default True


        pandas-indexer with -1's only.
        """
    def take(self, indexer: npt.NDArray[np.intp], axis: AxisInt = 1, verify: bool = True) -> Self:
        """
        Take items along any axis.
        """
    def _make_na_array(self, fill_value: Incomplete | None = None, use_na_proxy: bool = False): ...
    def _equal_values(self, other) -> bool:
        """
        Used in .equals defined in base class. Only check the column values
        assuming shape and indexes have already been checked.
        """

class ArrayManager(BaseArrayManager):
    @property
    def ndim(self) -> Literal[2]: ...
    _axes: Incomplete
    arrays: Incomplete
    def __init__(self, arrays: list[np.ndarray | ExtensionArray], axes: list[Index], verify_integrity: bool = True) -> None: ...
    def _verify_integrity(self) -> None: ...
    def fast_xs(self, loc: int) -> SingleArrayManager:
        """
        Return the array corresponding to `frame.iloc[loc]`.

        Parameters
        ----------
        loc : int

        Returns
        -------
        np.ndarray or ExtensionArray
        """
    def get_slice(self, slobj: slice, axis: AxisInt = 0) -> ArrayManager: ...
    def iget(self, i: int) -> SingleArrayManager:
        """
        Return the data as a SingleArrayManager.
        """
    def iget_values(self, i: int) -> ArrayLike:
        """
        Return the data for column i as the values (ndarray or ExtensionArray).
        """
    @property
    def column_arrays(self) -> list[ArrayLike]:
        """
        Used in the JSON C code to access column arrays.
        """
    def iset(self, loc: int | slice | np.ndarray, value: ArrayLike, inplace: bool = False, refs: Incomplete | None = None) -> None:
        """
        Set new column(s).

        This changes the ArrayManager in-place, but replaces (an) existing
        column(s), not changing column values in-place).

        Parameters
        ----------
        loc : integer, slice or boolean mask
            Positional location (already bounds checked)
        value : np.ndarray or ExtensionArray
        inplace : bool, default False
            Whether overwrite existing array as opposed to replacing it.
        """
    def column_setitem(self, loc: int, idx: int | slice | np.ndarray, value, inplace_only: bool = False) -> None:
        '''
        Set values ("setitem") into a single column (not setting the full column).

        This is a method on the ArrayManager level, to avoid creating an
        intermediate Series at the DataFrame level (`s = df[loc]; s[idx] = value`)
        '''
    def insert(self, loc: int, item: Hashable, value: ArrayLike, refs: Incomplete | None = None) -> None:
        """
        Insert item at selected position.

        Parameters
        ----------
        loc : int
        item : hashable
        value : np.ndarray or ExtensionArray
        """
    def idelete(self, indexer) -> ArrayManager:
        """
        Delete selected locations in-place (new block and array, same BlockManager)
        """
    def grouped_reduce(self, func: Callable) -> Self:
        """
        Apply grouped reduction function columnwise, returning a new ArrayManager.

        Parameters
        ----------
        func : grouped reduction function

        Returns
        -------
        ArrayManager
        """
    def reduce(self, func: Callable) -> Self:
        """
        Apply reduction function column-wise, returning a single-row ArrayManager.

        Parameters
        ----------
        func : reduction function

        Returns
        -------
        ArrayManager
        """
    def operate_blockwise(self, other: ArrayManager, array_op) -> ArrayManager:
        """
        Apply array_op blockwise with another (aligned) BlockManager.
        """
    def quantile(self, *, qs: Index, transposed: bool = False, interpolation: QuantileInterpolation = 'linear') -> ArrayManager: ...
    def unstack(self, unstacker, fill_value) -> ArrayManager:
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
    def as_array(self, dtype: Incomplete | None = None, copy: bool = False, na_value: object = ...) -> np.ndarray:
        """
        Convert the blockmanager data into an numpy array.

        Parameters
        ----------
        dtype : object, default None
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
    @classmethod
    def concat_horizontal(cls, mgrs: list[Self], axes: list[Index]) -> Self:
        """
        Concatenate uniformly-indexed ArrayManagers horizontally.
        """
    @classmethod
    def concat_vertical(cls, mgrs: list[Self], axes: list[Index]) -> Self:
        """
        Concatenate uniformly-indexed ArrayManagers vertically.
        """

class SingleArrayManager(BaseArrayManager, SingleDataManager):
    __slots__: Incomplete
    arrays: list[np.ndarray | ExtensionArray]
    _axes: list[Index]
    @property
    def ndim(self) -> Literal[1]: ...
    def __init__(self, arrays: list[np.ndarray | ExtensionArray], axes: list[Index], verify_integrity: bool = True) -> None: ...
    def _verify_integrity(self) -> None: ...
    @staticmethod
    def _normalize_axis(axis): ...
    def make_empty(self, axes: Incomplete | None = None) -> Self:
        """Return an empty ArrayManager with index/array of length 0"""
    @classmethod
    def from_array(cls, array, index) -> SingleArrayManager: ...
    @property
    def axes(self) -> list[Index]: ...
    @property
    def index(self) -> Index: ...
    @property
    def dtype(self): ...
    def external_values(self):
        """The array that Series.values returns"""
    def internal_values(self):
        """The array that Series._values returns"""
    def array_values(self):
        """The array that Series.array returns"""
    @property
    def _can_hold_na(self) -> bool: ...
    @property
    def is_single_block(self) -> bool: ...
    def fast_xs(self, loc: int) -> SingleArrayManager: ...
    def get_slice(self, slobj: slice, axis: AxisInt = 0) -> SingleArrayManager: ...
    def get_rows_with_mask(self, indexer: npt.NDArray[np.bool_]) -> SingleArrayManager: ...
    def apply(self, func, **kwargs) -> Self: ...
    def setitem(self, indexer, value, warn: bool = True) -> SingleArrayManager:
        """
        Set values with indexer.

        For SingleArrayManager, this backs s[indexer] = value

        See `setitem_inplace` for a version that works inplace and doesn't
        return a new Manager.
        """
    def idelete(self, indexer) -> SingleArrayManager:
        """
        Delete selected locations in-place (new array, same ArrayManager)
        """
    def _get_data_subset(self, predicate: Callable) -> SingleArrayManager: ...
    def set_values(self, values: ArrayLike) -> None:
        """
        Set (replace) the values of the SingleArrayManager in place.

        Use at your own risk! This does not check if the passed values are
        valid for the current SingleArrayManager (length, dtype, etc).
        """
    def to_2d_mgr(self, columns: Index) -> ArrayManager:
        """
        Manager analogue of Series.to_frame
        """

class NullArrayProxy:
    """
    Proxy object for an all-NA array.

    Only stores the length of the array, and not the dtype. The dtype
    will only be known when actually concatenating (after determining the
    common dtype, for which this proxy is ignored).
    Using this object avoids that the internals/concat.py needs to determine
    the proper dtype and array type.
    """
    ndim: int
    n: Incomplete
    def __init__(self, n: int) -> None: ...
    @property
    def shape(self) -> tuple[int]: ...
    def to_array(self, dtype: DtypeObj) -> ArrayLike:
        """
        Helper function to create the actual all-NA array from the NullArrayProxy
        object.

        Parameters
        ----------
        arr : NullArrayProxy
        dtype : the dtype for the resulting array

        Returns
        -------
        np.ndarray or ExtensionArray
        """

def concat_arrays(to_concat: list) -> ArrayLike:
    """
    Alternative for concat_compat but specialized for use in the ArrayManager.

    Differences: only deals with 1D arrays (no axis keyword), assumes
    ensure_wrapped_if_datetimelike and does not skip empty arrays to determine
    the dtype.
    In addition ensures that all NullArrayProxies get replaced with actual
    arrays.

    Parameters
    ----------
    to_concat : list of arrays

    Returns
    -------
    np.ndarray or ExtensionArray
    """
