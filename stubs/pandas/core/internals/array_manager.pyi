import np
import npt
import pandas._libs.lib as lib
import pandas.core.algorithms as algos
import pandas.core.internals.base
from _typeshed import Incomplete
from pandas._libs.algos import ensure_platform_int as ensure_platform_int
from pandas._libs.internals import BlockPlacement as BlockPlacement
from pandas._libs.lib import is_integer as is_integer
from pandas._libs.tslibs.nattype import NaT as NaT
from pandas.core.array_algos.quantile import quantile_compat as quantile_compat
from pandas.core.array_algos.take import take_1d as take_1d
from pandas.core.arrays.base import ExtensionArray as ExtensionArray
from pandas.core.arrays.datetimes import DatetimeArray as DatetimeArray
from pandas.core.arrays.numpy_ import NumpyExtensionArray as NumpyExtensionArray
from pandas.core.arrays.timedeltas import TimedeltaArray as TimedeltaArray
from pandas.core.construction import ensure_wrapped_if_datetimelike as ensure_wrapped_if_datetimelike, extract_array as extract_array, sanitize_array as sanitize_array
from pandas.core.dtypes.astype import astype_array as astype_array, astype_array_safe as astype_array_safe
from pandas.core.dtypes.base import ExtensionDtype as ExtensionDtype
from pandas.core.dtypes.cast import ensure_dtype_can_hold_na as ensure_dtype_can_hold_na, find_common_type as find_common_type, infer_dtype_from_scalar as infer_dtype_from_scalar, np_find_common_type as np_find_common_type
from pandas.core.dtypes.common import is_datetime64_ns_dtype as is_datetime64_ns_dtype, is_numeric_dtype as is_numeric_dtype, is_object_dtype as is_object_dtype, is_timedelta64_ns_dtype as is_timedelta64_ns_dtype
from pandas.core.dtypes.generic import ABCDataFrame as ABCDataFrame, ABCSeries as ABCSeries
from pandas.core.dtypes.missing import array_equals as array_equals, isna as isna, na_value_for_dtype as na_value_for_dtype
from pandas.core.indexers.utils import maybe_convert_indices as maybe_convert_indices, validate_indices as validate_indices
from pandas.core.indexes.base import Index as Index, ensure_index as ensure_index, get_values_for_csv as get_values_for_csv
from pandas.core.internals.base import DataManager as DataManager, SingleDataManager as SingleDataManager, ensure_np_dtype as ensure_np_dtype, interleaved_dtype as interleaved_dtype
from pandas.core.internals.blocks import ensure_block_shape as ensure_block_shape, external_values as external_values, extract_pandas_array as extract_pandas_array, maybe_coerce_values as maybe_coerce_values, new_block as new_block
from pandas.core.internals.managers import make_na_array as make_na_array
from typing import Callable, ClassVar, Literal

TYPE_CHECKING: bool

class BaseArrayManager(pandas.core.internals.base.DataManager):
    _axes: Incomplete
    arrays: Incomplete
    def __init__(self, arrays: list[np.ndarray | ExtensionArray], axes: list[Index], verify_integrity: bool = ...) -> None: ...
    def make_empty(self, axes) -> Self:
        """Return an empty ArrayManager with the items axis of len 0 (no columns)"""
    @staticmethod
    def _normalize_axis(axis: AxisInt) -> int: ...
    def set_axis(self, axis: AxisInt, new_labels: Index) -> None: ...
    def get_dtypes(self) -> npt.NDArray[np.object_]: ...
    def add_references(self, mgr: BaseArrayManager) -> None:
        """
        Only implemented on the BlockManager level
        """
    def apply(self, f, align_keys: list[str] | None, **kwargs) -> Self:
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
    def apply_with_block(self, f, align_keys, **kwargs) -> Self: ...
    def setitem(self, indexer, value, warn: bool = ...) -> Self: ...
    def diff(self, n: int) -> Self: ...
    def astype(self, dtype, copy: bool | None = ..., errors: str = ...) -> Self: ...
    def convert(self, copy: bool | None) -> Self: ...
    def get_values_for_csv(self, *, float_format, date_format, decimal, na_rep: str = ..., quoting) -> Self: ...
    def _get_data_subset(self, predicate: Callable) -> Self: ...
    def get_bool_data(self, copy: bool = ...) -> Self:
        """
        Select columns that are bool-dtype and object-dtype columns that are all-bool.

        Parameters
        ----------
        copy : bool, default False
            Whether to copy the blocks
        """
    def get_numeric_data(self, copy: bool = ...) -> Self:
        """
        Select columns that have a numeric dtype.

        Parameters
        ----------
        copy : bool, default False
            Whether to copy the blocks
        """
    def copy(self, deep: bool | Literal['all'] | None = ...) -> Self:
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
    def reindex_indexer(self, new_axis, indexer, axis: AxisInt, fill_value, allow_dups: bool = ..., copy: bool | None = ..., only_slice: bool = ..., use_na_proxy: bool = ...) -> Self: ...
    def _reindex_indexer(self, new_axis, indexer: npt.NDArray[np.intp] | None, axis: AxisInt, fill_value, allow_dups: bool = ..., copy: bool | None = ..., use_na_proxy: bool = ...) -> Self:
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
    def take(self, indexer: npt.NDArray[np.intp], axis: AxisInt = ..., verify: bool = ...) -> Self:
        """
        Take items along any axis.
        """
    def _make_na_array(self, fill_value, use_na_proxy: bool = ...): ...
    def _equal_values(self, other) -> bool:
        """
        Used in .equals defined in base class. Only check the column values
        assuming shape and indexes have already been checked.
        """
    @property
    def items(self): ...
    @property
    def axes(self): ...
    @property
    def shape_proper(self): ...
    @property
    def any_extension_types(self): ...
    @property
    def is_view(self): ...
    @property
    def is_single_block(self): ...

class ArrayManager(BaseArrayManager):
    def __init__(self, arrays: list[np.ndarray | ExtensionArray], axes: list[Index], verify_integrity: bool = ...) -> None: ...
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
    def get_slice(self, slobj: slice, axis: AxisInt = ...) -> ArrayManager: ...
    def iget(self, i: int) -> SingleArrayManager:
        """
        Return the data as a SingleArrayManager.
        """
    def iget_values(self, i: int) -> ArrayLike:
        """
        Return the data for column i as the values (ndarray or ExtensionArray).
        """
    def iset(self, loc: int | slice | np.ndarray, value: ArrayLike, inplace: bool = ..., refs) -> None:
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
    def column_setitem(self, loc: int, idx: int | slice | np.ndarray, value, inplace_only: bool = ...) -> None:
        '''
        Set values ("setitem") into a single column (not setting the full column).

        This is a method on the ArrayManager level, to avoid creating an
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
    def quantile(self, *, qs: Index, transposed: bool = ..., interpolation: QuantileInterpolation = ...) -> ArrayManager: ...
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
    def as_array(self, dtype, copy: bool = ..., na_value: object = ...) -> np.ndarray:
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
    @property
    def ndim(self): ...
    @property
    def column_arrays(self): ...

class SingleArrayManager(BaseArrayManager, pandas.core.internals.base.SingleDataManager):
    _axes: Incomplete
    arrays: Incomplete
    def __init__(self, arrays: list[np.ndarray | ExtensionArray], axes: list[Index], verify_integrity: bool = ...) -> None: ...
    def _verify_integrity(self) -> None: ...
    @staticmethod
    def _normalize_axis(axis): ...
    def make_empty(self, axes) -> Self:
        """Return an empty ArrayManager with index/array of length 0"""
    @classmethod
    def from_array(cls, array, index) -> SingleArrayManager: ...
    def external_values(self):
        """The array that Series.values returns"""
    def internal_values(self):
        """The array that Series._values returns"""
    def array_values(self):
        """The array that Series.array returns"""
    def fast_xs(self, loc: int) -> SingleArrayManager: ...
    def get_slice(self, slobj: slice, axis: AxisInt = ...) -> SingleArrayManager: ...
    def get_rows_with_mask(self, indexer: npt.NDArray[np.bool_]) -> SingleArrayManager: ...
    def apply(self, func, **kwargs) -> Self: ...
    def setitem(self, indexer, value, warn: bool = ...) -> SingleArrayManager:
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
    @property
    def ndim(self): ...
    @property
    def axes(self): ...
    @property
    def index(self): ...
    @property
    def dtype(self): ...
    @property
    def _can_hold_na(self): ...
    @property
    def is_single_block(self): ...

class NullArrayProxy:
    ndim: ClassVar[int] = ...
    def __init__(self, n: int) -> None: ...
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
    @property
    def shape(self): ...
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
