import numpy as np
from _typeshed import Incomplete
from pandas._config import using_copy_on_write as using_copy_on_write, warn_copy_on_write as warn_copy_on_write
from pandas._libs import lib as lib
from pandas._typing import ArrayLike as ArrayLike, AxisInt as AxisInt, DtypeObj as DtypeObj, Self as Self, Shape as Shape
from pandas.core.base import PandasObject as PandasObject
from pandas.core.construction import extract_array as extract_array
from pandas.core.dtypes.cast import find_common_type as find_common_type, np_can_hold_element as np_can_hold_element
from pandas.core.dtypes.dtypes import ExtensionDtype as ExtensionDtype, SparseDtype as SparseDtype
from pandas.core.indexes.api import Index as Index, default_index as default_index
from pandas.errors import AbstractMethodError as AbstractMethodError
from pandas.util._validators import validate_bool_kwarg as validate_bool_kwarg
from typing import Any, Literal

class _AlreadyWarned:
    warned_already: bool
    def __init__(self) -> None: ...

class DataManager(PandasObject):
    axes: list[Index]
    @property
    def items(self) -> Index: ...
    def __len__(self) -> int: ...
    @property
    def ndim(self) -> int: ...
    @property
    def shape(self) -> Shape: ...
    def _validate_set_axis(self, axis: AxisInt, new_labels: Index) -> None: ...
    def reindex_indexer(self, new_axis, indexer, axis: AxisInt, fill_value: Incomplete | None = None, allow_dups: bool = False, copy: bool = True, only_slice: bool = False) -> Self: ...
    def reindex_axis(self, new_index: Index, axis: AxisInt, fill_value: Incomplete | None = None, only_slice: bool = False) -> Self:
        """
        Conform data manager to new index.
        """
    def _equal_values(self, other: Self) -> bool:
        """
        To be implemented by the subclasses. Only check the column values
        assuming shape and indexes have already been checked.
        """
    def equals(self, other: object) -> bool:
        """
        Implementation for DataFrame.equals
        """
    def apply(self, f, align_keys: list[str] | None = None, **kwargs) -> Self: ...
    def apply_with_block(self, f, align_keys: list[str] | None = None, **kwargs) -> Self: ...
    def isna(self, func) -> Self: ...
    def fillna(self, value, limit: int | None, inplace: bool, downcast) -> Self: ...
    def where(self, other, cond, align: bool) -> Self: ...
    def putmask(self, mask, new, align: bool = True, warn: bool = True) -> Self: ...
    def round(self, decimals: int, using_cow: bool = False) -> Self: ...
    def replace(self, to_replace, value, inplace: bool) -> Self: ...
    def replace_regex(self, **kwargs) -> Self: ...
    def replace_list(self, src_list: list[Any], dest_list: list[Any], inplace: bool = False, regex: bool = False) -> Self:
        """do a list replace"""
    def interpolate(self, inplace: bool, **kwargs) -> Self: ...
    def pad_or_backfill(self, inplace: bool, **kwargs) -> Self: ...
    def shift(self, periods: int, fill_value) -> Self: ...
    def is_consolidated(self) -> bool: ...
    def consolidate(self) -> Self: ...
    def _consolidate_inplace(self) -> None: ...

class SingleDataManager(DataManager):
    @property
    def ndim(self) -> Literal[1]: ...
    @property
    def array(self) -> ArrayLike:
        """
        Quick access to the backing array of the Block or SingleArrayManager.
        """
    def setitem_inplace(self, indexer, value, warn: bool = True) -> None:
        """
        Set values with indexer.

        For Single[Block/Array]Manager, this backs s[indexer] = value

        This is an inplace version of `setitem()`, mutating the manager/values
        in place, not returning a new Manager (and Block), and thus never changing
        the dtype.
        """
    def grouped_reduce(self, func): ...
    @classmethod
    def from_array(cls, arr: ArrayLike, index: Index): ...

def interleaved_dtype(dtypes: list[DtypeObj]) -> DtypeObj | None:
    """
    Find the common dtype for `blocks`.

    Parameters
    ----------
    blocks : List[DtypeObj]

    Returns
    -------
    dtype : np.dtype, ExtensionDtype, or None
        None is returned when `blocks` is empty.
    """
def ensure_np_dtype(dtype: DtypeObj) -> np.dtype: ...
