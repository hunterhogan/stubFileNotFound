import np
import pandas._libs.algos as libalgos
import pandas._libs.lib as lib
import pandas.core.base
from pandas._config import using_copy_on_write as using_copy_on_write, warn_copy_on_write as warn_copy_on_write
from pandas.core.base import PandasObject as PandasObject
from pandas.core.construction import extract_array as extract_array
from pandas.core.dtypes.base import ExtensionDtype as ExtensionDtype
from pandas.core.dtypes.cast import find_common_type as find_common_type, np_can_hold_element as np_can_hold_element
from pandas.core.dtypes.dtypes import SparseDtype as SparseDtype
from pandas.core.indexes.api import default_index as default_index
from pandas.core.indexes.base import Index as Index
from pandas.errors import AbstractMethodError as AbstractMethodError
from pandas.util._validators import validate_bool_kwarg as validate_bool_kwarg
from typing import Any

TYPE_CHECKING: bool

class _AlreadyWarned:
    def __init__(self) -> None: ...

class DataManager(pandas.core.base.PandasObject):
    def __len__(self) -> int: ...
    def _validate_set_axis(self, axis: AxisInt, new_labels: Index) -> None: ...
    def reindex_indexer(self, new_axis, indexer, axis: AxisInt, fill_value, allow_dups: bool = ..., copy: bool = ..., only_slice: bool = ...) -> Self: ...
    def reindex_axis(self, new_index: Index, axis: AxisInt, fill_value, only_slice: bool = ...) -> Self:
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
    def apply(self, f, align_keys: list[str] | None, **kwargs) -> Self: ...
    def apply_with_block(self, f, align_keys: list[str] | None, **kwargs) -> Self: ...
    def isna(self, func) -> Self: ...
    def fillna(self, value, limit: int | None, inplace: bool, downcast) -> Self: ...
    def where(self, other, cond, align: bool) -> Self: ...
    def putmask(self, mask, new, align: bool = ..., warn: bool = ...) -> Self: ...
    def round(self, decimals: int, using_cow: bool = ...) -> Self: ...
    def replace(self, to_replace, value, inplace: bool) -> Self: ...
    def replace_regex(self, **kwargs) -> Self: ...
    def replace_list(self, src_list: list[Any], dest_list: list[Any], inplace: bool = ..., regex: bool = ...) -> Self:
        """do a list replace"""
    def interpolate(self, inplace: bool, **kwargs) -> Self: ...
    def pad_or_backfill(self, inplace: bool, **kwargs) -> Self: ...
    def shift(self, periods: int, fill_value) -> Self: ...
    def is_consolidated(self) -> bool: ...
    def consolidate(self) -> Self: ...
    def _consolidate_inplace(self) -> None: ...
    @property
    def items(self): ...
    @property
    def ndim(self): ...
    @property
    def shape(self): ...

class SingleDataManager(DataManager):
    def setitem_inplace(self, indexer, value, warn: bool = ...) -> None:
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
    @property
    def ndim(self): ...
    @property
    def array(self): ...
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
