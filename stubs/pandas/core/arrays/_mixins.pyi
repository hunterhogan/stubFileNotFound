import numpy as np
from _typeshed import Incomplete
from collections.abc import Sequence
from pandas import Series as Series
from pandas._libs import lib as lib
from pandas._libs.arrays import NDArrayBacked as NDArrayBacked
from pandas._libs.tslibs import is_supported_dtype as is_supported_dtype
from pandas._typing import ArrayLike as ArrayLike, AxisInt as AxisInt, Dtype as Dtype, F as F, FillnaOptions as FillnaOptions, NumpySorter as NumpySorter, NumpyValueArrayLike as NumpyValueArrayLike, PositionalIndexer2D as PositionalIndexer2D, PositionalIndexerTuple as PositionalIndexerTuple, ScalarIndexer as ScalarIndexer, Self as Self, SequenceIndexer as SequenceIndexer, Shape as Shape, TakeIndexer as TakeIndexer, npt as npt
from pandas.core import missing as missing
from pandas.core.algorithms import take as take, unique as unique
from pandas.core.array_algos.quantile import quantile_with_mask as quantile_with_mask
from pandas.core.array_algos.transforms import shift as shift
from pandas.core.arrays.base import ExtensionArray as ExtensionArray
from pandas.core.construction import extract_array as extract_array
from pandas.core.dtypes.common import pandas_dtype as pandas_dtype
from pandas.core.dtypes.dtypes import DatetimeTZDtype as DatetimeTZDtype, ExtensionDtype as ExtensionDtype, PeriodDtype as PeriodDtype
from pandas.core.dtypes.missing import array_equivalent as array_equivalent
from pandas.core.indexers import check_array_indexer as check_array_indexer
from pandas.core.sorting import nargminmax as nargminmax
from pandas.errors import AbstractMethodError as AbstractMethodError
from pandas.util._decorators import doc as doc
from pandas.util._validators import validate_bool_kwarg as validate_bool_kwarg, validate_fillna_kwargs as validate_fillna_kwargs, validate_insert_loc as validate_insert_loc
from typing import Any, Literal, overload

def ravel_compat(meth: F) -> F:
    """
    Decorator to ravel a 2D array before passing it to a cython operation,
    then reshape the result to our own shape.
    """

class NDArrayBackedExtensionArray(NDArrayBacked, ExtensionArray):
    """
    ExtensionArray that is backed by a single NumPy ndarray.
    """
    _ndarray: np.ndarray
    _internal_fill_value: Any
    def _box_func(self, x):
        """
        Wrap numpy type in our dtype.type if necessary.
        """
    def _validate_scalar(self, value) -> None: ...
    def view(self, dtype: Dtype | None = None) -> ArrayLike: ...
    def take(self, indices: TakeIndexer, *, allow_fill: bool = False, fill_value: Any = None, axis: AxisInt = 0) -> Self: ...
    def equals(self, other) -> bool: ...
    @classmethod
    def _from_factorized(cls, values, original): ...
    def _values_for_argsort(self) -> np.ndarray: ...
    def _values_for_factorize(self): ...
    def _hash_pandas_object(self, *, encoding: str, hash_key: str, categorize: bool) -> npt.NDArray[np.uint64]: ...
    def argmin(self, axis: AxisInt = 0, skipna: bool = True): ...
    def argmax(self, axis: AxisInt = 0, skipna: bool = True): ...
    def unique(self) -> Self: ...
    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[Self], axis: AxisInt = 0) -> Self: ...
    def searchsorted(self, value: NumpyValueArrayLike | ExtensionArray, side: Literal['left', 'right'] = 'left', sorter: NumpySorter | None = None) -> npt.NDArray[np.intp] | np.intp: ...
    def shift(self, periods: int = 1, fill_value: Incomplete | None = None): ...
    def __setitem__(self, key, value) -> None: ...
    def _validate_setitem_value(self, value): ...
    @overload
    def __getitem__(self, key: ScalarIndexer) -> Any: ...
    @overload
    def __getitem__(self, key: SequenceIndexer | PositionalIndexerTuple) -> Self: ...
    def _fill_mask_inplace(self, method: str, limit: int | None, mask: npt.NDArray[np.bool_]) -> None: ...
    def _pad_or_backfill(self, *, method: FillnaOptions, limit: int | None = None, limit_area: Literal['inside', 'outside'] | None = None, copy: bool = True) -> Self: ...
    def fillna(self, value: Incomplete | None = None, method: Incomplete | None = None, limit: int | None = None, copy: bool = True) -> Self: ...
    def _wrap_reduction_result(self, axis: AxisInt | None, result): ...
    def _putmask(self, mask: npt.NDArray[np.bool_], value) -> None:
        """
        Analogue to np.putmask(self, mask, value)

        Parameters
        ----------
        mask : np.ndarray[bool]
        value : scalar or listlike

        Raises
        ------
        TypeError
            If value cannot be cast to self.dtype.
        """
    def _where(self, mask: npt.NDArray[np.bool_], value) -> Self:
        """
        Analogue to np.where(mask, self, value)

        Parameters
        ----------
        mask : np.ndarray[bool]
        value : scalar or listlike

        Raises
        ------
        TypeError
            If value cannot be cast to self.dtype.
        """
    def insert(self, loc: int, item) -> Self:
        """
        Make new ExtensionArray inserting new item at location. Follows
        Python list.append semantics for negative values.

        Parameters
        ----------
        loc : int
        item : object

        Returns
        -------
        type(self)
        """
    def value_counts(self, dropna: bool = True) -> Series:
        """
        Return a Series containing counts of unique values.

        Parameters
        ----------
        dropna : bool, default True
            Don't include counts of NA values.

        Returns
        -------
        Series
        """
    def _quantile(self, qs: npt.NDArray[np.float64], interpolation: str) -> Self: ...
    def _cast_quantile_result(self, res_values: np.ndarray) -> np.ndarray:
        """
        Cast the result of quantile_with_mask to an appropriate dtype
        to pass to _from_backing_data in _quantile.
        """
    @classmethod
    def _empty(cls, shape: Shape, dtype: ExtensionDtype) -> Self:
        """
        Analogous to np.empty(shape, dtype=dtype)

        Parameters
        ----------
        shape : tuple[int]
        dtype : ExtensionDtype
        """
