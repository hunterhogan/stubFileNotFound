import np
import npt
import pandas._libs.arrays
import pandas._libs.lib as lib
import pandas.core.arrays.base
import pandas.core.missing as missing
from builtins import AxisInt, Shape
from pandas._libs.arrays import NDArrayBacked as NDArrayBacked
from pandas._libs.tslibs.np_datetime import is_supported_dtype as is_supported_dtype
from pandas._typing import F as F
from pandas.core.algorithms import take as take, unique as unique, value_counts as value_counts
from pandas.core.array_algos.quantile import quantile_with_mask as quantile_with_mask
from pandas.core.array_algos.transforms import shift as shift
from pandas.core.arrays.base import ExtensionArray as ExtensionArray
from pandas.core.construction import extract_array as extract_array
from pandas.core.dtypes.base import ExtensionDtype as ExtensionDtype
from pandas.core.dtypes.common import pandas_dtype as pandas_dtype
from pandas.core.dtypes.dtypes import DatetimeTZDtype as DatetimeTZDtype, PeriodDtype as PeriodDtype
from pandas.core.dtypes.missing import array_equivalent as array_equivalent
from pandas.core.indexers.utils import check_array_indexer as check_array_indexer
from pandas.core.sorting import nargminmax as nargminmax
from pandas.errors import AbstractMethodError as AbstractMethodError
from pandas.util._decorators import doc as doc
from pandas.util._validators import validate_bool_kwarg as validate_bool_kwarg, validate_fillna_kwargs as validate_fillna_kwargs, validate_insert_loc as validate_insert_loc
from typing import Any, ArrayLike, Dtype, FillnaOptions, Literal, PositionalIndexer2D, TakeIndexer

TYPE_CHECKING: bool
Self: None
npt: None
def ravel_compat(meth: F) -> F:
    """
    Decorator to ravel a 2D array before passing it to a cython operation,
    then reshape the result to our own shape.
    """

class NDArrayBackedExtensionArray(pandas._libs.arrays.NDArrayBacked, pandas.core.arrays.base.ExtensionArray):
    def _box_func(self, x):
        """
        Wrap numpy type in our dtype.type if necessary.
        """
    def _validate_scalar(self, value): ...
    def view(self, dtype: Dtype | None) -> ArrayLike: ...
    def take(self, indices: TakeIndexer, *, allow_fill: bool = ..., fill_value: Any, axis: AxisInt = ...) -> Self: ...
    def equals(self, other) -> bool: ...
    @classmethod
    def _from_factorized(cls, values, original): ...
    def _values_for_argsort(self) -> np.ndarray: ...
    def _values_for_factorize(self): ...
    def _hash_pandas_object(self, *, encoding: str, hash_key: str, categorize: bool) -> npt.NDArray[np.uint64]: ...
    def argmin(self, axis: AxisInt = ..., skipna: bool = ...): ...
    def argmax(self, axis: AxisInt = ..., skipna: bool = ...): ...
    def unique(self) -> Self: ...
    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[Self], axis: AxisInt = ...) -> Self:
        """
        Concatenate multiple array of this dtype.

        Parameters
        ----------
        to_concat : sequence of this type

        Returns
        -------
        ExtensionArray

        Examples
        --------
        >>> arr1 = pd.array([1, 2, 3])
        >>> arr2 = pd.array([4, 5, 6])
        >>> pd.arrays.IntegerArray._concat_same_type([arr1, arr2])
        <IntegerArray>
        [1, 2, 3, 4, 5, 6]
        Length: 6, dtype: Int64
        """
    def searchsorted(self, value: NumpyValueArrayLike | ExtensionArray, side: Literal['left', 'right'] = ..., sorter: NumpySorter | None) -> npt.NDArray[np.intp] | np.intp:
        """
        Find indices where elements should be inserted to maintain order.

        Find the indices into a sorted array `self` (a) such that, if the
        corresponding elements in `value` were inserted before the indices,
        the order of `self` would be preserved.

        Assuming that `self` is sorted:

        ======  ================================
        `side`  returned index `i` satisfies
        ======  ================================
        left    ``self[i-1] < value <= self[i]``
        right   ``self[i-1] <= value < self[i]``
        ======  ================================

        Parameters
        ----------
        value : array-like, list or scalar
            Value(s) to insert into `self`.
        side : {'left', 'right'}, optional
            If 'left', the index of the first suitable location found is given.
            If 'right', return the last such index.  If there is no suitable
            index, return either 0 or N (where N is the length of `self`).
        sorter : 1-D array-like, optional
            Optional array of integer indices that sort array a into ascending
            order. They are typically the result of argsort.

        Returns
        -------
        array of ints or int
            If value is array-like, array of insertion points.
            If value is scalar, a single integer.

        See Also
        --------
        numpy.searchsorted : Similar method from NumPy.

        Examples
        --------
        >>> arr = pd.array([1, 2, 3, 5])
        >>> arr.searchsorted([4])
        array([3])
        """
    def shift(self, periods: int = ..., fill_value):
        """
        Shift values by desired number.

        Newly introduced missing values are filled with
        ``self.dtype.na_value``.

        Parameters
        ----------
        periods : int, default 1
            The number of periods to shift. Negative values are allowed
            for shifting backwards.

        fill_value : object, optional
            The scalar value to use for newly introduced missing values.
            The default is ``self.dtype.na_value``.

        Returns
        -------
        ExtensionArray
            Shifted.

        Notes
        -----
        If ``self`` is empty or ``periods`` is 0, a copy of ``self`` is
        returned.

        If ``periods > len(self)``, then an array of size
        len(self) is returned, with all values filled with
        ``self.dtype.na_value``.

        For 2-dimensional ExtensionArrays, we are always shifting along axis=0.

        Examples
        --------
        >>> arr = pd.array([1, 2, 3])
        >>> arr.shift(2)
        <IntegerArray>
        [<NA>, <NA>, 1]
        Length: 3, dtype: Int64
        """
    def __setitem__(self, key, value) -> None: ...
    def _validate_setitem_value(self, value): ...
    def __getitem__(self, key: PositionalIndexer2D) -> Self | Any: ...
    def _fill_mask_inplace(self, method: str, limit: int | None, mask: npt.NDArray[np.bool_]) -> None: ...
    def _pad_or_backfill(self, *, method: FillnaOptions, limit: int | None, limit_area: Literal['inside', 'outside'] | None, copy: bool = ...) -> Self: ...
    def fillna(self, value, method, limit: int | None, copy: bool = ...) -> Self:
        '''
        Fill NA/NaN values using the specified method.

        Parameters
        ----------
        value : scalar, array-like
            If a scalar value is passed it is used to fill all missing values.
            Alternatively, an array-like "value" can be given. It\'s expected
            that the array-like have the same length as \'self\'.
        method : {\'backfill\', \'bfill\', \'pad\', \'ffill\', None}, default None
            Method to use for filling holes in reindexed Series:

            * pad / ffill: propagate last valid observation forward to next valid.
            * backfill / bfill: use NEXT valid observation to fill gap.

            .. deprecated:: 2.1.0

        limit : int, default None
            If method is specified, this is the maximum number of consecutive
            NaN values to forward/backward fill. In other words, if there is
            a gap with more than this number of consecutive NaNs, it will only
            be partially filled. If method is not specified, this is the
            maximum number of entries along the entire axis where NaNs will be
            filled.

            .. deprecated:: 2.1.0

        copy : bool, default True
            Whether to make a copy of the data before filling. If False, then
            the original should be modified and no new memory should be allocated.
            For ExtensionArray subclasses that cannot do this, it is at the
            author\'s discretion whether to ignore "copy=False" or to raise.
            The base class implementation ignores the keyword in pad/backfill
            cases.

        Returns
        -------
        ExtensionArray
            With NA/NaN filled.

        Examples
        --------
        >>> arr = pd.array([np.nan, np.nan, 2, 3, np.nan, np.nan])
        >>> arr.fillna(0)
        <IntegerArray>
        [0, 0, 2, 3, 0, 0]
        Length: 6, dtype: Int64
        '''
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
    def _where(self: Self, mask: npt.NDArray[np.bool_], value) -> Self:
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
    def value_counts(self, dropna: bool = ...) -> Series:
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
