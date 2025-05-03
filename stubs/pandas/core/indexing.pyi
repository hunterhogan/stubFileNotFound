import np
import pandas._libs.indexing
import pandas.core.algorithms as algos
import pandas.core.common as com
import typing
from pandas._config import using_copy_on_write as using_copy_on_write, warn_copy_on_write as warn_copy_on_write
from pandas._libs.indexing import NDFrameIndexerBase as NDFrameIndexerBase
from pandas._libs.lib import is_integer as is_integer, is_iterator as is_iterator, is_list_like as is_list_like, is_scalar as is_scalar, item_from_zerodim as item_from_zerodim
from pandas.core.construction import extract_array as extract_array, pd_array as pd_array
from pandas.core.dtypes.base import ExtensionDtype as ExtensionDtype
from pandas.core.dtypes.cast import can_hold_element as can_hold_element, maybe_promote as maybe_promote
from pandas.core.dtypes.common import is_bool_dtype as is_bool_dtype, is_numeric_dtype as is_numeric_dtype, is_object_dtype as is_object_dtype
from pandas.core.dtypes.concat import concat_compat as concat_compat
from pandas.core.dtypes.generic import ABCDataFrame as ABCDataFrame, ABCSeries as ABCSeries
from pandas.core.dtypes.inference import is_array_like as is_array_like, is_hashable as is_hashable, is_sequence as is_sequence
from pandas.core.dtypes.missing import construct_1d_array_from_inferred_fill_value as construct_1d_array_from_inferred_fill_value, infer_fill_value as infer_fill_value, is_valid_na_for_dtype as is_valid_na_for_dtype, isna as isna, na_value_for_dtype as na_value_for_dtype
from pandas.core.indexers.utils import check_array_indexer as check_array_indexer, is_list_like_indexer as is_list_like_indexer, is_scalar_indexer as is_scalar_indexer, length_of_indexer as length_of_indexer
from pandas.core.indexes.base import Index as Index
from pandas.core.indexes.multi import MultiIndex as MultiIndex
from pandas.errors import AbstractMethodError as AbstractMethodError, ChainedAssignmentError as ChainedAssignmentError, IndexingError as IndexingError, InvalidIndexError as InvalidIndexError, LossySetitemError as LossySetitemError, _check_cacher as _check_cacher
from pandas.util._decorators import doc as doc
from pandas.util._exceptions import find_stack_level as find_stack_level
from typing import Any, ClassVar

TYPE_CHECKING: bool
PYPY: bool
_chained_assignment_msg: str
_chained_assignment_warning_msg: str
T: typing.TypeVar
_NS: slice
_one_ellipsis_message: str

class _IndexSlice:
    def __getitem__(self, arg): ...
IndexSlice: _IndexSlice

class IndexingMixin:
    @property
    def iloc(self): ...
    @property
    def loc(self): ...
    @property
    def at(self): ...
    @property
    def iat(self): ...

class _LocationIndexer(pandas._libs.indexing.NDFrameIndexerBase):
    axis: ClassVar[None] = ...
    def __call__(self, axis: Axis | None) -> Self: ...
    def _get_setitem_indexer(self, key):
        """
        Convert a potentially-label-based key into a positional indexer.
        """
    def _maybe_mask_setitem_value(self, indexer, value):
        """
        If we have obj.iloc[mask] = series_or_frame and series_or_frame has the
        same length as obj, we treat this as obj.iloc[mask] = series_or_frame[mask],
        similar to Series.__setitem__.

        Note this is only for loc, not iloc.
        """
    def _ensure_listlike_indexer(self, key, axis, value) -> None:
        """
        Ensure that a list-like of column labels are all present by adding them if
        they do not already exist.

        Parameters
        ----------
        key : list-like of column labels
            Target labels.
        axis : key axis if known
        """
    def __setitem__(self, key, value) -> None: ...
    def _validate_key(self, key, axis: AxisInt):
        """
        Ensure that key is valid for current indexer.

        Parameters
        ----------
        key : scalar, slice or list-like
            Key requested.
        axis : int
            Dimension on which the indexing is being made.

        Raises
        ------
        TypeError
            If the key (or some element of it) has wrong type.
        IndexError
            If the key (or some element of it) is out of bounds.
        KeyError
            If the key was not found.
        """
    def _expand_ellipsis(self, tup: tuple) -> tuple:
        """
        If a tuple key includes an Ellipsis, replace it with an appropriate
        number of null slices.
        """
    def _validate_tuple_indexer(self, key: tuple) -> tuple:
        """
        Check the key for valid keys across my indexer.
        """
    def _is_nested_tuple_indexer(self, tup: tuple) -> bool:
        """
        Returns
        -------
        bool
        """
    def _convert_tuple(self, key: tuple) -> tuple: ...
    def _validate_key_length(self, key: tuple) -> tuple: ...
    def _getitem_tuple_same_dim(self, tup: tuple):
        """
        Index with indexers that should return an object of the same dimension
        as self.obj.

        This is only called after a failed call to _getitem_lowerdim.
        """
    def _getitem_lowerdim(self, tup: tuple): ...
    def _getitem_nested_tuple(self, tup: tuple): ...
    def _convert_to_indexer(self, key, axis: AxisInt): ...
    def _check_deprecated_callable_usage(self, key: Any, maybe_callable: T) -> T: ...
    def __getitem__(self, key): ...
    def _is_scalar_access(self, key: tuple): ...
    def _getitem_tuple(self, tup: tuple): ...
    def _getitem_axis(self, key, axis: AxisInt): ...
    def _has_valid_setitem_indexer(self, indexer) -> bool: ...
    def _getbool_axis(self, key, axis: AxisInt): ...

class _LocIndexer(_LocationIndexer):
    _takeable: ClassVar[bool] = ...
    _valid_types: ClassVar[str] = ...
    _docstring_components: ClassVar[list] = ...
    def _validate_key(self, key, axis: Axis):
        """
        Ensure that key is valid for current indexer.

        Parameters
        ----------
        key : scalar, slice or list-like
            Key requested.
        axis : int
            Dimension on which the indexing is being made.

        Raises
        ------
        TypeError
            If the key (or some element of it) has wrong type.
        IndexError
            If the key (or some element of it) is out of bounds.
        KeyError
            If the key was not found.
        """
    def _has_valid_setitem_indexer(self, indexer) -> bool: ...
    def _is_scalar_access(self, key: tuple) -> bool:
        """
        Returns
        -------
        bool
        """
    def _multi_take_opportunity(self, tup: tuple) -> bool:
        """
        Check whether there is the possibility to use ``_multi_take``.

        Currently the limit is that all axes being indexed, must be indexed with
        list-likes.

        Parameters
        ----------
        tup : tuple
            Tuple of indexers, one per axis.

        Returns
        -------
        bool
            Whether the current indexing,
            can be passed through `_multi_take`.
        """
    def _multi_take(self, tup: tuple):
        """
        Create the indexers for the passed tuple of keys, and
        executes the take operation. This allows the take operation to be
        executed all at once, rather than once for each dimension.
        Improving efficiency.

        Parameters
        ----------
        tup : tuple
            Tuple of indexers, one per axis.

        Returns
        -------
        values: same type as the object being indexed
        """
    def _getitem_iterable(self, key, axis: AxisInt):
        """
        Index current object with an iterable collection of keys.

        Parameters
        ----------
        key : iterable
            Targeted labels.
        axis : int
            Dimension on which the indexing is being made.

        Raises
        ------
        KeyError
            If no key was found. Will change in the future to raise if not all
            keys were found.

        Returns
        -------
        scalar, DataFrame, or Series: indexed value(s).
        """
    def _getitem_tuple(self, tup: tuple): ...
    def _get_label(self, label, axis: AxisInt): ...
    def _handle_lowerdim_multi_index_axis0(self, tup: tuple): ...
    def _getitem_axis(self, key, axis: AxisInt): ...
    def _get_slice_axis(self, slice_obj: slice, axis: AxisInt):
        """
        This is pretty simple as we just have to deal with labels.
        """
    def _convert_to_indexer(self, key, axis: AxisInt):
        """
        Convert indexing key into something we can use to do actual fancy
        indexing on a ndarray.

        Examples
        ix[:5] -> slice(0, 5)
        ix[[1,2,3]] -> [1,2,3]
        ix[['foo', 'bar', 'baz']] -> [i, j, k] (indices of foo, bar, baz)

        Going by Zen of Python?
        'In the face of ambiguity, refuse the temptation to guess.'
        raise AmbiguousIndexError with integer labels?
        - No, prefer label-based indexing
        """
    def _get_listlike_indexer(self, key, axis: AxisInt):
        """
        Transform a list-like of keys into a new index and an indexer.

        Parameters
        ----------
        key : list-like
            Targeted labels.
        axis:  int
            Dimension on which the indexing is being made.

        Raises
        ------
        KeyError
            If at least one key was requested but none was found.

        Returns
        -------
        keyarr: Index
            New index (coinciding with 'key' if the axis is unique).
        values : array-like
            Indexer for the return object, -1 denotes keys not found.
        """

class _iLocIndexer(_LocationIndexer):
    _valid_types: ClassVar[str] = ...
    _takeable: ClassVar[bool] = ...
    _docstring_components: ClassVar[list] = ...
    def _validate_key(self, key, axis: AxisInt): ...
    def _has_valid_setitem_indexer(self, indexer) -> bool:
        """
        Validate that a positional indexer cannot enlarge its target
        will raise if needed, does not modify the indexer externally.

        Returns
        -------
        bool
        """
    def _is_scalar_access(self, key: tuple) -> bool:
        """
        Returns
        -------
        bool
        """
    def _validate_integer(self, key: int | np.integer, axis: AxisInt) -> None:
        """
        Check that 'key' is a valid position in the desired axis.

        Parameters
        ----------
        key : int
            Requested position.
        axis : int
            Desired axis.

        Raises
        ------
        IndexError
            If 'key' is not a valid position in axis 'axis'.
        """
    def _getitem_tuple(self, tup: tuple): ...
    def _get_list_axis(self, key, axis: AxisInt):
        """
        Return Series values by list or array of integers.

        Parameters
        ----------
        key : list-like positional indexer
        axis : int

        Returns
        -------
        Series object

        Notes
        -----
        `axis` can only be zero.
        """
    def _getitem_axis(self, key, axis: AxisInt): ...
    def _get_slice_axis(self, slice_obj: slice, axis: AxisInt): ...
    def _convert_to_indexer(self, key, axis: AxisInt):
        """
        Much simpler as we only have to deal with our valid types.
        """
    def _get_setitem_indexer(self, key): ...
    def _setitem_with_indexer(self, indexer, value, name: str = ...):
        """
        _setitem_with_indexer is for setting values on a Series/DataFrame
        using positional indexers.

        If the relevant keys are not present, the Series/DataFrame may be
        expanded.

        This method is currently broken when dealing with non-unique Indexes,
        since it goes from positional indexers back to labels when calling
        BlockManager methods, see GH#12991, GH#22046, GH#15686.
        """
    def _setitem_with_indexer_split_path(self, indexer, value, name: str):
        """
        Setitem column-wise.
        """
    def _setitem_with_indexer_2d_value(self, indexer, value): ...
    def _setitem_with_indexer_frame_value(self, indexer, value: DataFrame, name: str): ...
    def _setitem_single_column(self, loc: int, value, plane_indexer) -> None:
        """

        Parameters
        ----------
        loc : int
            Indexer for column position
        plane_indexer : int, slice, listlike[int]
            The indexer we use for setitem along axis=0.
        """
    def _setitem_single_block(self, indexer, value, name: str) -> None:
        """
        _setitem_with_indexer for the case when we have a single Block.
        """
    def _setitem_with_indexer_missing(self, indexer, value):
        """
        Insert new row(s) or column(s) into the Series or DataFrame.
        """
    def _ensure_iterable_column_indexer(self, column_indexer):
        """
        Ensure that our column indexer is something that can be iterated over.
        """
    def _align_series(self, indexer, ser: Series, multiindex_indexer: bool = ..., using_cow: bool = ...):
        """
        Parameters
        ----------
        indexer : tuple, slice, scalar
            Indexer used to get the locations that will be set to `ser`.
        ser : pd.Series
            Values to assign to the locations specified by `indexer`.
        multiindex_indexer : bool, optional
            Defaults to False. Should be set to True if `indexer` was from
            a `pd.MultiIndex`, to avoid unnecessary broadcasting.

        Returns
        -------
        `np.array` of `ser` broadcast to the appropriate shape for assignment
        to the locations selected by `indexer`
        """
    def _align_frame(self, indexer, df: DataFrame) -> DataFrame: ...

class _ScalarAccessIndexer(pandas._libs.indexing.NDFrameIndexerBase):
    def _convert_key(self, key): ...
    def __getitem__(self, key): ...
    def __setitem__(self, key, value) -> None: ...

class _AtIndexer(_ScalarAccessIndexer):
    _takeable: ClassVar[bool] = ...
    _docstring_components: ClassVar[list] = ...
    def _convert_key(self, key):
        """
        Require they keys to be the same type as the index. (so we don't
        fallback)
        """
    def __getitem__(self, key): ...
    def __setitem__(self, key, value) -> None: ...
    @property
    def _axes_are_unique(self): ...

class _iAtIndexer(_ScalarAccessIndexer):
    _takeable: ClassVar[bool] = ...
    _docstring_components: ClassVar[list] = ...
    def _convert_key(self, key):
        """
        Require integer args. (and convert to label arguments)
        """
def _tuplify(ndim: int, loc: Hashable) -> tuple[Hashable | slice, ...]:
    """
    Given an indexer for the first dimension, create an equivalent tuple
    for indexing over all dimensions.

    Parameters
    ----------
    ndim : int
    loc : object

    Returns
    -------
    tuple
    """
def _tupleize_axis_indexer(ndim: int, axis: AxisInt, key) -> tuple:
    """
    If we have an axis, adapt the given key to be axis-independent.
    """
def check_bool_indexer(index: Index, key) -> np.ndarray:
    """
    Check if key is a valid boolean indexer for an object with such index and
    perform reindexing or conversion if needed.

    This function assumes that is_bool_indexer(key) == True.

    Parameters
    ----------
    index : Index
        Index of the object on which the indexing is done.
    key : list-like
        Boolean indexer to check.

    Returns
    -------
    np.array
        Resulting key.

    Raises
    ------
    IndexError
        If the key does not have the same length as index.
    IndexingError
        If the index of the key is unalignable to index.
    """
def convert_missing_indexer(indexer):
    """
    Reverse convert a missing indexer, which is a dict
    return the scalar indexer and a boolean indicating if we converted
    """
def convert_from_missing_indexer_tuple(indexer, axes):
    """
    Create a filtered indexer that doesn't have any missing indexers.
    """
def maybe_convert_ix(*args):
    """
    We likely want to take the cross-product.
    """
def is_nested_tuple(tup, labels) -> bool:
    """
    Returns
    -------
    bool
    """
def is_label_like(key) -> bool:
    """
    Returns
    -------
    bool
    """
def need_slice(obj: slice) -> bool:
    """
    Returns
    -------
    bool
    """
def check_dict_or_set_indexers(key) -> None:
    """
    Check if the indexer is or contains a dict or set, which is no longer allowed.
    """
