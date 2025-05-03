import groupby
import np
from _typeshed import Incomplete
from collections.abc import Iterable
from pandas._libs.lib import is_integer as is_integer, is_list_like as is_list_like
from pandas._libs.properties import cache_readonly as cache_readonly
from pandas.util._decorators import doc as doc
from typing import ClassVar, Literal

TYPE_CHECKING: bool

class GroupByIndexingMixin:
    _positional_selector: Incomplete
    _ascending_count: Incomplete
    _descending_count: Incomplete
    def _make_mask_from_positional_indexer(self, arg: PositionalIndexer | tuple) -> np.ndarray: ...
    def _make_mask_from_int(self, arg: int) -> np.ndarray: ...
    def _make_mask_from_list(self, args: Iterable[int]) -> bool | np.ndarray: ...
    def _make_mask_from_tuple(self, args: tuple) -> bool | np.ndarray: ...
    def _make_mask_from_slice(self, arg: slice) -> bool | np.ndarray: ...

class GroupByPositionalSelector:
    _docstring_components: ClassVar[list] = ...
    def __init__(self, groupby_object: groupby.GroupBy) -> None: ...
    def __getitem__(self, arg: PositionalIndexer | tuple) -> DataFrame | Series:
        """
        Select by positional index per group.

        Implements GroupBy._positional_selector

        Parameters
        ----------
        arg : PositionalIndexer | tuple
            Allowed values are:
            - int
            - int valued iterable such as list or range
            - slice with step either None or positive
            - tuple of integers and slices

        Returns
        -------
        Series
            The filtered subset of the original groupby Series.
        DataFrame
            The filtered subset of the original groupby DataFrame.

        See Also
        --------
        DataFrame.iloc : Integer-location based indexing for selection by position.
        GroupBy.head : Return first n rows of each group.
        GroupBy.tail : Return last n rows of each group.
        GroupBy._positional_selector : Return positional selection for each group.
        GroupBy.nth : Take the nth row from each group if n is an int, or a
            subset of rows, if n is a list of ints.
        """

class GroupByNthSelector:
    def __init__(self, groupby_object: groupby.GroupBy) -> None: ...
    def __call__(self, n: PositionalIndexer | tuple, dropna: Literal['any', 'all', None]) -> DataFrame | Series: ...
    def __getitem__(self, n: PositionalIndexer | tuple) -> DataFrame | Series: ...
