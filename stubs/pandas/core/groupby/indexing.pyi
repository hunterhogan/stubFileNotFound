from typing import (
    Any,
    Generic,
    Literal,
    TypeVar,
)

from pandas import (
    DataFrame,
    Series,
)
from pandas.core.groupby import groupby

from pandas._typing import PositionalIndexer

_GroupByT = TypeVar("_GroupByT", bound=groupby.GroupBy[Any])

class GroupByIndexingMixin: ...

class GroupByPositionalSelector:
    groupby_object: groupby.GroupBy[Any]
    def __getitem__(self, arg: PositionalIndexer | tuple[Any, ...]) -> DataFrame | Series: ...

class GroupByNthSelector(Generic[_GroupByT]):
    groupby_object: _GroupByT

    def __call__(
        self,
        n: PositionalIndexer | tuple[Any, ...],
        dropna: Literal["any", "all", None] = None,
    ) -> DataFrame | Series: ...
    def __getitem__(self, n: PositionalIndexer | tuple[Any, ...]) -> DataFrame | Series: ...
