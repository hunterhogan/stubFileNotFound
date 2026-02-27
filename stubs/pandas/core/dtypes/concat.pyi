from pandas import Categorical, CategoricalIndex, Series
from typing import Any, TypeVar

_CatT = TypeVar("_CatT", bound=Categorical | CategoricalIndex[Any] | Series)

def union_categoricals(
    to_union: list[_CatT],
    sort_categories: bool = False,
    ignore_order: bool = False,
) -> Categorical: ...
