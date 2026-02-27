from pandas.core.base import PandasObject
from typing import Generic, TypeVar

_T = TypeVar("_T")

class FrozenList(PandasObject, list[_T], Generic[_T]): ...
