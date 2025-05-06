from pandas._typing import Self as Self
from pandas.core.base import PandasObject as PandasObject
from typing import NoReturn

class FrozenList(PandasObject, list):
    """
    Container that doesn't allow setting item *but*
    because it's technically hashable, will be used
    for lookups, appropriately, etc.
    """
    def union(self, other) -> FrozenList:
        """
        Returns a FrozenList with other concatenated to the end of self.

        Parameters
        ----------
        other : array-like
            The array-like whose elements we are concatenating.

        Returns
        -------
        FrozenList
            The collection difference between self and other.
        """
    def difference(self, other) -> FrozenList:
        """
        Returns a FrozenList with elements from other removed from self.

        Parameters
        ----------
        other : array-like
            The array-like whose elements we are removing self.

        Returns
        -------
        FrozenList
            The collection difference between self and other.
        """
    __add__ = union
    __iadd__ = union
    def __getitem__(self, n): ...
    def __radd__(self, other) -> Self: ...
    def __eq__(self, other: object) -> bool: ...
    __req__ = __eq__
    def __mul__(self, other) -> Self: ...
    __imul__ = __mul__
    def __reduce__(self): ...
    def __hash__(self) -> int: ...
    def _disabled(self, *args, **kwargs) -> NoReturn:
        """
        This method will not function because object is immutable.
        """
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    __setitem__ = _disabled
    __setslice__ = _disabled
    __delitem__ = _disabled
    __delslice__ = _disabled
    pop = _disabled
    append = _disabled
    extend = _disabled
    remove = _disabled
    sort = _disabled
    insert = _disabled
