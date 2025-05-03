import pandas.core.base
from pandas.core.base import PandasObject as PandasObject
from pandas.io.formats.printing import pprint_thing as pprint_thing
from typing import NoReturn

TYPE_CHECKING: bool

class FrozenList(pandas.core.base.PandasObject, list):
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
    def __add__(self, other) -> FrozenList:
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
    def __iadd__(self, other) -> FrozenList:
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
    def __getitem__(self, n): ...
    def __radd__(self, other) -> Self: ...
    def __eq__(self, other: object) -> bool: ...
    def __req__(self, other: object) -> bool: ...
    def __mul__(self, other) -> Self: ...
    def __imul__(self, other) -> Self: ...
    def __reduce__(self): ...
    def __hash__(self) -> int: ...
    def _disabled(self, *args, **kwargs) -> NoReturn:
        """
        This method will not function because object is immutable.
        """
    def __setitem__(self, *args, **kwargs) -> NoReturn:
        """
        This method will not function because object is immutable.
        """
    def __setslice__(self, *args, **kwargs) -> NoReturn:
        """
        This method will not function because object is immutable.
        """
    def __delitem__(self, *args, **kwargs) -> NoReturn:
        """
        This method will not function because object is immutable.
        """
    def __delslice__(self, *args, **kwargs) -> NoReturn:
        """
        This method will not function because object is immutable.
        """
    def pop(self, *args, **kwargs) -> NoReturn:
        """
        This method will not function because object is immutable.
        """
    def append(self, *args, **kwargs) -> NoReturn:
        """
        This method will not function because object is immutable.
        """
    def extend(self, *args, **kwargs) -> NoReturn:
        """
        This method will not function because object is immutable.
        """
    def remove(self, *args, **kwargs) -> NoReturn:
        """
        This method will not function because object is immutable.
        """
    def sort(self, *args, **kwargs) -> NoReturn:
        """
        This method will not function because object is immutable.
        """
    def insert(self, *args, **kwargs) -> NoReturn:
        """
        This method will not function because object is immutable.
        """
