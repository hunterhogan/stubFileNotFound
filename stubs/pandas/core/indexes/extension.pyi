import numpy as np
from pandas._typing import ArrayLike as ArrayLike, npt as npt
from pandas.core.arrays import IntervalArray as IntervalArray
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray as NDArrayBackedExtensionArray
from pandas.core.indexes.base import Index as Index
from typing import TypeVar

from collections.abc import Callable

_ExtensionIndexT = TypeVar('_ExtensionIndexT', bound='ExtensionIndex')

def _inherit_from_data(name: str, delegate: type, cache: bool = False, wrap: bool = False):
    """
    Make an alias for a method of the underlying ExtensionArray.

    Parameters
    ----------
    name : str
        Name of an attribute the class should inherit from its EA parent.
    delegate : class
    cache : bool, default False
        Whether to convert wrapped properties into cache_readonly
    wrap : bool, default False
        Whether to wrap the inherited result in an Index.

    Returns
    -------
    attribute, method, property, or cache_readonly
    """
def inherit_names(names: list[str], delegate: type, cache: bool = False, wrap: bool = False) -> Callable[[type[_ExtensionIndexT]], type[_ExtensionIndexT]]:
    """
    Class decorator to pin attributes from an ExtensionArray to a Index subclass.

    Parameters
    ----------
    names : List[str]
    delegate : class
    cache : bool, default False
    wrap : bool, default False
        Whether to wrap the inherited result in an Index.
    """

class ExtensionIndex(Index):
    """
    Index subclass for indexes backed by ExtensionArray.
    """
    _data: IntervalArray | NDArrayBackedExtensionArray
    def _validate_fill_value(self, value):
        """
        Convert value to be insertable to underlying array.
        """
    def _isnan(self) -> npt.NDArray[np.bool_]: ...

class NDArrayBackedExtensionIndex(ExtensionIndex):
    """
    Index subclass for indexes backed by NDArrayBackedExtensionArray.
    """
    _data: NDArrayBackedExtensionArray
    def _get_engine_target(self) -> np.ndarray: ...
    def _from_join_target(self, result: np.ndarray) -> ArrayLike: ...
