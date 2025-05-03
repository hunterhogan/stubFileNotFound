import np
import pandas.core.indexes.base
import typing
from _typeshed import Incomplete
from pandas._libs.properties import cache_readonly as cache_readonly
from pandas.core.dtypes.generic import ABCDataFrame as ABCDataFrame
from pandas.core.indexes.base import Index as Index
from typing import Callable

TYPE_CHECKING: bool
_ExtensionIndexT: typing.TypeVar
def _inherit_from_data(name: str, delegate: type, cache: bool = ..., wrap: bool = ...):
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
def inherit_names(names: list[str], delegate: type, cache: bool = ..., wrap: bool = ...) -> Callable[[type[_ExtensionIndexT]], type[_ExtensionIndexT]]:
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

class ExtensionIndex(pandas.core.indexes.base.Index):
    _isnan: Incomplete
    def _validate_fill_value(self, value):
        """
        Convert value to be insertable to underlying array.
        """

class NDArrayBackedExtensionIndex(ExtensionIndex):
    def _get_engine_target(self) -> np.ndarray: ...
    def _from_join_target(self, result: np.ndarray) -> ArrayLike: ...
