import _abc
import collections
import typing
from _typeshed import Incomplete
from pandas._libs.tslibs.timestamps import Timestamp as Timestamp
from pandas.errors import UndefinedVariableError as UndefinedVariableError
from typing import ClassVar

_KT: typing.TypeVar
_VT: typing.TypeVar

class DeepChainMap(collections.ChainMap):
    __orig_bases__: ClassVar[tuple] = ...
    __abstractmethods__: ClassVar[frozenset] = ...
    _abc_impl: ClassVar[_abc._abc_data] = ...
    def __setitem__(self, key: _KT, value: _VT) -> None: ...
    def __delitem__(self, key: _KT) -> None:
        """
        Raises
        ------
        KeyError
            If `key` doesn't exist.
        """
def ensure_scope(level: int, global_dict, local_dict, resolvers: tuple = ..., target) -> Scope:
    """Ensure that we are grabbing the correct scope."""
def _replacer(x) -> str:
    """
    Replace a number with its hexadecimal representation. Used to tag
    temporary variables with their calling scope's id.
    """
def _raw_hex_id(obj) -> str:
    """Return the padded hexadecimal id of ``obj``."""

DEFAULT_GLOBALS: dict
def _get_pretty_string(obj) -> str:
    """
    Return a prettier version of obj.

    Parameters
    ----------
    obj : object
        Object to pretty print

    Returns
    -------
    str
        Pretty print object repr
    """

class Scope:
    level: Incomplete
    resolvers: Incomplete
    scope: Incomplete
    target: Incomplete
    temps: Incomplete
    def __init__(self, level: int, global_dict, local_dict, resolvers: tuple = ..., target) -> None: ...
    def resolve(self, key: str, is_local: bool):
        """
        Resolve a variable name in a possibly local context.

        Parameters
        ----------
        key : str
            A variable name
        is_local : bool
            Flag indicating whether the variable is local or not (prefixed with
            the '@' symbol)

        Returns
        -------
        value : object
            The value of a particular variable
        """
    def swapkey(self, old_key: str, new_key: str, new_value) -> None:
        """
        Replace a variable name, with a potentially new value.

        Parameters
        ----------
        old_key : str
            Current variable name to replace
        new_key : str
            New variable name to replace `old_key` with
        new_value : object
            Value to be replaced along with the possible renaming
        """
    def _get_vars(self, stack, scopes: list[str]) -> None:
        """
        Get specifically scoped variables from a list of stack frames.

        Parameters
        ----------
        stack : list
            A list of stack frames as returned by ``inspect.stack()``
        scopes : sequence of strings
            A sequence containing valid stack frame attribute names that
            evaluate to a dictionary. For example, ('locals', 'globals')
        """
    def _update(self, level: int) -> None:
        """
        Update the current scope by going back `level` levels.

        Parameters
        ----------
        level : int
        """
    def add_tmp(self, value) -> str:
        """
        Add a temporary variable to the scope.

        Parameters
        ----------
        value : object
            An arbitrary object to be assigned to a temporary variable.

        Returns
        -------
        str
            The name of the temporary variable created.
        """
    @property
    def has_resolvers(self): ...
    @property
    def ntemps(self): ...
    @property
    def full_scope(self): ...
