from _typeshed import Incomplete
from collections import ChainMap
from pandas._libs.tslibs import Timestamp as Timestamp
from pandas.errors import UndefinedVariableError as UndefinedVariableError
from typing import TypeVar

_KT = TypeVar('_KT')
_VT = TypeVar('_VT')

class DeepChainMap(ChainMap[_KT, _VT]):
    """
    Variant of ChainMap that allows direct updates to inner scopes.

    Only works when all passed mapping are mutable.
    """
    def __setitem__(self, key: _KT, value: _VT) -> None: ...
    def __delitem__(self, key: _KT) -> None:
        """
        Raises
        ------
        KeyError
            If `key` doesn't exist.
        """

def ensure_scope(level: int, global_dict: Incomplete | None = None, local_dict: Incomplete | None = None, resolvers=(), target: Incomplete | None = None) -> Scope:
    """Ensure that we are grabbing the correct scope."""
def _replacer(x) -> str:
    """
    Replace a number with its hexadecimal representation. Used to tag
    temporary variables with their calling scope's id.
    """
def _raw_hex_id(obj) -> str:
    """Return the padded hexadecimal id of ``obj``."""

DEFAULT_GLOBALS: Incomplete

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
    """
    Object to hold scope, with a few bells to deal with some custom syntax
    and contexts added by pandas.

    Parameters
    ----------
    level : int
    global_dict : dict or None, optional, default None
    local_dict : dict or Scope or None, optional, default None
    resolvers : list-like or None, optional, default None
    target : object

    Attributes
    ----------
    level : int
    scope : DeepChainMap
    target : object
    temps : dict
    """
    __slots__: Incomplete
    level: int
    scope: DeepChainMap
    resolvers: DeepChainMap
    temps: dict
    target: Incomplete
    def __init__(self, level: int, global_dict: Incomplete | None = None, local_dict: Incomplete | None = None, resolvers=(), target: Incomplete | None = None) -> None: ...
    def __repr__(self) -> str: ...
    @property
    def has_resolvers(self) -> bool:
        """
        Return whether we have any extra scope.

        For example, DataFrames pass Their columns as resolvers during calls to
        ``DataFrame.eval()`` and ``DataFrame.query()``.

        Returns
        -------
        hr : bool
        """
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
    def swapkey(self, old_key: str, new_key: str, new_value: Incomplete | None = None) -> None:
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
    def ntemps(self) -> int:
        """The number of temporary variables in this scope"""
    @property
    def full_scope(self) -> DeepChainMap:
        """
        Return the full scope for use with passing to engines transparently
        as a mapping.

        Returns
        -------
        vars : DeepChainMap
            All variables in this scope.
        """
