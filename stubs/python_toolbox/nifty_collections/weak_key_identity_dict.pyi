from _typeshed import Incomplete
from collections.abc import Generator
from typing import Any
import collections
import weakref

__all__ = ['WeakKeyIdentityDict']

class IdentityRef(weakref.ref):
    """A weak reference to an object, hashed by identity and not contents."""
    _hash: Incomplete
    def __init__(self, thing: Any, callback: Any=None) -> None: ...
    def __hash__(self) -> Any: ...

class WeakKeyIdentityDict(collections.abc.MutableMapping[Any, Any]):
    """
    A weak key dictionary which cares about the keys' identities.

    This is a fork of `weakref.WeakKeyDictionary`. Like in the original
    `WeakKeyDictionary`, the keys are referenced weakly, so if there are no
    more references to the key, it gets removed from this dict.

    The difference is that `WeakKeyIdentityDict` cares about the keys'
    identities and not their contents, so even unhashable objects like lists
    can be used as keys. The value will be tied to the object's identity and
    not its contents.
    """
    data: Incomplete
    _remove: Incomplete
    def __init__(self, dict_: Any=None) -> None: ...
    def __delitem__(self, key: Any) -> None: ...
    def __getitem__(self, key: Any) -> Any: ...
    def __repr__(self) -> str: ...
    def __setitem__(self, key: Any, value: Any) -> None: ...
    def copy(self) -> Any:
        """ D.copy() -> a shallow copy of D """
    def get(self, key: Any, default: Any=None) -> Any:
        """ D.get(k[,d]) -> D[k] if k in D, else d.  d defaults to None. """
    def __contains__(self, key: Any) -> bool: ...
    has_key = __contains__
    def items(self) -> Any:
        """ D.items() -> list of D's (key, value) pairs, as 2-tuples """
    def iteritems(self) -> Generator[Incomplete]:
        """ D.iteritems() -> an iterator over the (key, value) items of D """
    def iterkeyrefs(self) -> Any:
        """Return an iterator that yields the weak references to the keys.

        The references are not guaranteed to be 'live' at the time
        they are used, so the result of calling the references needs
        to be checked before being used.  This can be used to avoid
        creating references that will cause the garbage collector to
        keep the keys around longer than needed.

        """
    def iterkeys(self) -> Generator[Incomplete]:
        """ D.iterkeys() -> an iterator over the keys of D """
    def __iter__(self) -> Any: ...
    def itervalues(self) -> Any:
        """ D.itervalues() -> an iterator over the values of D """
    def keyrefs(self) -> Any:
        """Return a list of weak references to the keys.

        The references are not guaranteed to be 'live' at the time
        they are used, so the result of calling the references needs
        to be checked before being used.  This can be used to avoid
        creating references that will cause the garbage collector to
        keep the keys around longer than needed.

        """
    def keys(self) -> Any:
        """ D.keys() -> list of D's keys """
    def popitem(self) -> Any:
        """ D.popitem() -> (k, v), remove and return some (key, value) pair
        as a 2-tuple; but raise KeyError if D is empty """
    def pop(self, key: Any, *args: Any) -> Any:
        """ D.pop(k[,d]) -> v, remove specified key and return the
        corresponding value. If key is not found, d is returned if given,
        otherwise KeyError is raised """
    def setdefault(self, key: Any, default: Any=None) -> Any:
        """D.setdefault(k[,d]) -> D.get(k,d), also set D[k]=d if k not in D"""
    def update(self, dict: Any=None, **kwargs: Any) -> None:
        """ D.update(E, **F) -> None. Update D from E and F: for k in E: D[k] =
        E[k] (if E has keys else: for (k, v) in E: D[k] = v) then: for k in F:
        D[k] = F[k] """
    def __len__(self) -> int: ...



