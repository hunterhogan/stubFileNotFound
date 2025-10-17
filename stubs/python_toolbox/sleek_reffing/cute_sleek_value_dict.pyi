from .sleek_ref import SleekRef
from _typeshed import Incomplete
from collections.abc import Generator
from typing import Any
import collections

__all__ = ['CuteSleekValueDict']

class CuteSleekValueDict(collections.UserDict):
    """
    A dictionary which sleekrefs its values and propagates their callback.

    When a value is garbage-collected, it (1) removes itself from this dict and
    (2) calls the dict's own `callback` function.

    This class is like `weakref.WeakValueDictionary`, except (a) it uses
    sleekrefs instead of weakrefs and (b) when a value dies, it calls a
    callback.

    See documentation of `python_toolbox.sleek_reffing.SleekRef` for more
    details about sleekreffing.
    """

    callback: Incomplete
    _remove: Incomplete
    def __init__(self, callback: Any, *args: Any, **kwargs: Any) -> None: ...
    def __getitem__(self, key: Any) -> Any: ...
    def __contains__(self, key: Any) -> bool: ...
    def __eq__(self, other: object) -> Any: ...
    def __ne__(self, other: object) -> Any: ...
    has_key = __contains__
    def __setitem__(self, key: Any, value: Any) -> None: ...
    def copy(self) -> Any:
        """Shallow copy the `CuteSleekValueDict`."""
    __copy__ = copy
    def get(self, key: Any, default: Any=None) -> Any:
        """D.get(k[,d]) -> D[k] if k in D, else d.  d defaults to None."""
    def items(self) -> Any:
        """D.items() -> list of D's (key, value) pairs, as 2-tuples."""
    def iteritems(self) -> Generator[Incomplete]:
        """D.iteritems() -> an iterator over the (key, value) items of D."""
    def iterkeys(self) -> Any:
        """D.iterkeys() -> an iterator over the keys of D."""
    def __iter__(self) -> Any: ...
    def itervaluerefs(self) -> Any:
        """Return an iterator that yields the weak references to the values.

        The references are not guaranteed to be 'live' at the time
        they are used, so the result of calling the references needs
        to be checked before being used.  This can be used to avoid
        creating references that will cause the garbage collector to
        keep the values around longer than needed.

        """
    def itervalues(self) -> Generator[Incomplete]:
        """D.itervalues() -> an iterator over the values of D."""
    def popitem(self) -> Any:
        """D.popitem() -> (k, v), remove and return some (key, value) pair
        as a 2-tuple; but raise KeyError if D is empty.
        """
    def pop(self, key: Any, *args: Any) -> Any:
        """D.pop(k[,d]) -> v, remove specified key and return the
        corresponding value. If key is not found, d is returned if given,
        otherwise KeyError is raised.
        """
    def setdefault(self, key: Any, default: Any=None) -> Any:
        """D.setdefault(k[,d]) -> D.get(k,d), also set D[k]=d if k not in D."""
    def update(self, *other_dicts: Any, **kwargs: Any) -> None:
        """D.update(E, **F) -> None. Update D from E and F: for k in E: D[k] =
        E[k] (if E has keys else: for (k, v) in E: D[k] = v) then: for k in F:
        D[k] = F[k].
        """
    def valuerefs(self) -> Any:
        """Return a list of weak references to the values.

        The references are not guaranteed to be 'live' at the time
        they are used, so the result of calling the references needs
        to be checked before being used.  This can be used to avoid
        creating references that will cause the garbage collector to
        keep the values around longer than needed.

        """
    def values(self) -> Any:
        """D.values() -> list of D's values."""
    @classmethod
    def fromkeys(cls, iterable: Any, value: Any=None, callback: Any=...) -> Any:
        """dict.fromkeys(S[,v]) -> New csvdict with keys from S and values
        equal to v. v defaults to None.
        """

class KeyedSleekRef(SleekRef):
    """Sleekref whose weakref (if one exists) holds reference to a key."""

    def __new__(cls, thing: Any, callback: Any, key: Any) -> Any: ...
    def __init__(self, thing: Any, callback: Any, key: Any) -> None: ...



