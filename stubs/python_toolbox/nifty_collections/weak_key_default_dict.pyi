from _typeshed import Incomplete
from collections.abc import Generator
from typing import Any
import collections

class WeakKeyDefaultDict(collections.abc.MutableMapping[Any, Any]):
    '''
    A weak key dictionary which can use a default factory.

    This is a combination of `weakref.WeakKeyDictionary` and
    `collections.defaultdict`.

    The keys are referenced weakly, so if there are no more references to the
    key, it gets removed from this dict.

    If a "default factory" is supplied, when a key is attempted that doesn\'t
    exist the default factory will be called to create its new value.
    '''
    default_factory: Incomplete
    data: Incomplete
    _remove: Incomplete
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Construct the `WeakKeyDefaultDict`.

        You may supply a `default_factory` as a keyword argument.
        """
    def __missing__(self, key: Any) -> Any:
        """Get a value for a key which isn't currently registered."""
    def __repr__(self, recurse: Any=...) -> str: ...
    def copy(self) -> Any: ...
    __copy__ = copy
    def __reduce__(self) -> Any:
        """
        __reduce__ must return a 5-tuple as follows:

           - factory function
           - tuple of args for the factory function
           - additional state (here None)
           - sequence iterator (here None)
           - dictionary iterator (yielding successive (key, value) pairs

           This API is used by pickle.py and copy.py.
        """
    def __delitem__(self, key: Any) -> None: ...
    def __getitem__(self, key: Any) -> Any: ...
    def __setitem__(self, key: Any, value: Any) -> None: ...
    def get(self, key: Any, default: Any=None) -> Any: ...
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
        """D.update(E, **F) -> None. Update D from E and F: for k in E: D[k] =
        E[k] (if E has keys else: for (k, v) in E: D[k] = v) then: for k in F:
        D[k] = F[k] """
    def __len__(self) -> int: ...



