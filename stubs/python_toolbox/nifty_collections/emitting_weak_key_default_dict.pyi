from .weak_key_default_dict import WeakKeyDefaultDict as WeakKeyDefaultDict
from _typeshed import Incomplete
from typing import Any

class EmittingWeakKeyDefaultDict(WeakKeyDefaultDict):
    r"""
    A key that references keys weakly, has a default factory, and emits.

    This is a combination of `weakref.WeakKeyDictionary` and
    `collections.defaultdict`, which emits every time it\'s modified.

    The keys are referenced weakly, so if there are no more references to the
    key, it gets removed from this dict.

    If a "default factory" is supplied, when a key is attempted that doesn\'t
    exist the default factory will be called to create its new value.

    Every time that a change is made, like a key is added or removed or gets
    its value changed, we do `.emitter.emit()`.
    """

    emitter: Incomplete
    def __init__(self, emitter: Any, *args: Any, **kwargs: Any) -> None: ...
    def set_emitter(self, emitter: Any) -> None:
        """Set the emitter that will be emitted every time a change is made."""
    def __setitem__(self, key: Any, value: Any) -> None: ...
    def __delitem__(self, key: Any) -> None: ...
    def pop(self, key: Any, *args: Any) -> Any:
        """D.pop(k[,d]) -> v, remove specified key and return the
        corresponding value. If key is not found, d is returned if given,
        otherwise KeyError is raised.
        """
    def popitem(self) -> Any:
        """D.popitem() -> (k, v), remove and return some (key, value)
        pair as a 2-tuple; but raise KeyError if D is empty.
        """
    def clear(self) -> Any:
        """D.clear() -> None.  Remove all items from D."""
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



