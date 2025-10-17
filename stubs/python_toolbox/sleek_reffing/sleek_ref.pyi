from _typeshed import Incomplete
from typing import Any
import weakref

__all__ = ['SleekRef']

class Ref(weakref.ref):
    """
    A weakref.

    What this adds over `weakref.ref` is the ability to add custom attributes.
    """

class SleekRef:
    """
    Sleekref tries to reference an object weakly but if can't does it strongly.

    The problem with weakrefs is that some objects can't be weakreffed, for
    example `list` and `dict` objects. A sleekref tries to create a weakref to
    an object, but if it can't (like for a `list`) it creates a strong one
    instead.

    Thanks to sleekreffing you can avoid memory leaks when manipulating
    weakreffable object, but if you ever want to use non-weakreffable objects
    you are still able to. (Assuming you don't mind the memory leaks or stop
    them some other way.)

    When you call a dead sleekref, it doesn't return `None` like weakref; it
    raises `SleekRefDied`. Therefore, unlike weakref, you can store `None` in a
    sleekref.
    """

    callback: Incomplete
    is_none: Incomplete
    ref: Incomplete
    thing: Incomplete
    def __init__(self, thing: Any, callback: Any=None) -> None:
        """
        Construct the sleekref.

        `thing` is the object we want to sleekref. `callback` is the callable
        to call when the weakref to the object dies. (Only relevant for
        weakreffable objects.)
        """
    def __call__(self) -> Any:
        """Obtain the sleekreffed object. Raises `SleekRefDied` if reference died."""



