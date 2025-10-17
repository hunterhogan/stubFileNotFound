from _typeshed import Incomplete
from typing import Any

class ChangeTracker:
    """
    Tracks changes in objects that are registered with it.

    To register an object, use `.check_in(obj)`. It will return `True`. Every
    time `.check_in` will be called with the same object, it will return
    whether the object changed since the last time it was checked in.
    """

    library: Incomplete
    def __init__(self) -> None: ...
    def check_in(self, thing: Any) -> Any:
        """
        Check in an object for change tracking.

        The first time you check in an object, it will return `True`. Every
        time `.check_in` will be called with the same object, it will return
        whether the object changed since the last time it was checked in.
        """
    def __contains__(self, thing: Any) -> bool:
        """Return whether `thing` is tracked."""



