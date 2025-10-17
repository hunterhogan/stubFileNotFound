from .freezer import Freezer as Freezer
from _typeshed import Incomplete
from typing import Any

class FreezerPropertyFreezer(Freezer):
    """
    Freezer used internally by `FreezerProperty`.

    It uses the `FreezerProperty`'s internal freeze/thaw handlers as its own
    freeze/thaw handlers.
    """

    thing: Incomplete
    def __init__(self, thing: Any) -> None:
        """
        Construct the `FreezerPropertyFreezer`.

        `thing` is the object to whom the `FreezerProperty` belongs.
        """
    def freeze_handler(self) -> Any:
        """Call the `FreezerProperty`'s internal freeze handler."""
    def thaw_handler(self) -> Any:
        """Call the `FreezerProperty`'s internal thaw handler."""



