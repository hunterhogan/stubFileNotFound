from _typeshed import Incomplete
from typing import Any

class ReasonedBool:
    r"""
    A variation on `bool` that also gives a `.reason`.

    This is useful when you want to say "This is False because... (reason.)"

    Unfortunately this class is not a subclass of `bool`, since Python doesn\'t
    allow subclassing `bool`.
    """

    value: Incomplete
    reason: Incomplete
    def __init__(self, value: Any, reason: Any=None) -> None:
        """
        Construct the `ReasonedBool`.

        `reason` is the reason *why* it has a value of `True` or `False`. It is
        usually a string, but is allowed to be of any type.
        """
    def __eq__(self, other: object) -> Any: ...
    def __hash__(self) -> Any: ...
    def __neq__(self, other: Any) -> Any: ...
    def __bool__(self) -> bool: ...



