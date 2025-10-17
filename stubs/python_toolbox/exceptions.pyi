from _typeshed import Incomplete
from typing import Any

class CuteBaseException(BaseException):
    """Base exception that uses its first docstring line in lieu of a message."""

    message: Incomplete
    def __init__(self, message: Any=None) -> None: ...

class CuteException(CuteBaseException, Exception):
    """Exception that uses its first docstring line in lieu of a message."""



