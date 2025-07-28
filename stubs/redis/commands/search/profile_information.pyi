from typing import Any

class ProfileInformation:
    """
    Wrapper around FT.PROFILE response
    """
    _info: Any
    def __init__(self, info: Any) -> None: ...
    @property
    def info(self) -> Any: ...
