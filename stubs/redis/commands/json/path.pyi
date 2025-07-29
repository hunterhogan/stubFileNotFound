from typing import Any
class Path:
    """This class represents a path in a JSON value."""
    strPath: str
    @staticmethod
    def root_path() -> Any:
        """Return the root path's string representation."""
    def __init__(self, path: Any) -> None:
        """Make a new path based on the string representation in `path`."""
    def __repr__(self) -> str: ...
