from _typeshed import Incomplete
from typing import Any

class VersionInfo(tuple[Any, ...]):
    """
    Version number. This is a variation on a `namedtuple`.

    Example:

        VersionInfo(1, 2, 0) ==             VersionInfo(major=1, minor=2, micro=0, modifier='release') ==             (1, 2, 0)
    """

    __slots__: Incomplete
    _fields: Incomplete
    def __new__(cls, major: Any, minor: int = 0, micro: int = 0, modifier: str = 'release') -> Any:
        """Create new instance of `VersionInfo(major, minor, micro, modifier)`."""
    def _asdict(self) -> Any:
        """Return a new `OrderedDict` which maps field names to their values."""
    def __getnewargs__(self) -> Any:
        """Return self as a plain tuple. Used by copy and pickle."""
    @property
    def version_text(self) -> Any:
        """A textual description of the version, like '1.4.2 beta'."""
    major: Incomplete
    minor: Incomplete
    micro: Incomplete
    modifier: Incomplete



