from _typeshed import Incomplete
from collections.abc import Generator

_default_config: Incomplete
_current_config: Incomplete

def using(**kwargs) -> Generator[None]: ...
def setup(key, value: Incomplete | None = None) -> None:
    """Assign a value to (or reset) a configuration item. """
def query(key):
    """Ask for a value of the given configuration item. """
def configure() -> None:
    """Initialized configuration of polys module. """
