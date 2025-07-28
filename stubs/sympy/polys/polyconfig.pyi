from _typeshed import Incomplete
from collections.abc import Generator
from contextlib import contextmanager

_default_config: Incomplete
_current_config: Incomplete

@contextmanager
def using(**kwargs) -> Generator[None]: ...
def setup(key, value=None) -> None:
    """Assign a value to (or reset) a configuration item. """
def query(key):
    """Ask for a value of the given configuration item. """
def configure() -> None:
    """Initialized configuration of polys module. """
