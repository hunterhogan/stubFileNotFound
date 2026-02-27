
from typing import Tuple, TypeAlias, Union

__all__ = ["__version__", "__version_tuple__", "version", "version_tuple"]
TYPE_CHECKING = ...
if TYPE_CHECKING:
    VERSION_TUPLE: TypeAlias = tuple[int | str, ...]
else:
    ...
version: str
__version__: str
__version_tuple__: VERSION_TUPLE
version_tuple: VERSION_TUPLE
version = ...
version_tuple = ...
