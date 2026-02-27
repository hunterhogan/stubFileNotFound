
from typing import Tuple, TypeAlias, Union

__all__ = ["__commit_id__", "__version__", "__version_tuple__", "commit_id", "version", "version_tuple"]
TYPE_CHECKING = ...
if TYPE_CHECKING:
    VERSION_TUPLE: TypeAlias = tuple[int | str, ...]
    COMMIT_ID: TypeAlias = str | None
else:
    ...
version: str
__version__: str
__version_tuple__: VERSION_TUPLE
version_tuple: VERSION_TUPLE
commit_id: COMMIT_ID
__commit_id__: COMMIT_ID
version = ...
version_tuple = ...
commit_id = ...
