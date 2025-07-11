from typing import (
    IO,
    AnyStr,
    Generic,
)

from pandas._typing import CompressionDict
from typing import Any

class IOHandles(Generic[AnyStr]):
    handle: IO[AnyStr]
    compression: CompressionDict
    created_handles: list[IO[AnyStr]]
    is_wrapped: bool
    def close(self) -> None: ...
    def __enter__(self) -> IOHandles[AnyStr]: ...
    def __exit__(self, *args: object) -> None: ...
    def __init__(self, handle: Any, compression: Any, created_handles: Any, is_wrapped: Any) -> None: ...
