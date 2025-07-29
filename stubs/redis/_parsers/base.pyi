from ..exceptions import AskError as AskError, AuthenticationError as AuthenticationError, AuthenticationWrongNumberOfArgsError as AuthenticationWrongNumberOfArgsError, BusyLoadingError as BusyLoadingError, ClusterCrossSlotError as ClusterCrossSlotError, ClusterDownError as ClusterDownError, ConnectionError as ConnectionError, ExecAbortError as ExecAbortError, MasterDownError as MasterDownError, ModuleError as ModuleError, MovedError as MovedError, NoPermissionError as NoPermissionError, NoScriptError as NoScriptError, OutOfMemoryError as OutOfMemoryError, ReadOnlyError as ReadOnlyError, RedisError as RedisError, ResponseError as ResponseError, TryAgainError as TryAgainError
from ..typing import EncodableT as EncodableT
from .encoders import Encoder as Encoder
from .socket import SERVER_CLOSED_CONNECTION_ERROR as SERVER_CLOSED_CONNECTION_ERROR, SocketBuffer as SocketBuffer
from _typeshed import Incomplete
from abc import ABC
from asyncio import StreamReader
from typing import Callable, Protocol
from typing import Any

MODULE_LOAD_ERROR: str
NO_SUCH_MODULE_ERROR: str
MODULE_UNLOAD_NOT_POSSIBLE_ERROR: str
MODULE_EXPORTS_DATA_TYPES_ERROR: str
NO_AUTH_SET_ERROR: Incomplete

class BaseParser(ABC):
    EXCEPTION_CLASSES: Incomplete
    @classmethod
    def parse_error(cls, response: Any) -> Any:
        """Parse an error response"""
    def on_disconnect(self) -> None: ...
    def on_connect(self, connection: Any) -> None: ...

class _RESPBase(BaseParser):
    """Base class for sync-based resp parsing"""
    socket_read_size: Incomplete
    encoder: Incomplete
    _sock: Incomplete
    _buffer: Incomplete
    def __init__(self, socket_read_size: Any) -> None: ...
    def __del__(self) -> None: ...
    def on_connect(self, connection: Any) -> None:
        """Called when the socket connects"""
    def on_disconnect(self) -> None:
        """Called when the socket disconnects"""
    def can_read(self, timeout: Any) -> Any: ...

class AsyncBaseParser(BaseParser):
    """Base parsing class for the python-backed async parser"""
    __slots__: Incomplete
    _stream: StreamReader | None
    _read_size: Incomplete
    def __init__(self, socket_read_size: int) -> None: ...
    async def can_read_destructive(self) -> bool: ...
    async def read_response(self, disable_decoding: bool = False) -> EncodableT | ResponseError | None | list[EncodableT]: ...

_INVALIDATION_MESSAGE: Incomplete

class PushNotificationsParser(Protocol):
    """Protocol defining RESP3-specific parsing functionality"""
    pubsub_push_handler_func: Callable[..., Any]
    invalidation_push_handler_func: Callable[..., Any] | None
    def handle_pubsub_push_response(self, response: Any) -> Any:
        """Handle pubsub push responses"""
    def handle_push_response(self, response: Any, **kwargs: Any) -> Any: ...
    def set_pubsub_push_handler(self, pubsub_push_handler_func: Any) -> None: ...
    def set_invalidation_push_handler(self, invalidation_push_handler_func: Any) -> None: ...

class AsyncPushNotificationsParser(Protocol):
    """Protocol defining async RESP3-specific parsing functionality"""
    pubsub_push_handler_func: Callable[..., Any]
    invalidation_push_handler_func: Callable[..., Any] | None
    async def handle_pubsub_push_response(self, response: Any) -> Any:
        """Handle pubsub push responses asynchronously"""
    async def handle_push_response(self, response: Any, **kwargs: Any) -> Any:
        """Handle push responses asynchronously"""
    def set_pubsub_push_handler(self, pubsub_push_handler_func: Any) -> None:
        """Set the pubsub push handler function"""
    def set_invalidation_push_handler(self, invalidation_push_handler_func: Any) -> None:
        """Set the invalidation push handler function"""

class _AsyncRESPBase(AsyncBaseParser):
    """Base class for async resp parsing"""
    __slots__: Incomplete
    encoder: Encoder | None
    _buffer: bytes
    _chunks: Incomplete
    _pos: int
    def __init__(self, socket_read_size: int) -> None: ...
    def _clear(self) -> None: ...
    _stream: Incomplete
    _connected: bool
    def on_connect(self, connection: Any) -> None:
        """Called when the stream connects"""
    def on_disconnect(self) -> None:
        """Called when the stream disconnects"""
    async def can_read_destructive(self) -> bool: ...
    async def _read(self, length: int) -> bytes:
        """
        Read `length` bytes of data.  These are assumed to be followed
        by a '\r
' terminator which is subsequently discarded.
        """
    async def _readline(self) -> bytes:
        """
        read an unknown number of bytes up to the next '\r
'
        line separator, which is discarded.
        """
