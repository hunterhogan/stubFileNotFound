from ..exceptions import ConnectionError as ConnectionError, InvalidResponse as InvalidResponse, RedisError as RedisError
from ..typing import EncodableT as EncodableT
from ..utils import HIREDIS_AVAILABLE as HIREDIS_AVAILABLE
from .base import AsyncBaseParser as AsyncBaseParser, AsyncPushNotificationsParser as AsyncPushNotificationsParser, BaseParser as BaseParser, PushNotificationsParser as PushNotificationsParser
from .socket import NONBLOCKING_EXCEPTIONS as NONBLOCKING_EXCEPTIONS, NONBLOCKING_EXCEPTION_ERROR_NUMBERS as NONBLOCKING_EXCEPTION_ERROR_NUMBERS, SENTINEL as SENTINEL, SERVER_CLOSED_CONNECTION_ERROR as SERVER_CLOSED_CONNECTION_ERROR
from _typeshed import Incomplete
from typing import Callable, TypedDict

NOT_ENOUGH_DATA: Incomplete

class _HiredisReaderArgs(TypedDict, total=False):
    protocolError: Callable[[str], Exception]
    replyError: Callable[[str], Exception]
    encoding: str | None
    errors: str | None

class _HiredisParser(BaseParser, PushNotificationsParser):
    """Parser class for connections using Hiredis"""
    socket_read_size: Incomplete
    _buffer: Incomplete
    pubsub_push_handler_func: Incomplete
    invalidation_push_handler_func: Incomplete
    _hiredis_PushNotificationType: Incomplete
    def __init__(self, socket_read_size) -> None: ...
    def __del__(self) -> None: ...
    def handle_pubsub_push_response(self, response): ...
    _sock: Incomplete
    _socket_timeout: Incomplete
    _reader: Incomplete
    _next_response: Incomplete
    def on_connect(self, connection, **kwargs) -> None: ...
    def on_disconnect(self) -> None: ...
    def can_read(self, timeout): ...
    def read_from_socket(self, timeout=..., raise_on_timeout: bool = True): ...
    def read_response(self, disable_decoding: bool = False, push_request: bool = False): ...

class _AsyncHiredisParser(AsyncBaseParser, AsyncPushNotificationsParser):
    """Async implementation of parser class for connections using Hiredis"""
    __slots__: Incomplete
    _reader: Incomplete
    pubsub_push_handler_func: Incomplete
    invalidation_push_handler_func: Incomplete
    _hiredis_PushNotificationType: Incomplete
    def __init__(self, socket_read_size: int) -> None: ...
    async def handle_pubsub_push_response(self, response): ...
    _stream: Incomplete
    _connected: bool
    def on_connect(self, connection) -> None: ...
    def on_disconnect(self) -> None: ...
    async def can_read_destructive(self): ...
    async def read_from_socket(self): ...
    async def read_response(self, disable_decoding: bool = False, push_request: bool = False) -> EncodableT | list[EncodableT]: ...
