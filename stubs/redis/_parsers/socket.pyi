import socket
from ..exceptions import ConnectionError as ConnectionError, TimeoutError as TimeoutError
from ..utils import SSL_AVAILABLE as SSL_AVAILABLE
from _typeshed import Incomplete

NONBLOCKING_EXCEPTION_ERROR_NUMBERS: Incomplete
NONBLOCKING_EXCEPTIONS: Incomplete
SERVER_CLOSED_CONNECTION_ERROR: str
SENTINEL: Incomplete
SYM_CRLF: bytes

class SocketBuffer:
    _sock: Incomplete
    socket_read_size: Incomplete
    socket_timeout: Incomplete
    _buffer: Incomplete
    def __init__(self, socket: socket.socket, socket_read_size: int, socket_timeout: float) -> None: ...
    def unread_bytes(self) -> int:
        """
        Remaining unread length of buffer
        """
    def _read_from_socket(self, length: int | None = None, timeout: float | object = ..., raise_on_timeout: bool | None = True) -> bool: ...
    def can_read(self, timeout: float) -> bool: ...
    def read(self, length: int) -> bytes: ...
    def readline(self) -> bytes: ...
    def get_pos(self) -> int:
        """
        Get current read position
        """
    def rewind(self, pos: int) -> None:
        """
        Rewind the buffer to a specific position, to re-start reading
        """
    def purge(self) -> None:
        """
        After a successful read, purge the read part of buffer
        """
    def close(self) -> None: ...
