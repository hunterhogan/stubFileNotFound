from . import base_events
from socket import socket
import selectors

__all__ = ("BaseSelectorEventLoop",)

class BaseSelectorEventLoop(base_events.BaseEventLoop):
    def __init__(self, selector: selectors.BaseSelector | None = None) -> None: ...
    async def sock_recv(self, sock: socket, n: int) -> bytes: ...
