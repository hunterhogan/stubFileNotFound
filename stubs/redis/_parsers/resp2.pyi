from ..exceptions import ConnectionError as ConnectionError, InvalidResponse as InvalidResponse, ResponseError as ResponseError
from ..typing import EncodableT as EncodableT
from .base import _AsyncRESPBase as _AsyncRESPBase, _RESPBase as _RESPBase
from .socket import SERVER_CLOSED_CONNECTION_ERROR as SERVER_CLOSED_CONNECTION_ERROR
from typing import Any

class _RESP2Parser(_RESPBase):
    """RESP2 protocol implementation"""
    def read_response(self, disable_decoding: bool = False) -> Any: ...
    def _read_response(self, disable_decoding: bool = False) -> Any: ...

class _AsyncRESP2Parser(_AsyncRESPBase):
    """Async class for the RESP2 protocol"""
    _pos: int
    async def read_response(self, disable_decoding: bool = False) -> Any: ...
    async def _read_response(self, disable_decoding: bool = False) -> EncodableT | ResponseError | None: ...
