from ..exceptions import DataError as DataError
from _typeshed import Incomplete
from typing import Any

class Encoder:
    """Encode strings to bytes-like and decode bytes-like to strings"""
    __slots__: Incomplete
    encoding: Incomplete
    encoding_errors: Incomplete
    decode_responses: Incomplete
    def __init__(self, encoding: Any, encoding_errors: Any, decode_responses: Any) -> None: ...
    def encode(self, value: Any) -> Any:
        """Return a bytestring or bytes-like representation of the value"""
    def decode(self, value: Any, force: bool = False) -> Any:
        """Return a unicode string from the bytes-like representation"""
