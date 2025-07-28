from ..exceptions import DataError as DataError
from _typeshed import Incomplete

class Encoder:
    """Encode strings to bytes-like and decode bytes-like to strings"""
    __slots__: Incomplete
    encoding: Incomplete
    encoding_errors: Incomplete
    decode_responses: Incomplete
    def __init__(self, encoding, encoding_errors, decode_responses) -> None: ...
    def encode(self, value):
        """Return a bytestring or bytes-like representation of the value"""
    def decode(self, value, force: bool = False):
        """Return a unicode string from the bytes-like representation"""
