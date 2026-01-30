import codecs
from _typeshed import Incomplete

class ExtendCodec(codecs.Codec):
    name: Incomplete
    base_encoding: Incomplete
    mapping: Incomplete
    reverse: Incomplete
    max_len: Incomplete
    info: Incomplete
    def __init__(self, name, base_encoding, mapping) -> None: ...
    def _map(self, mapper, output_type, exc_type, input, errors): ...
    def encode(self, input, errors: str = 'strict'): ...
    def decode(self, input, errors: str = 'strict'): ...
    def error(self, e): ...

_extended_encodings: Incomplete
_cache: Incomplete

def search_function(name): ...
