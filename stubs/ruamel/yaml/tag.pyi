from _typeshed import Incomplete
from typing import Any

tag_attrib: str

class Tag:
    """store original tag information for roundtripping"""
    attrib = tag_attrib
    handle: Incomplete
    suffix: Incomplete
    handles: Incomplete
    _transform_type: bool | None
    def __init__(self, handle: Any = None, suffix: Any = None, handles: Any = None) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    _hash_id: Incomplete
    def __hash__(self) -> int: ...
    def __eq__(self, other: Any) -> bool: ...
    def startswith(self, x: str) -> bool: ...
    _trval: str | None
    @property
    def trval(self) -> str | None: ...
    value = trval
    _uri_decoded_suffix: str | None
    @property
    def uri_decoded_suffix(self) -> str | None: ...
    def select_transform(self, val: bool) -> None:
        """
        val: False -> non-round-trip
             True -> round-trip
        """
    def check_handle(self) -> bool: ...
