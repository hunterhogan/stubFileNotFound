from _typeshed import Incomplete
from typing import Any

class Document:
    """
    Represents a single document in a result set
    """
    id: Incomplete
    payload: Incomplete
    def __init__(self, id: Any, payload: Incomplete | None = None, **fields: Any) -> None: ...
    def __repr__(self) -> str: ...
    def __getitem__(self, item: Any) -> Any: ...
