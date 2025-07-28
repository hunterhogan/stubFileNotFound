from _typeshed import Incomplete

class Document:
    """
    Represents a single document in a result set
    """
    id: Incomplete
    payload: Incomplete
    def __init__(self, id, payload: Incomplete | None = None, **fields) -> None: ...
    def __repr__(self) -> str: ...
    def __getitem__(self, item): ...
