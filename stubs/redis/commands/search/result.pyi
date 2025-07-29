from ._util import to_string as to_string
from .document import Document as Document
from _typeshed import Incomplete
from typing import Any

class Result:
    """
    Represents the result of a search query, and has an array of Document
    objects
    """
    total: Incomplete
    duration: Incomplete
    docs: Incomplete
    def __init__(self, res: Any, hascontent: Any, duration: int = 0, has_payload: bool = False, with_scores: bool = False, field_encodings: dict[Any, Any] | None = None) -> None:
        """
        - duration: the execution time of the query
        - has_payload: whether the query has payloads
        - with_scores: whether the query has scores
        - field_encodings: a dictionary of field encodings if any is provided
        """
    def __repr__(self) -> str: ...
