from _typeshed import Incomplete
from enum import Enum
from typing import Any

class IndexType(Enum):
    """Enum of the currently supported index types."""
    HASH = 1
    JSON = 2

class IndexDefinition:
    """IndexDefinition is used to define a index definition for automatic
    indexing on Hash or Json update."""
    args: Incomplete
    def __init__(self, prefix: Any=[], filter: Incomplete | None = None, language_field: Incomplete | None = None, language: Incomplete | None = None, score_field: Incomplete | None = None, score: float = 1.0, payload_field: Incomplete | None = None, index_type: Incomplete | None = None) -> None: ...
    def _append_index_type(self, index_type: Any) -> None:
        """Append `ON HASH` or `ON JSON` according to the enum."""
    def _append_prefix(self, prefix: Any) -> None:
        """Append PREFIX."""
    def _append_filter(self, filter: Any) -> None:
        """Append FILTER."""
    def _append_language(self, language_field: Any, language: Any) -> None:
        """Append LANGUAGE_FIELD and LANGUAGE."""
    def _append_score(self, score_field: Any, score: Any) -> None:
        """Append SCORE_FIELD and SCORE."""
    def _append_payload(self, payload_field: Any) -> None:
        """Append PAYLOAD_FIELD."""
