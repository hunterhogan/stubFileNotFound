from ._util import to_string as to_string
from _typeshed import Incomplete

class Suggestion:
    """
    Represents a single suggestion being sent or returned from the
    autocomplete server
    """
    string: Incomplete
    payload: Incomplete
    score: Incomplete
    def __init__(self, string: str, score: float = 1.0, payload: str | None = None) -> None: ...
    def __repr__(self) -> str: ...

class SuggestionParser:
    """
    Internal class used to parse results from the `SUGGET` command.
    This needs to consume either 1, 2, or 3 values at a time from
    the return value depending on what objects were requested
    """
    with_scores: Incomplete
    with_payloads: Incomplete
    sugsize: int
    _scoreidx: int
    _payloadidx: int
    _sugs: Incomplete
    def __init__(self, with_scores: bool, with_payloads: bool, ret) -> None: ...
    def __iter__(self): ...
