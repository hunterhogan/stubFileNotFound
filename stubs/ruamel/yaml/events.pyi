from _typeshed import Incomplete
from ruamel.yaml.tag import Tag as Tag
from typing import Any

SHOW_LINES: bool

def CommentCheck() -> None: ...

class Event:
    __slots__: Incomplete
    crepr: str
    start_mark: Incomplete
    end_mark: Incomplete
    comment: Incomplete
    def __init__(self, start_mark: Any = None, end_mark: Any = None, comment: Any = ...) -> None: ...
    def __repr__(self) -> Any: ...
    def compact_repr(self) -> str: ...

class NodeEvent(Event):
    __slots__: Incomplete
    anchor: Incomplete
    def __init__(self, anchor: Any, start_mark: Any = None, end_mark: Any = None, comment: Any = None) -> None: ...

class CollectionStartEvent(NodeEvent):
    __slots__: Incomplete
    ctag: Incomplete
    implicit: Incomplete
    flow_style: Incomplete
    nr_items: Incomplete
    def __init__(self, anchor: Any, tag: Any, implicit: Any, start_mark: Any = None, end_mark: Any = None, flow_style: Any = None, comment: Any = None, nr_items: int | None = None) -> None: ...
    @property
    def tag(self) -> str | None: ...

class CollectionEndEvent(Event):
    __slots__: Incomplete

class StreamStartEvent(Event):
    __slots__: Incomplete
    crepr: str
    encoding: Incomplete
    def __init__(self, start_mark: Any = None, end_mark: Any = None, encoding: Any = None, comment: Any = None) -> None: ...

class StreamEndEvent(Event):
    __slots__: Incomplete
    crepr: str

class DocumentStartEvent(Event):
    __slots__: Incomplete
    crepr: str
    explicit: Incomplete
    version: Incomplete
    tags: Incomplete
    def __init__(self, start_mark: Any = None, end_mark: Any = None, explicit: Any = None, version: Any = None, tags: Any = None, comment: Any = None) -> None: ...
    def compact_repr(self) -> str: ...

class DocumentEndEvent(Event):
    __slots__: Incomplete
    crepr: str
    explicit: Incomplete
    def __init__(self, start_mark: Any = None, end_mark: Any = None, explicit: Any = None, comment: Any = None) -> None: ...
    def compact_repr(self) -> str: ...

class AliasEvent(NodeEvent):
    __slots__: str
    crepr: str
    style: Incomplete
    def __init__(self, anchor: Any, start_mark: Any = None, end_mark: Any = None, style: Any = None, comment: Any = None) -> None: ...
    def compact_repr(self) -> str: ...

class ScalarEvent(NodeEvent):
    __slots__: Incomplete
    crepr: str
    ctag: Incomplete
    implicit: Incomplete
    value: Incomplete
    style: Incomplete
    def __init__(self, anchor: Any, tag: Any, implicit: Any, value: Any, start_mark: Any = None, end_mark: Any = None, style: Any = None, comment: Any = None) -> None: ...
    @property
    def tag(self) -> str | None: ...
    @tag.setter
    def tag(self, val: Any) -> None: ...
    def compact_repr(self) -> str: ...

class SequenceStartEvent(CollectionStartEvent):
    __slots__: Incomplete
    crepr: str
    def compact_repr(self) -> str: ...

class SequenceEndEvent(CollectionEndEvent):
    __slots__: Incomplete
    crepr: str

class MappingStartEvent(CollectionStartEvent):
    __slots__: Incomplete
    crepr: str
    def compact_repr(self) -> str: ...

class MappingEndEvent(CollectionEndEvent):
    __slots__: Incomplete
    crepr: str
