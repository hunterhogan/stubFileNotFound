from _typeshed import Incomplete
from ruamel.yaml.tag import Tag as Tag
from typing import Any

class Node:
    __slots__: Incomplete
    ctag: Incomplete
    value: Incomplete
    start_mark: Incomplete
    end_mark: Incomplete
    comment: Incomplete
    anchor: Incomplete
    def __init__(self, tag: Any, value: Any, start_mark: Any, end_mark: Any, comment: Any = None, anchor: Any = None) -> None: ...
    @property
    def tag(self) -> str | None: ...
    @tag.setter
    def tag(self, val: Any) -> None: ...
    def __repr__(self) -> Any: ...
    def dump(self, indent: int = 0) -> None: ...

class ScalarNode(Node):
    '''
    styles:
      ? -> set() ? key, no value
      - -> suppressable null value in set
      " -> double quoted
      \' -> single quoted
      | -> literal style
      > -> folding style
    '''
    __slots__: Incomplete
    id: str
    style: Incomplete
    def __init__(self, tag: Any, value: Any, start_mark: Any = None, end_mark: Any = None, style: Any = None, comment: Any = None, anchor: Any = None) -> None: ...

class CollectionNode(Node):
    __slots__: Incomplete
    flow_style: Incomplete
    anchor: Incomplete
    def __init__(self, tag: Any, value: Any, start_mark: Any = None, end_mark: Any = None, flow_style: Any = None, comment: Any = None, anchor: Any = None) -> None: ...

class SequenceNode(CollectionNode):
    __slots__: Incomplete
    id: str

class MappingNode(CollectionNode):
    __slots__: Incomplete
    id: str
    merge: Incomplete
    def __init__(self, tag: Any, value: Any, start_mark: Any = None, end_mark: Any = None, flow_style: Any = None, comment: Any = None, anchor: Any = None) -> None: ...
