from ..helpers import nativestr as nativestr
from _typeshed import Incomplete
from typing import Any

class BFInfo:
    capacity: Incomplete
    size: Incomplete
    filterNum: Incomplete
    insertedNum: Incomplete
    expansionRate: Incomplete
    def __init__(self, args: Any) -> None: ...
    def get(self, item: Any) -> Any: ...
    def __getitem__(self, item: Any) -> Any: ...

class CFInfo:
    size: Incomplete
    bucketNum: Incomplete
    filterNum: Incomplete
    insertedNum: Incomplete
    deletedNum: Incomplete
    bucketSize: Incomplete
    expansionRate: Incomplete
    maxIteration: Incomplete
    def __init__(self, args: Any) -> None: ...
    def get(self, item: Any) -> Any: ...
    def __getitem__(self, item: Any) -> Any: ...

class CMSInfo:
    width: Incomplete
    depth: Incomplete
    count: Incomplete
    def __init__(self, args: Any) -> None: ...
    def __getitem__(self, item: Any) -> Any: ...

class TopKInfo:
    k: Incomplete
    width: Incomplete
    depth: Incomplete
    decay: Incomplete
    def __init__(self, args: Any) -> None: ...
    def __getitem__(self, item: Any) -> Any: ...

class TDigestInfo:
    compression: Incomplete
    capacity: Incomplete
    merged_nodes: Incomplete
    unmerged_nodes: Incomplete
    merged_weight: Incomplete
    unmerged_weight: Incomplete
    total_compressions: Incomplete
    memory_usage: Incomplete
    def __init__(self, args: Any) -> None: ...
    def get(self, item: Any) -> Any: ...
    def __getitem__(self, item: Any) -> Any: ...
