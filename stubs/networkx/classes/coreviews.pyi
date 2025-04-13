from _typeshed import Incomplete
from collections.abc import Mapping

__all__ = ['AtlasView', 'AdjacencyView', 'MultiAdjacencyView', 'UnionAtlas', 'UnionAdjacency', 'UnionMultiInner', 'UnionMultiAdjacency', 'FilterAtlas', 'FilterAdjacency', 'FilterMultiInner', 'FilterMultiAdjacency']

class AtlasView(Mapping):
    """An AtlasView is a Read-only Mapping of Mappings.

    It is a View into a dict-of-dict data structure.
    The inner level of dict is read-write. But the
    outer level is read-only.

    See Also
    ========
    AdjacencyView: View into dict-of-dict-of-dict
    MultiAdjacencyView: View into dict-of-dict-of-dict-of-dict
    """
    __slots__: Incomplete
    def __getstate__(self): ...
    _atlas: Incomplete
    def __setstate__(self, state) -> None: ...
    def __init__(self, d) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self): ...
    def __getitem__(self, key): ...
    def copy(self): ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class AdjacencyView(AtlasView):
    """An AdjacencyView is a Read-only Map of Maps of Maps.

    It is a View into a dict-of-dict-of-dict data structure.
    The inner level of dict is read-write. But the
    outer levels are read-only.

    See Also
    ========
    AtlasView: View into dict-of-dict
    MultiAdjacencyView: View into dict-of-dict-of-dict-of-dict
    """
    __slots__: Incomplete
    def __getitem__(self, name): ...
    def copy(self): ...

class MultiAdjacencyView(AdjacencyView):
    """An MultiAdjacencyView is a Read-only Map of Maps of Maps of Maps.

    It is a View into a dict-of-dict-of-dict-of-dict data structure.
    The inner level of dict is read-write. But the
    outer levels are read-only.

    See Also
    ========
    AtlasView: View into dict-of-dict
    AdjacencyView: View into dict-of-dict-of-dict
    """
    __slots__: Incomplete
    def __getitem__(self, name): ...
    def copy(self): ...

class UnionAtlas(Mapping):
    """A read-only union of two atlases (dict-of-dict).

    The two dict-of-dicts represent the inner dict of
    an Adjacency:  `G.succ[node]` and `G.pred[node]`.
    The inner level of dict of both hold attribute key:value
    pairs and is read-write. But the outer level is read-only.

    See Also
    ========
    UnionAdjacency: View into dict-of-dict-of-dict
    UnionMultiAdjacency: View into dict-of-dict-of-dict-of-dict
    """
    __slots__: Incomplete
    def __getstate__(self): ...
    _succ: Incomplete
    _pred: Incomplete
    def __setstate__(self, state) -> None: ...
    def __init__(self, succ, pred) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self): ...
    def __getitem__(self, key): ...
    def copy(self): ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class UnionAdjacency(Mapping):
    """A read-only union of dict Adjacencies as a Map of Maps of Maps.

    The two input dict-of-dict-of-dicts represent the union of
    `G.succ` and `G.pred`. Return values are UnionAtlas
    The inner level of dict is read-write. But the
    middle and outer levels are read-only.

    succ : a dict-of-dict-of-dict {node: nbrdict}
    pred : a dict-of-dict-of-dict {node: nbrdict}
    The keys for the two dicts should be the same

    See Also
    ========
    UnionAtlas: View into dict-of-dict
    UnionMultiAdjacency: View into dict-of-dict-of-dict-of-dict
    """
    __slots__: Incomplete
    def __getstate__(self): ...
    _succ: Incomplete
    _pred: Incomplete
    def __setstate__(self, state) -> None: ...
    def __init__(self, succ, pred) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self): ...
    def __getitem__(self, nbr): ...
    def copy(self): ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class UnionMultiInner(UnionAtlas):
    """A read-only union of two inner dicts of MultiAdjacencies.

    The two input dict-of-dict-of-dicts represent the union of
    `G.succ[node]` and `G.pred[node]` for MultiDiGraphs.
    Return values are UnionAtlas.
    The inner level of dict is read-write. But the outer levels are read-only.

    See Also
    ========
    UnionAtlas: View into dict-of-dict
    UnionAdjacency:  View into dict-of-dict-of-dict
    UnionMultiAdjacency:  View into dict-of-dict-of-dict-of-dict
    """
    __slots__: Incomplete
    def __getitem__(self, node): ...
    def copy(self): ...

class UnionMultiAdjacency(UnionAdjacency):
    """A read-only union of two dict MultiAdjacencies.

    The two input dict-of-dict-of-dict-of-dicts represent the union of
    `G.succ` and `G.pred` for MultiDiGraphs. Return values are UnionAdjacency.
    The inner level of dict is read-write. But the outer levels are read-only.

    See Also
    ========
    UnionAtlas:  View into dict-of-dict
    UnionMultiInner:  View into dict-of-dict-of-dict
    """
    __slots__: Incomplete
    def __getitem__(self, node): ...

class FilterAtlas(Mapping):
    """A read-only Mapping of Mappings with filtering criteria for nodes.

    It is a view into a dict-of-dict data structure, and it selects only
    nodes that meet the criteria defined by ``NODE_OK``.

    See Also
    ========
    FilterAdjacency
    FilterMultiInner
    FilterMultiAdjacency
    """
    _atlas: Incomplete
    NODE_OK: Incomplete
    def __init__(self, d, NODE_OK) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self): ...
    def __getitem__(self, key): ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class FilterAdjacency(Mapping):
    """A read-only Mapping of Mappings with filtering criteria for nodes and edges.

    It is a view into a dict-of-dict-of-dict data structure, and it selects nodes
    and edges that satisfy specific criteria defined by ``NODE_OK`` and ``EDGE_OK``,
    respectively.

    See Also
    ========
    FilterAtlas
    FilterMultiInner
    FilterMultiAdjacency
    """
    _atlas: Incomplete
    NODE_OK: Incomplete
    EDGE_OK: Incomplete
    def __init__(self, d, NODE_OK, EDGE_OK) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self): ...
    def __getitem__(self, node): ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class FilterMultiInner(FilterAdjacency):
    """A read-only Mapping of Mappings with filtering criteria for nodes and edges.

    It is a view into a dict-of-dict-of-dict-of-dict data structure, and it selects nodes
    and edges that meet specific criteria defined by ``NODE_OK`` and ``EDGE_OK``.

    See Also
    ========
    FilterAtlas
    FilterAdjacency
    FilterMultiAdjacency
    """
    def __iter__(self): ...
    def __getitem__(self, nbr): ...

class FilterMultiAdjacency(FilterAdjacency):
    """A read-only Mapping of Mappings with filtering criteria
    for nodes and edges.

    It is a view into a dict-of-dict-of-dict-of-dict data structure,
    and it selects nodes and edges that satisfy specific criteria
    defined by ``NODE_OK`` and ``EDGE_OK``, respectively.

    See Also
    ========
    FilterAtlas
    FilterAdjacency
    FilterMultiInner
    """
    def __getitem__(self, node): ...
