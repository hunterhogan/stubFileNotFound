
from typing import FrozenSet, Generic, Tuple, TypeVar

T = TypeVar("T")
class DisjointSet(Generic[T]):
    def __init__(self) -> None:
        ...

    def make_set(self, e: T): # -> None:
        ...

    def find(self, e: T): # -> T:
        ...

    def union(self, x: T, y: T): # -> None:
        ...

    def sets(self) -> frozenset[frozenset[T]]:
        ...

    def sorted(self) -> tuple[tuple[T, ...], ...]:
        """Sorted tuple of sorted tuples edition of sets()."""
