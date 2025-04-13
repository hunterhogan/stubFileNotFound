from _typeshed import Incomplete

__all__ = ['MinHeap', 'PairingHeap', 'BinaryHeap']

class MinHeap:
    """Base class for min-heaps.

    A MinHeap stores a collection of key-value pairs ordered by their values.
    It supports querying the minimum pair, inserting a new pair, decreasing the
    value in an existing pair and deleting the minimum pair.
    """
    class _Item:
        """Used by subclassess to represent a key-value pair."""
        __slots__: Incomplete
        key: Incomplete
        value: Incomplete
        def __init__(self, key, value) -> None: ...
        def __repr__(self) -> str: ...
    _dict: Incomplete
    def __init__(self) -> None:
        """Initialize a new min-heap."""
    def min(self) -> None:
        """Query the minimum key-value pair.

        Returns
        -------
        key, value : tuple
            The key-value pair with the minimum value in the heap.

        Raises
        ------
        NetworkXError
            If the heap is empty.
        """
    def pop(self) -> None:
        """Delete the minimum pair in the heap.

        Returns
        -------
        key, value : tuple
            The key-value pair with the minimum value in the heap.

        Raises
        ------
        NetworkXError
            If the heap is empty.
        """
    def get(self, key, default: Incomplete | None = None) -> None:
        """Returns the value associated with a key.

        Parameters
        ----------
        key : hashable object
            The key to be looked up.

        default : object
            Default value to return if the key is not present in the heap.
            Default value: None.

        Returns
        -------
        value : object.
            The value associated with the key.
        """
    def insert(self, key, value, allow_increase: bool = False) -> None:
        """Insert a new key-value pair or modify the value in an existing
        pair.

        Parameters
        ----------
        key : hashable object
            The key.

        value : object comparable with existing values.
            The value.

        allow_increase : bool
            Whether the value is allowed to increase. If False, attempts to
            increase an existing value have no effect. Default value: False.

        Returns
        -------
        decreased : bool
            True if a pair is inserted or the existing value is decreased.
        """
    def __nonzero__(self):
        """Returns whether the heap if empty."""
    def __bool__(self) -> bool:
        """Returns whether the heap if empty."""
    def __len__(self) -> int:
        """Returns the number of key-value pairs in the heap."""
    def __contains__(self, key) -> bool:
        """Returns whether a key exists in the heap.

        Parameters
        ----------
        key : any hashable object.
            The key to be looked up.
        """

class PairingHeap(MinHeap):
    """A pairing heap."""
    class _Node(MinHeap._Item):
        """A node in a pairing heap.

        A tree in a pairing heap is stored using the left-child, right-sibling
        representation.
        """
        __slots__: Incomplete
        left: Incomplete
        next: Incomplete
        prev: Incomplete
        parent: Incomplete
        def __init__(self, key, value) -> None: ...
    _root: Incomplete
    def __init__(self) -> None:
        """Initialize a pairing heap."""
    def min(self): ...
    def pop(self): ...
    def get(self, key, default: Incomplete | None = None): ...
    def insert(self, key, value, allow_increase: bool = False): ...
    def _link(self, root, other):
        """Link two nodes, making the one with the smaller value the parent of
        the other.
        """
    def _merge_children(self, root):
        """Merge the subtrees of the root using the standard two-pass method.
        The resulting subtree is detached from the root.
        """
    def _cut(self, node) -> None:
        """Cut a node from its parent."""

class BinaryHeap(MinHeap):
    """A binary heap."""
    _heap: Incomplete
    _count: Incomplete
    def __init__(self) -> None:
        """Initialize a binary heap."""
    def min(self): ...
    def pop(self): ...
    def get(self, key, default: Incomplete | None = None): ...
    def insert(self, key, value, allow_increase: bool = False): ...
