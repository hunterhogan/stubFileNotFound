__all__ = ['tree_leaves', 'ltree_leaves', 'tree_nodes', 'ltree_nodes']
from collections.abc import Generator
from typing import Any


def tree_leaves(root: Any, follow: Any=..., children: Any=...) -> Generator[Any, Any, None]:
    """Iterates over tree leaves."""
    ...

def ltree_leaves(root: Any, follow: Any=..., children: Any=...) -> list[Any]:
    """Lists tree leaves."""
    ...

def tree_nodes(root: Any, follow: Any=..., children: Any=...) -> Generator[Any, Any, None]:
    """Iterates over all tree nodes."""
    ...

def ltree_nodes(root: Any, follow: Any=..., children: Any=...) -> list[Any]:
    """Lists all tree nodes."""
    ...

