from _typeshed import Incomplete
from collections.abc import Generator

class UnionFind:
    """Union-find data structure.

    Each unionFind instance X maintains a family of disjoint sets of
    hashable objects, supporting the following two methods:

    - X[item] returns a name for the set containing the given item.
      Each set is named by an arbitrarily-chosen one of its members; as
      long as the set remains unchanged it will keep the same name. If
      the item is not yet part of a set in X, a new singleton set is
      created for it.

    - X.union(item1, item2, ...) merges the sets containing each item
      into a single larger set.  If any item is not yet part of a set
      in X, it is added to X as one of the members of the merged set.

      Union-find data structure. Based on Josiah Carlson's code,
      https://code.activestate.com/recipes/215912/
      with significant additional changes by D. Eppstein.
      http://www.ics.uci.edu/~eppstein/PADS/UnionFind.py

    """
    parents: Incomplete
    weights: Incomplete
    def __init__(self, elements: Incomplete | None = None) -> None:
        """Create a new empty union-find structure.

        If *elements* is an iterable, this structure will be initialized
        with the discrete partition on the given set of elements.

        """
    def __getitem__(self, object):
        """Find and return the name of the set containing the object."""
    def __iter__(self):
        """Iterate through all items ever found or unioned by this structure."""
    def to_sets(self) -> Generator[Incomplete, Incomplete]:
        '''Iterates over the sets stored in this structure.

        For example::

            >>> partition = UnionFind("xyz")
            >>> sorted(map(sorted, partition.to_sets()))
            [[\'x\'], [\'y\'], [\'z\']]
            >>> partition.union("x", "y")
            >>> sorted(map(sorted, partition.to_sets()))
            [[\'x\', \'y\'], [\'z\']]

        '''
    def union(self, *objects):
        """Find the sets containing the objects and merge them all."""
