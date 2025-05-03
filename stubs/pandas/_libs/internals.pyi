import _cython_3_0_11
from _typeshed import Incomplete
from pandas._libs.algos import ensure_int64 as ensure_int64
from typing import ClassVar

__reduce_cython__: _cython_3_0_11.cython_function_or_method
__setstate_cython__: _cython_3_0_11.cython_function_or_method
__test__: dict
_unpickle_block: _cython_3_0_11.cython_function_or_method
get_blkno_indexers: _cython_3_0_11.cython_function_or_method
get_blkno_placements: _cython_3_0_11.cython_function_or_method
get_concat_blkno_indexers: _cython_3_0_11.cython_function_or_method
slice_len: _cython_3_0_11.cython_function_or_method
update_blklocs_and_blknos: _cython_3_0_11.cython_function_or_method

class Block:
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    _mgr_locs: Incomplete
    ndim: Incomplete
    refs: Incomplete
    values: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def slice_block_rows(self, *args, **kwargs):
        """
        Perform __getitem__-like specialized to slicing along index.

        Assumes self.ndim == 2
        """
    def __reduce__(self): ...

class BlockManager:
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    _blklocs: Incomplete
    _blknos: Incomplete
    _is_consolidated: Incomplete
    _known_consolidated: Incomplete
    axes: Incomplete
    blocks: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def _post_setstate(self, *args, **kwargs): ...
    def _rebuild_blknos_and_blklocs(self, *args, **kwargs):
        """
        Update mgr._blknos / mgr._blklocs.
        """
    def get_slice(self, *args, **kwargs): ...
    def __reduce__(self): ...

class BlockPlacement:
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    as_array: Incomplete
    as_slice: Incomplete
    indexer: Incomplete
    is_slice_like: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def add(self, *args, **kwargs): ...
    def append(self, *args, **kwargs): ...
    def delete(self, *args, **kwargs): ...
    def increment_above(self, *args, **kwargs):
        """
        Increment any entries of 'loc' or above by one.
        """
    def tile_for_unstack(self, *args, **kwargs):
        """
        Find the new mgr_locs for the un-stacked version of a Block.
        """
    def __getitem__(self, index):
        """Return self[key]."""
    def __iter__(self):
        """Implement iter(self)."""
    def __len__(self) -> int:
        """Return len(self)."""
    def __reduce__(self): ...

class BlockValuesRefs:
    clear_counter: Incomplete
    referenced_blocks: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def _clear_dead_references(self, *args, **kwargs): ...
    def add_index_reference(self, *args, **kwargs):
        """Adds a new reference to our reference collection when creating an index.

                Parameters
                ----------
                index : Index
                    The index that the new reference should point to.
        """
    def add_reference(self, *args, **kwargs):
        """Adds a new reference to our reference collection.

                Parameters
                ----------
                blk : Block
                    The block that the new references should point to.
        """
    def has_reference(self, *args, **kwargs):
        """Checks if block has foreign references.

                A reference is only relevant if it is still alive. The reference to
                ourselves does not count.

                Returns
                -------
                bool
        """
    def __reduce__(self): ...
