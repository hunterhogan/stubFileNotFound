import _cython_3_0_11
from _typeshed import Incomplete
from typing import ClassVar

__pyx_unpickle_BlockMerge: _cython_3_0_11.cython_function_or_method
__pyx_unpickle_BlockUnion: _cython_3_0_11.cython_function_or_method
__pyx_unpickle_SparseIndex: _cython_3_0_11.cython_function_or_method
__test__: dict
get_blocks: _cython_3_0_11.cython_function_or_method
make_mask_object_ndarray: _cython_3_0_11.cython_function_or_method
sparse_add_float64: _cython_3_0_11.cython_function_or_method
sparse_add_int64: _cython_3_0_11.cython_function_or_method
sparse_and_int64: _cython_3_0_11.cython_function_or_method
sparse_and_uint8: _cython_3_0_11.cython_function_or_method
sparse_div_float64: _cython_3_0_11.cython_function_or_method
sparse_div_int64: _cython_3_0_11.cython_function_or_method
sparse_eq_float64: _cython_3_0_11.cython_function_or_method
sparse_eq_int64: _cython_3_0_11.cython_function_or_method
sparse_floordiv_float64: _cython_3_0_11.cython_function_or_method
sparse_floordiv_int64: _cython_3_0_11.cython_function_or_method
sparse_ge_float64: _cython_3_0_11.cython_function_or_method
sparse_ge_int64: _cython_3_0_11.cython_function_or_method
sparse_gt_float64: _cython_3_0_11.cython_function_or_method
sparse_gt_int64: _cython_3_0_11.cython_function_or_method
sparse_le_float64: _cython_3_0_11.cython_function_or_method
sparse_le_int64: _cython_3_0_11.cython_function_or_method
sparse_lt_float64: _cython_3_0_11.cython_function_or_method
sparse_lt_int64: _cython_3_0_11.cython_function_or_method
sparse_mod_float64: _cython_3_0_11.cython_function_or_method
sparse_mod_int64: _cython_3_0_11.cython_function_or_method
sparse_mul_float64: _cython_3_0_11.cython_function_or_method
sparse_mul_int64: _cython_3_0_11.cython_function_or_method
sparse_ne_float64: _cython_3_0_11.cython_function_or_method
sparse_ne_int64: _cython_3_0_11.cython_function_or_method
sparse_or_int64: _cython_3_0_11.cython_function_or_method
sparse_or_uint8: _cython_3_0_11.cython_function_or_method
sparse_pow_float64: _cython_3_0_11.cython_function_or_method
sparse_pow_int64: _cython_3_0_11.cython_function_or_method
sparse_sub_float64: _cython_3_0_11.cython_function_or_method
sparse_sub_int64: _cython_3_0_11.cython_function_or_method
sparse_truediv_float64: _cython_3_0_11.cython_function_or_method
sparse_truediv_int64: _cython_3_0_11.cython_function_or_method
sparse_xor_int64: _cython_3_0_11.cython_function_or_method
sparse_xor_uint8: _cython_3_0_11.cython_function_or_method

class BlockIndex(SparseIndex):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    blengths: Incomplete
    blocs: Incomplete
    indices: Incomplete
    length: Incomplete
    nblocks: Incomplete
    nbytes: Incomplete
    ngaps: Incomplete
    npoints: Incomplete
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def equals(self, *args, **kwargs): ...
    def intersect(self, *args, **kwargs):
        """
        Intersect two BlockIndex objects

        Returns
        -------
        BlockIndex
        """
    def lookup(self, *args, **kwargs):
        """
        Return the internal location if value exists on given index.
        Return -1 otherwise.
        """
    def lookup_array(self, *args, **kwargs):
        """
        Vectorized lookup, returns ndarray[int32_t]
        """
    def make_union(self, *args, **kwargs):
        """
        Combine together two BlockIndex objects, accepting indices if contained
        in one or the other

        Parameters
        ----------
        other : SparseIndex

        Notes
        -----
        union is a protected keyword in Cython, hence make_union

        Returns
        -------
        BlockIndex
        """
    def to_block_index(self, *args, **kwargs): ...
    def to_int_index(self, *args, **kwargs): ...
    def __reduce__(self): ...

class IntIndex(SparseIndex):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    indices: Incomplete
    length: Incomplete
    nbytes: Incomplete
    ngaps: Incomplete
    npoints: Incomplete
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def equals(self, *args, **kwargs): ...
    def intersect(self, *args, **kwargs): ...
    def lookup(self, *args, **kwargs):
        """
        Return the internal location if value exists on given index.
        Return -1 otherwise.
        """
    def lookup_array(self, *args, **kwargs):
        """
        Vectorized lookup, returns ndarray[int32_t]
        """
    def make_union(self, *args, **kwargs): ...
    def to_block_index(self, *args, **kwargs): ...
    def to_int_index(self, *args, **kwargs): ...
    def __reduce__(self): ...

class SparseIndex:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def __reduce__(self): ...
    def __reduce_cython__(self, *args, **kwargs): ...
    def __setstate_cython__(self, *args, **kwargs): ...
