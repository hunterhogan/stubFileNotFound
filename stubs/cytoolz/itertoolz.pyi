import _cython_3_0_11
from typing import Any, ClassVar

__pyx_capi__: dict
__pyx_unpickle__getter_null: _cython_3_0_11.cython_function_or_method
__reduce_cython__: _cython_3_0_11.cython_function_or_method
__setstate_cython__: _cython_3_0_11.cython_function_or_method
__test__: dict
concat: _cython_3_0_11.cython_function_or_method
concatv: _cython_3_0_11.cython_function_or_method
cons: _cython_3_0_11.cython_function_or_method
count: _cython_3_0_11.cython_function_or_method
diff: _cython_3_0_11.cython_function_or_method
drop: _cython_3_0_11.cython_function_or_method
first: _cython_3_0_11.cython_function_or_method
frequencies: _cython_3_0_11.cython_function_or_method
get: _cython_3_0_11.cython_function_or_method
getter: _cython_3_0_11.cython_function_or_method
groupby: _cython_3_0_11.cython_function_or_method
identity: _cython_3_0_11.cython_function_or_method
isdistinct: _cython_3_0_11.cython_function_or_method
isiterable: _cython_3_0_11.cython_function_or_method
join: _cython_3_0_11.cython_function_or_method
last: _cython_3_0_11.cython_function_or_method
mapcat: _cython_3_0_11.cython_function_or_method
merge_sorted: _cython_3_0_11.cython_function_or_method
no_default: str
no_pad: str
nth: _cython_3_0_11.cython_function_or_method
partition: _cython_3_0_11.cython_function_or_method
peek: _cython_3_0_11.cython_function_or_method
peekn: _cython_3_0_11.cython_function_or_method
pluck: _cython_3_0_11.cython_function_or_method
reduceby: _cython_3_0_11.cython_function_or_method
rest: _cython_3_0_11.cython_function_or_method
second: _cython_3_0_11.cython_function_or_method
tail: _cython_3_0_11.cython_function_or_method
take: _cython_3_0_11.cython_function_or_method
take_nth: _cython_3_0_11.cython_function_or_method
topk: _cython_3_0_11.cython_function_or_method
unique: _cython_3_0_11.cython_function_or_method

class _diff_identity:
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __iter__(self):
        """Implement iter(self)."""
    def __next__(self): ...
    def __reduce__(self):
        """_diff_identity.__reduce_cython__(self)"""

class _diff_key:
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __iter__(self):
        """Implement iter(self)."""
    def __next__(self): ...
    def __reduce__(self):
        """_diff_key.__reduce_cython__(self)"""

class _getter_index:
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __call__(self, *args, **kwargs):
        """Call self as a function."""
    def __reduce__(self):
        """_getter_index.__reduce_cython__(self)"""

class _getter_list:
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __call__(self, *args, **kwargs):
        """Call self as a function."""
    def __reduce__(self):
        """_getter_list.__reduce_cython__(self)"""

class _getter_null:
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __call__(self, *args, **kwargs):
        """Call self as a function."""
    def __reduce__(self):
        """_getter_null.__reduce_cython__(self)"""
    def __reduce_cython__(self) -> Any:
        """_getter_null.__reduce_cython__(self)"""
    def __setstate_cython__(self, __pyx_state) -> Any:
        """_getter_null.__setstate_cython__(self, __pyx_state)"""

class _inner_join(_join):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __next__(self): ...
    def __reduce__(self):
        """_inner_join.__reduce_cython__(self)"""

class _inner_join_index(_inner_join):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __reduce__(self):
        """_inner_join_index.__reduce_cython__(self)"""

class _inner_join_indices(_inner_join):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __reduce__(self):
        """_inner_join_indices.__reduce_cython__(self)"""

class _inner_join_key(_inner_join):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __reduce__(self):
        """_inner_join_key.__reduce_cython__(self)"""

class _join:
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __iter__(self):
        """Implement iter(self)."""
    def __reduce__(self):
        """_join.__reduce_cython__(self)"""

class _left_outer_join(_join):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __next__(self): ...
    def __reduce__(self):
        """_left_outer_join.__reduce_cython__(self)"""

class _left_outer_join_index(_left_outer_join):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __reduce__(self):
        """_left_outer_join_index.__reduce_cython__(self)"""

class _left_outer_join_indices(_left_outer_join):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __reduce__(self):
        """_left_outer_join_indices.__reduce_cython__(self)"""

class _left_outer_join_key(_left_outer_join):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __reduce__(self):
        """_left_outer_join_key.__reduce_cython__(self)"""

class _merge_sorted:
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __iter__(self):
        """Implement iter(self)."""
    def __next__(self): ...
    def __reduce__(self):
        """_merge_sorted.__reduce_cython__(self)"""

class _merge_sorted_key:
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __iter__(self):
        """Implement iter(self)."""
    def __next__(self): ...
    def __reduce__(self):
        """_merge_sorted_key.__reduce_cython__(self)"""

class _outer_join(_join):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __next__(self): ...
    def __reduce__(self):
        """_outer_join.__reduce_cython__(self)"""

class _outer_join_index(_outer_join):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __reduce__(self):
        """_outer_join_index.__reduce_cython__(self)"""

class _outer_join_indices(_outer_join):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __reduce__(self):
        """_outer_join_indices.__reduce_cython__(self)"""

class _outer_join_key(_outer_join):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __reduce__(self):
        """_outer_join_key.__reduce_cython__(self)"""

class _pluck_index:
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __iter__(self):
        """Implement iter(self)."""
    def __next__(self): ...
    def __reduce__(self):
        """_pluck_index.__reduce_cython__(self)"""

class _pluck_index_default:
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __iter__(self):
        """Implement iter(self)."""
    def __next__(self): ...
    def __reduce__(self):
        """_pluck_index_default.__reduce_cython__(self)"""

class _pluck_list:
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __iter__(self):
        """Implement iter(self)."""
    def __next__(self): ...
    def __reduce__(self):
        """_pluck_list.__reduce_cython__(self)"""

class _pluck_list_default:
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __iter__(self):
        """Implement iter(self)."""
    def __next__(self): ...
    def __reduce__(self):
        """_pluck_list_default.__reduce_cython__(self)"""

class _right_outer_join(_join):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __next__(self): ...
    def __reduce__(self):
        """_right_outer_join.__reduce_cython__(self)"""

class _right_outer_join_index(_right_outer_join):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __reduce__(self):
        """_right_outer_join_index.__reduce_cython__(self)"""

class _right_outer_join_indices(_right_outer_join):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __reduce__(self):
        """_right_outer_join_indices.__reduce_cython__(self)"""

class _right_outer_join_key(_right_outer_join):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __reduce__(self):
        """_right_outer_join_key.__reduce_cython__(self)"""

class _unique_identity:
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __iter__(self):
        """Implement iter(self)."""
    def __next__(self): ...
    def __reduce__(self):
        """_unique_identity.__reduce_cython__(self)"""

class _unique_key:
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __iter__(self):
        """Implement iter(self)."""
    def __next__(self): ...
    def __reduce__(self):
        """_unique_key.__reduce_cython__(self)"""

class accumulate:
    @classmethod
    def __init__(cls, binop, seq, initial=...) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __iter__(self):
        """Implement iter(self)."""
    def __next__(self): ...
    def __reduce__(self):
        """accumulate.__reduce_cython__(self)"""

class interleave:
    @classmethod
    def __init__(cls, seqs) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __iter__(self):
        """Implement iter(self)."""
    def __next__(self): ...
    def __reduce__(self):
        """interleave.__reduce_cython__(self)"""

class interpose:
    @classmethod
    def __init__(cls, el, seq) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __iter__(self):
        """Implement iter(self)."""
    def __next__(self): ...
    def __reduce__(self):
        """interpose.__reduce_cython__(self)"""

class iterate:
    @classmethod
    def __init__(cls, func, x) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __iter__(self):
        """Implement iter(self)."""
    def __next__(self): ...
    def __reduce__(self):
        """iterate.__reduce_cython__(self)"""

class partition_all:
    @classmethod
    def __init__(cls, n, seq) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __iter__(self):
        """Implement iter(self)."""
    def __next__(self): ...
    def __reduce__(self):
        """partition_all.__reduce_cython__(self)"""

class random_sample:
    @classmethod
    def __init__(cls, prob, seq, random_state=...) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __iter__(self):
        """Implement iter(self)."""
    def __next__(self): ...
    def __reduce__(self):
        """random_sample.__reduce_cython__(self)"""

class remove:
    @classmethod
    def __init__(cls, predicate, seq) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __iter__(self):
        """Implement iter(self)."""
    def __next__(self): ...
    def __reduce__(self):
        """remove.__reduce_cython__(self)"""

class sliding_window:
    @classmethod
    def __init__(cls, n, seq) -> Any:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __iter__(self):
        """Implement iter(self)."""
    def __next__(self): ...
    def __reduce__(self):
        """sliding_window.__reduce_cython__(self)"""
