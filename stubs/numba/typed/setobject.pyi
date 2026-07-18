from _typeshed import Incomplete
from enum import IntEnum
from numba.core import cgutils as cgutils, types as types, typing as typing
from numba.core.errors import TypingError as TypingError
from numba.core.extending import intrinsic as intrinsic, lower_builtin as lower_builtin, make_attribute_wrapper as make_attribute_wrapper, models as models, overload as overload, overload_method as overload_method, register_model as register_model
from numba.core.imputils import RefType as RefType, impl_ret_borrowed as impl_ret_borrowed, iternext_impl as iternext_impl
from numba.core.types import SetIterableType as SetIterableType, SetIteratorType as SetIteratorType, SetType as SetType, Type as Type

ll_set_type: Incomplete
ll_setiter_type: Incomplete
ll_voidptr_type: Incomplete
ll_status: Incomplete
ll_ssize_t: Incomplete
ll_hash = ll_ssize_t
ll_bytes: Incomplete

class Status(IntEnum):
    """Status code for set operations.
    """
    ENTRY_PRESENT = 1
    OK = 0
    ERR_KEY_NOT_FOUND = -1
    ERR_SET_MUTATED = -2
    ERR_ITER_EXHAUSTED = -3
    ERR_SET_EMPTY = -4
    ERR_CMP_FAILED = -5

def new_set(key):
    """Construct a new set.

    Parameters
    ----------
    key : TypeRef
        Key type of the new set.
    """

class SetModel(models.StructModel):
    def __init__(self, dmm, fe_type) -> None: ...

class SetIterModel(models.StructModel):
    def __init__(self, dmm, fe_type) -> None: ...

def impl_new_set(value, n_keys: int = 8):
    """Creates a new set with *value* as the type
    of the set value.
    """
def impl_len(setp):
    """len(set)
    """
def impl_len_iters(s): ...
def impl_set_add(s, key): ...
def impl_discard(s, key): ...
def impl_contains(s, key): ...
def impl_equal(set_a, set_b): ...
def impl_not_equal(da, db): ...
def impl_copy(s): ...
def impl_iterable_getiter(context, builder, sig, args): ...
def impl_set_getiter(context, builder, sig, args): ...
def impl_iterator_iternext(context, builder, sig, args, result) -> None: ...
